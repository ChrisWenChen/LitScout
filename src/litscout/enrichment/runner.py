from __future__ import annotations

import hashlib
import json
from typing import Iterable
import asyncio

from pydantic import ValidationError

from litscout.core.models import CanonicalPaper
from litscout.core.rate_limit import RateLimiter
from litscout.ranking import build_ranking
from litscout.classification import rule_classify

from .llm import LLMClient
from .models import (
    CanonicalSnapshot,
    ClassificationResult,
    EnrichedPaper,
    LLMOutputBatch,
    LiteInsight,
    LiteGoalScore,
    Provenance,
    RankingBreakdown,
    SkillProfile,
)
from .prompt import build_prompt_batch, build_prompt_single, canonical_input


def _hash_input(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _snapshot(paper: CanonicalPaper) -> CanonicalSnapshot:
    return CanonicalSnapshot(
        canonical_id=str(paper.canonical_id),
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        abstract=paper.abstract,
        doi=paper.doi,
        arxiv_id=paper.arxiv_id,
        url_primary=paper.url_primary,
        venue=paper.venue,
        sources=[
            {
                "source_name": s.source_name,
                "source_url": s.source_url,
                "extra": s.extra,
            }
            for s in paper.sources
        ],
    )


def _evidence_fields(paper: CanonicalPaper) -> list[str]:
    fields = ["title", "authors", "year", "venue", "abstract", "sources"]
    if paper.doi:
        fields.append("doi")
    if paper.arxiv_id:
        fields.append("arxiv_id")
    return fields


def _build_provenance(
    skill: SkillProfile,
    prompt_hash: str,
    input_hash: str,
    time_window: str | None,
    retries: int,
) -> Provenance:
    return Provenance(
        provider=skill.llm.provider,
        model=skill.llm.model,
        temperature=skill.llm.temperature,
        seed=skill.llm.seed,
        prompt_hash=prompt_hash,
        skill_hash=skill.skill_hash or "",
        input_hash=input_hash,
        time_window=time_window,
        evidence=[],
        retries=retries,
    )


def _merge_classification(paper: CanonicalPaper) -> ClassificationResult:
    rule_path, rule_rationale = rule_classify(paper)
    rationale = []
    rationale.append(f"rule_path={'/'.join(rule_path)}")
    rationale.extend(rule_rationale)
    return ClassificationResult(category_path=rule_path, tags=[], rationale=rationale)


def _keyword_goal_score(text: str, signals: list[str], negative: list[str]) -> float:
    if not signals and not negative:
        return 0.0
    pos = sum(1 for s in signals if s and s.lower() in text)
    neg = sum(1 for s in negative if s and s.lower() in text)
    if pos == 0 and neg == 0:
        return 0.0
    denom = max(1, len(signals))
    score = (pos - neg) / float(denom)
    return max(0.0, min(1.0, score))


def _ensure_goal_scores(ai: LiteInsight, skill: SkillProfile, paper: CanonicalPaper) -> None:
    if not skill.goals:
        return
    existing = {gs.goal_id for gs in ai.goal_scores}
    text = f"{paper.title or ''} {paper.abstract or ''}".lower()
    for goal in skill.goals:
        if goal.goal_id not in existing:
            score = _keyword_goal_score(text, goal.signals, goal.negative_signals)
            ai.goal_scores.append(LiteGoalScore(goal_id=goal.goal_id, score=score))
    # enforce fixed order and full coverage
    order = {goal.goal_id: idx for idx, goal in enumerate(skill.goals)}
    ai.goal_scores.sort(key=lambda gs: order.get(gs.goal_id, 999))


def _max_goal_score(ai: LiteInsight) -> float:
    if not ai.goal_scores:
        return 0.0
    return max(gs.score for gs in ai.goal_scores)


def _finalize_enriched(
    paper: CanonicalPaper,
    ai: LiteInsight,
    skill: SkillProfile,
    time_window: str | None,
    prompt_hash: str,
    input_hash: str,
    retries: int,
    use_goals: bool,
    goal_weight: float,
) -> EnrichedPaper:
    _ensure_goal_scores(ai, skill, paper)
    max_goal_score = _max_goal_score(ai)
    ranking_dict = build_ranking(
        paper,
        skill.scoring.weights,
        ai.score,
        [f"llm_score={ai.score:.3f}"],
        skill.scoring.recency_half_life_years,
    )
    if use_goals:
        base_total = ranking_dict["total"]
        ranking_dict["total"] = max(
            0.0,
            min(1.0, (1 - goal_weight) * base_total + goal_weight * max_goal_score),
        )
        ranking_dict["rationale"].append(
            f"goal_match_score={max_goal_score:.3f} weighted by {goal_weight:.2f}"
        )
    ranking = RankingBreakdown.model_validate(ranking_dict)
    classification = _merge_classification(paper)
    provenance = _build_provenance(skill, prompt_hash, input_hash, time_window, retries)
    provenance.evidence = _evidence_fields(paper)
    return EnrichedPaper(
        paper_id=str(paper.canonical_id),
        canonical=_snapshot(paper),
        ai=ai,
        ranking=ranking,
        classification=classification,
        provenance=provenance,
    )


def _error_enriched(
    paper: CanonicalPaper,
    skill: SkillProfile,
    time_window: str | None,
    prompt_hash: str,
    input_hash: str,
    retries: int,
    error: str,
) -> EnrichedPaper:
    provenance = _build_provenance(skill, prompt_hash, input_hash, time_window, retries)
    provenance.evidence = _evidence_fields(paper)
    return EnrichedPaper(
        paper_id=str(paper.canonical_id),
        canonical=_snapshot(paper),
        provenance=provenance,
        error=error,
    )


def _validate_single(payload: dict) -> LiteInsight:
    return LiteInsight.model_validate(payload)


def _validate_batch(payload: dict) -> LLMOutputBatch:
    return LLMOutputBatch.model_validate(payload)


def _batch_iter(items: list[CanonicalPaper], batch_size: int) -> Iterable[list[CanonicalPaper]]:
    if batch_size <= 1:
        for item in items:
            yield [item]
        return
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


async def enrich_papers(
    papers: list[CanonicalPaper],
    goal: str,
    skill: SkillProfile,
    client: LLMClient,
    batch_size: int = 1,
    time_window: str | None = None,
    max_retries: int = 2,
    use_goals: bool = False,
    goal_weight: float = 0.2,
    max_concurrency: int = 1,
) -> list[EnrichedPaper]:
    async def _process_batch(batch: list[CanonicalPaper]) -> list[EnrichedPaper]:
        if len(batch) == 1:
            prompt = build_prompt_single(batch[0], goal, skill)
            schema = LiteInsight.model_json_schema()
        else:
            prompt = build_prompt_batch(batch, goal, skill)
            schema = LLMOutputBatch.model_json_schema()

        prompt_hash = client.prompt_hash(prompt)
        input_hashes = {
            str(p.canonical_id): _hash_input({"goal": goal, "paper": canonical_input(p)})
            for p in batch
        }

        attempt = 0
        last_error: str | None = None
        while attempt <= max_retries:
            try:
                payload = await client.generate_json(prompt, schema)
                if len(batch) == 1:
                    ai_items = [_validate_single(payload)]
                else:
                    ai_items = _validate_batch(payload).items
                break
            except (ValidationError, ValueError, RuntimeError) as exc:
                last_error = str(exc)
                attempt += 1
                if attempt > max_retries:
                    ai_items = []
                    break

        by_id = {item.paper_id: item for item in ai_items}
        enriched_batch: list[EnrichedPaper] = []
        for paper in batch:
            paper_id = str(paper.canonical_id)
            input_hash = input_hashes[paper_id]
            if paper_id in by_id:
                enriched = _finalize_enriched(
                    paper,
                    by_id[paper_id],
                    skill,
                    time_window,
                    prompt_hash,
                    input_hash,
                    attempt,
                    use_goals,
                    goal_weight,
                )
            else:
                enriched = _error_enriched(
                    paper,
                    skill,
                    time_window,
                    prompt_hash,
                    input_hash,
                    attempt,
                    last_error or "LLM output validation failed",
                )
            enriched_batch.append(enriched)
        return enriched_batch

    batches = list(_batch_iter(papers, batch_size))
    print("=" * 60)
    print(
        "LLM batching",
        {
            "total_papers": len(papers),
            "batch_size": batch_size,
            "total_batches": len(batches),
        },
    )
    print("=" * 60)
    if max_concurrency <= 1:
        results: list[EnrichedPaper] = []
        for idx, batch in enumerate(batches, start=1):
            print("-" * 60)
            print("LLM batch start", {"batch_index": idx, "total_batches": len(batches)})
            print("-" * 60)
            results.extend(await _process_batch(batch))
        return results

    sem = asyncio.Semaphore(max_concurrency)

    async def _bounded(batch: list[CanonicalPaper], idx: int) -> list[EnrichedPaper]:
        async with sem:
            print("-" * 60)
            print("LLM batch start", {"batch_index": idx, "total_batches": len(batches)})
            print("-" * 60)
            return await _process_batch(batch)

    tasks = [
        asyncio.create_task(_bounded(batch, idx)) for idx, batch in enumerate(batches, start=1)
    ]
    batch_results = await asyncio.gather(*tasks)
    results = []
    for batch in batch_results:
        results.extend(batch)
    return results


def load_jsonl(path: str) -> list[CanonicalPaper]:
    papers: list[CanonicalPaper] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(CanonicalPaper.model_validate_json(line))
    return papers


def dump_enriched_jsonl(path: str, enriched: list[EnrichedPaper]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in enriched:
            f.write(item.model_dump_json())
            f.write("\n")
