from __future__ import annotations

import json
from typing import Any

from litscout.core.models import CanonicalPaper

from .models import SkillProfile


def _truncate(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def canonical_input(paper: CanonicalPaper) -> dict[str, Any]:
    authors = paper.authors[:8] if paper.authors else []
    abstract = _truncate(paper.abstract, 1500) if paper.abstract else None
    sources = []
    for src in paper.sources:
        sources.append(
            {
                "source_name": src.source_name,
                "source_url": src.source_url,
                "extra": src.extra,
            }
        )
    return {
        "paper_id": str(paper.canonical_id),
        "title": paper.title,
        "authors": authors,
        "year": paper.year,
        "abstract": abstract,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "url_primary": paper.url_primary,
        "venue": paper.venue,
        "sources": sources,
    }


def skill_summary(skill: SkillProfile) -> dict[str, Any]:
    return {
        "language": skill.language,
        "scoring": {
            "weights": skill.scoring.weights,
            "recency_half_life_years": skill.scoring.recency_half_life_years,
        },
        "classification": {
            "taxonomy": skill.classification.taxonomy,
            "max_tags": skill.classification.max_tags,
        },
        "output": {
            "uncertainty_style": skill.output.uncertainty_style,
            "evidence_style": skill.output.evidence_style,
        },
    }


def _goals_payload(skill: SkillProfile) -> list[dict[str, Any]]:
    return [
        {
            "goal_id": g.goal_id,
            "name": g.name,
            "description": g.description,
            "signals": g.signals,
            "negative_signals": g.negative_signals,
        }
        for g in skill.goals
    ]


def build_prompt_single(paper: CanonicalPaper, goal: str, skill: SkillProfile) -> str:
    payload = {
        "research_goal": goal,
        "goals": _goals_payload(skill),
        "skill": skill_summary(skill),
        "paper": canonical_input(paper),
    }
    return (
        "You are an expert research analyst. Only use the provided metadata. "
        "Do not assume you read the full paper. Return STRICT JSON only.\n"
        "Output ONLY a single JSON object. Do NOT include any extra text, markdown, or comments.\n"
        "Return a single JSON object that matches the provided schema.\n"
        "Required fields: paper_id, summary, methods, priority, score, goal_scores.\n"
        "If goals are provided, you MUST output goal_scores with entries for all goals in order (G1,G2,G3...).\n"
        "Each goal score must be a decimal between 0 and 1 with two digits (e.g., 0.00, 0.50, 1.00).\n"
        "Score must be between 0 and 1. Priority must be one of P0/P1/P2/P3.\n"
        "Summary should be one sentence. Methods should be short phrases.\n"
        "Output format example (JSON only):\n"
        "{\"paper_id\":\"...\",\"summary\":\"...\",\"methods\":[\"...\"],\"priority\":\"P2\",\"score\":0.50,"
        "\"goal_scores\":[{\"goal_id\":\"G1\",\"score\":0.50},{\"goal_id\":\"G2\",\"score\":0.00},{\"goal_id\":\"G3\",\"score\":1.00}]}\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def build_prompt_batch(papers: list[CanonicalPaper], goal: str, skill: SkillProfile) -> str:
    payload = {
        "research_goal": goal,
        "goals": _goals_payload(skill),
        "skill": skill_summary(skill),
        "papers": [canonical_input(p) for p in papers],
    }
    return (
        "You are an expert research analyst. Only use the provided metadata. "
        "Do not assume you read the full papers. Return STRICT JSON only.\n"
        "Output ONLY a single JSON object. Do NOT include any extra text, markdown, or comments.\n"
        "Return a single JSON object with an 'items' array matching the schema.\n"
        "Each item must include the same paper_id as the input.\n"
        "Required fields per item: paper_id, summary, methods, priority, score, goal_scores.\n"
        "If goals are provided, you MUST output goal_scores with entries for all goals in order (G1,G2,G3...).\n"
        "Each goal score must be a decimal between 0 and 1 with two digits (e.g., 0.00, 0.50, 1.00).\n"
        "Score must be between 0 and 1. Priority must be one of P0/P1/P2/P3.\n"
        "Summary should be one sentence. Methods should be short phrases.\n"
        "Output format example (JSON only):\n"
        "{\"items\":[{\"paper_id\":\"...\",\"summary\":\"...\",\"methods\":[\"...\"],\"priority\":\"P2\",\"score\":0.50,"
        "\"goal_scores\":[{\"goal_id\":\"G1\",\"score\":0.50},{\"goal_id\":\"G2\",\"score\":0.00},{\"goal_id\":\"G3\",\"score\":1.00}]}]}\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )
