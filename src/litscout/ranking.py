from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from litscout.core.models import CanonicalPaper


def _extract_citations(paper: CanonicalPaper) -> int | None:
    for source in paper.sources:
        if source.source_name == "semantic_scholar":
            value = source.extra.get("citationCount")
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
    return None


def score_recency(year: int | None, half_life_years: float) -> float:
    if not year:
        return 0.0
    current_year = datetime.now(timezone.utc).year
    age = max(0, current_year - year)
    if half_life_years <= 0:
        return 0.0
    return float(math.exp(-math.log(2) * age / half_life_years))


def score_importance(paper: CanonicalPaper) -> tuple[float, list[str]]:
    citations = _extract_citations(paper)
    rationale: list[str] = []
    if citations is None:
        score = 0.2
        rationale.append("citationCount missing; using baseline importance 0.2")
    else:
        score = min(1.0, math.log1p(citations) / math.log1p(1000))
        rationale.append(f"citationCount={citations} scaled via log1p/1000")

    venue = (paper.venue or "").lower()
    if venue:
        if any(k in venue for k in ["nature", "science", "cell", "neurips", "icml", "iclr"]):
            score = min(1.0, score + 0.1)
            rationale.append("venue bonus applied")
    return score, rationale


def score_relevance(relevance: float) -> float:
    return max(0.0, min(1.0, relevance))


def total_score(weights: dict[str, float], relevance: float, importance: float, recency: float) -> float:
    return max(0.0, min(1.0, (
        weights.get("relevance", 0.0) * relevance
        + weights.get("importance", 0.0) * importance
        + weights.get("recency", 0.0) * recency
    )))


def ranking_rationale(
    relevance_rationale: list[str],
    importance_rationale: list[str],
    recency_year: int | None,
    recency_score: float,
) -> list[str]:
    rationale: list[str] = []
    rationale.extend(relevance_rationale)
    rationale.extend(importance_rationale)
    if recency_year:
        rationale.append(f"recency based on year={recency_year}, score={recency_score:.3f}")
    else:
        rationale.append("recency unavailable (missing year)")
    return rationale


def build_ranking(
    paper: CanonicalPaper,
    weights: dict[str, float],
    relevance: float,
    relevance_rationale: list[str],
    half_life_years: float,
) -> dict[str, Any]:
    relevance_score = score_relevance(relevance)
    importance_score, importance_rationale = score_importance(paper)
    recency_score = score_recency(paper.year, half_life_years)
    total = total_score(weights, relevance_score, importance_score, recency_score)
    rationale = ranking_rationale(relevance_rationale, importance_rationale, paper.year, recency_score)
    return {
        "relevance": relevance_score,
        "importance": importance_score,
        "recency": recency_score,
        "total": total,
        "rationale": rationale,
    }
