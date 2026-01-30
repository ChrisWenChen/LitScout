from __future__ import annotations

from typing import Any

from litscout.core.models import CanonicalPaper


_ARXIV_PREFIX_MAP = {
    "cs": "CS",
    "math": "Math",
    "stat": "Statistics",
    "physics": "Physics",
    "q-bio": "Biology",
    "q-fin": "Finance",
}


def rule_classify(paper: CanonicalPaper) -> tuple[list[str], list[str]]:
    path: list[str] = []
    rationale: list[str] = []

    for source in paper.sources:
        primary = source.extra.get("primary_category")
        if isinstance(primary, str) and "." in primary:
            prefix = primary.split(".", 1)[0]
            mapped = _ARXIV_PREFIX_MAP.get(prefix)
            if mapped:
                path = [mapped, primary]
                rationale.append(f"arXiv primary_category={primary}")
                return path, rationale

    venue = (paper.venue or "").lower()
    if venue:
        if any(k in venue for k in ["neurips", "icml", "iclr", "acl", "emnlp"]):
            path = ["CS", "AI/ML"]
            rationale.append(f"venue match: {paper.venue}")
        elif any(k in venue for k in ["nature", "science", "cell"]):
            path = ["General", "High-Impact"]
            rationale.append(f"venue match: {paper.venue}")
        elif any(k in venue for k in ["phys", "prl", "aps"]):
            path = ["Physics"]
            rationale.append(f"venue match: {paper.venue}")

    if not path:
        path = ["Unknown"]
        rationale.append("no rule-based match")

    return path, rationale


def normalize_ai_classification(
    category_path: list[str],
    tags: list[str],
    taxonomy: list[str],
    max_tags: int,
) -> tuple[list[str], list[str], list[str]]:
    rationale: list[str] = []
    clean_tags = [t.strip() for t in tags if t.strip()]
    clean_tags = clean_tags[: max(0, max_tags)]

    if taxonomy:
        valid = set(taxonomy)
        if category_path:
            leaf = category_path[-1]
            if leaf not in valid:
                rationale.append("category not in taxonomy; mapped to Other")
                category_path = ["Other"]
        else:
            category_path = ["Other"]
            rationale.append("category missing; mapped to Other")

    return category_path, clean_tags, rationale
