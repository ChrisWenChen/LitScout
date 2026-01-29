from __future__ import annotations

from typing import Iterable

from .models import CanonicalPaper


def render_jsonl(papers: Iterable[CanonicalPaper]) -> str:
    lines = [paper.model_dump_json() for paper in papers]
    return "\n".join(lines) + ("\n" if lines else "")
