from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from .models import CanonicalPaper


def render_markdown(
    papers: Iterable[CanonicalPaper],
    query: str,
    sources_used: list[str],
    notes_by_id: dict[str, list[str]] | None = None,
) -> str:
    notes_by_id = notes_by_id or {}
    now = datetime.now(timezone.utc).isoformat()
    source_line = " | ".join(sources_used) if sources_used else "N/A"

    lines: list[str] = []
    lines.append("# LitScout Results (RAW)")
    lines.append(f"_Query_: `{query}`")
    lines.append(f"_Generated_: `{now}`")
    lines.append(f"_Sources_: `{source_line}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    for idx, paper in enumerate(papers, start=1):
        lines.append(f"## Paper {idx} — `{paper.title}`")
        lines.append(f"- **Canonical ID**: `{paper.canonical_id}`")
        lines.append(f"- **Year**: `{paper.year if paper.year is not None else 'N/A'}`")
        authors = "; ".join(paper.authors) if paper.authors else "N/A"
        lines.append(f"- **Authors**: `{authors}`")
        lines.append(f"- **DOI**: `{paper.doi or 'N/A'}`")
        lines.append(f"- **arXiv**: `{paper.arxiv_id or 'N/A'}`")
        lines.append(f"- **ADS Bibcode**: `{paper.bibcode or 'N/A'}`")
        lines.append(f"- **Venue**: `{paper.venue or 'N/A'}`")
        lines.append(f"- **Primary URL**: `{paper.url_primary or 'N/A'}`")
        lines.append("")
        lines.append("### Abstract")
        lines.append(paper.abstract or "N/A")
        lines.append("")
        lines.append("### Source Records")
        for source in paper.sources:
            if source.source_name == "semantic_scholar":
                citations = source.extra.get("citationCount")
                citations_value = citations if citations is not None else "N/A"
                lines.append(
                    f"- **Semantic Scholar**: retrieved `{source.retrieved_at}` | url: `{source.source_url or 'N/A'}` | citations: `{citations_value}`"
                )
            elif source.source_name == "inspire":
                control = source.extra.get("control_number") or "N/A"
                lines.append(
                    f"- **INSPIRE-HEP**: retrieved `{source.retrieved_at}` | url: `{source.source_url or 'N/A'}` | control: `{control}`"
                )
            elif source.source_name == "ads":
                bibcode = source.extra.get("bibcode") or "N/A"
                lines.append(
                    f"- **NASA ADS**: retrieved `{source.retrieved_at}` | url: `{source.source_url or 'N/A'}` | bibcode: `{bibcode}`"
                )
            else:
                lines.append(
                    f"- **arXiv**: retrieved `{source.retrieved_at}` | url: `{source.source_url or 'N/A'}`"
                )
        lines.append("")
        lines.append("### Notes")
        lines.append(f"- dedup_cluster: `{paper.dedup_cluster_id}`")
        lines.append(f"- merge_confidence: `{paper.merge_confidence}`")
        for note in notes_by_id.get(str(paper.canonical_id), []):
            lines.append(f"- {note}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
