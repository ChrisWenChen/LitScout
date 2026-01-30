from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

from .models import EnrichedPaper, ResearchGoal


def render_enriched_markdown(
    enriched: Iterable[EnrichedPaper],
    goal: str,
    goals: Sequence[ResearchGoal] | None = None,
) -> str:
    items = list(enriched)
    items.sort(key=lambda p: (p.ranking.total if p.ranking else -1), reverse=True)

    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append("# LitScout Results (ENRICHED)")
    lines.append(f"_Generated_: `{now}`")
    lines.append(f"_Goal_: `{goal}`")

    if items:
        prov = items[0].provenance
        lines.append("_Provenance_:")
        lines.append(
            f"- provider/model: `{prov.provider}` / `{prov.model}` | skill_hash: `{prov.skill_hash}` | input_hash: `{prov.input_hash}`"
        )
    lines.append("")
    def _format_authors(authors: list[str]) -> str:
        if not authors:
            return "N/A"
        if len(authors) <= 4:
            return ", ".join(authors)
        head = ", ".join(authors[:2])
        tail = ", ".join(authors[-2:])
        return f"{head}, …, {tail}"

    def _format_link(paper: EnrichedPaper) -> str:
        if paper.canonical.doi:
            return f"https://doi.org/{paper.canonical.doi}"
        if paper.canonical.arxiv_id:
            return f"arxiv:{paper.canonical.arxiv_id}"
        if paper.canonical.url_primary:
            return paper.canonical.url_primary
        return "N/A"

    if items:
        lines.append("## Highlights")
        lines.append("")
        most_important = items[0]
        lines.append(
            f"- Top paper: {most_important.canonical.title} — {most_important.ai.summary if most_important.ai else 'N/A'}"
        )
        lines.append("")

    goal_ids: list[str]
    if goals:
        goal_ids = [g.goal_id for g in goals]
    else:
        goal_set = []
        for paper in items:
            if not paper.ai or not paper.ai.goal_scores:
                continue
            for gs in paper.ai.goal_scores:
                if gs.goal_id not in goal_set:
                    goal_set.append(gs.goal_id)
        goal_ids = goal_set

    goal_desc_map = {g.goal_id: g.description for g in goals or []}
    for goal_id in goal_ids:
        lines.append(f"# {goal_id}")
        desc = goal_desc_map.get(goal_id)
        if desc:
            desc_one = " ".join(desc.split())
            lines.append(desc_one)
        lines.append("")
        scored: list[tuple[float, EnrichedPaper]] = []
        for paper in items:
            if not paper.ai or not paper.ai.goal_scores:
                continue
            score = next((g.score for g in paper.ai.goal_scores if g.goal_id == goal_id), None)
            if score is None:
                continue
            scored.append((score, paper))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            lines.append("- Best match: N/A")
            lines.append("")
            continue
        lines.append("- Best match:")
        for score, paper in scored[:3]:
            one_liner = paper.ai.summary if paper.ai else "N/A"
            authors = _format_authors(paper.canonical.authors)
            link = _format_link(paper)
            lines.append(
                f"  - {paper.canonical.title} | authors: {authors} | {link} | {one_liner}"
            )
        lines.append("")

    lines.append("## Ranked Results")
    lines.append("")

    for idx, paper in enumerate(items, start=1):
        total = f"{paper.ranking.total:.3f}" if paper.ranking else "N/A"
        one_liner = paper.ai.summary if paper.ai else "N/A"
        authors = _format_authors(paper.canonical.authors)
        link = _format_link(paper)
        lines.append(
            f"{idx}. **{paper.canonical.title}** | score={total} | authors: {authors} | {link} | {one_liner}"
        )
        if paper.error:
            lines.append(f"   - ERROR: {paper.error}")

    return "\n".join(lines).rstrip() + "\n"
