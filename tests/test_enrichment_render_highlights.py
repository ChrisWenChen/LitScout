from datetime import datetime, timezone

from litscout.enrichment.models import (
    CanonicalSnapshot,
    EnrichedPaper,
    LiteGoalScore,
    LiteInsight,
    Provenance,
    RankingBreakdown,
    ResearchGoal,
)
from litscout.enrichment.render_md import render_enriched_markdown


def _paper_snapshot(
    pid: str,
    title: str,
    authors: list[str],
    doi: str | None = None,
    arxiv_id: str | None = None,
    url_primary: str | None = None,
) -> CanonicalSnapshot:
    return CanonicalSnapshot(
        canonical_id=pid,
        title=title,
        authors=authors,
        year=2024,
        abstract="Abstract",
        doi=doi,
        arxiv_id=arxiv_id,
        url_primary=url_primary,
        venue=None,
        sources=[],
    )


def _provenance() -> Provenance:
    return Provenance(
        provider="mimo",
        model="mimo-v2-flash",
        temperature=0.1,
        prompt_hash="hash",
        skill_hash="skill",
        input_hash="input",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _ranking(total: float) -> RankingBreakdown:
    return RankingBreakdown(
        relevance=total,
        importance=0.0,
        recency=0.0,
        total=total,
        rationale=["llm_score"],
    )


def test_render_highlights_goal_lists_and_authors_links():
    authors = ["A1", "A2", "A3", "A4", "A5", "A6"]
    p1 = EnrichedPaper(
        paper_id="1",
        canonical=_paper_snapshot("1", "Paper One", authors, doi="10.1000/test"),
        ai=LiteInsight(
            paper_id="1",
            summary="Summary one.",
            methods=["M1"],
            priority="P2",
            score=0.6,
            goal_scores=[LiteGoalScore(goal_id="G1", score=0.9)],
        ),
        ranking=_ranking(0.9),
        classification=None,
        provenance=_provenance(),
    )
    p2 = EnrichedPaper(
        paper_id="2",
        canonical=_paper_snapshot("2", "Paper Two", authors, arxiv_id="1234.5678"),
        ai=LiteInsight(
            paper_id="2",
            summary="Summary two.",
            methods=["M2"],
            priority="P2",
            score=0.5,
            goal_scores=[LiteGoalScore(goal_id="G1", score=0.8)],
        ),
        ranking=_ranking(0.8),
        classification=None,
        provenance=_provenance(),
    )
    p3 = EnrichedPaper(
        paper_id="3",
        canonical=_paper_snapshot("3", "Paper Three", authors, url_primary="https://example.com/p3"),
        ai=LiteInsight(
            paper_id="3",
            summary="Summary three.",
            methods=["M3"],
            priority="P3",
            score=0.4,
            goal_scores=[LiteGoalScore(goal_id="G1", score=0.7)],
        ),
        ranking=_ranking(0.7),
        classification=None,
        provenance=_provenance(),
    )

    goals = [
        ResearchGoal(goal_id="G1", name="G1", description="Test goal description", signals=[], negative_signals=[]),
    ]
    md = render_enriched_markdown([p2, p3, p1], "goal", goals)

    assert "# G1" in md
    assert "Test goal description" in md
    assert "authors: A1, A2, …, A5, A6" in md
    assert "https://doi.org/10.1000/test" in md
    assert "arxiv:1234.5678" in md
