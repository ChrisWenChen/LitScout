from litscout.core.dedupe import dedupe_papers
from litscout.core.models import CanonicalPaper, SourceRecord, utc_now_iso


def _paper(title: str, doi: str | None, arxiv_id: str | None, year: int | None, source: str):
    return CanonicalPaper(
        title=title,
        authors=["A"],
        year=year,
        abstract="short",
        doi=doi,
        arxiv_id=arxiv_id,
        url_primary=None,
        venue=None,
        sources=[
            SourceRecord(
                source_name=source,
                retrieved_at=utc_now_iso(),
                source_url=None,
                extra={},
            )
        ],
        dedup_cluster_id="tmp",
        merge_confidence="low",
    )


def test_dedupe_by_doi():
    p1 = _paper("Title", "10.1/abc", None, 2021, "arxiv")
    p2 = _paper("Title", "10.1/abc", None, 2020, "semantic_scholar")
    merged, notes = dedupe_papers([p1, p2])
    assert len(merged) == 1
    assert merged[0].merge_confidence == "high"
    assert merged[0].year == 2020
    assert notes


def test_dedupe_by_title():
    p1 = _paper("Graph Neural Networks", None, None, 2022, "arxiv")
    p2 = _paper("Graph Neural Networks", None, None, 2022, "semantic_scholar")
    merged, _ = dedupe_papers([p1, p2])
    assert len(merged) == 1
    assert merged[0].merge_confidence == "medium"


def test_dedupe_strict_disables_title_merge():
    p1 = _paper("Graph Neural Networks", None, None, 2022, "arxiv")
    p2 = _paper("Graph Neural Networks", None, None, 2022, "semantic_scholar")
    merged, _ = dedupe_papers([p1, p2], strict=True)
    assert len(merged) == 2
