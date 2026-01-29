from datetime import datetime, timezone

from litscout.core.models import CanonicalPaper, SourceRecord
from litscout.core.render_jsonl import render_jsonl
from litscout.core.render_md import render_markdown


def _paper(title: str, source_name: str) -> CanonicalPaper:
    source = SourceRecord(
        source_name=source_name,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        source_url="https://example.com/paper",
        extra={"citationCount": 3},
    )
    return CanonicalPaper(
        title=title,
        authors=["Ada Lovelace", "Alan Turing"],
        year=2024,
        abstract="Test abstract.",
        doi="10.1000/test",
        arxiv_id="1234.5678",
        url_primary="https://example.com/pdf",
        venue="TestConf",
        sources=[source],
        dedup_cluster_id="arxiv:1234.5678",
        merge_confidence="low",
    )


def test_render_markdown_includes_fields():
    paper = _paper("A Test Paper", "semantic_scholar")
    notes = {str(paper.canonical_id): ["note-1"]}
    md = render_markdown([paper], "test query", ["arXiv", "Semantic Scholar"], notes)

    assert md.startswith("# LitScout Results (RAW)")
    assert "_Query_: `test query`" in md
    assert "_Sources_: `arXiv | Semantic Scholar`" in md
    assert "## Paper 1" in md
    assert "**Canonical ID**" in md
    assert "**Authors**: `Ada Lovelace; Alan Turing`" in md
    assert "**DOI**: `10.1000/test`" in md
    assert "### Abstract" in md
    assert "Test abstract." in md
    assert "### Source Records" in md
    assert "**Semantic Scholar**" in md
    assert "### Notes" in md
    assert "note-1" in md


def test_render_jsonl_one_line_per_paper():
    p1 = _paper("Paper One", "arxiv")
    p2 = _paper("Paper Two", "semantic_scholar")
    jsonl = render_jsonl([p1, p2])

    lines = [line for line in jsonl.splitlines() if line.strip()]
    assert len(lines) == 2
    assert str(p1.canonical_id) in lines[0]
    assert "Paper One" in lines[0]
    assert "Paper Two" in lines[1]
    assert jsonl.endswith("\n")
