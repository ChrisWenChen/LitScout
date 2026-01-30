from typer.testing import CliRunner

import litscout.cli as cli
from litscout.core.models import CanonicalPaper, SourceRecord
from datetime import datetime, timezone
from uuid import uuid4


runner = CliRunner()


def _paper(title: str, source_name: str, cluster: str) -> CanonicalPaper:
    source = SourceRecord(
        source_name=source_name,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        source_url="https://example.com/paper",
        extra={},
    )
    return CanonicalPaper(
        canonical_id=uuid4(),
        title=title,
        authors=["Author"],
        year=2024,
        abstract="Abstract",
        doi=None,
        arxiv_id=None,
        bibcode=None,
        url_primary="https://example.com/pdf",
        venue=None,
        sources=[source],
        dedup_cluster_id=cluster,
        merge_confidence="low",
    )


def test_cli_sources_only_arxiv(tmp_path, monkeypatch):
    async def _fake_arxiv(*_args, **_kwargs):
        return [_paper("Paper A", "arxiv", "arxiv:1")]

    async def _should_not_call(*_args, **_kwargs):
        raise AssertionError("semantic scholar should not be called")

    monkeypatch.setenv("S2_API_KEY", "test")
    monkeypatch.setattr(cli, "search_arxiv", _fake_arxiv)
    monkeypatch.setattr(cli, "search_semantic_scholar", _should_not_call)

    out_md = tmp_path / "out.md"
    out_jsonl = tmp_path / "out.jsonl"
    db_path = tmp_path / "litscout.db"

    result = runner.invoke(
        cli.app,
        [
            "search",
            "test query",
            "--sources",
            "arxiv",
            "--out-md",
            str(out_md),
            "--out-jsonl",
            str(out_jsonl),
            "--db",
            str(db_path),
        ],
    )

    assert result.exit_code == 0
    assert out_md.exists()
    assert out_jsonl.exists()


def test_cli_sources_invalid():
    result = runner.invoke(cli.app, ["search", "test", "--sources", "unknown"])
    assert result.exit_code != 0
    assert "Unknown sources" in result.output


def test_cli_sources_ads_invalid_key(tmp_path, monkeypatch):
    async def _should_not_call(*_args, **_kwargs):
        raise AssertionError("ads should not be called with invalid key")

    monkeypatch.setenv("ADS_API_TOKEN", "short")
    monkeypatch.setattr(cli, "search_ads", _should_not_call)

    out_md = tmp_path / "out.md"
    out_jsonl = tmp_path / "out.jsonl"
    db_path = tmp_path / "litscout.db"

    result = runner.invoke(
        cli.app,
        [
            "search",
            "test query",
            "--sources",
            "ads",
            "--out-md",
            str(out_md),
            "--out-jsonl",
            str(out_jsonl),
            "--db",
            str(db_path),
        ],
    )

    assert result.exit_code == 0
    assert out_md.exists()
    assert out_jsonl.exists()
