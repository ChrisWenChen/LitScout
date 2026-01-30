import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from typer.testing import CliRunner

import litscout.cli as cli
from litscout.core.models import CanonicalPaper, SourceRecord
from litscout.core.storage import Storage


runner = CliRunner()


def _paper(title: str, source_name: str, cluster: str) -> CanonicalPaper:
    source = SourceRecord(
        source_name=source_name,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        source_url="https://example.com/paper",
        extra={"citationCount": 2},
    )
    return CanonicalPaper(
        canonical_id=uuid4(),
        title=title,
        authors=["Test Author"],
        year=2024,
        abstract="Abstract",
        doi=f"10.1000/{cluster}",
        arxiv_id="1234.5678",
        url_primary="https://example.com/pdf",
        venue="TestConf",
        sources=[source],
        dedup_cluster_id=cluster,
        merge_confidence="low",
    )


def test_cli_search_full_flow(tmp_path, monkeypatch):
    async def _fake_arxiv(*_args, **_kwargs):
        return [_paper("Paper A", "arxiv", "arxiv:1")]

    async def _fake_s2(*_args, **_kwargs):
        return [_paper("Paper B", "semantic_scholar", "s2:1")]

    monkeypatch.setenv("S2_API_KEY", "test")
    monkeypatch.setattr(cli, "search_arxiv", _fake_arxiv)
    monkeypatch.setattr(cli, "search_semantic_scholar", _fake_s2)

    out_md = tmp_path / "out.md"
    out_jsonl = tmp_path / "out.jsonl"
    db_path = tmp_path / "litscout.db"

    result = runner.invoke(
        cli.app,
        [
            "search",
            "test query",
            "--out-md",
            str(out_md),
            "--out-jsonl",
            str(out_jsonl),
            "--db",
            str(db_path),
            "--since",
            "2020",
            "--year-to",
            "2024",
        ],
    )

    assert result.exit_code == 0
    assert out_md.exists()
    assert out_jsonl.exists()

    md_text = out_md.read_text(encoding="utf-8")
    assert "LitScout Results" in md_text
    assert "Paper A" in md_text or "Paper B" in md_text

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT COUNT(*) FROM papers")
    assert cur.fetchone()[0] >= 1
    conn.close()


def test_cli_cache_hit_skips_fetch(tmp_path, monkeypatch):
    async def _fake_arxiv(*_args, **_kwargs):
        return [_paper("Paper A", "arxiv", "arxiv:1")]

    async def _fake_s2(*_args, **_kwargs):
        return [_paper("Paper B", "semantic_scholar", "s2:1")]

    monkeypatch.setenv("S2_API_KEY", "test")
    monkeypatch.setattr(cli, "search_arxiv", _fake_arxiv)
    monkeypatch.setattr(cli, "search_semantic_scholar", _fake_s2)

    out_md = tmp_path / "out.md"
    out_jsonl = tmp_path / "out.jsonl"
    db_path = tmp_path / "litscout.db"

    asyncio.run(
        cli._run_search(
            ["cache test"],
            "cache test",
            2,
            None,
            None,
            str(out_md),
            str(out_jsonl),
            str(db_path),
            False,
            10.0,
            False,
            3600,
            2000,
            False,
        )
    )

    async def _should_not_call(*_args, **_kwargs):
        raise AssertionError("fetch should be skipped due to cache")

    monkeypatch.setattr(cli, "search_arxiv", _should_not_call)
    monkeypatch.setattr(cli, "search_semantic_scholar", _should_not_call)

    asyncio.run(
        cli._run_search(
            ["cache test"],
            "cache test",
            2,
            None,
            None,
            str(out_md),
            str(out_jsonl),
            str(db_path),
            False,
            10.0,
            False,
            3600,
            2000,
            False,
        )
    )


def test_cli_logs_sqlite_error(tmp_path, monkeypatch, caplog):
    async def _fake_arxiv(*_args, **_kwargs):
        return [_paper("Paper A", "arxiv", "arxiv:1")]

    monkeypatch.setattr(cli, "search_arxiv", _fake_arxiv)
    monkeypatch.setenv("S2_API_KEY", "")

    def _boom(self, *_args, **_kwargs):
        raise sqlite3.OperationalError("boom")

    monkeypatch.setattr(Storage, "save_papers", _boom, raising=True)

    out_md = tmp_path / "out.md"
    out_jsonl = tmp_path / "out.jsonl"
    db_path = tmp_path / "litscout.db"

    caplog.set_level(logging.ERROR)
    result = runner.invoke(
        cli.app,
        [
            "search",
            "test query",
            "--out-md",
            str(out_md),
            "--out-jsonl",
            str(out_jsonl),
            "--db",
            str(db_path),
            "--no-cache",
        ],
    )

    assert result.exit_code != 0
    assert "SQLite operational error" in caplog.text


def test_cli_filters_year_range(tmp_path, monkeypatch):
    async def _fake_arxiv(*_args, **_kwargs):
        return [
            _paper("Paper Old", "arxiv", "arxiv:old").model_copy(
                update={"year": 1990, "arxiv_id": "old.0001", "doi": "10.1000/old"}
            ),
            _paper("Paper InRange", "arxiv", "arxiv:mid").model_copy(
                update={"year": 2000, "arxiv_id": "mid.0001", "doi": "10.1000/mid"}
            ),
            _paper("Paper New", "arxiv", "arxiv:new").model_copy(
                update={"year": 2010, "arxiv_id": "new.0001", "doi": "10.1000/new"}
            ),
        ]

    monkeypatch.setattr(cli, "search_arxiv", _fake_arxiv)
    monkeypatch.setenv("S2_API_KEY", "")

    out_md = tmp_path / "out.md"
    out_jsonl = tmp_path / "out.jsonl"
    db_path = tmp_path / "litscout.db"

    asyncio.run(
        cli._run_search(
            ["year range"],
            "year range",
            50,
            1995,
            2005,
            str(out_md),
            str(out_jsonl),
            str(db_path),
            True,
            10.0,
            False,
            3600,
            2000,
            False,
        )
    )

    md_text = out_md.read_text(encoding="utf-8")
    assert "Paper InRange" in md_text
    assert "Paper Old" not in md_text
    assert "Paper New" not in md_text
