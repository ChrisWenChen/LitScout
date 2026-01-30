import asyncio
import builtins
import errno
import os

import pytest
import typer

import litscout.cli as cli
from litscout.core.models import CanonicalPaper, SourceRecord, utc_now_iso


def _paper(title: str, source_name: str, cluster: str) -> CanonicalPaper:
    source = SourceRecord(
        source_name=source_name,
        retrieved_at=utc_now_iso(),
        source_url=None,
        extra={},
    )
    return CanonicalPaper(
        title=title,
        authors=["A"],
        year=2024,
        abstract="Abstract",
        doi=None,
        arxiv_id=None,
        url_primary=None,
        venue=None,
        sources=[source],
        dedup_cluster_id=cluster,
        merge_confidence="low",
    )


def test_cli_empty_results(tmp_path, monkeypatch):
    async def _fake_arxiv(*_args, **_kwargs):
        return []

    monkeypatch.setattr(cli, "search_arxiv", _fake_arxiv)
    monkeypatch.setenv("S2_API_KEY", "")

    out_md = tmp_path / "out.md"
    out_jsonl = tmp_path / "out.jsonl"
    db_path = tmp_path / "litscout.db"

    asyncio.run(
        cli._run_search(
            ["empty query"],
            "empty query",
            5,
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
            ["arxiv"],
        )
    )

    assert out_md.exists()
    assert out_jsonl.exists()
    assert "LitScout Results" in out_md.read_text(encoding="utf-8")


def test_cli_handles_source_failure(tmp_path, monkeypatch):
    async def _fail_arxiv(*_args, **_kwargs):
        raise RuntimeError("network down")

    async def _fake_s2(*_args, **_kwargs):
        return [_paper("Paper S2", "semantic_scholar", "s2:1")]

    monkeypatch.setattr(cli, "search_arxiv", _fail_arxiv)
    monkeypatch.setattr(cli, "search_semantic_scholar", _fake_s2)
    monkeypatch.setenv("S2_API_KEY", "valid_key_12345")

    out_md = tmp_path / "out.md"
    out_jsonl = tmp_path / "out.jsonl"
    db_path = tmp_path / "litscout.db"

    asyncio.run(
        cli._run_search(
            ["test query"],
            "test query",
            5,
            None,
            None,
            str(out_md),
            str(out_jsonl),
            str(db_path),
            True,
            10.0,
            False,
            3600,
            2000,
            False,
            ["arxiv", "semantic_scholar"],
        )
    )

    assert "Paper S2" in out_md.read_text(encoding="utf-8")


def test_cli_disk_full(tmp_path, monkeypatch):
    async def _fake_arxiv(*_args, **_kwargs):
        return [_paper("Paper A", "arxiv", "arxiv:1")]

    monkeypatch.setattr(cli, "search_arxiv", _fake_arxiv)

    out_md = tmp_path / "out.md"
    out_jsonl = tmp_path / "out.jsonl"
    db_path = tmp_path / "litscout.db"

    real_open = builtins.open

    def _raising_open(path, *args, **kwargs):
        if os.fspath(path) in (str(out_md), str(out_jsonl)):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _raising_open)

    with pytest.raises(typer.Exit):
        asyncio.run(
            cli._run_search(
                ["disk full"],
                "disk full",
                1,
                None,
                None,
                str(out_md),
                str(out_jsonl),
                str(db_path),
                True,
                10.0,
                False,
                3600,
                2000,
                False,
                ["arxiv"],
            )
        )
