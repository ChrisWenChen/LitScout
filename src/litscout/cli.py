from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import httpx
import typer

from litscout.connectors.arxiv import search_arxiv
from litscout.connectors.semanticscholar import search_semantic_scholar
from litscout.core.cache import CacheStore
from litscout.core.dedupe import dedupe_papers
from litscout.core.models import CanonicalPaper
from litscout.core.rate_limit import RateLimiter
from litscout.core.render_jsonl import render_jsonl
from litscout.core.render_md import render_markdown
from litscout.core.storage import Storage

app = typer.Typer(add_completion=False)

CACHE_VERSION = 1
MAX_QUERY_LEN = 500


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(os.path.expanduser(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _cache_key(source: str, query: str, topk: int, since: Optional[int]) -> str:
    return json.dumps(
        {
            "version": CACHE_VERSION,
            "source": source,
            "query": query,
            "topk": topk,
            "since": since,
        },
        sort_keys=True,
    )


def _sort_key(paper: CanonicalPaper) -> tuple:
    year_none = paper.year is None
    year_val = -(paper.year or 0)
    return (year_none, year_val, paper.title.lower())


async def _run_search(
    query: str,
    topk: int,
    since: Optional[int],
    out_md: str,
    out_jsonl: str,
    db_path: str,
    no_cache: bool,
    max_rps: float,
    strict: bool,
    cache_ttl: Optional[int],
    cache_max_entries: Optional[int],
) -> None:
    storage = Storage(db_path)
    try:
        cache = CacheStore(storage.conn)
        limiter = RateLimiter(max_rps)

        sources_used: list[str] = []
        collected: list[CanonicalPaper] = []

        async with httpx.AsyncClient() as client:
            # arXiv
            arxiv_key = _cache_key("arxiv", query, topk, since)
            arxiv_data = None if no_cache else cache.get(arxiv_key, cache_ttl)
            if arxiv_data is None:
                logging.info("Fetching arXiv results")
                arxiv_papers = await search_arxiv(query, topk, since, client, limiter)
                cache.set(arxiv_key, [p.model_dump(mode="json") for p in arxiv_papers])
            else:
                logging.info("Using cached arXiv results")
                arxiv_papers = [CanonicalPaper.model_validate(p) for p in arxiv_data]
            if arxiv_papers:
                sources_used.append("arXiv")
                collected.extend(arxiv_papers)

        s2_api_key = os.environ.get("S2_API_KEY")
        if not s2_api_key:
            typer.echo("Warning: S2_API_KEY is not set. Skipping Semantic Scholar.")
        else:
            headers = {"x-api-key": s2_api_key}
            async with httpx.AsyncClient(headers=headers) as client:
                s2_key = _cache_key("semantic_scholar", query, topk, since)
                s2_data = None if no_cache else cache.get(s2_key, cache_ttl)
                if s2_data is None:
                    logging.info("Fetching Semantic Scholar results")
                    s2_papers = await search_semantic_scholar(
                        query, topk, since, client, limiter
                    )
                    cache.set(s2_key, [p.model_dump(mode="json") for p in s2_papers])
                else:
                    logging.info("Using cached Semantic Scholar results")
                    s2_papers = [CanonicalPaper.model_validate(p) for p in s2_data]
                if s2_papers:
                    sources_used.append("Semantic Scholar")
                    collected.extend(s2_papers)

        merged, notes_by_id = dedupe_papers(collected, strict=strict)
        merged.sort(key=_sort_key)

        _ensure_parent(out_md)
        _ensure_parent(out_jsonl)

        md_text = render_markdown(merged, query, sources_used, notes_by_id)
        jsonl_text = render_jsonl(merged)

        with open(os.path.expanduser(out_md), "w", encoding="utf-8") as f:
            f.write(md_text)
        with open(os.path.expanduser(out_jsonl), "w", encoding="utf-8") as f:
            f.write(jsonl_text)

        try:
            storage.save_papers(merged)
        except sqlite3.OperationalError as exc:
            logging.error("SQLite operational error: %s", exc)
            raise

        if not no_cache:
            removed = cache.cleanup(cache_ttl, cache_max_entries)
            if removed:
                logging.info("Cache cleanup removed %s entries", removed)
        typer.echo(f"Saved {len(merged)} papers to {out_md}, {out_jsonl}, and {db_path}.")
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    finally:
        storage.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    topk: int = typer.Option(100, "--topk", help="Max results per source"),
    since: Optional[int] = typer.Option(None, "--since", help="Filter by year >= since"),
    out_md: Optional[str] = typer.Option(
        None,
        "--out-md",
        help="Output Markdown path (default: output-<timestamp>/results_raw.md)",
    ),
    out_jsonl: Optional[str] = typer.Option(
        None,
        "--out-jsonl",
        help="Output JSONL path (default: output-<timestamp>/results_raw.jsonl)",
    ),
    db_path: str = typer.Option(
        "~/.litscout/litscout.db", "--db", help="SQLite database path"
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore cache"),
    max_rps: float = typer.Option(
        1 / 3, "--max-requests-per-second", help="Max requests per second"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Disable title-based deduplication"
    ),
    cache_ttl: Optional[int] = typer.Option(
        86400, "--cache-ttl-seconds", help="Cache TTL in seconds"
    ),
    cache_max_entries: Optional[int] = typer.Option(
        2000, "--cache-max-entries", help="Max cache entries"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR)"
    ),
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not query or not query.strip():
        raise typer.BadParameter("query must be non-empty")
    if len(query) > MAX_QUERY_LEN:
        raise typer.BadParameter(f"query is too long (>{MAX_QUERY_LEN})")
    if topk <= 0:
        raise typer.BadParameter("topk must be > 0")
    if since is not None:
        current_year = datetime.now(timezone.utc).year
        if since > current_year:
            raise typer.BadParameter(f"since cannot be in the future ({current_year})")

    if out_md is None or out_jsonl is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = f"output-{ts}"
        if out_md is None:
            out_md = os.path.join(out_dir, "results_raw.md")
        if out_jsonl is None:
            out_jsonl = os.path.join(out_dir, "results_raw.jsonl")

    asyncio.run(
        _run_search(
            query,
            topk,
            since,
            out_md,
            out_jsonl,
            db_path,
            no_cache,
            max_rps,
            strict,
            cache_ttl,
            cache_max_entries,
        )
    )


@app.command()
def cache_cleanup(
    db_path: str = typer.Option(
        "~/.litscout/litscout.db", "--db", help="SQLite database path"
    ),
    cache_ttl: Optional[int] = typer.Option(
        86400, "--cache-ttl-seconds", help="Cache TTL in seconds"
    ),
    cache_max_entries: Optional[int] = typer.Option(
        2000, "--cache-max-entries", help="Max cache entries"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR)"
    ),
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    storage = Storage(db_path)
    try:
        cache = CacheStore(storage.conn)
        removed = cache.cleanup(cache_ttl, cache_max_entries)
        typer.echo(f"Cache cleanup removed {removed} entries.")
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    finally:
        storage.close()


if __name__ == "__main__":
    app()
