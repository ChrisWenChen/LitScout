from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
from litscout.enrichment import load_skill_profile, enrich_papers, load_jsonl, dump_enriched_jsonl, render_enriched_markdown
from litscout.enrichment.llm import LLMConfig, build_client

app = typer.Typer(
    add_completion=False,
    help=(
        "LitScout CLI. Examples:\n"
        "  litscout search \"graph neural networks\" --topk 20\n"
        "  litscout search --queries-file queries.txt --keyword-rank\n"
        "  litscout enrich --goal \"sign problem\" --skill docs/skill.md.example"
    ),
)

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


def _load_queries(query: Optional[str], queries: list[str], queries_file: Optional[str]) -> list[str]:
    collected: list[str] = []
    if query:
        collected.append(query)
    if queries:
        collected.extend(queries)
    if queries_file:
        with open(queries_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                collected.append(line)
    return [q for q in (q.strip() for q in collected) if q]


def _extract_keywords(queries: list[str]) -> list[str]:
    keywords: list[str] = []
    for q in queries:
        for phrase in re.findall(r"\"([^\"]+)\"", q):
            keywords.append(phrase.lower())
        stripped = re.sub(r"\"[^\"]+\"", " ", q)
        for token in re.split(r"\s+", stripped):
            token = token.strip()
            if token and token.upper() not in ("AND", "OR"):
                keywords.append(token.lower())
    return list(dict.fromkeys(keywords))


def _keyword_score(paper: CanonicalPaper, keywords: list[str]) -> float:
    title = (paper.title or "").lower()
    abstract = (paper.abstract or "").lower()
    score = 0.0
    for kw in keywords:
        if not kw:
            continue
        score += 2.0 * title.count(kw)
        score += 1.0 * abstract.count(kw)
    return score


def _sort_key_with_keywords(paper: CanonicalPaper, score: float) -> tuple:
    year_none = paper.year is None
    year_val = -(paper.year or 0)
    return (-score, year_none, year_val, paper.title.lower())


async def _run_search(
    queries: list[str],
    display_query: str,
    topk: int,
    since: Optional[int],
    year_to: Optional[int],
    out_md: str,
    out_jsonl: str,
    db_path: str,
    no_cache: bool,
    max_rps: float,
    strict: bool,
    cache_ttl: Optional[int],
    cache_max_entries: Optional[int],
    keyword_rank: bool,
) -> None:
    storage = Storage(db_path)
    try:
        cache = CacheStore(storage.conn)
        limiter = RateLimiter(max_rps)

        sources_used: list[str] = []
        collected: list[CanonicalPaper] = []

        queries_to_run = list(queries)
        async with httpx.AsyncClient() as client:
            for query in queries_to_run:
                # arXiv
                arxiv_key = _cache_key("arxiv", query, topk, since)
                arxiv_data = None if no_cache else cache.get(arxiv_key, cache_ttl)
                if arxiv_data is None:
                    logging.info("Fetching arXiv results for query: %s", query)
                    arxiv_papers = await search_arxiv(query, topk, since, client, limiter)
                    cache.set(arxiv_key, [p.model_dump(mode="json") for p in arxiv_papers])
                else:
                    logging.info("Using cached arXiv results for query: %s", query)
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
                for query in queries_to_run:
                    s2_key = _cache_key("semantic_scholar", query, topk, since)
                    s2_data = None if no_cache else cache.get(s2_key, cache_ttl)
                    if s2_data is None:
                        logging.info("Fetching Semantic Scholar results for query: %s", query)
                        s2_papers = await search_semantic_scholar(
                            query, topk, since, client, limiter
                        )
                        cache.set(s2_key, [p.model_dump(mode="json") for p in s2_papers])
                    else:
                        logging.info("Using cached Semantic Scholar results for query: %s", query)
                        s2_papers = [CanonicalPaper.model_validate(p) for p in s2_data]
                    if s2_papers:
                        sources_used.append("Semantic Scholar")
                        collected.extend(s2_papers)

        merged, notes_by_id = dedupe_papers(collected, strict=strict)
        if since is not None or year_to is not None:
            filtered: list[CanonicalPaper] = []
            for paper in merged:
                if paper.year is None:
                    continue
                if since is not None and paper.year < since:
                    continue
                if year_to is not None and paper.year > year_to:
                    continue
                filtered.append(paper)
            merged = filtered
        if keyword_rank:
            keywords = _extract_keywords(queries)
            scores = {str(p.canonical_id): _keyword_score(p, keywords) for p in merged}
            merged.sort(key=lambda p: _sort_key_with_keywords(p, scores.get(str(p.canonical_id), 0.0)))
        else:
            merged.sort(key=_sort_key)

        _ensure_parent(out_md)
        _ensure_parent(out_jsonl)

        md_text = render_markdown(merged, display_query, sources_used, notes_by_id)
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


@app.command(
    help=(
        "Search papers from arXiv and Semantic Scholar.\n"
        "Examples:\n"
        "  litscout search \"graph neural networks\" --topk 20\n"
        "  litscout search --query \"sign problem\" --query \"phase transition\" --keyword-rank\n"
        "  litscout search --queries-file queries.txt"
    )
)
def search(
    query: Optional[str] = typer.Argument(None, help="Search query"),
    queries: list[str] = typer.Option(
        [], "--query", help="Additional search query (repeatable)"
    ),
    queries_file: Optional[str] = typer.Option(
        None, "--queries-file", help="Path to file with one query per line"
    ),
    topk: int = typer.Option(100, "--topk", help="Max results per source"),
    since: Optional[int] = typer.Option(None, "--since", help="Filter by year >= since"),
    year_to: Optional[int] = typer.Option(None, "--year-to", help="Filter by year <= year-to"),
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
    keyword_rank: bool = typer.Option(
        False, "--keyword-rank", help="Rank by keyword hits (title>abstract)"
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
        if not queries and not queries_file:
            raise typer.BadParameter("query must be non-empty")
    if topk <= 0:
        raise typer.BadParameter("topk must be > 0")
    if since is not None:
        current_year = datetime.now(timezone.utc).year
        if since > current_year:
            raise typer.BadParameter(f"since cannot be in the future ({current_year})")
    if year_to is not None:
        current_year = datetime.now(timezone.utc).year
        if year_to > current_year:
            raise typer.BadParameter(f"year-to cannot be in the future ({current_year})")
    if since is not None and year_to is not None and since > year_to:
        raise typer.BadParameter("since cannot be greater than year-to")

    if out_md is None or out_jsonl is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = f"output-{ts}"
        if out_md is None:
            out_md = os.path.join(out_dir, "results_raw.md")
        if out_jsonl is None:
            out_jsonl = os.path.join(out_dir, "results_raw.jsonl")

    query_list = _load_queries(query, queries or [], queries_file)
    if not query_list:
        raise typer.BadParameter("no queries provided")
    if any(len(q) > MAX_QUERY_LEN for q in query_list):
        raise typer.BadParameter(f"query is too long (>{MAX_QUERY_LEN})")

    display_query = " | ".join(query_list)
    asyncio.run(
        _run_search(
            query_list,
            display_query,
            topk,
            since,
            year_to,
            out_md,
            out_jsonl,
            db_path,
            no_cache,
            max_rps,
            strict,
            cache_ttl,
            cache_max_entries,
            keyword_rank,
        )
    )


async def _run_enrich(
    goal: str,
    skill_path: str,
    in_jsonl: str,
    out_jsonl: str,
    out_md: str,
    provider: Optional[str],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    base_url: Optional[str],
    batch_size: int,
    llm_max_rps: float,
    time_window: Optional[str],
    use_goals: bool,
    goal_weight: Optional[float],
    llm_concurrency: int,
) -> None:
    skill = load_skill_profile(skill_path)
    if provider:
        skill.llm.provider = provider
    if model:
        skill.llm.model = model
    if temperature is not None:
        skill.llm.temperature = temperature
    if max_tokens is not None:
        skill.llm.max_tokens = max_tokens
    if base_url:
        skill.llm.base_url = base_url
    if goal_weight is not None:
        skill.scoring.goal_match_weight = goal_weight
    use_goals = use_goals or skill.scoring.use_goals

    if skill.llm.provider == "mimo" and not skill.llm.base_url:
        skill.llm.base_url = "https://api.xiaomimimo.com/v1"
    if skill.llm.provider == "deepseek" and not skill.llm.base_url:
        skill.llm.base_url = "https://api.deepseek.com/v1"
    if skill.llm.provider == "grok" and not skill.llm.base_url:
        skill.llm.base_url = "https://api.x.ai/v1"

    config = LLMConfig(
        provider=skill.llm.provider,
        model=skill.llm.model,
        temperature=skill.llm.temperature,
        max_tokens=skill.llm.max_tokens,
        seed=skill.llm.seed,
        base_url=skill.llm.base_url or "https://api.openai.com",
    )
    limiter = RateLimiter(llm_max_rps)
    api_key = os.environ.get("OPENAI_API_KEY")
    if skill.llm.provider == "mimo":
        api_key = os.environ.get("MIMO_API_KEY")
    if skill.llm.provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
    if skill.llm.provider == "grok":
        api_key = os.environ.get("XAI_API_KEY")
    if api_key is not None:
        logging.debug("LLM API key length: %s", len(api_key))
    client = build_client(config, api_key=api_key, limiter=limiter)

    papers = load_jsonl(in_jsonl)
    enriched = await enrich_papers(
        papers,
        goal,
        skill,
        client,
        batch_size=batch_size,
        time_window=time_window,
        use_goals=use_goals,
        goal_weight=skill.scoring.goal_match_weight,
        max_concurrency=llm_concurrency,
    )

    _ensure_parent(out_jsonl)
    _ensure_parent(out_md)
    dump_enriched_jsonl(out_jsonl, enriched)
    md_text = render_enriched_markdown(enriched, goal)
    with open(os.path.expanduser(out_md), "w", encoding="utf-8") as f:
        f.write(md_text)

    typer.echo(f"Saved {len(enriched)} enriched papers to {out_jsonl} and {out_md}.")


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


@app.command(
    help=(
        "Enrich results with AI ranking and classification.\n"
        "Outputs a lightweight JSON schema (summary/methods/priority/score/goal_scores).\n"
        "Examples:\n"
        "  litscout enrich --goal \"sign problem\" --skill docs/skill.md.example\n"
        "  litscout enrich --goal \"...\" --skill skill.md --in-jsonl results_raw.jsonl"
    )
)
def enrich(
    goal: str = typer.Option(..., "--goal", help="Research goal for ranking/classification"),
    skill_path: str = typer.Option(..., "--skill", help="Path to skill.md"),
    in_jsonl: str = typer.Option("results_raw.jsonl", "--in-jsonl", help="Input JSONL path"),
    out_jsonl: str = typer.Option(
        "results_enriched.jsonl", "--out-jsonl", help="Output enriched JSONL path"
    ),
    out_md: str = typer.Option(
        "results_enriched.md", "--out-md", help="Output enriched Markdown path"
    ),
    provider: Optional[str] = typer.Option(None, "--provider", help="Override provider"),
    model: Optional[str] = typer.Option(None, "--model", help="Override model"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Override temperature"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Override max tokens"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Override base URL"),
    batch_size: int = typer.Option(5, "--batch-size", help="Max papers per LLM request"),
    llm_max_rps: float = typer.Option(
        1000 / 60, "--llm-max-requests-per-second", help="LLM request rate limit"
    ),
    llm_concurrency: int = typer.Option(
        1, "--llm-concurrency", help="Max concurrent LLM requests"
    ),
    time_window: Optional[str] = typer.Option(None, "--time-window", help="Time window label"),
    use_goals: bool = typer.Option(False, "--use-goals", help="Use goal match in sorting"),
    goal_weight: Optional[float] = typer.Option(
        None, "--goal-weight", help="Weight for goal match when --use-goals is set"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR)"
    ),
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not goal.strip():
        raise typer.BadParameter("goal must be non-empty")
    if batch_size <= 0:
        raise typer.BadParameter("batch-size must be > 0")

    asyncio.run(
        _run_enrich(
            goal,
            skill_path,
            in_jsonl,
            out_jsonl,
            out_md,
            provider,
            model,
            temperature,
            max_tokens,
            base_url,
            batch_size,
            llm_max_rps,
            time_window,
            use_goals,
            goal_weight,
            llm_concurrency,
        )
    )


if __name__ == "__main__":
    app()
