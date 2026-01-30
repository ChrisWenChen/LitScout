from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import typer

from litscout.connectors.arxiv import search_arxiv
from litscout.connectors.inspire import search_inspire
from litscout.connectors.semanticscholar import search_semantic_scholar
from litscout.connectors.ads import search_ads
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
SOURCE_ALIASES = {
    "arxiv": "arxiv",
    "s2": "semantic_scholar",
    "semantic_scholar": "semantic_scholar",
    "semantic-scholar": "semantic_scholar",
    "semanticscholar": "semantic_scholar",
    "inspire": "inspire",
    "inspire-hep": "inspire",
    "inspirehep": "inspire",
    "ads": "ads",
    "nasa-ads": "ads",
    "nasa_ads": "ads",
}
SOURCE_LABELS = {
    "arxiv": "arXiv",
    "semantic_scholar": "Semantic Scholar",
    "inspire": "INSPIRE-HEP",
    "ads": "NASA ADS",
}


def _is_pytest_running() -> bool:
    return os.environ.get("PYTEST_CURRENT_TEST") is not None


def _configure_logging(log_level: str, log_file: str | None) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    formatter = _RedactingFormatter("%(asctime)s %(levelname)s %(message)s")
    if root.handlers:
        for handler in root.handlers:
            handler.setLevel(level)
            handler.setFormatter(formatter)
        if log_file and not _is_pytest_running():
            log_path = os.path.abspath(os.path.expanduser(log_file))
            _ensure_parent(log_path)
            for handler in root.handlers:
                if isinstance(handler, logging.FileHandler):
                    try:
                        if os.path.abspath(handler.baseFilename) == log_path:
                            return
                    except Exception:
                        continue
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        return
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file and not _is_pytest_running():
        _ensure_parent(log_file)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root.addHandler(handler)


def _slugify_query(text: str, max_len: int = 64) -> str:
    lowered = text.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    if not cleaned:
        return "query"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("-")
    return cleaned


def _is_probably_valid_key(key: str | None) -> bool:
    if not key:
        return False
    if any(ch.isspace() for ch in key):
        return False
    return len(key.strip()) >= 10


class _RedactingFormatter(logging.Formatter):
    _SENSITIVE_KEYS = ("key", "token", "secret", "authorization")

    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.args, dict):
            record.args = self._redact_dict(record.args)
        elif isinstance(record.args, tuple) and len(record.args) == 1:
            arg = record.args[0]
            if isinstance(arg, dict):
                record.args = (self._redact_dict(arg),)
        message = super().format(record)
        return self._redact_message(message)

    def _redact_message(self, message: str) -> str:
        if "Bearer " in message:
            prefix = message.split("Bearer ", 1)[0]
            return f"{prefix}Bearer ***"
        return message

    def _redact_dict(self, data: dict) -> dict:
        redacted: dict[str, object] = {}
        for key, value in data.items():
            if any(token in key.lower() for token in self._SENSITIVE_KEYS):
                redacted[key] = "***"
            else:
                redacted[key] = value
        return redacted


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


def _parse_sources(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    items = [s.strip().lower() for s in value.split(",") if s.strip()]
    if not items:
        raise typer.BadParameter("sources must be non-empty")
    normalized: list[str] = []
    unknown: list[str] = []
    for item in items:
        mapped = SOURCE_ALIASES.get(item)
        if not mapped:
            unknown.append(item)
            continue
        if mapped not in normalized:
            normalized.append(mapped)
    if unknown:
        allowed = ", ".join(sorted(set(SOURCE_ALIASES.keys())))
        raise typer.BadParameter(f"Unknown sources: {', '.join(unknown)}. Allowed: {allowed}")
    return normalized


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
    sources: Optional[list[str]],
) -> None:
    logger = logging.getLogger(__name__)
    started = time.monotonic()
    storage = Storage(db_path)
    try:
        cache = CacheStore(storage.conn)
        limiter = RateLimiter(max_rps)

        sources_used: list[str] = []
        collected: list[CanonicalPaper] = []
        selected_sources = sources or ["arxiv", "semantic_scholar"]
        request_count = 0
        cache_hit_count = 0
        source_counts: dict[str, int] = {}
        logger.info(
            "Search start %s",
            {"sources": selected_sources, "topk": topk, "since": since, "year_to": year_to},
        )
        logger.info("SQLite database: %s", db_path)

        queries_to_run = list(queries)
        if "arxiv" in selected_sources:
            async with httpx.AsyncClient() as client:
                for query in queries_to_run:
                    arxiv_key = _cache_key("arxiv", query, topk, since)
                    arxiv_data = None if no_cache else cache.get(arxiv_key, cache_ttl)
                    if arxiv_data is None:
                        logging.info("Fetching arXiv results for query: %s", query)
                        try:
                            request_count += 1
                            arxiv_papers = await search_arxiv(query, topk, since, client, limiter)
                        except Exception as exc:
                            logging.warning("arXiv search failed: %s", exc)
                            arxiv_papers = []
                        else:
                            cache.set(arxiv_key, [p.model_dump(mode="json") for p in arxiv_papers])
                    else:
                        logging.info("Using cached arXiv results for query: %s", query)
                        cache_hit_count += 1
                        arxiv_papers = [CanonicalPaper.model_validate(p) for p in arxiv_data]
                    if arxiv_papers:
                        if SOURCE_LABELS["arxiv"] not in sources_used:
                            sources_used.append(SOURCE_LABELS["arxiv"])
                        collected.extend(arxiv_papers)
                        source_counts["arxiv"] = source_counts.get("arxiv", 0) + len(arxiv_papers)

        s2_api_key = os.environ.get("S2_API_KEY")
        if "semantic_scholar" in selected_sources:
            if not _is_probably_valid_key(s2_api_key):
                typer.echo("Warning: S2_API_KEY is missing or invalid. Skipping Semantic Scholar.")
            else:
                headers = {"x-api-key": s2_api_key}
                async with httpx.AsyncClient(headers=headers) as client:
                    for query in queries_to_run:
                        s2_key = _cache_key("semantic_scholar", query, topk, since)
                        s2_data = None if no_cache else cache.get(s2_key, cache_ttl)
                        if s2_data is None:
                            logging.info("Fetching Semantic Scholar results for query: %s", query)
                            try:
                                request_count += 1
                                s2_papers = await search_semantic_scholar(
                                    query, topk, since, client, limiter
                                )
                            except Exception as exc:
                                logging.warning("Semantic Scholar search failed: %s", exc)
                                s2_papers = []
                            else:
                                cache.set(s2_key, [p.model_dump(mode="json") for p in s2_papers])
                        else:
                            logging.info("Using cached Semantic Scholar results for query: %s", query)
                            cache_hit_count += 1
                            s2_papers = [CanonicalPaper.model_validate(p) for p in s2_data]
                        if s2_papers:
                            if SOURCE_LABELS["semantic_scholar"] not in sources_used:
                                sources_used.append(SOURCE_LABELS["semantic_scholar"])
                            collected.extend(s2_papers)
                            source_counts["semantic_scholar"] = source_counts.get("semantic_scholar", 0) + len(
                                s2_papers
                            )

        if "inspire" in selected_sources:
            inspire_api_key = os.environ.get("INSPIRE_API_KEY")
            if inspire_api_key and not _is_probably_valid_key(inspire_api_key):
                typer.echo("Warning: INSPIRE_API_KEY looks invalid. Continuing without key.")
                inspire_api_key = None
            if not inspire_api_key:
                typer.echo("Warning: INSPIRE_API_KEY is not set. Continuing without key.")
            headers = {"Authorization": f"Bearer {inspire_api_key}"} if inspire_api_key else None
            async with httpx.AsyncClient(headers=headers) as client:
                for query in queries_to_run:
                    inspire_key = _cache_key("inspire", query, topk, since)
                    inspire_data = None if no_cache else cache.get(inspire_key, cache_ttl)
                    if inspire_data is None:
                        logging.info("Fetching INSPIRE-HEP results for query: %s", query)
                        try:
                            request_count += 1
                            inspire_papers = await search_inspire(query, topk, since, client, limiter)
                        except Exception as exc:
                            logging.warning("INSPIRE-HEP search failed: %s", exc)
                            inspire_papers = []
                        else:
                            cache.set(inspire_key, [p.model_dump(mode="json") for p in inspire_papers])
                    else:
                        logging.info("Using cached INSPIRE-HEP results for query: %s", query)
                        cache_hit_count += 1
                        inspire_papers = [CanonicalPaper.model_validate(p) for p in inspire_data]
                    if inspire_papers:
                        if SOURCE_LABELS["inspire"] not in sources_used:
                            sources_used.append(SOURCE_LABELS["inspire"])
                        collected.extend(inspire_papers)
                        source_counts["inspire"] = source_counts.get("inspire", 0) + len(inspire_papers)

        if "ads" in selected_sources:
            ads_api_token = os.environ.get("ADS_API_TOKEN")
            if not _is_probably_valid_key(ads_api_token):
                typer.echo("Warning: ADS_API_TOKEN is missing or invalid. Skipping NASA ADS.")
            else:
                headers = {"Authorization": f"Bearer {ads_api_token}"}
                async with httpx.AsyncClient(headers=headers) as client:
                    for query in queries_to_run:
                        ads_key = _cache_key("ads", query, topk, since)
                        ads_data = None if no_cache else cache.get(ads_key, cache_ttl)
                        if ads_data is None:
                            logging.info("Fetching NASA ADS results for query: %s", query)
                            try:
                                request_count += 1
                                ads_papers = await search_ads(query, topk, since, client, limiter)
                            except Exception as exc:
                                logging.warning("NASA ADS search failed: %s", exc)
                                ads_papers = []
                            else:
                                cache.set(ads_key, [p.model_dump(mode="json") for p in ads_papers])
                        else:
                            logging.info("Using cached NASA ADS results for query: %s", query)
                            cache_hit_count += 1
                            ads_papers = [CanonicalPaper.model_validate(p) for p in ads_data]
                        if ads_papers:
                            if SOURCE_LABELS["ads"] not in sources_used:
                                sources_used.append(SOURCE_LABELS["ads"])
                            collected.extend(ads_papers)
                            source_counts["ads"] = source_counts.get("ads", 0) + len(ads_papers)

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
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "Search done %s",
            {
                "elapsed_ms": elapsed_ms,
                "papers": len(merged),
                "requests": request_count,
                "cache_hits": cache_hit_count,
                "source_counts": source_counts,
            },
        )
        typer.echo(f"Saved {len(merged)} papers to {out_md}, {out_jsonl}, and {db_path}.")
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    finally:
        storage.close()


@app.command(
    help=(
        "Search papers from arXiv, Semantic Scholar, INSPIRE-HEP, and NASA ADS.\n"
        "Examples:\n"
        "  litscout search \"graph neural networks\" --topk 20\n"
        "  litscout search --query \"sign problem\" --query \"phase transition\" --keyword-rank\n"
        "  litscout search --queries-file queries.txt\n"
        "  litscout search \"higgs\" --sources arxiv,inspire,ads"
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
    sources: Optional[str] = typer.Option(
        None,
        "--sources",
        help="Comma-separated sources: arxiv,s2,inspire,ads (default: arxiv + s2 if key)",
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

    query_list = _load_queries(query, queries or [], queries_file)
    if not query_list:
        raise typer.BadParameter("no queries provided")
    if any(len(q) > MAX_QUERY_LEN for q in query_list):
        raise typer.BadParameter(f"query is too long (>{MAX_QUERY_LEN})")

    display_query = " | ".join(query_list)
    if out_md is None or out_jsonl is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = _slugify_query(query_list[0])
        out_dir = f"{slug}-{ts}"
        if out_md is None:
            out_md = os.path.join(out_dir, f"{slug}-{ts}.results_raw.md")
        if out_jsonl is None:
            out_jsonl = os.path.join(out_dir, f"{slug}-{ts}.results_raw.jsonl")
    log_file = None
    if out_md:
        log_dir = os.path.dirname(os.path.abspath(os.path.expanduser(out_md)))
        if log_dir:
            base = os.path.splitext(os.path.basename(out_md))[0]
            log_file = os.path.join(log_dir, f"{base}.log")
    _configure_logging(log_level, log_file)
    sources_selected = _parse_sources(sources)
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
            sources_selected,
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
    logger = logging.getLogger(__name__)
    started = time.monotonic()
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
    if not _is_probably_valid_key(api_key):
        raise RuntimeError(f"Invalid or missing API key for provider={skill.llm.provider}")
    client = build_client(config, api_key=api_key, limiter=limiter)
    logger.info(
        "Enrich start %s",
        {
            "provider": config.provider,
            "model": config.model,
            "base_url": config.base_url,
            "batch_size": batch_size,
        },
    )

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
    md_text = render_enriched_markdown(enriched, goal, skill.goals)
    with open(os.path.expanduser(out_md), "w", encoding="utf-8") as f:
        f.write(md_text)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    total_batches = (len(papers) + max(1, batch_size) - 1) // max(1, batch_size)
    logger.info(
        "Enrich done %s",
        {
            "elapsed_ms": elapsed_ms,
            "papers": len(enriched),
            "batches": total_batches,
            "requests": total_batches,
        },
    )
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
    log_file = None
    if out_md:
        log_dir = os.path.dirname(os.path.abspath(os.path.expanduser(out_md)))
        if log_dir:
            base = os.path.splitext(os.path.basename(out_md))[0]
            log_file = os.path.join(log_dir, f"{base}.log")
    _configure_logging(log_level, log_file)
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
