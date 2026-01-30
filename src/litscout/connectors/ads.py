from __future__ import annotations

import asyncio
from typing import Any

import httpx

from litscout.core.models import CanonicalPaper, SourceRecord, utc_now_iso
from litscout.core.normalize import normalize_doi
from litscout.core.rate_limit import RateLimiter


ADS_API_URL = "https://api.adsabs.harvard.edu/v1/search/query"


def _parse_year(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_title(value: Any) -> str:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str):
            return first.strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _extract_first(value: Any) -> str | None:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_arxiv_id(identifiers: list[Any] | None) -> str | None:
    if not identifiers:
        return None
    for ident in identifiers:
        if not isinstance(ident, str):
            continue
        lower = ident.lower()
        if lower.startswith("arxiv:"):
            return ident.split(":", 1)[-1].strip()
    return None


async def _fetch_with_retries(
    client: httpx.AsyncClient, url: str, limiter: RateLimiter, retries: int = 2
) -> httpx.Response:
    backoff = 0.5
    for attempt in range(retries + 1):
        await limiter.acquire()
        try:
            resp = await client.get(url, timeout=30.0)
            if resp.status_code in (429,) or resp.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp
        except (httpx.RequestError, httpx.HTTPStatusError):
            if attempt == retries:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


def _parse_ads_payload(payload: dict[str, Any], since: int | None) -> list[CanonicalPaper]:
    results: list[CanonicalPaper] = []
    retrieved_at = utc_now_iso()

    docs = ((payload.get("response") or {}).get("docs")) or []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        year = _parse_year(doc.get("year"))
        if since and year and year < since:
            continue

        title = _extract_title(doc.get("title"))
        authors: list[str] = []
        authors_raw = doc.get("author")
        if isinstance(authors_raw, list):
            authors = [a.strip() for a in authors_raw if isinstance(a, str) and a.strip()]
        elif isinstance(authors_raw, str) and authors_raw.strip():
            authors = [authors_raw.strip()]
        abstract = doc.get("abstract")
        doi = normalize_doi(_extract_first(doc.get("doi")))
        arxiv_id = _extract_arxiv_id(doc.get("identifier") or [])
        bibcode = doc.get("bibcode") if isinstance(doc.get("bibcode"), str) else None
        venue = _extract_first(doc.get("pub")) or _extract_first(doc.get("pub_raw"))

        primary_url = None
        if bibcode:
            primary_url = f"https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract"

        source = SourceRecord(
            source_name="ads",
            retrieved_at=retrieved_at,
            source_url=primary_url,
            extra={"bibcode": bibcode},
        )

        paper = CanonicalPaper(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            doi=doi,
            arxiv_id=arxiv_id,
            bibcode=bibcode,
            url_primary=primary_url,
            venue=venue,
            sources=[source],
            dedup_cluster_id=f"ads:{bibcode or ''}",
            merge_confidence="low",
        )
        results.append(paper)

    return results


async def search_ads(
    query: str, topk: int, since: int | None, client: httpx.AsyncClient, limiter: RateLimiter
) -> list[CanonicalPaper]:
    fields = "title,author,year,abstract,doi,identifier,bibcode,pub,pub_raw,arxiv_class"
    params = {"q": query, "rows": str(topk), "fl": fields}
    url = httpx.URL(ADS_API_URL, params=params)
    resp = await _fetch_with_retries(client, str(url), limiter)
    return _parse_ads_payload(resp.json(), since)
