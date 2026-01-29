from __future__ import annotations

import asyncio
from typing import Any

import httpx

from litscout.core.models import CanonicalPaper, SourceRecord, utc_now_iso
from litscout.core.normalize import normalize_doi
from litscout.core.rate_limit import RateLimiter


S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


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


def _parse_s2_payload(payload: dict[str, Any], since: int | None) -> list[CanonicalPaper]:
    results: list[CanonicalPaper] = []
    retrieved_at = utc_now_iso()

    for item in payload.get("data", []) or []:
        year = item.get("year")
        if since and year and year < since:
            continue

        authors = [a.get("name", "") for a in item.get("authors", [])]
        authors = [a.strip() for a in authors if a.strip()]

        external_ids = item.get("externalIds") or {}
        doi = normalize_doi(item.get("doi") or external_ids.get("DOI"))
        arxiv_id = external_ids.get("ArXiv") or external_ids.get("arXiv")

        source = SourceRecord(
            source_name="semantic_scholar",
            retrieved_at=retrieved_at,
            source_url=item.get("url"),
            extra={
                "paperId": item.get("paperId"),
                "citationCount": item.get("citationCount"),
            },
        )

        paper = CanonicalPaper(
            title=item.get("title") or "",
            authors=authors,
            year=year,
            abstract=item.get("abstract"),
            doi=doi,
            arxiv_id=arxiv_id,
            url_primary=item.get("url"),
            venue=item.get("venue"),
            sources=[source],
            dedup_cluster_id=f"s2:{item.get('paperId')}",
            merge_confidence="low",
        )
        results.append(paper)

    return results


async def search_semantic_scholar(
    query: str,
    topk: int,
    since: int | None,
    client: httpx.AsyncClient,
    limiter: RateLimiter,
) -> list[CanonicalPaper]:
    # Use a conservative field list to avoid 400s on unsupported fields.
    fields = "title,authors,year,abstract,url,citationCount,venue,externalIds"
    params = {"query": query, "limit": str(topk), "fields": fields}
    url = httpx.URL(S2_API_URL, params=params)
    resp = await _fetch_with_retries(client, str(url), limiter)
    return _parse_s2_payload(resp.json(), since)
