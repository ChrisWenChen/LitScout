from __future__ import annotations

import asyncio
from typing import Any

import httpx

from litscout.core.models import CanonicalPaper, SourceRecord, utc_now_iso
from litscout.core.normalize import normalize_doi
from litscout.core.rate_limit import RateLimiter


INSPIRE_API_URL = "https://inspirehep.net/api/literature"


def _first_string(values: list[Any] | None, key: str) -> str | None:
    if not values:
        return None
    for item in values:
        if isinstance(item, dict):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        elif isinstance(item, str) and item.strip():
            return item.strip()
    return None


def _parse_year(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_authors(authors: list[dict[str, Any]] | None) -> list[str]:
    if not authors:
        return []
    names: list[str] = []
    for author in authors:
        if not isinstance(author, dict):
            continue
        full_name = author.get("full_name") or author.get("name")
        if isinstance(full_name, str) and full_name.strip():
            names.append(full_name.strip())
            continue
        first = author.get("first_name")
        last = author.get("last_name")
        if isinstance(first, str) and isinstance(last, str):
            combined = f"{first.strip()} {last.strip()}".strip()
            if combined:
                names.append(combined)
    return [n for n in (name.strip() for name in names) if n]


def _extract_primary_url(item: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    links = item.get("links") or {}
    html_url = links.get("html")
    if isinstance(html_url, str) and html_url.strip():
        return html_url.strip()
    urls = metadata.get("urls") or []
    url = _first_string(urls, "value") or _first_string(urls, "url")
    return url


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


def _parse_inspire_payload(payload: dict[str, Any], since: int | None) -> list[CanonicalPaper]:
    results: list[CanonicalPaper] = []
    retrieved_at = utc_now_iso()

    hits = ((payload.get("hits") or {}).get("hits")) or []
    for item in hits:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}

        title = None
        titles = metadata.get("titles")
        if isinstance(titles, list):
            title = _first_string(titles, "title")
        if not title:
            fallback_title = metadata.get("title")
            if isinstance(fallback_title, str):
                title = fallback_title.strip()
        if not title:
            title = ""

        authors = _extract_authors(metadata.get("authors") or [])

        year = None
        publication_info = metadata.get("publication_info") or []
        if isinstance(publication_info, list):
            for info in publication_info:
                if not isinstance(info, dict):
                    continue
                year = _parse_year(info.get("year"))
                if year is not None:
                    break
        if year is None:
            preprint_date = metadata.get("preprint_date")
            if isinstance(preprint_date, str) and len(preprint_date) >= 4:
                year = _parse_year(preprint_date[:4])

        if since and year and year < since:
            continue

        abstract = None
        abstracts = metadata.get("abstracts")
        if isinstance(abstracts, list):
            abstract = _first_string(abstracts, "value")
        if not abstract:
            fallback_abs = metadata.get("abstract")
            if isinstance(fallback_abs, str) and fallback_abs.strip():
                abstract = fallback_abs.strip()

        doi = None
        dois = metadata.get("dois")
        if isinstance(dois, list):
            doi = _first_string(dois, "value")
        if not doi:
            fallback_doi = metadata.get("doi")
            if isinstance(fallback_doi, str):
                doi = fallback_doi.strip()
        doi = normalize_doi(doi)

        arxiv_id = None
        arxiv_eprints = metadata.get("arxiv_eprints")
        if isinstance(arxiv_eprints, list):
            arxiv_id = _first_string(arxiv_eprints, "value")

        venue = None
        if isinstance(publication_info, list):
            for info in publication_info:
                if not isinstance(info, dict):
                    continue
                venue = info.get("journal_title") or info.get("pub_title")
                if isinstance(venue, str) and venue.strip():
                    venue = venue.strip()
                    break

        primary_url = _extract_primary_url(item, metadata)
        control_number = metadata.get("control_number") or item.get("id")

        primary_category = None
        primary_arxiv_category = metadata.get("primary_arxiv_category") or {}
        if isinstance(primary_arxiv_category, dict):
            primary_category = primary_arxiv_category.get("term")

        source = SourceRecord(
            source_name="inspire",
            retrieved_at=retrieved_at,
            source_url=primary_url or (item.get("links") or {}).get("self"),
            extra={
                "control_number": control_number,
                "primary_category": primary_category,
            },
        )

        paper = CanonicalPaper(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            doi=doi,
            arxiv_id=arxiv_id,
            bibcode=None,
            url_primary=primary_url,
            venue=venue,
            sources=[source],
            dedup_cluster_id=f"inspire:{control_number or ''}",
            merge_confidence="low",
        )
        results.append(paper)

    return results


async def search_inspire(
    query: str, topk: int, since: int | None, client: httpx.AsyncClient, limiter: RateLimiter
) -> list[CanonicalPaper]:
    # TODO: add pagination when topk exceeds single-page size.
    params = {"q": query, "size": str(topk)}
    url = httpx.URL(INSPIRE_API_URL, params=params)
    resp = await _fetch_with_retries(client, str(url), limiter)
    return _parse_inspire_payload(resp.json(), since)
