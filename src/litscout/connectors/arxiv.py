from __future__ import annotations

import asyncio
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Iterable

import httpx

from litscout.core.models import CanonicalPaper, SourceRecord, utc_now_iso
from litscout.core.normalize import normalize_doi
from litscout.core.rate_limit import RateLimiter


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


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


def _parse_arxiv_feed(xml_text: str, since: int | None) -> list[CanonicalPaper]:
    root = ET.fromstring(xml_text)
    entries = root.findall(f"{ATOM_NS}entry")
    results: list[CanonicalPaper] = []
    retrieved_at = utc_now_iso()

    for entry in entries:
        title = (entry.findtext(f"{ATOM_NS}title") or "").strip().replace("\n", " ")
        summary = (entry.findtext(f"{ATOM_NS}summary") or "").strip().replace("\n", " ")
        published = entry.findtext(f"{ATOM_NS}published")
        year = None
        if published and len(published) >= 4:
            try:
                year = int(published[:4])
            except ValueError:
                year = None
        if since and year and year < since:
            continue

        authors = [a.findtext(f"{ATOM_NS}name") or "" for a in entry.findall(f"{ATOM_NS}author")]
        authors = [a.strip() for a in authors if a.strip()]

        arxiv_id = None
        entry_id = entry.findtext(f"{ATOM_NS}id")
        if entry_id:
            arxiv_id = entry_id.rsplit("/", 1)[-1]

        pdf_url = None
        html_url = None
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
            if link.attrib.get("rel") == "alternate":
                html_url = link.attrib.get("href")

        primary_cat = None
        primary_elem = entry.find(f"{ARXIV_NS}primary_category")
        if primary_elem is not None:
            primary_cat = primary_elem.attrib.get("term")

        doi_elem = entry.find(f"{ARXIV_NS}doi")
        doi = normalize_doi(doi_elem.text if doi_elem is not None else None)

        source = SourceRecord(
            source_name="arxiv",
            retrieved_at=retrieved_at,
            source_url=html_url or pdf_url,
            extra={"primary_category": primary_cat},
        )

        paper = CanonicalPaper(
            title=title,
            authors=authors,
            year=year,
            abstract=summary or None,
            doi=doi,
            arxiv_id=arxiv_id,
            url_primary=pdf_url or html_url,
            venue=None,
            sources=[source],
            dedup_cluster_id=f"arxiv:{arxiv_id or ''}",
            merge_confidence="low",
        )
        results.append(paper)

    return results


async def search_arxiv(
    query: str, topk: int, since: int | None, client: httpx.AsyncClient, limiter: RateLimiter
) -> list[CanonicalPaper]:
    encoded_query = urllib.parse.quote(f"all:{query}")
    url = f"{ARXIV_API_URL}?search_query={encoded_query}&start=0&max_results={topk}"
    resp = await _fetch_with_retries(client, url, limiter)
    return _parse_arxiv_feed(resp.text, since)
