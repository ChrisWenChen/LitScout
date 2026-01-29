import asyncio

import httpx
import pytest

from litscout.connectors.arxiv import _fetch_with_retries as arxiv_fetch
from litscout.connectors.semanticscholar import _fetch_with_retries as s2_fetch
from litscout.core.rate_limit import RateLimiter


async def _always_500(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, request=request)


def _run(coro):
    return asyncio.run(coro)


def test_arxiv_fetch_retries(monkeypatch):
    async def _fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    transport = httpx.MockTransport(_always_500)

    async def _call():
        limiter = RateLimiter(1000)
        async with httpx.AsyncClient(transport=transport) as client:
            await arxiv_fetch(client, "http://example.com", limiter, retries=1)

    with pytest.raises(httpx.HTTPStatusError):
        _run(_call())


def test_s2_fetch_retries(monkeypatch):
    async def _fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    transport = httpx.MockTransport(_always_500)

    async def _call():
        limiter = RateLimiter(1000)
        async with httpx.AsyncClient(transport=transport) as client:
            await s2_fetch(client, "http://example.com", limiter, retries=1)

    with pytest.raises(httpx.HTTPStatusError):
        _run(_call())
