import asyncio
import time

from litscout.core.rate_limit import RateLimiter


def test_rate_limiter_enforces_min_interval(monkeypatch):
    sleeps: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    limiter = RateLimiter(2.0)
    limiter._last_ts = time.monotonic()

    async def _run() -> None:
        await limiter.acquire()
        await limiter.acquire()

    asyncio.run(_run())

    assert len(sleeps) == 2
    assert all(delay > 0 for delay in sleeps)
    assert all(delay <= 0.5 for delay in sleeps)
