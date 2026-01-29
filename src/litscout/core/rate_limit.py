import asyncio
import time


class RateLimiter:
    def __init__(self, max_per_second: float) -> None:
        self._min_interval = 1.0 / max_per_second if max_per_second > 0 else 0
        self._lock = asyncio.Semaphore(1)
        self._last_ts = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait_for = self._min_interval - (now - self._last_ts)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_ts = time.monotonic()
