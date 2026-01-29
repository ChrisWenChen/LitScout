from datetime import datetime, timezone, timedelta
import sqlite3

from litscout.core.cache import CacheStore


def test_cache_ttl_expiry():
    conn = sqlite3.connect(":memory:")
    cache = CacheStore(conn)
    cache.set("k1", {"value": 1})

    old_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    conn.execute("UPDATE cache SET created_at = ? WHERE key = ?", (old_time, "k1"))
    conn.commit()

    assert cache.get("k1", ttl_seconds=60) is None
    cur = conn.execute("SELECT COUNT(*) FROM cache WHERE key = ?", ("k1",))
    assert cur.fetchone()[0] == 0


def test_cache_size_cap():
    conn = sqlite3.connect(":memory:")
    cache = CacheStore(conn)
    cache.set("k1", {"value": 1})
    cache.set("k2", {"value": 2})
    cache.set("k3", {"value": 3})

    removed = cache.cleanup(ttl_seconds=None, max_entries=2)
    assert removed == 1
    cur = conn.execute("SELECT COUNT(*) FROM cache")
    assert cur.fetchone()[0] == 2
