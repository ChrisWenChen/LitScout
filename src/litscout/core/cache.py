import json
from datetime import datetime, timezone
from typing import Any, Optional

import sqlite3


class CacheStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._init_table()

    def _init_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_created_at ON cache(created_at)"
        )
        self._conn.commit()

    def get(self, key: str, ttl_seconds: int | None = None) -> Optional[Any]:
        cur = self._conn.execute(
            "SELECT value_json, created_at FROM cache WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        if not row:
            return None
        value_json, created_at = row[0], row[1]
        if ttl_seconds is not None and ttl_seconds > 0:
            try:
                created_ts = datetime.fromisoformat(created_at).timestamp()
            except ValueError:
                created_ts = 0
            cutoff = datetime.now(timezone.utc).timestamp() - ttl_seconds
            if created_ts < cutoff:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._conn.commit()
                return None
        return json.loads(value_json)

    def set(self, key: str, value: Any) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(value, ensure_ascii=False)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value_json, created_at) VALUES (?, ?, ?)",
            (key, payload, created_at),
        )
        self._conn.commit()

    def cleanup(self, ttl_seconds: int | None, max_entries: int | None) -> int:
        removed = 0
        if ttl_seconds is not None and ttl_seconds > 0:
            cutoff = datetime.now(timezone.utc).timestamp() - ttl_seconds
            cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
            cur = self._conn.execute(
                "DELETE FROM cache WHERE created_at < ?", (cutoff_iso,)
            )
            removed += cur.rowcount if cur.rowcount is not None else 0
        if max_entries is not None and max_entries > 0:
            cur = self._conn.execute("SELECT COUNT(*) FROM cache")
            total = int(cur.fetchone()[0])
            if total > max_entries:
                to_delete = total - max_entries
                del_cur = self._conn.execute(
                    """
                    DELETE FROM cache
                    WHERE key IN (
                        SELECT key FROM cache
                        ORDER BY created_at ASC
                        LIMIT ?
                    )
                    """,
                    (to_delete,),
                )
                removed += del_cur.rowcount if del_cur.rowcount is not None else 0
        self._conn.commit()
        return removed
