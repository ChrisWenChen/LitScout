from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

import sqlite3

from litscout.core.models import CanonicalPaper, SourceRecord
from litscout.core.storage import Storage


def _paper(idx: int) -> CanonicalPaper:
    source = SourceRecord(
        source_name="arxiv",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        source_url=f"https://example.com/{idx}",
        extra={},
    )
    return CanonicalPaper(
        canonical_id=uuid4(),
        title=f"Title {idx}",
        authors=["Author One", "Author Two"],
        year=2024,
        abstract="Abstract",
        doi=None,
        arxiv_id=f"1234.{idx:04d}",
        url_primary=f"https://example.com/{idx}.pdf",
        venue=None,
        sources=[source],
        dedup_cluster_id=f"arxiv:{idx}",
        merge_confidence="low",
    )


def test_storage_concurrency_settings(tmp_path):
    db_path = tmp_path / "litscout.db"
    storage = Storage(str(db_path))
    try:
        mode = storage.conn.execute("PRAGMA journal_mode").fetchone()[0]
        timeout = storage.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert str(mode).lower() == "wal"
        assert int(timeout) == 5000
    finally:
        storage.close()


def test_storage_concurrent_writes(tmp_path):
    db_path = tmp_path / "litscout.db"

    try:
        papers_a = [_paper(i) for i in range(5)]
        papers_b = [_paper(i + 100) for i in range(5)]

        def _save(papers):
            storage = Storage(str(db_path))
            try:
                storage.save_papers(papers)
            finally:
                storage.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(_save, papers_a)
            fut_b = pool.submit(_save, papers_b)
            fut_a.result()
            fut_b.result()

        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT COUNT(*) FROM papers")
        assert cur.fetchone()[0] >= 10
        conn.close()
    finally:
        pass


def test_storage_bulk_insert(tmp_path):
    db_path = tmp_path / "litscout.db"
    storage = Storage(str(db_path))
    try:
        papers = [_paper(i) for i in range(200)]
        storage.save_papers(papers)

        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT COUNT(*) FROM papers")
        assert cur.fetchone()[0] >= 200
        conn.close()
    finally:
        storage.close()
