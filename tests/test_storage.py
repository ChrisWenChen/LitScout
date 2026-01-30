import sqlite3

from litscout.core.models import CanonicalPaper, SourceRecord, utc_now_iso
from litscout.core.storage import Storage


def test_storage_roundtrip(tmp_path):
    db_path = tmp_path / "litscout.db"
    storage = Storage(str(db_path))
    paper = CanonicalPaper(
        title="Sample",
        authors=["Alice", "Bob"],
        year=2023,
        abstract="abstract",
        doi="10.1/xyz",
        arxiv_id="1234.5678",
        bibcode="2024A&A...123..456A",
        url_primary="http://example.com",
        venue="TestConf",
        sources=[
            SourceRecord(
                source_name="arxiv",
                retrieved_at=utc_now_iso(),
                source_url="http://arxiv.org/abs/1234.5678",
                extra={},
            )
        ],
        dedup_cluster_id="arxiv:1234.5678",
        merge_confidence="low",
    )
    storage.save_papers([paper])
    storage.close()

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT COUNT(*) FROM papers")
    assert cur.fetchone()[0] == 1
    cur = conn.execute("SELECT bibcode FROM papers")
    assert cur.fetchone()[0] == "2024A&A...123..456A"
    cur = conn.execute("SELECT COUNT(*) FROM authors")
    assert cur.fetchone()[0] == 2
    cur = conn.execute("SELECT COUNT(*) FROM sources")
    assert cur.fetchone()[0] == 1
    conn.close()
