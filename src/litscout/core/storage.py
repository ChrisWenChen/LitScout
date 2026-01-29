from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Iterable
import hashlib

from .models import CanonicalPaper, SourceRecord


class Storage:
    def __init__(self, db_path: str) -> None:
        self.db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
                canonical_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                year INTEGER,
                abstract TEXT,
                doi TEXT,
                arxiv_id TEXT,
                url_primary TEXT,
                venue TEXT,
                dedup_cluster_id TEXT NOT NULL,
                merge_confidence TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS authors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS paper_authors (
                paper_id TEXT NOT NULL,
                author_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                PRIMARY KEY (paper_id, author_id),
                FOREIGN KEY (paper_id) REFERENCES papers(canonical_id),
                FOREIGN KEY (author_id) REFERENCES authors(id)
            );

            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                source_url TEXT,
                extra_json TEXT,
                source_hash TEXT,
                FOREIGN KEY (paper_id) REFERENCES papers(canonical_id)
            );
            """
        )
        # Ensure source_hash exists and is indexed for idempotency across runs.
        if not self._column_exists("sources", "source_hash"):
            self.conn.execute("ALTER TABLE sources ADD COLUMN source_hash TEXT")
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_paper_hash ON sources(paper_id, source_hash)"
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def save_papers(self, papers: Iterable[CanonicalPaper]) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self.conn:
            for paper in papers:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO papers (
                        canonical_id, title, year, abstract, doi, arxiv_id, url_primary,
                        venue, dedup_cluster_id, merge_confidence, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(paper.canonical_id),
                        paper.title,
                        paper.year,
                        paper.abstract,
                        paper.doi,
                        paper.arxiv_id,
                        paper.url_primary,
                        paper.venue,
                        paper.dedup_cluster_id,
                        paper.merge_confidence,
                        created_at,
                    ),
                )

                for idx, author in enumerate(paper.authors):
                    author_id = self._get_or_create_author(author)
                    self.conn.execute(
                        "INSERT OR REPLACE INTO paper_authors (paper_id, author_id, position) VALUES (?, ?, ?)",
                        (str(paper.canonical_id), author_id, idx),
                    )

                for source in paper.sources:
                    self._insert_source(paper.canonical_id, source)

    def _get_or_create_author(self, name: str) -> int:
        cur = self.conn.execute("SELECT id FROM authors WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            return int(row[0])
        cur = self.conn.execute("INSERT INTO authors (name) VALUES (?)", (name,))
        return int(cur.lastrowid)

    def _insert_source(self, paper_id: str, source: SourceRecord) -> None:
        source_hash = self._hash_source(source)
        self.conn.execute(
            """
            INSERT OR IGNORE INTO sources (
                paper_id, source_name, retrieved_at, source_url, extra_json, source_hash
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(paper_id),
                source.source_name,
                source.retrieved_at,
                source.source_url,
                json.dumps(source.extra, ensure_ascii=False),
                source_hash,
            ),
        )

    def _hash_source(self, source: SourceRecord) -> str:
        payload = {
            "source_name": source.source_name,
            "source_url": source.source_url,
            "extra": source.extra,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _column_exists(self, table: str, column: str) -> bool:
        cur = self.conn.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]
        return column in cols
