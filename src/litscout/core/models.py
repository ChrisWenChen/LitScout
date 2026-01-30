from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SourceRecord(BaseModel):
    source_name: Literal["arxiv", "semantic_scholar", "inspire", "ads"]
    retrieved_at: str
    source_url: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class CanonicalPaper(BaseModel):
    canonical_id: UUID = Field(default_factory=uuid4)
    title: str
    authors: list[str]
    year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    bibcode: Optional[str] = None
    url_primary: Optional[str] = None
    venue: Optional[str] = None
    sources: list[SourceRecord] = Field(default_factory=list)
    dedup_cluster_id: str
    merge_confidence: Literal["high", "medium", "low"]

    def model_dump_jsonl(self) -> str:
        return self.model_dump_json()
