import asyncio
from datetime import datetime, timezone

from litscout.core.models import CanonicalPaper, SourceRecord
from litscout.enrichment.models import SkillProfile
from litscout.enrichment.runner import enrich_papers


class _MockClient:
    def __init__(self):
        self.calls = 0

    def prompt_hash(self, prompt: str) -> str:
        return "hash"

    async def generate_json(self, prompt: str, schema: dict):
        self.calls += 1
        if self.calls == 1:
            return {"bad": "payload"}
        return {
            "paper_id": "00000000-0000-0000-0000-000000000001",
            "summary": "One line.",
            "methods": ["method"],
            "priority": "P1",
            "score": 0.8,
            "goal_scores": [{"goal_id": "G1", "score": 0.7}],
        }


def _paper() -> CanonicalPaper:
    source = SourceRecord(
        source_name="arxiv",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        source_url="https://example.com",
        extra={},
    )
    return CanonicalPaper(
        canonical_id="00000000-0000-0000-0000-000000000001",
        title="Test",
        authors=["A"],
        year=2024,
        abstract="Abstract",
        doi=None,
        arxiv_id="1234.5678",
        url_primary="https://example.com",
        venue=None,
        sources=[source],
        dedup_cluster_id="arxiv:1234.5678",
        merge_confidence="low",
    )


def test_schema_retry_and_enrich():
    client = _MockClient()
    skill = SkillProfile()
    enriched = asyncio.run(
        enrich_papers([_paper()], "goal", skill, client, batch_size=1, time_window="2026-01-29")
    )
    assert len(enriched) == 1
    assert enriched[0].ai is not None
    assert enriched[0].ranking is not None
    assert enriched[0].classification is not None
    assert enriched[0].provenance.retries == 1
