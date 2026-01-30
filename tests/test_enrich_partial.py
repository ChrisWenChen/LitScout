import asyncio

from litscout.core.models import CanonicalPaper, SourceRecord, utc_now_iso
from litscout.enrichment.models import SkillProfile
from litscout.enrichment.runner import enrich_papers


class _PartialClient:
    def __init__(self, include_id: str) -> None:
        self.include_id = include_id

    def prompt_hash(self, prompt: str) -> str:
        return "hash"

    async def generate_json(self, prompt: str, schema: dict):
        item = {
            "paper_id": self.include_id,
            "summary": "One line.",
            "methods": ["method"],
            "priority": "P2",
            "score": 0.4,
            "goal_scores": [{"goal_id": "G1", "score": 0.25}],
        }
        if "items" in schema.get("properties", {}):
            return {"items": [item]}
        return item


def _paper(title: str) -> CanonicalPaper:
    source = SourceRecord(
        source_name="arxiv",
        retrieved_at=utc_now_iso(),
        source_url=None,
        extra={},
    )
    return CanonicalPaper(
        title=title,
        authors=["A"],
        year=2024,
        abstract="Abstract",
        doi=None,
        arxiv_id=None,
        url_primary=None,
        venue=None,
        sources=[source],
        dedup_cluster_id="tmp",
        merge_confidence="low",
    )


def test_enrich_partial_batch_missing_items():
    p1 = _paper("Paper One")
    p2 = _paper("Paper Two")
    client = _PartialClient(str(p1.canonical_id))
    skill = SkillProfile()
    enriched = asyncio.run(
        enrich_papers([p1, p2], "goal", skill, client, batch_size=2, time_window="2026-01-29")
    )
    assert len(enriched) == 2
    by_id = {e.paper_id: e for e in enriched}
    assert by_id[str(p1.canonical_id)].ai is not None
    assert by_id[str(p2.canonical_id)].error is not None
