import asyncio

from litscout.enrichment.models import SkillProfile
from litscout.enrichment.runner import enrich_papers, load_jsonl


class _BatchClient:
    def __init__(self, paper_ids: list[str]):
        self.paper_ids = paper_ids

    def prompt_hash(self, prompt: str) -> str:
        return "hash"

    async def generate_json(self, prompt: str, schema: dict):
        items = []
        for pid in self.paper_ids:
            items.append(
                {
                    "paper_id": pid,
                    "summary": "One line.",
                    "methods": ["method"],
                    "priority": "P2",
                    "score": 0.4,
                    "goal_scores": [{"goal_id": "G1", "score": 0.25}],
                }
            )
        if "items" in schema.get("properties", {}):
            return {"items": items}
        for item in items:
            if item["paper_id"] in prompt:
                return item
        return items[0]


def test_enrich_batch_from_fixture():
    papers = load_jsonl("tests/fixtures/raw_sample.jsonl")
    ids = [str(p.canonical_id) for p in papers]
    client = _BatchClient(ids)
    skill = SkillProfile()
    enriched = asyncio.run(
        enrich_papers(papers, "goal", skill, client, batch_size=2, time_window="2026-01-29")
    )
    assert len(enriched) == 3
    assert all(e.ai is not None for e in enriched)
    assert all(e.ranking is not None for e in enriched)
    assert all(e.classification is not None for e in enriched)
    assert all(e.provenance.input_hash for e in enriched)
