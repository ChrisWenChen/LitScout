import asyncio

from litscout.enrichment.models import SkillProfile
from litscout.enrichment.runner import enrich_papers
from litscout.core.models import CanonicalPaper, SourceRecord
from datetime import datetime, timezone


class _BadBatchClient:
    def __init__(self):
        self.calls = 0

    def prompt_hash(self, prompt: str) -> str:
        return "hash"

    async def generate_json(self, prompt: str, schema: dict):
        self.calls += 1
        if self.calls == 1:
            return {"items": [{"paper_id": "bad"}]}  # malformed items (missing fields)
        return {"items": []}  # valid shape but empty -> triggers errors for all papers


def _paper(pid: str) -> CanonicalPaper:
    source = SourceRecord(
        source_name="arxiv",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        source_url="https://example.com",
        extra={},
    )
    return CanonicalPaper(
        canonical_id=pid,
        title=f"Title {pid}",
        authors=["A"],
        year=2024,
        abstract="Abstract",
        doi=None,
        arxiv_id=None,
        url_primary="https://example.com",
        venue=None,
        sources=[source],
        dedup_cluster_id=f"arxiv:{pid}",
        merge_confidence="low",
    )


def test_batch_parse_failure_marks_errors():
    client = _BadBatchClient()
    skill = SkillProfile()
    papers = [_paper("00000000-0000-0000-0000-000000000001"), _paper("00000000-0000-0000-0000-000000000002")]
    enriched = asyncio.run(enrich_papers(papers, "goal", skill, client, batch_size=2, max_retries=1))
    assert len(enriched) == 2
    assert all(e.ai is None for e in enriched)
    assert all(e.error is not None for e in enriched)
