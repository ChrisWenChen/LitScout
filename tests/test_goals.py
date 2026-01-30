import asyncio
from datetime import datetime, timezone

from litscout.core.models import CanonicalPaper, SourceRecord
from litscout.enrichment.models import ResearchGoal, SkillProfile
from litscout.enrichment.runner import enrich_papers
from litscout.enrichment.skill import load_skill_profile


class _MockClient:
    def __init__(self, payload):
        self.payload = payload

    def prompt_hash(self, prompt: str) -> str:
        return "hash"

    async def generate_json(self, prompt: str, schema: dict):
        return self.payload


def _paper(year=None, abstract="Abstract"):
    source = SourceRecord(
        source_name="arxiv",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        source_url="https://example.com",
        extra={},
    )
    return CanonicalPaper(
        canonical_id="00000000-0000-0000-0000-000000000010",
        title="Test",
        authors=["A"],
        year=year,
        abstract=abstract,
        doi=None,
        arxiv_id="1234.5678",
        url_primary="https://example.com",
        venue=None,
        sources=[source],
        dedup_cluster_id="arxiv:1234.5678",
        merge_confidence="low",
    )


def test_skill_goal_parsing(tmp_path):
    text = """
version: "1.0"

# Research Goals
## G1: Sign problem

description: |
  focus on sign problem mitigation
signals:
  - "sign problem"
negative_signals:
  - "classical"
""".strip()
    path = tmp_path / "skill.md"
    path.write_text(text, encoding="utf-8")
    profile = load_skill_profile(str(path))
    assert len(profile.goals) == 1
    assert profile.goals[0].goal_id == "G1"
    assert "sign problem" in profile.goals[0].signals


def test_goal_scores_present():
    payload = {
        "paper_id": "00000000-0000-0000-0000-000000000010",
        "summary": "One line.",
        "methods": ["method"],
        "priority": "P3",
        "score": 0.1,
        "goal_scores": [],
    }
    client = _MockClient(payload)
    skill = SkillProfile(goals=[ResearchGoal(goal_id="G1", name="G1", description="d", signals=[])])
    paper = _paper(year=None, abstract=None)
    enriched = asyncio.run(enrich_papers([paper], "goal", skill, client))
    assert enriched[0].ai is not None
    assert enriched[0].ai.priority == "P3"
    assert enriched[0].ai.goal_scores
    assert [gs.goal_id for gs in enriched[0].ai.goal_scores] == ["G1"]


def test_goal_scores_order_and_rounding():
    payload = {
        "paper_id": "00000000-0000-0000-0000-000000000010",
        "summary": "One line.",
        "methods": ["method"],
        "priority": "P2",
        "score": 0.3,
        "goal_scores": [
            {"goal_id": "G2", "score": 0.777},
            {"goal_id": "G1", "score": 0.333},
        ],
    }
    client = _MockClient(payload)
    skill = SkillProfile(
        goals=[
            ResearchGoal(goal_id="G1", name="G1", description="d", signals=[]),
            ResearchGoal(goal_id="G2", name="G2", description="d", signals=[]),
        ]
    )
    paper = _paper(year=2024, abstract="Abstract")
    enriched = asyncio.run(enrich_papers([paper], "goal", skill, client))
    goal_scores = enriched[0].ai.goal_scores
    assert [gs.goal_id for gs in goal_scores] == ["G1", "G2"]
    assert goal_scores[0].score == 0.33
    assert goal_scores[1].score == 0.78
