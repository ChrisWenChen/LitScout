from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkillScoring(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {"relevance": 0.5, "importance": 0.3, "recency": 0.2}
    )
    recency_half_life_years: float = 4.0
    use_goals: bool = False
    goal_match_weight: float = 0.2


class SkillClassification(BaseModel):
    taxonomy: list[str] = Field(
        default_factory=lambda: [
            "Survey/Meta",
            "Theory",
            "Method",
            "System",
            "Benchmark/Dataset",
            "Application",
        ]
    )
    max_tags: int = 6


class SkillOutput(BaseModel):
    uncertainty_style: str = "explicit"
    evidence_style: str = "metadata_only"


class SkillLLM(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    max_tokens: int = 800
    seed: Optional[int] = None
    base_url: Optional[str] = None


class ResearchGoal(BaseModel):
    goal_id: str
    name: str
    description: str
    signals: list[str] = Field(default_factory=list)
    negative_signals: list[str] = Field(default_factory=list)


class SkillProfile(BaseModel):
    version: str = "1.0"
    language: str = "zh"
    scoring: SkillScoring = Field(default_factory=SkillScoring)
    classification: SkillClassification = Field(default_factory=SkillClassification)
    output: SkillOutput = Field(default_factory=SkillOutput)
    llm: SkillLLM = Field(default_factory=SkillLLM)
    goals: list[ResearchGoal] = Field(default_factory=list)
    skill_hash: Optional[str] = None


class CanonicalSnapshot(BaseModel):
    canonical_id: str
    title: str
    authors: list[str]
    year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url_primary: Optional[str] = None
    venue: Optional[str] = None
    sources: list[dict[str, Any]] = Field(default_factory=list)


class RankingBreakdown(BaseModel):
    relevance: float
    importance: float
    recency: float
    total: float
    rationale: list[str]

    @field_validator("relevance", "importance", "recency", "total")
    @classmethod
    def _bounded(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("score must be within [0,1]")
        return v


class ClassificationResult(BaseModel):
    category_path: list[str]
    tags: list[str]
    rationale: list[str]


class GoalScore(BaseModel):
    goal_id: str
    score: float
    evidence: list[str]
    confidence: Literal["high", "medium", "low"]

    @field_validator("score")
    @classmethod
    def _score_bounded(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("goal score must be within [0,1]")
        return v


class LiteGoalScore(BaseModel):
    goal_id: str
    score: float = 0.0

    @field_validator("score")
    @classmethod
    def _score_bounded(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("goal score must be within [0,1]")
        return round(v, 2)


class LiteInsight(BaseModel):
    paper_id: str
    summary: str
    methods: list[str] = Field(default_factory=list)
    priority: Literal["P0", "P1", "P2", "P3"]
    score: float
    goal_scores: list[LiteGoalScore] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def _score_bounded(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("score must be within [0,1]")
        return v

    @model_validator(mode="after")
    def _clean_goal_scores(self) -> "LiteInsight":
        if not self.goal_scores:
            return self
        cleaned: list[LiteGoalScore] = []
        for gs in self.goal_scores:
            if not gs.goal_id or not gs.goal_id.strip():
                continue
            cleaned.append(gs)
        self.goal_scores = cleaned
        return self


class Provenance(BaseModel):
    provider: str
    model: str
    temperature: float
    seed: Optional[int] = None
    prompt_hash: str
    skill_hash: str
    input_hash: str
    timestamp: str = Field(default_factory=utc_now_iso)
    time_window: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)
    retries: int = 0


class EnrichedPaper(BaseModel):
    paper_id: str
    canonical: CanonicalSnapshot
    ai: Optional[LiteInsight] = None
    ranking: Optional[RankingBreakdown] = None
    classification: Optional[ClassificationResult] = None
    provenance: Provenance
    error: Optional[str] = None


class LLMOutputBatch(BaseModel):
    items: list[LiteInsight]

    @model_validator(mode="after")
    def _ensure_unique_ids(self) -> "LLMOutputBatch":
        ids = [item.paper_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("paper_id must be unique in batch")
        return self
