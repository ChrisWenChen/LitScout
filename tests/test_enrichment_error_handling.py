import json

import pytest
from pydantic import ValidationError

from litscout.enrichment.models import LiteInsight


def test_handle_missing_field():
    response = {}
    with pytest.raises(ValidationError):
        LiteInsight.model_validate(response)


def test_handle_malformed_response():
    with open("tests/fixtures/malformed_llm_response.json") as f:
        response = json.load(f)
    with pytest.raises(ValidationError):
        LiteInsight.model_validate(response)


def test_handle_edge_case():
    with open("tests/fixtures/edge_case_papers.jsonl") as f:
        for line in f:
            response = json.loads(line)
            with pytest.raises(ValidationError):
                LiteInsight.model_validate(response)


def test_handle_unexpected_data_type():
    response = {
        "paper_id": 123,
        "summary": "One line.",
        "methods": [],
        "priority": "P2",
        "score": 0.5,
        "goal_scores": [],
    }
    with pytest.raises(ValidationError):
        LiteInsight.model_validate(response)
