from datetime import datetime, timezone

from litscout.ranking import score_recency, total_score


def test_recency_half_life():
    current_year = datetime.now(timezone.utc).year
    assert score_recency(current_year, 4) == 1.0
    score = score_recency(current_year - 4, 4)
    assert 0.49 <= score <= 0.51


def test_total_score_weights():
    weights = {"relevance": 0.5, "importance": 0.3, "recency": 0.2}
    total = total_score(weights, 1.0, 0.0, 0.0)
    assert total == 0.5
