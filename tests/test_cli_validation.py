from datetime import datetime, timezone

from typer.testing import CliRunner

from litscout.cli import app, MAX_QUERY_LEN


runner = CliRunner()


def test_validation_topk_positive():
    result = runner.invoke(app, ["search", "test", "--topk", "0"])
    assert result.exit_code != 0
    assert "topk must be > 0" in result.output


def test_validation_since_not_future():
    future_year = datetime.now(timezone.utc).year + 1
    result = runner.invoke(app, ["search", "test", "--since", str(future_year)])
    assert result.exit_code != 0
    assert "since cannot be in the future" in result.output


def test_validation_year_to_not_future():
    future_year = datetime.now(timezone.utc).year + 1
    result = runner.invoke(app, ["search", "test", "--year-to", str(future_year)])
    assert result.exit_code != 0
    assert "year-to cannot be in the future" in result.output


def test_validation_since_not_greater_than_year_to():
    result = runner.invoke(app, ["search", "test", "--since", "2020", "--year-to", "2010"])
    assert result.exit_code != 0
    assert "since cannot be greater than year-to" in result.output


def test_validation_query_non_empty():
    result = runner.invoke(app, ["search", ""])  # empty query
    assert result.exit_code != 0
    assert "query must be non-empty" in result.output


def test_validation_query_max_length():
    long_query = "a" * (MAX_QUERY_LEN + 1)
    result = runner.invoke(app, ["search", long_query])
    assert result.exit_code != 0
    assert "query is too long" in result.output
