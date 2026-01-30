from typer.testing import CliRunner

import litscout.cli as cli


runner = CliRunner()


def test_enrich_requires_valid_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "short")
    result = runner.invoke(
        cli.app,
        [
            "enrich",
            "--goal",
            "test goal",
            "--skill",
            "docs/skill.md.example",
            "--provider",
            "openai",
            "--in-jsonl",
            "tests/fixtures/raw_sample.jsonl",
            "--out-jsonl",
            str(tmp_path / "out.jsonl"),
            "--out-md",
            str(tmp_path / "out.md"),
        ],
    )
    assert result.exit_code != 0
    output = result.output or ""
    if result.exception:
        output += str(result.exception)
    assert "Invalid or missing API key" in output
