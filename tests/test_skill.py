from litscout.enrichment.skill import load_skill_profile


def test_skill_defaults_and_hash(tmp_path):
    path = tmp_path / "skill.md"
    path.write_text(
        """
version: "1.0"
llm:
  provider: "openai"
  model: "gpt-4.1-mini"
""".strip(),
        encoding="utf-8",
    )
    profile = load_skill_profile(str(path))
    assert profile.scoring.weights["relevance"] == 0.5
    assert profile.classification.max_tags == 6
    assert profile.skill_hash

    profile2 = load_skill_profile(str(path))
    assert profile.skill_hash == profile2.skill_hash
