import json

from litscout.cli import CACHE_VERSION, _cache_key


def test_cache_key_includes_version():
    key = _cache_key("arxiv", "test", 5, 2020)
    payload = json.loads(key)
    assert payload["version"] == CACHE_VERSION
