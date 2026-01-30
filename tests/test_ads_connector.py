import json
from pathlib import Path

from litscout.connectors.ads import _parse_ads_payload


def test_parse_ads_payload_mapping():
    payload = json.loads(Path("tests/fixtures/ads_sample.json").read_text(encoding="utf-8"))
    papers = _parse_ads_payload(payload, since=None)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.title == "ADS Title"
    assert paper.authors == ["Alice", "Bob"]
    assert paper.year == 2022
    assert paper.abstract == "ADS abstract"
    assert paper.doi == "10.9999/ads"
    assert paper.arxiv_id == "2201.00001"
    assert paper.bibcode == "2022ApJ...123..456A"
    assert paper.venue == "ApJ"
    assert paper.url_primary == "https://ui.adsabs.harvard.edu/abs/2022ApJ...123..456A/abstract"
    assert paper.sources[0].source_name == "ads"
    assert paper.sources[0].extra.get("bibcode") == "2022ApJ...123..456A"
