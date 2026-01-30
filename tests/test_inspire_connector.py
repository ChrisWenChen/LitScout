import json
from pathlib import Path

from litscout.connectors.inspire import _parse_inspire_payload


def test_parse_inspire_payload_mapping():
    payload = json.loads(Path("tests/fixtures/inspire_sample.json").read_text(encoding="utf-8"))
    papers = _parse_inspire_payload(payload, since=None)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.title == "Inspire Title"
    assert paper.authors == ["Alice A.", "Bob B."]
    assert paper.year == 2021
    assert paper.abstract == "Inspire abstract."
    assert paper.doi == "10.1234/insp"
    assert paper.arxiv_id == "2101.00001"
    assert paper.venue == "Phys Rev D"
    assert paper.url_primary == "https://inspirehep.net/literature/123456"
    assert paper.sources[0].source_name == "inspire"
    assert paper.sources[0].extra.get("control_number") == 123456
