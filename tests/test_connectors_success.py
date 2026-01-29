from litscout.connectors.arxiv import _parse_arxiv_feed
from litscout.connectors.semanticscholar import _parse_s2_payload


def test_parse_arxiv_feed_success_filters_since():
    xml = """
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/1234.5678v1</id>
        <title>Test Title</title>
        <summary>Test summary</summary>
        <published>2022-05-01T00:00:00Z</published>
        <author><name>Alice</name></author>
        <author><name>Bob</name></author>
        <link rel="alternate" href="https://arxiv.org/abs/1234.5678v1" />
        <link title="pdf" href="https://arxiv.org/pdf/1234.5678v1" />
        <arxiv:primary_category term="cs.LG" />
        <arxiv:doi>10.1000/ABC</arxiv:doi>
      </entry>
      <entry>
        <id>http://arxiv.org/abs/1111.2222v1</id>
        <title>Old Paper</title>
        <summary>Old summary</summary>
        <published>2018-01-01T00:00:00Z</published>
        <author><name>Carol</name></author>
        <link rel="alternate" href="https://arxiv.org/abs/1111.2222v1" />
      </entry>
    </feed>
    """

    papers = _parse_arxiv_feed(xml, since=2020)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.title == "Test Title"
    assert paper.authors == ["Alice", "Bob"]
    assert paper.year == 2022
    assert paper.doi == "10.1000/abc"
    assert paper.arxiv_id == "1234.5678v1"
    assert paper.url_primary == "https://arxiv.org/pdf/1234.5678v1"
    assert paper.sources[0].extra.get("primary_category") == "cs.LG"


def test_parse_s2_payload_success():
    payload = {
        "data": [
            {
                "title": "S2 Paper",
                "authors": [{"name": "Dana"}],
                "year": 2023,
                "abstract": "S2 abstract",
                "url": "https://www.semanticscholar.org/paper/abc",
                "citationCount": 5,
                "venue": "S2Conf",
                "externalIds": {"DOI": "10.5555/XYZ", "ArXiv": "9999.8888"},
                "paperId": "abc",
            }
        ]
    }

    papers = _parse_s2_payload(payload, since=None)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.title == "S2 Paper"
    assert paper.authors == ["Dana"]
    assert paper.year == 2023
    assert paper.abstract == "S2 abstract"
    assert paper.doi == "10.5555/xyz"
    assert paper.arxiv_id == "9999.8888"
    assert paper.venue == "S2Conf"
    assert paper.sources[0].extra.get("citationCount") == 5
