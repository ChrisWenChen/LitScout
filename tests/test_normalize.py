from litscout.core.normalize import normalize_doi, normalize_title


def test_normalize_title():
    assert normalize_title("  Graph-Neural, Networks! ") == "graph neural networks"
    assert normalize_title("A   B\nC") == "a b c"


def test_normalize_doi():
    assert normalize_doi(" 10.1000/XYZ ") == "10.1000/xyz"
    assert normalize_doi(None) is None
