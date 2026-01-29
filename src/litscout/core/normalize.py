import re


_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    lowered = title.strip().lower()
    no_punct = _PUNCT_RE.sub(" ", lowered)
    collapsed = _WS_RE.sub(" ", no_punct).strip()
    return collapsed


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.strip().lower()
