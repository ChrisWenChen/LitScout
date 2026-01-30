from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Tuple
from uuid import uuid4

from .models import CanonicalPaper, SourceRecord
from .normalize import normalize_title


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _choose_longer(a: str | None, b: str | None) -> str | None:
    if not a:
        return b
    if not b:
        return a
    return a if len(a) >= len(b) else b


def _prefer_arxiv_authors(papers: list[CanonicalPaper]) -> list[str]:
    for paper in papers:
        if any(src.source_name == "arxiv" for src in paper.sources) and paper.authors:
            return paper.authors
    longest = max(papers, key=lambda p: len(p.authors) if p.authors else 0)
    return longest.authors


def _earliest_year(papers: list[CanonicalPaper]) -> tuple[int | None, list[str]]:
    years = [p.year for p in papers if p.year is not None]
    if not years:
        return None, []
    unique = sorted(set(years))
    notes = []
    if len(unique) > 1:
        notes.append(f"year_conflict: {', '.join(str(y) for y in unique)}")
    return min(unique), notes


def dedupe_papers(
    papers: Iterable[CanonicalPaper], strict: bool = False
) -> tuple[list[CanonicalPaper], dict[str, list[str]]]:
    papers_list = list(papers)
    if not papers_list:
        return [], {}

    dsu = _DSU(len(papers_list))
    key_map: dict[str, int] = {}

    for idx, paper in enumerate(papers_list):
        keys = []
        if paper.doi:
            keys.append(f"doi:{paper.doi}")
        if paper.arxiv_id:
            keys.append(f"arxiv:{paper.arxiv_id}")
        if paper.bibcode:
            keys.append(f"bibcode:{paper.bibcode}")
        if not strict:
            norm_title = normalize_title(paper.title)
            if norm_title:
                keys.append(f"title:{norm_title}")
        for key in keys:
            if key in key_map:
                dsu.union(idx, key_map[key])
            else:
                key_map[key] = idx

    clusters: dict[int, list[CanonicalPaper]] = defaultdict(list)
    for idx, paper in enumerate(papers_list):
        clusters[dsu.find(idx)].append(paper)

    merged: list[CanonicalPaper] = []
    notes_by_id: dict[str, list[str]] = {}

    for group in clusters.values():
        group_dois = {p.doi for p in group if p.doi}
        group_arxiv = {p.arxiv_id for p in group if p.arxiv_id}
        group_bibcodes = {p.bibcode for p in group if p.bibcode}
        norm_titles = {normalize_title(p.title) for p in group if p.title}
        doi_count = sum(1 for p in group if p.doi)
        arxiv_count = sum(1 for p in group if p.arxiv_id)
        bibcode_count = sum(1 for p in group if p.bibcode)

        if len(group) > 1 and (
            (doi_count >= 2 and len(group_dois) == 1)
            or (arxiv_count >= 2 and len(group_arxiv) == 1)
            or (bibcode_count >= 2 and len(group_bibcodes) == 1)
        ):
            confidence = "high"
        elif not strict and len(group) > 1 and len(norm_titles) == 1:
            confidence = "medium"
        else:
            confidence = "low"

        title = group[0].title
        abstract = group[0].abstract
        url_primary = group[0].url_primary
        venue = group[0].venue
        doi = group[0].doi
        arxiv_id = group[0].arxiv_id
        bibcode = group[0].bibcode

        for paper in group[1:]:
            title = _choose_longer(title, paper.title)
            abstract = _choose_longer(abstract, paper.abstract)
            url_primary = _choose_longer(url_primary, paper.url_primary)
            venue = _choose_longer(venue, paper.venue)
            doi = doi or paper.doi
            arxiv_id = arxiv_id or paper.arxiv_id
            bibcode = bibcode or paper.bibcode

        authors = _prefer_arxiv_authors(group)
        year, year_notes = _earliest_year(group)

        sources: list[SourceRecord] = []
        for paper in group:
            sources.extend(paper.sources)

        if doi:
            dedup_cluster_id = f"doi:{doi}"
        elif arxiv_id:
            dedup_cluster_id = f"arxiv:{arxiv_id}"
        elif bibcode:
            dedup_cluster_id = f"bibcode:{bibcode}"
        else:
            dedup_cluster_id = f"title:{normalize_title(title)}"

        merged_paper = CanonicalPaper(
            canonical_id=uuid4(),
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            doi=doi,
            arxiv_id=arxiv_id,
            bibcode=bibcode,
            url_primary=url_primary,
            venue=venue,
            sources=sources,
            dedup_cluster_id=dedup_cluster_id,
            merge_confidence=confidence,
        )
        merged.append(merged_paper)
        if year_notes:
            notes_by_id[str(merged_paper.canonical_id)] = year_notes

    return merged, notes_by_id
