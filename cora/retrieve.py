"""BM25 retrieval over stored abstracts. Small corpus, no vector store needed in P0.5."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]*")
_STOP = {
    "the", "a", "an", "of", "in", "and", "or", "to", "is", "are", "was", "were", "for", "on",
    "with", "by", "as", "that", "this", "these", "those", "we", "our", "be", "it", "its", "at",
    "from", "which", "than", "but", "not", "has", "have", "had", "been", "their", "also", "may",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class Index:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        self._tokens = [tokenize(f"{d.get('title', '')} {d.get('abstract', '')}") for d in docs]
        self._bm25 = BM25Okapi(self._tokens) if docs else None

    def search(self, query: str, k: int = 8, species_keys: list[str] | None = None) -> list[tuple[dict, float]]:
        """Rank by BM25, but *filter* by term overlap: on a tiny corpus BM25's IDF goes
        non-positive for terms every document contains, so `score > 0` would drop real hits."""
        if not self.docs or self._bm25 is None:
            return []
        q = tokenize(query)
        if not q:
            return []
        qset = set(q)
        scores = self._bm25.get_scores(q)
        wanted = set(species_keys) if species_keys else None
        hits = []
        for d, s, toks in zip(self.docs, scores, self._tokens):
            overlap = len(qset & set(toks))
            if overlap == 0:
                continue
            if wanted is not None and not (wanted & set(d.get("species", []))):
                continue
            hits.append((d, float(s), overlap))
        hits.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [(d, s) for d, s, _ in hits[:k]]


def build_index(conn, species_keys: list[str] | None = None) -> Index:
    from . import db

    return Index(db.get_docs(conn, species_keys))
