"""Pure-Python Okapi BM25 (k1=1.25, b=0.75 per ADR-001)."""
from __future__ import annotations

import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, k1: float = 1.25, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.n = 0

    def fit(self, docs: list[list[str]]) -> "BM25":
        self.postings = defaultdict(list)
        self.doc_len = [len(d) for d in docs]
        self.n = len(docs)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        return self

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], top_k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            idf = self.idf(term)
            for doc, tf in self.postings.get(term, ()):
                if allowed is not None and doc not in allowed:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc] / (self.avgdl or 1))
                scores[doc] += idf * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
