"""Cross-encoder stand-in. Interface matches an ONNX/Triton reranker (score in [0,1], sigmoid-calibrated).
Features: idf-weighted query coverage, bigram overlap, dense cosine, exact-ID match."""
from __future__ import annotations

import math
from typing import Callable

from ..textutil import tokens


def _bigrams(t: list[str]) -> set[tuple[str, str]]:
    return set(zip(t, t[1:]))


class Reranker:
    def __init__(self, idf: Callable[[str], float]):
        self.idf = idf

    def score(self, query: str, text: str, cosine: float = 0.0) -> float:
        q, d = tokens(query), tokens(text)
        if not q:
            return 0.0
        dset = set(d)
        weights = {t: self.idf(t) for t in set(q)}
        total = sum(weights.values()) or 1.0
        cov = sum(w for t, w in weights.items() if t in dset) / total
        qb = _bigrams(q)
        big = len(qb & _bigrams(d)) / len(qb) if qb else 0.0
        z = 5.0 * (0.62 * cov + 0.2 * big + 0.18 * max(cosine, 0.0) - 0.36)
        return 1.0 / (1.0 + math.exp(-z))
