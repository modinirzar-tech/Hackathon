"""Dense retrieval. Default embedder is a deterministic hashing-trick encoder (no model download);
swap in a BGE/ONNX encoder by implementing the `Embedder` protocol. Index = FAISS HNSW (efSearch=64)."""
from __future__ import annotations

import zlib
from typing import Protocol

import numpy as np

from ..textutil import tokens

try:  # pragma: no cover - exercised implicitly
    import faiss
except Exception:  # faiss optional; numpy brute force is the fallback
    faiss = None


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = zlib.crc32(f.encode())
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]).astype(np.float32) if texts else np.zeros((0, self.dim), np.float32)


class DenseIndex:
    def __init__(self, embedder: Embedder, ef_search: int = 64):
        self.embedder = embedder
        self.ef_search = ef_search
        self.matrix = np.zeros((0, embedder.dim), np.float32)
        self.index = None

    def fit(self, texts: list[str]) -> "DenseIndex":
        self.matrix = self.embedder.encode(texts)
        if faiss is not None and len(texts):
            self.index = faiss.IndexHNSWFlat(self.embedder.dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = self.ef_search
            self.index.add(self.matrix)
        return self

    def search(self, query: str, top_k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        n = len(self.matrix)
        if n == 0:
            return []
        q = self.embedder.encode([query])
        if self.index is not None and (allowed is None or n > 1000):
            k = min(n, top_k * 4 if allowed is not None else top_k)
            sims, ids = self.index.search(q, k)
            hits = [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i >= 0]
        else:  # exact search; also used for tenant-filtered small corpora so filtering never drops recall
            sims = (self.matrix @ q[0])
            hits = [(i, float(sims[i])) for i in np.argsort(-sims)[:n]]
        if allowed is not None:
            hits = [h for h in hits if h[0] in allowed]
        return hits[:top_k]

    def cosine(self, query: str, doc_idx: int) -> float:
        return float(self.matrix[doc_idx] @ self.embedder.encode([query])[0])
