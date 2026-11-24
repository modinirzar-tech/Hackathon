"""Tenant-isolated semantic cache (ADR-003). In-memory TTL store with Redis-compatible key scheme:
    key = cache:{tenant}:{SHA256(tenant:user:normalized_query)}
Semantic candidates are searched ONLY inside the caller's own (tenant,user) namespace. Jittered TTL; single-flight."""
from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time

import numpy as np

from ..rag.dense import HashingEmbedder


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def cache_key(tenant: str, user: str, query: str) -> str:
    digest = hashlib.sha256(f"{tenant}:{user}:{normalize_query(query)}".encode()).hexdigest()
    return f"cache:{tenant}:{digest}"


class SemanticCache:
    def __init__(self, ttl_s: float = 300.0, jitter_s: float = 45.0, cosine: float = 0.95, max_entries: int = 50000):
        self.ttl, self.jitter, self.cosine, self.max = ttl_s, jitter_s, cosine, max_entries
        self.emb = HashingEmbedder()
        self.entries: dict[str, tuple[float, np.ndarray, dict]] = {}
        self.ns: dict[tuple[str, str], list[str]] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.available = True
        self.hits = self.misses = 0

    def _alive(self, key: str, now: float):
        e = self.entries.get(key)
        if e and e[0] > now:
            return e
        if e:
            self.entries.pop(key, None)
        return None

    def get(self, tenant: str, user: str, query: str) -> dict | None:
        if not self.available:  # FMEA #1: bypass, never fall back to a shared key
            return None
        now = time.time()
        e = self._alive(cache_key(tenant, user, query), now)
        if e is None:
            qv = self.emb.encode([normalize_query(query)])[0]
            best, best_sim = None, self.cosine
            for k in list(self.ns.get((tenant, user), ())):
                cand = self._alive(k, now)
                if cand is not None:
                    sim = float(cand[1] @ qv)
                    if sim >= best_sim:
                        best, best_sim = cand, sim
            e = best
        if e is None:
            self.misses += 1
            return None
        self.hits += 1
        return dict(e[2])

    def put(self, tenant: str, user: str, query: str, value: dict) -> None:
        if not self.available:
            return
        if len(self.entries) >= self.max:
            self.entries.pop(next(iter(self.entries)))
        key = cache_key(tenant, user, query)
        ttl = self.ttl + random.uniform(-self.jitter, self.jitter)
        self.entries[key] = (time.time() + ttl, self.emb.encode([normalize_query(query)])[0], value)
        lst = self.ns.setdefault((tenant, user), [])
        if key not in lst:
            lst.append(key)

    def lock_for(self, key: str) -> asyncio.Lock:  # single-flight guard per key
        return self.locks.setdefault(key, asyncio.Lock())

    @property
    def hit_ratio(self) -> float:
        t = self.hits + self.misses
        return self.hits / t if t else 0.0
