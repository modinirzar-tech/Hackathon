"""3-stage hybrid retrieval: BM25 ‖ dense -> RRF (dedupe, pool) -> rerank -> abstain gate (design §1.4)."""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

from ..config import Settings
from ..graph.vault import Note
from ..textutil import tokens
from .bm25 import BM25
from .dense import DenseIndex, HashingEmbedder
from .rerank import Reranker


@dataclass
class IndexedChunk:
    chunk_id: str
    chunk_hash: str
    note_id: str
    tenant: str
    text: str
    source_uri: str


def chunk_note(note: Note, max_tokens: int, overlap: int) -> list[str]:
    words = note.body.split()
    if len(words) <= max_tokens:
        return [note.body]
    step = max_tokens - overlap
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), step)]


class RagEngine:
    def __init__(self, notes: list[Note], settings: Settings):
        self.s = settings
        self.chunks: list[IndexedChunk] = []
        for n in notes:
            for part in chunk_note(n, settings.chunk_tokens, settings.chunk_overlap):
                h = hashlib.sha256(part.encode()).hexdigest()
                cid = hashlib.sha256(f"{n.id}:{h}".encode()).hexdigest()[:32]
                self.chunks.append(IndexedChunk(cid, h, n.id, n.tenant, part, f"vault://{n.id}"))
        self.by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        texts = [c.text for c in self.chunks]
        self.bm25 = BM25(settings.bm25_k1, settings.bm25_b).fit([tokens(t) for t in texts])
        self.dense = DenseIndex(HashingEmbedder(), settings.hnsw_ef_search).fit(texts)
        self.reranker = Reranker(self.bm25.idf)
        self.fault_reranker_down = False

    def _allowed(self, tenant_id: str) -> set[int]:
        return {i for i, c in enumerate(self.chunks) if c.tenant in ("all", tenant_id)}

    def get_chunk(self, chunk_id: str, tenant_id: str) -> IndexedChunk | None:
        i = self.by_id.get(chunk_id)
        if i is None or self.chunks[i].tenant not in ("all", tenant_id):
            return None  # tenant-scoped: foreign chunks look non-existent
        return self.chunks[i]

    def retrieve(self, tenant_id: str, query: str, bm25_top_k=50, dense_top_k=50, rrf_k=60,
                 fusion_pool=40, rerank_k=4, threshold=0.42) -> dict:
        allowed = self._allowed(tenant_id)
        q_tokens = tokens(query)
        t0 = time.perf_counter()
        bm = self.bm25.search(q_tokens, bm25_top_k, allowed)
        t1 = time.perf_counter()
        dn = self.dense.search(query, dense_top_k, allowed)
        t2 = time.perf_counter()

        rrf: dict[int, float] = {}
        for ranked in (bm, dn):
            for rank, (doc, _) in enumerate(ranked, 1):
                rrf[doc] = rrf.get(doc, 0.0) + 1.0 / (rrf_k + rank)
        seen_hash: set[str] = set()
        pool: list[int] = []
        for doc, _ in sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0])):
            h = self.chunks[doc].chunk_hash
            if h not in seen_hash:
                seen_hash.add(h)
                pool.append(doc)
            if len(pool) >= fusion_pool:
                break
        t3 = time.perf_counter()

        bm_s, dn_s = dict(bm), dict(dn)
        unreranked = self.fault_reranker_down
        scored = []
        for rank, doc in enumerate(pool, 1):
            cos = dn_s.get(doc)
            if cos is None:
                cos = self.dense.cosine(query, doc)
            if unreranked:  # FMEA #3: keep RRF order, flag it, and grade by cheap idf coverage vs a stricter bar
                dset = set(tokens(self.chunks[doc].text))
                w = {t: self.bm25.idf(t) for t in set(q_tokens)}
                score = sum(v for t, v in w.items() if t in dset) / (sum(w.values()) or 1.0)
            else:
                score = self.reranker.score(query, self.chunks[doc].text, cos)
            scored.append((doc, rank, score, bm_s.get(doc), cos))
        scored.sort(key=(lambda x: x[1]) if unreranked else (lambda x: (-x[2], x[1])))
        top = scored[:rerank_k]
        t4 = time.perf_counter()

        out_chunks = [{
            "chunk_id": self.chunks[d].chunk_id, "chunk_hash": self.chunks[d].chunk_hash,
            "source_uri": self.chunks[d].source_uri, "text": self.chunks[d].text,
            "bm25": b, "cosine": c, "rrf_rank": r, "rerank_score": round(s, 4),
        } for d, r, s, b, c in top]
        top_score = out_chunks[0]["rerank_score"] if out_chunks else 0.0
        eff_threshold = threshold if not unreranked else max(threshold, 0.55)
        ms = lambda a, b: round((b - a) * 1000, 3)
        return {
            "chunks": out_chunks, "top_score": top_score,
            "abstain": (not out_chunks) or top_score < eff_threshold,
            "unreranked": unreranked,
            "stage_ms": {"bm25": ms(t0, t1), "dense": ms(t1, t2), "rrf": ms(t2, t3), "rerank": ms(t3, t4)},
        }


def note_id_from_uri(uri: str) -> str:
    return re.sub(r"^vault://", "", uri)
