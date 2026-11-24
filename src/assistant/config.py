"""Runtime settings. Every value is overridable via environment variable."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None:
        return default
    if cast is bool:
        return raw.lower() in ("1", "true", "yes", "on")
    return cast(raw)


@dataclass
class Settings:
    vault_dir: Path = field(default_factory=lambda: Path(_env("VAULT_DIR", ROOT / "knowledge_vault", Path)))
    graph_path: Path = field(default_factory=lambda: Path(_env("GRAPH_PATH", ROOT / "data" / "graph.json", Path)))
    mcp_config_path: Path = field(default_factory=lambda: Path(_env("MCP_CONFIG", ROOT / "mcp_config.json", Path)))
    manifest_lock_path: Path = field(default_factory=lambda: Path(_env("MANIFEST_LOCK", ROOT / "data" / "mcp_manifest.lock.json", Path)))
    db_path: str = field(default_factory=lambda: _env("DB_PATH", str(ROOT / "data" / "state.db")))
    auth_secret: str = field(default_factory=lambda: _env("AUTH_SECRET", "dev-only-secret-change-me"))
    dev_mode: bool = field(default_factory=lambda: _env("DEV_MODE", True, bool))

    # retrieval (ADR-001)
    bm25_k1: float = 1.25
    bm25_b: float = 0.75
    rrf_k: int = 60
    top_k: int = 50
    fusion_pool: int = 40
    rerank_k: int = 4
    threshold: float = field(default_factory=lambda: _env("ABSTAIN_THRESHOLD", 0.42, float))
    hnsw_ef_search: int = 64
    chunk_tokens: int = 512
    chunk_overlap: int = 64

    # loop / budgets (ADR-004)
    max_steps: int = 5
    step_budget_ms: float = field(default_factory=lambda: _env("STEP_BUDGET_MS", 120.0, float))
    loop_budget_ms: float = field(default_factory=lambda: _env("LOOP_BUDGET_MS", 450.0, float))
    mcp_timeout_ms: float = field(default_factory=lambda: _env("MCP_TIMEOUT_MS", 150.0, float))
    observation_tokens: int = 180
    graph_token_cap: int = 320

    # cache (ADR-003)
    cache_ttl_s: float = 300.0
    cache_jitter_s: float = 45.0
    cache_cosine: float = field(default_factory=lambda: _env("CACHE_COSINE", 0.95, float))

    # bulkheads / shedding
    rag_pool: int = 64
    db_pool: int = 128
    llm_pool: int = 256
    max_inflight: int = field(default_factory=lambda: _env("MAX_INFLIGHT", 2000, int))

    # approvals
    approval_ttl_s: float = field(default_factory=lambda: _env("APPROVAL_TTL_S", 86400.0, float))

    # optional OpenAI-compatible endpoint (vLLM / TRT-LLM). Empty => deterministic extractive generator.
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", ""))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "local-model"))

    def mcp_config(self) -> dict:
        return json.loads(self.mcp_config_path.read_text())
