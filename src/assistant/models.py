"""Pydantic v2 strict contracts (design §1.2). Every model forbids extras and is frozen."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

UUID4_RE = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"


class Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


TenantId = Annotated[str, StringConstraints(pattern=r"^tenant_[A-Za-z0-9]{1,32}$")]
UserId = Annotated[str, StringConstraints(pattern=r"^user_[A-Za-z0-9]{1,32}$")]
SKU = Annotated[str, StringConstraints(pattern=r"^SKU-\d{3}$")]
LOT = Annotated[str, StringConstraints(pattern=r"^LOT-\d{5}$")]
ChunkId = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{16,64}$")]
NodeId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9\-_.]{2,63}$")]
Uuid4Str = Annotated[str, StringConstraints(pattern=UUID4_RE)]
Tier = Literal["TIER_1_READ", "TIER_2_COMPUTE", "TIER_3_MUTATE"]
TIER_RANK = {"TIER_1_READ": 1, "TIER_2_COMPUTE": 2, "TIER_3_MUTATE": 3}
Role = Literal["user", "approver", "readonly"]


class Envelope(Strict):
    """Carried in JSON-RPC `_meta`. tenant_id/user_id are bound from auth, never from LLM output."""

    tenant_id: TenantId
    user_id: UserId
    session_id: Uuid4Str
    idempotency_key: Uuid4Str
    traceparent: Annotated[str, StringConstraints(pattern=r"^00-[a-f0-9]{32}-[a-f0-9]{16}-0[01]$")]


# ---- hybrid_retrieve ------------------------------------------------------
class HybridRetrieveIn(Strict):
    env: Envelope
    query: str = Field(min_length=1, max_length=2000)
    bm25_top_k: int = Field(default=50, ge=10, le=50)
    dense_top_k: int = Field(default=50, ge=10, le=50)
    rrf_k: int = Field(default=60, ge=1, le=200)
    fusion_pool: int = Field(default=40, ge=4, le=50)
    rerank_k: int = Field(default=4, ge=1, le=8)
    threshold: float = Field(default=0.42, ge=0.0, le=1.0)


class Chunk(Strict):
    chunk_id: ChunkId
    chunk_hash: str
    source_uri: str
    text: str
    bm25: float | None = None
    cosine: float | None = None
    rrf_rank: int | None = None
    rerank_score: float


class HybridRetrieveOut(Strict):
    chunks: list[Chunk] = Field(max_length=8)
    top_score: float
    abstain: bool
    unreranked: bool = False
    stage_ms: dict[Literal["bm25", "dense", "rrf", "rerank"], float]


# ---- graph_expand ---------------------------------------------------------
class GraphExpandIn(Strict):
    env: Envelope
    anchor: NodeId
    depth: int = Field(default=1, ge=1, le=2)
    token_cap: int = Field(default=320, ge=1, le=400)


class GraphNode(Strict):
    id: str
    domain: str
    risk_tier: str
    version: int
    summary: str


class GraphEdge(Strict):
    src: str
    dst: str
    kind: str


class GraphExpandOut(Strict):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    tokens: int


# ---- state store ----------------------------------------------------------
class SubmitFormPayload(Strict):
    form_type: Literal["return_request", "expense_report", "incident_report", "pto_request", "warranty_claim"]
    notes: str = Field(default="", max_length=200)


class TransitionPayload(Strict):
    to_state: Literal["DRAFT", "REVIEW", "RELEASED", "QUARANTINED", "RECALLED"]
    reason: str = Field(default="", max_length=200)


class ExecuteTransactionIn(Strict):
    env: Envelope
    action: Literal["submit_form", "transition_workflow"]
    tier: Tier
    target: SKU | LOT
    payload: dict[str, Any]


class ExecuteTransactionOut(Strict):
    status: Literal["EXECUTED", "PENDING_APPROVAL", "DENIED"]
    approval_id: Uuid4Str | None = None
    transaction_id: Uuid4Str | None = None
    effective_tier: Tier
    replayed: bool = False


class ApprovalDecideIn(Strict):
    env: Envelope
    approval_id: Uuid4Str
    decision: Literal["approve", "reject"]
    comment: str = Field(default="", max_length=200)


class ApprovalDecideOut(Strict):
    status: Literal["EXECUTED", "DENIED"]
    approval_id: Uuid4Str
    transaction_id: Uuid4Str | None = None


class ApprovalListIn(Strict):
    env: Envelope


class ApprovalRecord(Strict):
    approval_id: Uuid4Str
    requester: str
    action: str
    target: str
    payload: dict[str, Any]
    status: str
    created_at: float
    expires_at: float


class ApprovalListOut(Strict):
    approvals: list[ApprovalRecord]


class FactsPutIn(Strict):
    env: Envelope
    fact: str = Field(min_length=1, max_length=300)
    source_chunk_ids: list[ChunkId] = Field(min_length=1)
    verified_by: Literal["VERIFICATION_SUBAGENT"]


class FactsPutOut(Strict):
    stored: bool


class FactsGetIn(Strict):
    env: Envelope


class FactRecord(Strict):
    fact: str
    source_chunk_ids: list[str]
    created_at: float


class FactsGetOut(Strict):
    facts: list[FactRecord]


class AuditAppendIn(Strict):
    env: Envelope
    event: dict[str, Any]


class AuditAppendOut(Strict):
    seq: int
    hash: str


class AuditVerifyIn(Strict):
    env: Envelope


class AuditVerifyOut(Strict):
    ok: bool
    entries: int
    first_bad_seq: int | None = None


# ---- public API -----------------------------------------------------------
class AskRequest(BaseModel):
    """Public HTTP request. Lenient on purpose; tenant/user come from the auth token, not the body."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class Citation(BaseModel):
    n: int
    chunk_id: str
    source_uri: str


class AskResponse(BaseModel):
    status: Literal[
        "ANSWERED", "ABSTAINED", "REFUSED", "REJECTED_OOS", "PENDING_APPROVAL",
        "EXECUTED", "DENIED", "HALTED", "HALTED_BREAKER", "NEEDS_INFO", "DIRECT", "AUDIT",
    ]
    answer: str
    route: str = ""
    citations: list[Citation] = []
    trace_id: str = ""
    session_id: str = ""
    approval_id: str | None = None
    faithfulness: float | None = None
    cached: bool = False
    steps: int = 0
    latency_ms: float = 0.0
    detail: dict[str, Any] = {}


def make_envelope_stub() -> dict:
    """Placeholder envelope used only to validate LLM-supplied tool arguments (real env is bound by the host)."""
    return {"tenant_id": "tenant_stub", "user_id": "user_stub", "session_id": "00000000-0000-4000-8000-000000000000",
            "idempotency_key": "00000000-0000-4000-8000-000000000000",
            "traceparent": "00-" + "0" * 32 + "-" + "0" * 16 + "-01"}
