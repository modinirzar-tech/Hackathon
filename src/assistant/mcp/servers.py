"""The three MCP servers (design §1.2.2). Each is self-contained and re-checks tenant/RBAC itself."""
from __future__ import annotations

import time

from ..errors import Code, McpError
from ..faults import Faults
from ..graph.vault import GraphStore
from ..models import (ApprovalDecideIn, ApprovalListIn, AuditAppendIn, AuditVerifyIn, ExecuteTransactionIn,
                      FactsGetIn, FactsPutIn, GraphExpandIn, HybridRetrieveIn, SubmitFormPayload, TransitionPayload)
from ..rag.engine import RagEngine
from ..state.store import StateStore
from .base import McpServer


def build_rag_server(engine: RagEngine, secret: str, faults: Faults) -> McpServer:
    srv = McpServer("mcp-hybrid-rag", secret)

    @srv.tool("hybrid_retrieve", HybridRetrieveIn, "TIER_1_READ")
    def hybrid_retrieve(a: HybridRetrieveIn, auth):
        if "rag_hang" in faults:
            time.sleep(1.0)
        engine.fault_reranker_down = "reranker_down" in faults
        return engine.retrieve(a.env.tenant_id, a.query, a.bm25_top_k, a.dense_top_k, a.rrf_k,
                               a.fusion_pool, a.rerank_k, a.threshold)

    @srv.resource("rag://chunks/")
    def chunk_resource(uri: str, auth):
        c = engine.get_chunk(uri.removeprefix("rag://chunks/"), auth.tenant_id)
        if not c:
            raise McpError(Code.INVALID_PARAMS, "chunk not found")
        return {"chunk_id": c.chunk_id, "source_uri": c.source_uri, "text": c.text}

    return srv


def build_graph_server(graph: GraphStore, secret: str, faults: Faults) -> McpServer:
    srv = McpServer("mcp-graphify-vault", secret)

    @srv.tool("graph_expand", GraphExpandIn, "TIER_1_READ")
    def graph_expand(a: GraphExpandIn, auth):
        if "graph_down" in faults:
            raise RuntimeError("graph unavailable")
        return graph.expand(a.anchor, a.depth, a.token_cap, a.env.tenant_id)

    return srv


def build_state_server(store: StateStore, secret: str, faults: Faults) -> McpServer:
    srv = McpServer("mcp-state-store", secret)

    def guard():
        if "state_down" in faults:
            raise RuntimeError("state store unavailable")

    def idempotent(method: str, a, fn):
        key, tenant = a.env.idempotency_key, a.env.tenant_id
        prior = store.idem_get(tenant, key, method)
        if prior is not None:
            return {**prior, "replayed": True}
        out = fn()
        store.idem_put(tenant, key, method, out)
        return out

    @srv.tool("session_facts_get", FactsGetIn, "TIER_1_READ")
    def facts_get(a: FactsGetIn, auth):
        guard()
        return {"facts": store.facts_get(a.env.tenant_id, a.env.session_id)}

    @srv.tool("session_facts_put", FactsPutIn, "TIER_2_COMPUTE")
    def facts_put(a: FactsPutIn, auth):
        guard()
        store.facts_put(a.env.tenant_id, a.env.session_id, a.fact, a.source_chunk_ids)
        return {"stored": True}

    @srv.tool("execute_transaction", ExecuteTransactionIn, "TIER_3_MUTATE")
    def execute_transaction(a: ExecuteTransactionIn, auth):
        guard()
        sub = SubmitFormPayload if a.action == "submit_form" else TransitionPayload
        try:
            sub.model_validate(a.payload)
        except Exception as exc:
            raise McpError(Code.INVALID_PARAMS, "payload invalid for action", {"detail": str(exc)[:200]}) from exc
        if a.action == "submit_form" and not a.target.startswith("SKU-") and not a.target.startswith("LOT-"):
            raise McpError(Code.INVALID_PARAMS, "bad target")

        def run():
            r = store.execute_transaction(a.env.tenant_id, a.env.user_id, a.action, a.tier, a.target, a.payload)
            store.audit_append(a.env.tenant_id, {"event": "execute_transaction", "user": a.env.user_id, "action": a.action,
                                                 "target": a.target, "status": r["status"], "effective_tier": r["effective_tier"],
                                                 "approval_id": r["approval_id"]})
            return {**r, "replayed": False}
        return idempotent("execute_transaction", a, run)

    @srv.tool("approval_list", ApprovalListIn, "TIER_3_MUTATE", role="approver")
    def approval_list(a: ApprovalListIn, auth):
        guard()
        return {"approvals": store.approvals_list(a.env.tenant_id)}

    @srv.tool("approval_decide", ApprovalDecideIn, "TIER_3_MUTATE", role="approver")
    def approval_decide(a: ApprovalDecideIn, auth):
        guard()

        def run():
            r = store.approval_decide(a.env.tenant_id, a.env.user_id, a.approval_id, a.decision)
            store.audit_append(a.env.tenant_id, {"event": "approval_decide", "approver": a.env.user_id,
                                                 "approval_id": a.approval_id, "decision": a.decision, "status": r["status"]})
            return r
        return idempotent("approval_decide", a, run)

    @srv.tool("audit_append", AuditAppendIn, "TIER_1_READ")
    def audit_append(a: AuditAppendIn, auth):
        guard()
        seq, h = store.audit_append(a.env.tenant_id, a.event)
        return {"seq": seq, "hash": h}

    @srv.tool("audit_verify", AuditVerifyIn, "TIER_1_READ")
    def audit_verify(a: AuditVerifyIn, auth):
        guard()
        return store.audit_verify()

    return srv
