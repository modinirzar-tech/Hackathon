"""Request orchestration (design §1.4): pre-hooks -> cache -> triage -> subagents -> generate -> verify -> post-hooks
-> audit / facts / cache. Subagents: Retrieval (read-only), Execution (state-store only, T3 => HITL), Verification (veto)."""
from __future__ import annotations

import re
import time
import uuid

from ..config import Settings
from ..errors import BreakerOpen, BulkheadFull, Code, McpError, PolicyBlocked
from ..models import AskRequest, AskResponse, Citation, UUID4_RE
from ..observability.tracing import Metrics, Tracer, make_traceparent
from ..plugins.host import HookCtx, PluginHost
from ..security import injection
from ..security.auth import AuthContext, verify_token
from ..state.cache import SemanticCache
from ..textutil import content_tokens, estimate_tokens, sentences, unwiki
from .harness import CallContext, ToolHarness
from .llm import LLM, ContextItem
from .loop import GoalAnchor, ReActLoop, RetrievalPolicy
from .triage import TARGET, parse_transaction, triage
from .verifier import verify

ABSTAIN_MSG = "I can't find grounded information in the knowledge base to answer that, so I won't guess. Please rephrase or contact the policy owner."
OOS_MSG = "That request is outside what this assistant supports (enterprise policy, SOP and product questions, and gated workflow actions)."
CITE = re.compile(r"\s*\[\d+\]")
_PII_CLAUSE = re.compile(r"(?:\b(?:my|our)\s+)?(?:\b(?:e-?mail|phone|number|ssn|card)\b\s*(?:is|at|:)?\s*)?\[(?:EMAIL|SSN|PHONE|CARD)\][,;.\s]*", re.I)


class Orchestrator:
    def __init__(self, settings: Settings, harness: ToolHarness, loop: ReActLoop, plugins: PluginHost, cache: SemanticCache,
                 llm: LLM, tracer: Tracer, metrics: Metrics, vault_notes: dict, graph, host):
        self.s, self.h, self.loop, self.plugins, self.cache = settings, harness, loop, plugins, cache
        self.llm, self.tracer, self.metrics, self.notes, self.graph, self.host = llm, tracer, metrics, vault_notes, graph, host
        self.wal: list[dict] = []  # audit events buffered while the state store is down (read-only paths)

    # ------------------------------------------------------------------
    async def ask(self, req: AskRequest, token: str) -> AskResponse:
        t0 = time.perf_counter()
        auth = verify_token(self.s.auth_secret, token)
        session_id = req.session_id if req.session_id and re.match(UUID4_RE, req.session_id) else str(uuid.uuid4())
        trace_id = self.tracer.start_trace()
        ctx = CallContext(auth, token, session_id, trace_id)
        hc = HookCtx(auth.tenant_id, auth.user_id, session_id, req.query)
        self.metrics.inc("requests")
        with self.tracer.span("request"):
            resp = await self._handle(ctx, hc, auth)
        resp["trace_id"], resp["session_id"] = trace_id, session_id
        resp["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        self.metrics.observe("e2e", resp["latency_ms"])
        self.metrics.inc(f"status_{resp['status'].lower()}")
        return AskResponse(**resp)

    async def _handle(self, ctx: CallContext, hc: HookCtx, auth: AuthContext) -> dict:
        try:
            self.plugins.run_pre(hc)  # PII redact -> injection shield (fail-closed)
        except PolicyBlocked as e:
            resp = {"status": "REFUSED", "answer": "Request refused by security policy.", "route": "BLOCKED", "detail": e.data}
            await self._finish(ctx, hc, resp, final=True)
            return resp

        with self.tracer.span("cache.lookup"):
            hit = self.cache.get(auth.tenant_id, auth.user_id, hc.query)
        if hit:
            hit["cached"] = True
            await self._finish(ctx, hc, hit, final=True)
            return hit

        with self.tracer.span("triage.classify"):
            tr = triage(hc.query)
        route = tr.route
        if route == "DIRECT_ANSWER":
            resp = {"status": "DIRECT", "answer": "Hello! Ask me about company policy, SOPs or products, or request a workflow action.", "route": route}
        elif route == "REJECT_OOS":
            resp = {"status": "REJECTED_OOS", "answer": OOS_MSG, "route": route}
        elif route == "AUDIT_VERIFY":
            resp = await self._audit(ctx)
        elif route == "TRANSACTION_EXEC":
            resp = await self._execute(ctx, hc, tr)
        else:
            resp = await self._rag(ctx, hc, tr)
        resp["route"] = route
        await self._finish(ctx, hc, resp)
        if resp["status"] == "ANSWERED" and not resp.get("cached"):
            self.cache.put(auth.tenant_id, auth.user_id, hc.query, dict(resp))
        return resp

    # ---- finish: post hooks + audit ------------------------------------------
    async def _finish(self, ctx: CallContext, hc: HookCtx, resp: dict, final: bool = False) -> None:
        with self.tracer.span("hook.post"):
            self.plugins.run("post_execution", ctx=hc, response=resp)
        event = {"event": "ask", "route": resp.get("route"), "status": resp["status"], "user": hc.user_id, "session": hc.session_id,
                 "query_redacted": hc.query[:200], "pii_types": hc.pii_found, "cached": resp.get("cached", False),
                 "faithfulness": resp.get("faithfulness")}
        await self._audit_append(ctx, event)

    async def _audit_append(self, ctx: CallContext, event: dict) -> None:
        self.wal.append(event)
        try:
            while self.wal:
                await self.host.call("state-store", "audit_append", {"event": self.wal[0]}, ctx.envelope(str(uuid.uuid4())), ctx.token)
                self.wal.pop(0)
        except Exception:
            self.metrics.inc("audit_buffered")  # local WAL until the store recovers (FMEA #6)

    # ---- Retrieval subagent + generation + Verification subagent ---------------
    async def _rag(self, ctx: CallContext, hc: HookCtx, tr) -> dict:
        question = _PII_CLAUSE.sub("", hc.query).strip() or hc.query  # redaction placeholders carry no retrieval signal
        goal = GoalAnchor(question, tr.route, tuple(sorted({m.group(0).upper() for m in TARGET.finditer(hc.query)})))
        res = await self.loop.run(ctx, goal, RetrievalPolicy(self.s, tr.complexity))
        detail = {"steps": res.steps, "states": res.states, "halt_reason": res.halt_reason,
                  "unreranked": res.unreranked, "stage_ms": res.tool_results.get("stage_ms", [])}
        if res.status == "HALTED_BREAKER":
            return {"status": "HALTED_BREAKER", "answer": res.remediation["message"], "steps": res.steps,
                    "detail": {**detail, "remediation": res.remediation}}
        # indirect-injection scan of retrieved text before it can reach the LLM
        safe = []
        for c in res.chunks:
            bad, rules = injection.is_attack(c["text"])
            if bad:
                self.metrics.inc("indirect_injection_dropped")
                detail.setdefault("dropped_chunks", []).append({"chunk_id": c["chunk_id"], "rules": rules})
            else:
                safe.append(c)
        usable = [c for c in safe if c["rerank_score"] >= (max(self.s.threshold, 0.55) if res.unreranked else self.s.threshold)][:3]
        if res.status == "HALTED" and not usable:
            return {"status": "HALTED", "answer": "I couldn't complete retrieval within the latency budget, so I won't guess. Please retry.",
                    "steps": res.steps, "detail": detail}
        if res.abstain or not usable:
            return {"status": "ABSTAINED", "answer": ABSTAIN_MSG, "steps": res.steps, "detail": detail}
        if res.status == "HALTED":
            detail["incomplete"] = True

        items = [ContextItem(i + 1, c["chunk_id"], c["source_uri"], _snippet(c["text"], question)) for i, c in enumerate(usable)]
        if tr.complexity == "multi_hop" and res.graph:
            used = {c["source_uri"].removeprefix("vault://") for c in usable}
            for n in res.graph["nodes"]:
                if n["id"] not in used:
                    items.append(ContextItem(len(items) + 1, "graph:" + n["id"], f"vault://{n['id']}", n["summary"]))
        self._record_token_savings(res, items, usable)

        with self.tracer.span("llm.generate"):
            draft = await self.llm.generate(question, items)
        with self.tracer.span("verify.nli"):
            evidence = {i.n: (usable[i.n - 1]["text"] if i.n <= len(usable) else i.text) for i in items}
            v = verify(draft, evidence)
        if v["total"] == 0 or v["unsupported"]:
            self.metrics.inc("verifier_vetoes")
            return {"status": "ABSTAINED", "answer": ABSTAIN_MSG, "faithfulness": v["faithfulness"], "steps": res.steps,
                    "detail": {**detail, "veto": v["unsupported"]}}
        cites = [Citation(n=i.n, chunk_id=i.chunk_id, source_uri=i.source_uri) for i in items if f"[{i.n}]" in draft]
        resp = {"status": "ANSWERED", "answer": draft, "citations": [c.model_dump() for c in cites],
                "faithfulness": v["faithfulness"], "steps": res.steps, "detail": detail}
        await self._remember(ctx, resp, usable)
        return resp

    def _record_token_savings(self, res, items, usable) -> None:
        """Baseline = naive stuffing (full text of reranked chunks + full body of every 1-hop neighbour note)."""
        base = sum(estimate_tokens(c["text"]) for c in usable)
        if res.graph:
            anchor = usable[0]["source_uri"].removeprefix("vault://")
            for nid in {n["id"] for n in res.graph["nodes"]} - {anchor}:
                note = self.notes.get(nid)
                if note:
                    base += estimate_tokens(note.body)
        opt = sum(estimate_tokens(i.text) for i in items)
        self.metrics.inc("tokens_baseline", base)
        self.metrics.inc("tokens_optimized", opt)

    async def _remember(self, ctx: CallContext, resp: dict, usable: list[dict]) -> None:
        fact = CITE.sub("", sentences(resp["answer"])[0])[:300]
        try:
            await self.h.call(ctx, "session_facts_put", {"fact": fact, "source_chunk_ids": [c["chunk_id"] for c in usable],
                                                         "verified_by": "VERIFICATION_SUBAGENT"})
        except Exception:
            self.metrics.inc("facts_put_failed")

    # ---- Execution subagent (state-store only; T3 => HITL freeze) -------------------
    async def _execute(self, ctx: CallContext, hc: HookCtx, tr) -> dict:
        slots = parse_transaction(hc.query, tr.action)
        if not slots["target"] or not slots["payload"]:
            return {"status": "NEEDS_INFO", "answer": "Which SKU/LOT and what exact action or form? e.g. 'recall LOT-12345' or 'submit a return request for SKU-042'."}
        args = {"action": tr.action, "tier": tr.risk_tier, "target": slots["target"], "payload": slots["payload"]}
        try:
            out = await self.h.call(ctx, "execute_transaction", args)
        except BreakerOpen as e:
            return {"status": "HALTED_BREAKER", "answer": "Repeated identical failures; stopped. Please correct the request.", "detail": e.data}
        except McpError as e:
            if e.code == Code.INVALID_PARAMS:
                return {"status": "NEEDS_INFO", "answer": "The identifiers or values were invalid. Use SKU-### (3 digits) or LOT-##### (5 digits).",
                        "detail": e.data}
            if e.code == Code.RBAC_DENIED:
                return {"status": "DENIED", "answer": "You are not permitted to perform that action.", "detail": {"error": e.name}}
            return {"status": "DENIED", "answer": "The state service is unavailable, so the action was refused (fail-closed). Nothing was changed.",
                    "detail": {"error": e.name}}
        except BulkheadFull:
            return {"status": "DENIED", "answer": "System is shedding load; action refused. Please retry.", "detail": {"error": "BULKHEAD"}}
        if out["status"] == "PENDING_APPROVAL":
            return {"status": "PENDING_APPROVAL", "approval_id": out["approval_id"], "detail": {"code": int(Code.HITL_PENDING)},
                    "answer": f"This is a Tier-3 action ({tr.action} on {args['target'].upper()}). It is frozen until an authorized approver decides. Ticket: {out['approval_id']}."}
        return {"status": "EXECUTED", "answer": f"Done: {tr.action} executed (transaction {out['transaction_id']}).",
                "detail": {"transaction_id": out["transaction_id"]}}

    async def _audit(self, ctx: CallContext) -> dict:
        try:
            out = await self.h.call(ctx, "audit_verify", {})
        except Exception:
            return {"status": "DENIED", "answer": "Audit service unavailable.", "detail": {}}
        msg = "Audit chain intact" if out["ok"] else f"AUDIT CHAIN TAMPERED at seq {out['first_bad_seq']}"
        return {"status": "AUDIT", "answer": f"{msg} ({out['entries']} entries verified).", "detail": out}


def _snippet(text: str, query: str, max_sentences: int = 2) -> str:
    """Compressed context: heading dropped, top query-overlap sentences kept in original order."""
    q = content_tokens(query)
    sents = [unwiki(s) for s in sentences(re.sub(r"^#.*\n?", "", text))]
    ranked = sorted(range(len(sents)), key=lambda i: -len(q & content_tokens(sents[i])))[:max_sentences]
    return " ".join(sents[i] for i in sorted(ranked))
