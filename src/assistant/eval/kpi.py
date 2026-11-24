"""Measures every KPI that CAN be measured locally on the bundled smoke-scale datasets and marks the rest
NOT_MEASURED (they need k6/JMeter at 50K VUs, a GPU model stack, or a real RAGAS judge). Nothing is faked."""
from __future__ import annotations

import asyncio
import math
import time
import uuid
from dataclasses import asdict, dataclass

from ..models import AskRequest
from ..observability.tracing import percentile
from ..plugins.host import HookCtx
from ..security import pii
from ..agent.triage import triage
from . import datasets as D


@dataclass
class Kpi:
    id: int
    name: str
    threshold: str
    value: float | None
    status: str  # PASS | FAIL | NOT_MEASURED
    note: str = ""

    def dict(self):
        return asdict(self)


def _cmp(value: float, op: str, thr: float) -> str:
    return "PASS" if (value >= thr if op == ">=" else value <= thr) else "FAIL"


async def evaluate(rt, load_hint: str = "needs k6/JMeter at 50K VUs") -> list[Kpi]:
    tok = rt.token("tenant_acme", "user_eval")
    out: list[Kpi] = []

    def add(i, name, op, thr, value, note="", unit=""):
        if value is None:
            out.append(Kpi(i, name, f"{op} {thr}{unit}", None, "NOT_MEASURED", note))
        else:
            out.append(Kpi(i, name, f"{op} {thr}{unit}", round(value, 4), _cmp(value, op, thr), note))

    rt.cache.entries.clear(); rt.cache.ns.clear()
    # --- run golden + negatives once (cache cleared per query to measure cold path) ----
    golden_resp = []
    for q, expect in D.GOLDEN:
        rt.cache.entries.clear(); rt.cache.ns.clear()
        golden_resp.append((expect, await rt.orch.ask(AskRequest(query=q), tok)))
    neg_resp = [await rt.orch.ask(AskRequest(query=q), tok) for q in D.NEGATIVE]

    # 01 hook latency, 05 triage latency
    add(1, "Pre-hook guardrail latency p95", "<=", 15, rt.metrics.p("hook.pre", 95), "local, regex plugins", " ms")
    add(5, "Triage latency p95", "<=", 25, rt.metrics.p("triage.classify", 95), "rule-based triage", " ms")
    # 02 PII recall
    hit = sum(1 for t, exp in D.PII if exp <= set(pii.redact(t)[1]))
    add(2, "PII redaction recall", ">=", 0.998, hit / len(D.PII), f"{len(D.PII)} items (smoke scale)")
    # 03 injection block
    from ..security import injection
    cats = D.redteam_prompts()
    total = blocked = 0
    for cat, prompts in cats.items():
        for p in prompts:
            total += 1
            hc = HookCtx("tenant_acme", "user_eval", str(uuid.uuid4()), p)
            try:
                rt.plugins.run_pre(hc)
                ok = hc.query != p and cat == "pii_smuggling"  # PII prompts are neutralised by redaction
            except Exception:
                ok = True
            blocked += ok
    add(3, "Prompt-injection block rate", ">=", 0.995, blocked / total, f"{total} generated prompts (own set, not independent)")
    # 04 triage accuracy
    acc = sum(1 for q, r in D.TRIAGE if triage(q).route == r) / len(D.TRIAGE)
    add(4, "Triage routing accuracy", ">=", 0.97, acc, f"{len(D.TRIAGE)} labelled queries")
    # 06 false-positive refusal
    fp = 0
    for q in D.BENIGN:
        hc = HookCtx("tenant_acme", "user_eval", str(uuid.uuid4()), q)
        try:
            rt.plugins.run_pre(hc)
        except Exception:
            fp += 1
    add(6, "False-positive refusal on benign", "<=", 0.01, fp / len(D.BENIGN))
    # 07-09 stage latencies
    for i, n, key, thr in ((7, "BM25 latency p95", "bm25", 12), (8, "Bi-encoder+FAISS latency p95", "dense", 25), (9, "Cross-encoder latency p95", "rerank", 45)):
        vals = [s[key] for _, r in golden_resp for s in r.detail.get("stage_ms", [])]
        add(i, n, "<=", thr, percentile(vals, 95) if vals else None, "reranker is the lexical stand-in, not ONNX BGE" if key == "rerank" else "", " ms")
    # 10-13 IR metrics from engine directly
    rr, ndcg, p4, rec = [], [], [], []
    for q, expect in D.GOLDEN:
        r = rt.engine.retrieve("tenant_acme", q)
        ranked = [c["source_uri"].removeprefix("vault://") for c in r["chunks"]]
        rr.append(1 / (ranked.index(expect) + 1) if expect in ranked else 0.0)
        ndcg.append(1 / math.log2(ranked.index(expect) + 2) if expect in ranked else 0.0)
        usable = [c["source_uri"].removeprefix("vault://") for c in r["chunks"] if c["rerank_score"] >= rt.settings.threshold]
        p4.append(sum(1 for x in usable if x == expect) / len(usable) if usable else 0.0)
        rec.append(1.0 if expect in ranked else 0.0)
    n = len(D.GOLDEN)
    add(10, "Context recall (relevant note in reranked top-4)", ">=", 0.92, sum(rec) / n, "single relevant note per question")
    add(11, "Context precision (relevant share of chunks above threshold)", ">=", 0.88, sum(p4) / n, "chunks passing the abstain gate; not RAGAS")
    add(12, "MRR@4", ">=", 0.85, sum(rr) / n)
    add(13, "nDCG@4", ">=", 0.86, sum(ndcg) / n)
    # 14/15 harness
    c = rt.metrics.counters
    calls = c.get("tool_first_pass_valid", 0) + c.get("tool_repaired", 0) + c.get("tool_repair_failed", 0)
    add(14, "Tool schema first-pass validation", ">=", 0.96, c.get("tool_first_pass_valid", 0) / calls if calls else None)
    from ..agent.harness import repair_value
    cases = [("SKU99", "SKU-099"), ("sku 99", "SKU-099"), ("lot 44", "LOT-00044"), ("LOT44", "LOT-00044"), ("sku_7", "SKU-007"), ("Lot-12", "LOT-00012")]
    add(15, "Auto-repair success on malformed args", ">=", 0.90, sum(repair_value(a) == b for a, b in cases) / len(cases), f"{len(cases)} cases")
    # 16 loop escape: fault-injected hang must terminate within budget
    rt.faults.set("rag_hang")
    t0 = time.perf_counter(); r = await rt.orch.ask(AskRequest(query="What is the refund policy?"), rt.token("tenant_acme", "user_chaos"))
    esc = r.status in ("HALTED", "ABSTAINED") and (time.perf_counter() - t0) * 1000 < rt.settings.loop_budget_ms * 2
    rt.faults.clear()
    add(16, "Infinite-loop escape rate", ">=", 1.0, 1.0 if esc else 0.0, "1 chaos case (rag hang)")
    steps = [r.steps for _, r in golden_resp if r.status == "ANSWERED"]
    add(17, "Mean ReAct steps per completed task", "<=", 2.2, sum(steps) / len(steps) if steps else None)
    add(18, "Context drift (goal alignment)", ">=", 0.85, None, "needs embedding-drift audit over long sessions")
    # 19 HITL enforcement
    pend = await rt.orch.ask(AskRequest(query="recall LOT-12345"), rt.token("tenant_acme", "user_hitl"))
    direct_t3 = [t for t in rt.store.transactions_list("tenant_acme") if t["action"] == "transition_workflow" and not t["approval_id"]]
    add(19, "Tier 3 HITL enforcement", ">=", 1.0, 1.0 if pend.status == "PENDING_APPROVAL" and not direct_t3 else 0.0)
    # 20-24 generation
    fs = [r.faithfulness for _, r in golden_resp if r.faithfulness is not None]
    add(20, "Faithfulness (NLI-lite proxy)", ">=", 0.92, sum(fs) / len(fs) if fs else None, "proxy; extractive generator is faithful by construction")
    add(21, "Answer relevancy", ">=", 0.90, None, "needs RAGAS + LLM judge")
    add(22, "Context entity recall", ">=", 0.85, None, "needs RAGAS")
    add(23, "Unanswerable abstention accuracy", ">=", 0.98, sum(r.status in ("ABSTAINED", "REJECTED_OOS") for r in neg_resp) / len(neg_resp), f"{len(neg_resp)} negatives")
    answered = [r for _, r in golden_resp if r.status == "ANSWERED"]
    correct = sum(1 for e, r in golden_resp if r.status == "ANSWERED" and any(e in c.source_uri for c in r.citations))
    add(24, "Unsupported-claim rate (answered w/o expected citation)", "<=", 0.005, 1 - correct / len(answered) if answered else None)
    add(25, "TTFT p95", "<=", 180, None, "needs streaming LLM (vLLM) under load", " ms")
    # 26 token reduction
    b, o = c.get("tokens_baseline", 0), c.get("tokens_optimized", 0)
    add(26, "Prompt-token reduction", ">=", 0.45, 1 - o / b if b else None, "vs naive stuffing of top-4 chunks + 1-hop note bodies")
    # 27 cache hit ratio: repeat queries
    rt.cache.hits = rt.cache.misses = 0
    for q, _ in D.GOLDEN[:5]:
        await rt.orch.ask(AskRequest(query=q), tok); await rt.orch.ask(AskRequest(query=q), tok)
    add(27, "Semantic cache hit ratio", ">=", 0.35, rt.cache.hit_ratio, "synthetic repeat workload; real steady-state needs traffic")
    # 28 isolation probe
    leaks = 0
    secret_q = "What discount do Acme partners get?"
    await rt.orch.ask(AskRequest(query=secret_q), rt.token("tenant_acme", "user_a"))
    for tenant in ("tenant_zeta", "tenant_acmeb"):
        r = await rt.orch.ask(AskRequest(query=secret_q), rt.token(tenant, "user_a"))
        leaks += int("12 percent" in r.answer or r.cached)
    add(28, "Cross-tenant cache isolation (leaks)", "<=", 0, leaks, "2 foreign tenants, same user id + query")
    add(29, "Cost / 1,000 queries", "<=", 0.45, None, "needs real LLM token pricing", " USD")
    add(30, "MCP JSON-RPC serialization p95", "<=", 4, rt.metrics.p("mcp.serialize", 95), "", " ms")
    for i, nme, thr, unit in ((31, "E2E p50 @ 50K VUs", 240, " ms"), (32, "E2E p95 @ 50K VUs", 450, " ms"), (33, "E2E p99 @ 50K VUs", 590, " ms"),
                              (34, "HTTP 5xx @ 50K VUs", 0.0005, ""), (35, "Sustained throughput", 8500, " RPS"), (36, "RSS growth 1-hr soak", 0.03, "")):
        add(i, nme, ">=" if i == 35 else "<=", thr, None, load_hint, unit)
    out.sort(key=lambda k: k.id)
    return out


def summarize(kpis: list[Kpi]) -> dict:
    return {"pass": sum(k.status == "PASS" for k in kpis), "fail": sum(k.status == "FAIL" for k in kpis),
            "not_measured": sum(k.status == "NOT_MEASURED" for k in kpis), "total": len(kpis)}
