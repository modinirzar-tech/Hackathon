"""Streamlit developer workbench (design §2.8). Run: streamlit run app.py"""
import asyncio
import os
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
os.environ.setdefault("DB_PATH", str(Path(__file__).resolve().parent / "data" / "workbench.db"))

from assistant.eval import datasets as D  # noqa: E402
from assistant.eval.kpi import evaluate, summarize  # noqa: E402
from assistant.faults import KNOWN  # noqa: E402
from assistant.models import AskRequest  # noqa: E402
from assistant.runtime import build_runtime  # noqa: E402
from assistant.security import injection, pii  # noqa: E402

st.set_page_config(page_title="Agentic Platform Workbench", layout="wide")


@st.cache_resource
def runtime():
    return build_runtime()


rt = runtime()
run = asyncio.run
with st.sidebar:
    st.header("Identity")
    tenant = st.text_input("tenant_id", "tenant_acme")
    user = st.text_input("user_id", "user_alice")
    role = st.selectbox("role", ["user", "approver", "readonly"])
    session = st.session_state.setdefault("session", str(uuid.uuid4()))
    st.caption(f"session {session[:8]}…")
tok = rt.token(tenant, user, role)

tabs = st.tabs(["1 Architecture & Workflow", "2 Hybrid RAG Lab", "3 MCP & Plugin Inspector", "4 Memory & Graphify",
                "5 Red Team & Breaker", "6 KPI Scorecard"])

with tabs[0]:
    st.subheader("Live triage console")
    q = st.text_input("Query", "What is the refund policy?")
    if st.button("Ask") and q:
        r = run(rt.orch.ask(AskRequest(query=q, session_id=session), tok))
        st.session_state["last"] = r
    r = st.session_state.get("last")
    if r:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", r.status); c2.metric("Route", r.route); c3.metric("Steps", r.steps); c4.metric("Latency ms", r.latency_ms)
        st.write(r.answer)
        if r.citations:
            st.json([c.model_dump() for c in r.citations])
        with st.expander("ReAct trace / spans"):
            st.json(r.detail)
            st.table([{"span": s.name, "ms": round(s.duration_ms, 2)} for s in rt.tracer.get(r.trace_id)])
    st.subheader("Tier-3 HITL queue")
    if role == "approver":
        pend = run(rt.host.call("state-store", "approval_list", {}, __import__("assistant.agent.harness", fromlist=["CallContext"]).CallContext(
            __import__("assistant.security.auth", fromlist=["verify_token"]).verify_token(rt.settings.auth_secret, tok), tok, session, "d" * 32
        ).envelope(str(uuid.uuid4())), tok))["approvals"]
        for a in pend:
            st.write(a)
            cols = st.columns(2)
            from assistant.agent.harness import CallContext
            from assistant.security.auth import verify_token
            cx = CallContext(verify_token(rt.settings.auth_secret, tok), tok, session, "d" * 32)
            for col, dec in zip(cols, ("approve", "reject")):
                if col.button(f"{dec} {a['approval_id'][:6]}"):
                    try:
                        st.success(run(rt.host.call("state-store", "approval_decide", {"approval_id": a["approval_id"], "decision": dec}, cx.envelope(str(uuid.uuid4())), tok)))
                    except Exception as e:
                        st.error(str(e))
        if not pend:
            st.info("No pending approvals.")
    else:
        st.info("Switch role to 'approver' (different user than the requester) to review frozen actions.")
    with st.expander("Workflow diagram (Mermaid source)"):
        st.code(Path("docs/workflow_orchestration.mmd").read_text(), language="mermaid")

with tabs[1]:
    st.subheader("Hybrid RAG lab")
    q2 = st.text_input("Retrieval query", "How many days of PTO do employees get?", key="q2")
    c = st.columns(4)
    k1 = c[0].slider("BM25 k1", 0.5, 2.0, rt.settings.bm25_k1); b = c[1].slider("BM25 b", 0.0, 1.0, rt.settings.bm25_b)
    rrf_k = c[2].slider("RRF k", 1, 200, rt.settings.rrf_k); thr = c[3].slider("abstain threshold", 0.0, 1.0, rt.settings.threshold)
    rt.engine.bm25.k1, rt.engine.bm25.b = k1, b
    out = rt.engine.retrieve(tenant, q2, rrf_k=rrf_k, threshold=thr)
    st.write(f"top score **{out['top_score']}** → {'ABSTAIN' if out['abstain'] else 'ANSWER'}  | stage ms {out['stage_ms']}")
    st.dataframe([{k: ch[k] for k in ("source_uri", "bm25", "cosine", "rrf_rank", "rerank_score")} for ch in out["chunks"]])

with tabs[2]:
    st.subheader("MCP servers")
    st.table([{"alias": a, "server": s.name, "tools": ", ".join(s.tools), "pool": rt.host.pool_size[a], "calls": s.calls} for a, s in rt.host.servers.items()])
    st.subheader("Plugins (hot toggle)")
    for name, p in rt.plugins.plugins.items():
        rt.plugins.toggle(name, st.checkbox(f"{name} (prio {p.priority}, fail_closed={p.fail_closed})", p.enabled, key=f"pl_{name}"))
    st.subheader("Fault injection (FMEA)")
    for f in sorted(KNOWN):
        rt.faults.set(f, st.checkbox(f, f in rt.faults, key=f"f_{f}"))
    rt.cache.available = not st.checkbox("redis_down (cache bypass)", not rt.cache.available)

with tabs[3]:
    st.subheader("Graphify subgraph")
    anchor = st.selectbox("anchor", sorted(rt.notes)); depth = st.radio("depth", [1, 2], horizontal=True)
    g = rt.graph.expand(anchor, depth, 320, tenant)
    st.write(f"{len(g['nodes'])} nodes, {g['tokens']} tokens (cap 320)")
    st.graphviz_chart("digraph{rankdir=LR;" + "".join(f'"{e["src"]}"->"{e["dst"]}";' for e in g["edges"]) + "}")
    st.markdown(f"**Note**\n\n{rt.notes[anchor].body}")
    st.subheader("Session facts (verified only)")
    st.json(rt.store.facts_get(tenant, session))
    b_, o_ = rt.metrics.counters.get("tokens_baseline", 0), rt.metrics.counters.get("tokens_optimized", 0)
    st.metric("Prompt-token reduction (measured this process)", f"{(1 - o_ / b_) * 100:.1f}%" if b_ else "n/a")

with tabs[4]:
    st.subheader("Red team")
    cat = st.selectbox("Category", list(D.redteam_prompts()))
    if st.button("Run category"):
        rows = []
        for p in D.redteam_prompts()[cat]:
            att, hits = injection.is_attack(p); red, found = pii.redact(p)
            rows.append({"prompt": p[:70], "blocked": att, "pii": found, "neutralised": att or red != p})
        st.dataframe(rows)
    st.subheader("Circuit breaker")
    if st.button("Reset breaker"):
        rt.plugins.plugins["circuit_breaker"].reset(); st.success("reset")
    st.json({str(k): v for k, v in rt.plugins.plugins["circuit_breaker"].failures.items()})
    if st.button("Verify audit chain"):
        st.json(rt.store.audit_verify())

with tabs[5]:
    st.subheader("36-KPI scorecard")
    st.caption("Measured on bundled smoke-scale datasets. KPIs 18, 21, 22, 25, 29, 31–36 are NOT_MEASURED here (need load rig / GPU / RAGAS judge).")
    if st.button("Run evaluation"):
        st.session_state["kpis"] = run(evaluate(rt))
    k = st.session_state.get("kpis")
    if k:
        st.json(summarize(k))
        st.dataframe([x.dict() for x in k])
