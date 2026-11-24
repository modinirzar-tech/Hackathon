# Enterprise Agentic Assistant

Implementation of [design.md](design.md): an MCP-structured, plugin-extensible, multi-tenant RAG + gated-transaction assistant
with bounded ReAct loop, fail-closed security hooks, HITL for Tier-3 actions, hash-chained audit and a KPI scorecard.

## Quick start
```bash
pip install -r requirements.txt
make test      # 40 tests: contracts, security, RAG, loop, HITL, tenant isolation, FMEA fault injection
make demo      # end-to-end scenarios in-process
make run       # API on :8000   (POST /v1/dev/token -> POST /v1/ask)
make ui        # Streamlit workbench (6 tabs)
make kpi       # scorecard for every KPI measurable locally
```
```bash
TOK=$(curl -s -XPOST :8000/v1/dev/token -H 'content-type: application/json' -d '{"tenant_id":"tenant_acme","user_id":"user_alice"}' | jq -r .token)
curl -s -XPOST :8000/v1/ask -H "Authorization: Bearer $TOK" -H 'content-type: application/json' -d '{"query":"How many days of PTO do I get?"}'
```
Approver flow: issue a token with `"role":"approver"` for a *different* user, then `GET /v1/approvals` and
`POST /v1/approvals/{id}/decision {"decision":"approve"}`.

## Layout
| Path | Role (design ref) |
|---|---|
| `src/assistant/models.py`, `errors.py` | Strict Pydantic contracts, error codes (§1.2) |
| `src/assistant/mcp/` | JSON-RPC 2.0 server core, 3 servers, host with bulkheads/timeouts/signed manifests (§1.3) |
| `mcp_servers/*.py` | Standalone stdio / HTTP workers matching `mcp_config.json` |
| `src/assistant/rag/` | BM25 → dense (FAISS HNSW) → RRF → rerank → abstain gate (ADR-001/005) |
| `src/assistant/graph/`, `knowledge_vault/` | Obsidian notes + Graphify-style graph, ≤2-hop, token cap (§2.4) |
| `src/assistant/state/` | SQLite state store (hash-chained audit, approvals, facts, idempotency), tenant cache (ADR-003) |
| `src/assistant/plugins/` | Pluggy hooks: PII, injection (fail-closed), circuit breaker, RAGAS proxy, OTel (ADR-006) |
| `src/assistant/agent/` | Triage, tool harness (validate→repair→RBAC→idempotency→breaker), bounded loop, verifier, orchestrator (§1.4, ADR-004) |
| `src/assistant/api/` | FastAPI gateway, load shedding, `/metrics` |
| `src/assistant/eval/`, `scripts/kpi_report.py` | Datasets + KPI evaluator |
| `app.py` | Streamlit workbench (§2.8) |
| `load_tests/k6_50k_load.js` | k6 scenario with KPI-31..34 thresholds |

## What is real vs. a stand-in (read before trusting any number)
This runs with **no GPU, Redis, Postgres, or model downloads**, so several production components are deliberate stand-ins behind the
same interfaces:

| Design component | Here | To go to production |
|---|---|---|
| BGE-Large bi-encoder (ONNX) | Deterministic hashing-trick embedder (`rag/dense.py`, `Embedder` protocol) | Implement `Embedder` with BGE |
| `bge-reranker-v2-m3` (ONNX/Triton) | Lexical cross-encoder stand-in (idf coverage + bigrams + cosine), same `[0,1]` contract and 0.42 gate | Replace `Reranker.score` |
| vLLM / TRT-LLM | Deterministic **extractive** generator; `LLM_BASE_URL` enables an OpenAI-compatible client with 429 backoff + extractive fallback | Point at vLLM |
| Redis Cluster | In-process TTL cache with the same key scheme / jitter / single-flight | Back `SemanticCache` with Redis |
| PostgreSQL + pgvector | SQLite, same schema, `tenant_id` in every query | Swap `StateStore` driver |
| MCP stdio/HTTP transports | Servers run **in-process** through real JSON-RPC (serialised) with timeouts; stdio/HTTP workers exist in `mcp_servers/` but the host does not spawn them | Add a stdio/HTTP client transport in `McpHost` |
| RAGAS | Local proxy (NLI-lite faithfulness, token-coverage relevancy) | Run RAGAS off-path with an LLM judge |
| OTel | In-process tracer + Prometheus text; JSONL export via `OTEL_JSONL_EXPORT` | Add OTLP exporter to `Tracer.exporters` |
| Triage / injection classifier | Deterministic rules | Distilled classifier |
| JMeter plan, Graphviz PNG | **Not included** (Mermaid source in `docs/`) | — |

**KPIs:** `make kpi` measures 25 of the 36 and all pass on the bundled *smoke-scale* sets (26 golden questions, 8 negatives,
100 template-generated red-team prompts authored alongside the rules — not an independent benchmark, and not the 250-question golden set).
KPIs 18, 21, 22, 25, 29 and 31–36 report `NOT_MEASURED`: the 50K-VU latency/throughput/soak gates need `load_tests/k6_50k_load.js`
run distributed against a scaled deployment. **No claim is made that the 450 ms p95 @ 50K VU target is met.**

## Behaviours worth knowing
- `tenant_id`/`user_id` come only from the signed token; MCP servers re-verify, and arguments containing `env` are rejected.
- Tier is floored server-side (`transition_workflow` is always Tier 3 → `PENDING_APPROVAL`); approver must differ from requester; approvals expire to `DENIED`.
- Fail-closed: PII/injection plugin errors refuse the request; state-store outage refuses mutations while read-only answers continue with audit buffered in a local WAL.
- Limit enforcement (5 steps / 120 ms per call / 450 ms loop / breaker at 2 identical failures) lives in code, not prompts. The 120 ms bound applies per tool call.
- Manifests are pinned in `data/mcp_manifest.lock.json` (auto-created in `DEV_MODE`; refused if missing when `DEV_MODE=0`). After changing a tool, run `make pin`.
- `mcp_config.json` sets the state-store `pool_size` to 128 (design bulkhead `db_pool`); the blueprint had 200.
- Open questions Q1–Q10 in design.md were resolved as proposed there (HNSW, 25 ms triage, Tab 6 scorecard, etc.); Q3 (tenant-shared cache) is **not** implemented — cache is strictly per tenant+user.
