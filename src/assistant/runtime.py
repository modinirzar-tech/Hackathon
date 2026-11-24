"""Composition root: builds every component once. Used by the API, Streamlit workbench, tests and eval scripts."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .agent.harness import ToolHarness
from .agent.llm import LLM, ExtractiveLLM, OpenAICompatLLM
from .agent.loop import ReActLoop
from .agent.orchestrator import Orchestrator
from .config import Settings
from .errors import Code, McpError
from .faults import Faults
from .graph.vault import GraphStore, build_graph, load_vault, write_graph
from .mcp.host import McpHost
from .mcp.servers import build_graph_server, build_rag_server, build_state_server
from .observability.tracing import Metrics, Tracer
from .plugins.host import CircuitBreaker, InjectionShield, OtelExporter, PiiRedactor, PluginHost, RagasAuditor
from .rag.engine import RagEngine
from .security.auth import issue_token
from .state.cache import SemanticCache
from .state.store import StateStore


@dataclass
class Runtime:
    settings: Settings
    metrics: Metrics
    tracer: Tracer
    faults: Faults
    plugins: PluginHost
    host: McpHost
    store: StateStore
    cache: SemanticCache
    engine: RagEngine
    graph: GraphStore
    notes: dict
    llm: LLM
    harness: ToolHarness
    loop: ReActLoop
    orch: Orchestrator

    def token(self, tenant: str = "tenant_acme", user: str = "user_alice", role: str = "user") -> str:
        return issue_token(self.settings.auth_secret, tenant, user, role)


def _verify_or_pin_manifests(settings: Settings, sigs: dict[str, str]) -> None:
    """Signed-manifest pinning: first run pins (dev only); afterwards any change refuses startup."""
    lock = settings.manifest_lock_path
    if lock.exists():
        pinned = json.loads(lock.read_text())
        for alias, sig in sigs.items():
            if pinned.get(alias) != sig:
                raise McpError(Code.POLICY_BLOCKED, f"manifest for '{alias}' changed since it was pinned ({lock}); "
                                                    "review and run scripts/pin_manifests.py")
    elif settings.dev_mode:
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps(sigs, indent=2))
    else:
        raise McpError(Code.POLICY_BLOCKED, "unsigned/unpinned MCP manifests in non-dev mode")


def build_runtime(settings: Settings | None = None, llm: LLM | None = None) -> Runtime:
    s = settings or Settings()
    cfg = s.mcp_config()
    metrics = Metrics()
    tracer = Tracer(metrics)
    faults = Faults()

    notes_list = load_vault(s.vault_dir)
    notes = {n.id: n for n in notes_list}
    graph_json = write_graph(notes_list, s.graph_path) if not s.graph_path.exists() else json.loads(s.graph_path.read_text())
    if {n["id"] for n in graph_json["nodes"]} != set(notes):  # stale graph => rebuild
        graph_json = write_graph(notes_list, s.graph_path)
    graph = GraphStore(graph_json)
    engine = RagEngine(notes_list, s)
    store = StateStore(s.db_path, s.approval_ttl_s)
    cache = SemanticCache(s.cache_ttl_s, s.cache_jitter_s, s.cache_cosine)

    host = McpHost(s.auth_secret, tracer, metrics, s.mcp_timeout_ms)
    servers = cfg["mcpServers"]
    sigs = {
        "hybrid-rag": host.connect("hybrid-rag", build_rag_server(engine, s.auth_secret, faults), servers["hybrid-rag"].get("pool_size", s.rag_pool)),
        "graphify-vault": host.connect("graphify-vault", build_graph_server(graph, s.auth_secret, faults), servers["graphify-vault"].get("pool_size", s.rag_pool)),
        "state-store": host.connect("state-store", build_state_server(store, s.auth_secret, faults), servers["state-store"].get("pool_size", s.db_pool)),
    }
    _verify_or_pin_manifests(s, sigs)

    plugins = PluginHost(metrics, tracer)
    pcfg = {p["name"]: p for p in cfg.get("plugins", [])}
    breaker = CircuitBreaker(pcfg.get("circuit_breaker", {}).get("max_identical_calls", 2))
    for plugin in (PiiRedactor(faults), InjectionShield(faults), breaker,
                   RagasAuditor(pcfg.get("ragas_auditor", {}).get("min_faithfulness", 0.92), metrics),
                   OtelExporter(tracer, pcfg.get("otel_exporter", {}).get("endpoint", ""))):
        plugins.register(plugin, {k: v for k, v in pcfg.get(plugin.name, {}).items() if k in ("priority", "fail_closed")})

    if llm is None:
        base = ExtractiveLLM(faults)
        llm = OpenAICompatLLM(s.llm_base_url, s.llm_model, base, faults) if s.llm_base_url else base
    harness = ToolHarness(host, plugins, tracer, metrics)
    loop = ReActLoop(harness, s, tracer, metrics)
    orch = Orchestrator(s, harness, loop, plugins, cache, llm, tracer, metrics, notes, graph, host)
    return Runtime(s, metrics, tracer, faults, plugins, host, store, cache, engine, graph, notes, llm, harness, loop, orch)
