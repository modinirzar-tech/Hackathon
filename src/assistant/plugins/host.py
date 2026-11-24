"""Pluggy-based lifecycle hooks (ADR-006). Plugins run outside the LLM, ordered by `priority`, and
`fail_closed` plugins turn any internal error into POLICY_BLOCKED instead of a silent pass-through."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pluggy

from ..errors import BreakerOpen, McpError, PolicyBlocked
from ..faults import Faults
from ..observability.tracing import Metrics, Tracer
from ..security import injection, pii
from ..textutil import content_tokens

hookspec = pluggy.HookspecMarker("assistant")
hookimpl = pluggy.HookimplMarker("assistant")


@dataclass
class HookCtx:
    tenant_id: str
    user_id: str
    session_id: str
    query: str
    pii_found: list[str] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)


class Specs:
    @hookspec
    def pre_execution(self, ctx: HookCtx): ...

    @hookspec
    def tool_dispatch(self, session_id: str, tool: str, args_hash: str): ...

    @hookspec
    def tool_result(self, session_id: str, tool: str, args_hash: str, ok: bool): ...

    @hookspec
    def post_execution(self, ctx: HookCtx, response: dict): ...


class PiiRedactor:
    name, priority, fail_closed, enabled = "pii_redactor", 10, True, True

    def __init__(self, faults: Faults):
        self.faults = faults

    @hookimpl
    def pre_execution(self, ctx: HookCtx):
        if "pii_plugin_error" in self.faults:
            raise RuntimeError("injected plugin failure")
        ctx.query, ctx.pii_found = pii.redact(ctx.query)

    @hookimpl
    def post_execution(self, ctx: HookCtx, response: dict):
        response["answer"], found = pii.redact(response.get("answer", ""))
        if found:
            ctx.flags["output_pii_scrubbed"] = found


class InjectionShield:
    name, priority, fail_closed, enabled = "injection_shield", 20, True, True

    def __init__(self, faults: Faults):
        self.faults = faults

    @hookimpl
    def pre_execution(self, ctx: HookCtx):
        if "injection_plugin_error" in self.faults:
            raise RuntimeError("injected plugin failure")
        attack, hits = injection.is_attack(ctx.query)
        if attack:
            raise PolicyBlocked("request blocked by injection shield", {"rules": hits})


class CircuitBreaker:
    """Blocks a (tool, args_hash) pair after `max_identical_calls` failures within a session."""
    name, priority, fail_closed, enabled = "circuit_breaker", 10, True, True

    def __init__(self, max_identical_calls: int = 2):
        self.max = max_identical_calls
        self.failures: dict[tuple[str, str, str], int] = {}

    @hookimpl
    def tool_dispatch(self, session_id: str, tool: str, args_hash: str):
        n = self.failures.get((session_id, tool, args_hash), 0)
        if n >= self.max:
            raise BreakerOpen(f"{tool} failed {n}x with identical arguments", {"tool": tool, "args_hash": args_hash})

    @hookimpl
    def tool_result(self, session_id: str, tool: str, args_hash: str, ok: bool):
        key = (session_id, tool, args_hash)
        if ok:
            self.failures.pop(key, None)
        else:
            self.failures[key] = self.failures.get(key, 0) + 1

    def reset(self):
        self.failures.clear()


class RagasAuditor:
    """Local proxy for RAGAS (the real library needs an LLM judge): faithfulness comes from the NLI-lite
    verifier, answer relevancy from question-token coverage. Production wires RAGAS asynchronously off-path."""
    name, priority, fail_closed, enabled = "ragas_auditor", 10, False, True

    def __init__(self, min_faithfulness: float, metrics: Metrics):
        self.min, self.metrics = min_faithfulness, metrics

    @hookimpl
    def post_execution(self, ctx: HookCtx, response: dict):
        f = response.get("faithfulness")
        if f is None:
            return
        self.metrics.observe("ragas.faithfulness", f)
        q, a = content_tokens(ctx.query), content_tokens(response.get("answer", ""))
        self.metrics.observe("ragas.answer_relevancy", len(q & a) / len(q) if q else 1.0)
        if f < self.min:
            self.metrics.inc("ragas_below_min")


class OtelExporter:
    name, priority, fail_closed, enabled = "otel_exporter", 100, False, True

    def __init__(self, tracer: Tracer, endpoint: str = ""):
        self.tracer, self.endpoint = tracer, endpoint

    @hookimpl
    def post_execution(self, ctx: HookCtx, response: dict):
        # Spans are already collected by Tracer; an OTLP adapter would flush them to `endpoint` here.
        response.setdefault("detail", {})["otel_endpoint"] = self.endpoint


class PluginHost:
    def __init__(self, metrics: Metrics, tracer: Tracer):
        self.pm = pluggy.PluginManager("assistant")
        self.pm.add_hookspecs(Specs)
        self.metrics, self.tracer = metrics, tracer
        self.plugins: dict[str, Any] = {}

    def register(self, plugin, config: dict | None = None):
        for k, v in (config or {}).items():
            if k in ("priority", "fail_closed", "enabled"):
                setattr(plugin, k, v)
        self.pm.register(plugin, name=plugin.name)
        self.plugins[plugin.name] = plugin

    def toggle(self, name: str, enabled: bool):
        self.plugins[name].enabled = enabled

    def _impls(self, stage: str):
        impls = [i for i in getattr(self.pm.hook, stage).get_hookimpls() if i.plugin.enabled]
        return sorted(impls, key=lambda i: i.plugin.priority)

    def run(self, stage: str, **kwargs):
        """Runs every enabled impl in priority order. fail_closed plugin errors => PolicyBlocked."""
        for impl in self._impls(stage):
            p = impl.plugin
            try:
                impl.function(**kwargs)
            except McpError:
                raise
            except Exception as exc:
                self.metrics.inc(f"plugin_error_{p.name}")
                if p.fail_closed:
                    raise PolicyBlocked(f"{p.name} failed; request refused (fail-closed)", {"plugin": p.name}) from exc

    def run_pre(self, ctx: HookCtx):
        t0 = time.perf_counter()
        with self.tracer.span("hook.pre"):
            self.run("pre_execution", ctx=ctx)
        self.metrics.observe("hook.pre.total", (time.perf_counter() - t0) * 1000)
