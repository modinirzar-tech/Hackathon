"""Bounded ReAct loop (ADR-004): INIT->REASONING->TOOL_CALL->OBSERVATION->VERIFY->DONE/HALTED.
Limits are enforced HERE, not in a prompt: max 5 steps, per-call 120 ms, loop 450 ms, breaker on identical failures.
The reasoner is a deterministic policy; an LLM planner can replace `Policy.next_actions` without touching limits."""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from ..config import Settings
from ..errors import BreakerOpen, BulkheadFull, Code, McpError
from ..observability.tracing import Metrics, Tracer
from ..textutil import content_tokens, estimate_tokens, sentences
from .harness import CallContext, ToolHarness


@dataclass(frozen=True)
class GoalAnchor:
    """Immutable block pinned at index 0 of every reasoning prompt (anti-drift)."""
    query: str
    route: str
    entities: tuple[str, ...]
    constraints: tuple[str, ...] = ("answer only from retrieved evidence", "never mutate without approval")

    def render(self) -> str:
        return f"GOAL: {self.query}\nROUTE: {self.route}\nENTITIES: {', '.join(self.entities)}\nCONSTRAINTS: {'; '.join(self.constraints)}"


@dataclass
class LoopResult:
    status: str  # DONE | HALTED | HALTED_BREAKER
    steps: int = 0
    states: list[str] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    graph: dict | None = None
    abstain: bool = False
    unreranked: bool = False
    remediation: dict | None = None
    observations: list[str] = field(default_factory=list)
    tool_results: dict = field(default_factory=dict)
    error: McpError | None = None
    halt_reason: str = ""


def compress_observation(chunks: list[dict], query: str, max_tokens: int) -> str:
    """Extractive, strips JSON boilerplate; keeps highest query-overlap sentences within the token budget."""
    q = content_tokens(query)
    cands = []
    for c in chunks:
        for s in sentences(re.sub(r"^#.*\n?", "", c["text"])):
            cands.append((len(q & content_tokens(s)) / (len(q) or 1), c["chunk_id"][:8], s))
    cands.sort(key=lambda x: -x[0])
    out, used = [], 0
    for _, cid, s in cands:
        cost = estimate_tokens(s)
        if used + cost > max_tokens:
            continue
        out.append(f"({cid}) {s}")
        used += cost
    return " ".join(out)


class RetrievalPolicy:
    """Step 1: hybrid_retrieve (+ graph_expand on the top chunk). Multi-hop: one follow-up retrieval seeded by neighbours."""

    def __init__(self, settings: Settings, complexity: str):
        self.s, self.complexity = settings, complexity

    async def run(self, loop: "ReActLoop", ctx: CallContext, goal: GoalAnchor, res: LoopResult) -> None:
        query = goal.query
        for step in range(1, self.s.max_steps + 1):
            res.steps = step
            res.states.append("REASONING")
            retr = await loop.tool(ctx, "hybrid_retrieve", {"query": query, "threshold": self.s.threshold}, res)
            res.chunks = _merge(res.chunks, retr["chunks"])
            res.abstain, res.unreranked = retr["abstain"], retr["unreranked"]
            res.tool_results.setdefault("stage_ms", []).append(retr["stage_ms"])
            res.states.append("OBSERVATION")
            if retr["abstain"]:
                res.observations.append("no sufficiently relevant evidence")
                res.states += ["VERIFY", "DONE"]
                res.status = "DONE"
                return
            res.observations.append(compress_observation(retr["chunks"], goal.query, self.s.observation_tokens))
            if step == 1:
                anchor = retr["chunks"][0]["source_uri"].removeprefix("vault://")
                try:
                    res.graph = await loop.tool(ctx, "graph_expand", {"anchor": anchor, "depth": 2 if self.complexity == "multi_hop" else 1,
                                                                      "token_cap": self.s.graph_token_cap}, res)
                except BreakerOpen:
                    raise
                except McpError:
                    res.graph = None  # FMEA #5: Graphify is an optimisation, never a correctness dependency
            res.states.append("VERIFY")
            if self.complexity == "multi_hop" and step == 1 and res.graph:
                seen = {c["source_uri"].removeprefix("vault://") for c in res.chunks}
                extra = [n["id"].replace("-", " ") for n in res.graph["nodes"] if n["id"] not in seen]
                if extra:
                    query = f"{goal.query} {extra[0]}"
                    continue  # needs more evidence and steps < max
            res.states.append("DONE")
            res.status = "DONE"
            return
        res.status, res.halt_reason = "HALTED", "max_steps"


def _merge(old: list[dict], new: list[dict]) -> list[dict]:
    seen = {c["chunk_id"] for c in old}
    merged = old + [c for c in new if c["chunk_id"] not in seen]
    return sorted(merged, key=lambda c: -c["rerank_score"])[:8]


class ReActLoop:
    def __init__(self, harness: ToolHarness, settings: Settings, tracer: Tracer, metrics: Metrics):
        self.h, self.s, self.tracer, self.metrics = harness, settings, tracer, metrics
        self._deadline = 0.0

    async def tool(self, ctx: CallContext, name: str, args: dict, res: LoopResult) -> dict:
        remaining = self._deadline - time.perf_counter()
        if remaining <= 0:
            raise _Halt("loop_budget")
        res.states.append("TOOL_CALL")
        budget = min(self.s.step_budget_ms / 1000, remaining)
        try:
            return await asyncio.wait_for(self.h.call(ctx, name, args), budget)
        except asyncio.TimeoutError:
            raise _Halt("step_budget") from None

    async def run(self, ctx: CallContext, goal: GoalAnchor, policy: RetrievalPolicy) -> LoopResult:
        res = LoopResult(status="HALTED", states=["INIT", "TRIAGE"])
        self._deadline = time.perf_counter() + self.s.loop_budget_ms / 1000
        with self.tracer.span("agent.react.loop") as sp:
            try:
                await policy.run(self, ctx, goal, res)
            except BreakerOpen as e:
                res.status, res.states = "HALTED_BREAKER", res.states + ["HALTED_BREAKER"]
                res.error = e
                res.remediation = {"tool": e.data.get("tool"), "message": "Identical call failed repeatedly; stopped. "
                                   "Correct the arguments (SKU-### / LOT-#####) or ask the user for clarification."}
            except BulkheadFull:
                res.status, res.halt_reason = "HALTED", "load_shed"
                res.states.append("HALTED")
            except _Halt as h:
                res.status, res.halt_reason = "HALTED", h.reason
                res.states.append("HALTED")
            except McpError as e:
                if e.code == Code.MCP_TIMEOUT:
                    res.status, res.halt_reason = "HALTED", "mcp_timeout"
                    res.states.append("HALTED")
                res.error = e
            sp.attrs.update(steps=res.steps, status=res.status)
        self.metrics.observe("react.steps", res.steps)
        self.metrics.inc(f"loop_{res.status.lower()}")
        return res


class _Halt(Exception):
    def __init__(self, reason: str):
        self.reason = reason
