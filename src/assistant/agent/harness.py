"""ToolExecutionHarness (design §2.1): strict Pydantic -> regex auto-repair -> RBAC tier -> UUIDv4 idempotency
-> circuit breaker -> dispatch. Telemetry feeds KPI-14/15."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from ..errors import BreakerOpen, Code, McpError
from ..mcp.host import McpHost
from ..models import (TIER_RANK, AuditVerifyIn, ExecuteTransactionIn, Envelope, FactsPutIn, GraphExpandIn, HybridRetrieveIn,
                      make_envelope_stub)
from ..observability.tracing import Metrics, Tracer, make_traceparent
from ..plugins.host import PluginHost
from ..security.auth import AuthContext

TOOLS: dict[str, tuple[str, type[BaseModel], str]] = {
    # tool -> (server alias, model, tier)
    "hybrid_retrieve": ("hybrid-rag", HybridRetrieveIn, "TIER_1_READ"),
    "graph_expand": ("graphify-vault", GraphExpandIn, "TIER_1_READ"),
    "execute_transaction": ("state-store", ExecuteTransactionIn, "TIER_3_MUTATE"),
    "session_facts_put": ("state-store", FactsPutIn, "TIER_2_COMPUTE"),
    "audit_verify": ("state-store", AuditVerifyIn, "TIER_1_READ"),
}
_SKU = re.compile(r"^\s*(?:sku)[\s_-]*(\d{1,3})\s*$", re.I)
_LOT = re.compile(r"^\s*(?:lot)[\s_-]*(\d{1,5})\s*$", re.I)


def repair_value(v: Any) -> Any:
    """Deterministic, regex-only: SKU99/sku 99 -> SKU-099 ; lot 44/LOT44 -> LOT-00044. Recurses into dict/list."""
    if isinstance(v, str):
        if m := _SKU.match(v):
            return f"SKU-{int(m.group(1)):03d}"
        if m := _LOT.match(v):
            return f"LOT-{int(m.group(1)):05d}"
        return v
    if isinstance(v, dict):
        return {k: repair_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [repair_value(x) for x in v]
    return v


def args_hash(tool: str, args: dict) -> str:
    return hashlib.sha256(f"{tool}:{json.dumps(args, sort_keys=True, default=str)}".encode()).hexdigest()[:16]


@dataclass
class CallContext:
    auth: AuthContext
    token: str
    session_id: str
    trace_id: str
    idem_keys: dict[str, str] = field(default_factory=dict)  # args_hash -> stable UUIDv4 (safe retries)

    def envelope(self, key: str) -> Envelope:
        return Envelope(tenant_id=self.auth.tenant_id, user_id=self.auth.user_id, session_id=self.session_id,
                        idempotency_key=key, traceparent=make_traceparent(self.trace_id))


class ToolHarness:
    def __init__(self, host: McpHost, plugins: PluginHost, tracer: Tracer, metrics: Metrics):
        self.host, self.plugins, self.tracer, self.metrics = host, plugins, tracer, metrics

    def _validate(self, model: type[BaseModel], tool: str, args: dict) -> tuple[dict, bool]:
        """Returns (args, repaired). Raises McpError(-32602) with remediation if repair does not fix it."""
        probe = {"env": make_envelope_stub(), **args}
        try:
            model.model_validate(probe)
            self.metrics.inc("tool_first_pass_valid")
            return args, False
        except ValidationError as first:
            fixed = repair_value(args)
            try:
                model.model_validate({"env": make_envelope_stub(), **fixed})
            except ValidationError as second:
                self.metrics.inc("tool_repair_failed")
                raise McpError(Code.INVALID_PARAMS, f"{tool}: arguments invalid after auto-repair",
                               {"errors": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in second.errors()][:5],
                                "remediation": "Provide arguments matching the tool schema exactly "
                                               "(SKU-### and LOT-#####)."}) from first
            self.metrics.inc("tool_repaired")
            return fixed, True

    async def call(self, ctx: CallContext, tool: str, args: dict) -> dict:
        alias, model, tier = TOOLS[tool]
        self.metrics.inc("tool_calls")
        args, _ = self._validate(model, tool, args)                                    # 1+2
        if TIER_RANK[tier] > TIER_RANK[ctx.auth.max_tier] and not (tool == "execute_transaction" and ctx.auth.allows("TIER_2_COMPUTE")):
            raise McpError(Code.RBAC_DENIED, f"{tool} requires {tier}")                # 3
        h = args_hash(tool, args)
        key = ctx.idem_keys.setdefault(h, str(uuid.uuid4()))                           # 4 (stable per identical call)
        self.plugins.run("tool_dispatch", session_id=ctx.session_id, tool=tool, args_hash=h)  # breaker
        try:
            out = await self.host.call(alias, tool, args, ctx.envelope(key), ctx.token)
        except BreakerOpen:
            raise
        except Exception:
            self.plugins.run("tool_result", session_id=ctx.session_id, tool=tool, args_hash=h, ok=False)
            raise
        self.plugins.run("tool_result", session_id=ctx.session_id, tool=tool, args_hash=h, ok=True)
        return out
