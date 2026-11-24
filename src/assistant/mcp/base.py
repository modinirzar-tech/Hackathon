"""Minimal MCP-style JSON-RPC 2.0 server core: tools/list, tools/call, resources/read, ping.
Pipeline per call: auth verify -> tenant bind (-32001) -> tool lookup -> strict validation (-32602)
-> RBAC tier (-32002) -> handler (run in a worker thread so host timeouts are real)."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sys
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ValidationError

from ..errors import Code, McpError
from ..models import TIER_RANK
from ..security.auth import AuthContext, verify_token


@dataclass
class ToolSpec:
    name: str
    model: type[BaseModel]
    handler: Callable[[BaseModel, AuthContext], dict]
    tier: str
    role: str | None = None  # required role, e.g. "approver"


class McpServer:
    def __init__(self, name: str, auth_secret: str):
        self.name = name
        self.auth_secret = auth_secret
        self.tools: dict[str, ToolSpec] = {}
        self.resources: dict[str, Callable[[str, AuthContext], dict]] = {}
        self.calls = 0

    # ---- registration ----------------------------------------------------
    def tool(self, name: str, model: type[BaseModel], tier: str, role: str | None = None):
        def deco(fn):
            self.tools[name] = ToolSpec(name, model, fn, tier, role)
            return fn
        return deco

    def resource(self, prefix: str):
        def deco(fn):
            self.resources[prefix] = fn
            return fn
        return deco

    # ---- manifest (signed; host pins it) ----------------------------------
    def manifest(self) -> dict:
        return {"server": self.name, "tools": {
            n: {"tier": t.tier, "role": t.role,
                "schema_sha256": hashlib.sha256(json.dumps(t.model.model_json_schema(), sort_keys=True).encode()).hexdigest()}
            for n, t in sorted(self.tools.items())}, "resources": sorted(self.resources)}

    def signed_manifest(self) -> dict:
        m = self.manifest()
        sig = hmac.new(self.auth_secret.encode(), json.dumps(m, sort_keys=True).encode(), hashlib.sha256).hexdigest()
        return {"manifest": m, "signature": sig}

    # ---- JSON-RPC ----------------------------------------------------------
    @staticmethod
    def _err(rid, err: McpError) -> dict:
        return {"jsonrpc": "2.0", "id": rid, "error": err.to_json()}

    async def handle_raw(self, raw: str) -> str:
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            return json.dumps(self._err(None, McpError(Code.PARSE_ERROR, "invalid JSON")))
        return json.dumps(await self.handle(req))

    async def handle(self, req: dict) -> dict:
        rid = req.get("id") if isinstance(req, dict) else None
        try:
            if not isinstance(req, dict) or req.get("jsonrpc") != "2.0" or "method" not in req:
                raise McpError(Code.INVALID_REQUEST, "not a JSON-RPC 2.0 request")
            method, params = req["method"], req.get("params") or {}
            self.calls += 1
            if method == "ping":
                return {"jsonrpc": "2.0", "id": rid, "result": {}}
            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": rid, "result": self.signed_manifest()}
            if method not in ("tools/call", "resources/read"):
                raise McpError(Code.METHOD_NOT_FOUND, f"unknown method {method}")
            meta = dict(params.get("_meta") or {})
            token = meta.pop("auth_token", None)
            if not token:
                raise McpError(Code.RBAC_DENIED, "missing auth token")
            auth = verify_token(self.auth_secret, token)
            if meta.get("tenant_id") != auth.tenant_id or meta.get("user_id") != auth.user_id:
                raise McpError(Code.TENANT_MISMATCH, "tenant/user in request differs from authenticated identity")
            if method == "resources/read":
                uri = params.get("uri", "")
                for prefix, fn in self.resources.items():
                    if uri.startswith(prefix):
                        return {"jsonrpc": "2.0", "id": rid, "result": await asyncio.to_thread(fn, uri, auth)}
                raise McpError(Code.METHOD_NOT_FOUND, f"no resource for {uri}")
            spec = self.tools.get(params.get("name", ""))
            if not spec:
                raise McpError(Code.METHOD_NOT_FOUND, f"unknown tool {params.get('name')}")
            args = params.get("arguments") or {}
            if "env" in args:
                raise McpError(Code.INVALID_PARAMS, "arguments may not contain 'env'")
            try:
                model = spec.model.model_validate({"env": meta, **args})
            except ValidationError as exc:
                raise McpError(Code.INVALID_PARAMS, "validation failed",
                               {"errors": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()][:8]}) from exc
            if model.env.tenant_id != auth.tenant_id:  # defense in depth
                raise McpError(Code.TENANT_MISMATCH, "tenant mismatch")
            if TIER_RANK[spec.tier] > TIER_RANK[auth.max_tier] and not (spec.tier == "TIER_3_MUTATE" and spec.role is None
                                                                      and TIER_RANK[auth.max_tier] >= 2):
                raise McpError(Code.RBAC_DENIED, f"tier {spec.tier} exceeds grant {auth.max_tier}")
            if spec.role and auth.role != spec.role:
                raise McpError(Code.RBAC_DENIED, f"role '{spec.role}' required")
            result = await asyncio.to_thread(spec.handler, model, auth)
            return {"jsonrpc": "2.0", "id": rid, "result": {"structuredContent": result}}
        except McpError as e:
            return self._err(rid, e)
        except Exception as e:  # never leak internals
            return self._err(rid, McpError(-32603, "internal error", {"type": type(e).__name__}))

    # ---- stdio transport (local workers) ----------------------------------
    def serve_stdio(self) -> None:  # pragma: no cover - exercised by scripts/run_mcp_stdio.py
        async def loop():
            for line in sys.stdin:
                if line.strip():
                    sys.stdout.write(await self.handle_raw(line) + "\n")
                    sys.stdout.flush()
        asyncio.run(loop())
