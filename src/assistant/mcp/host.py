"""MCP host/client inside FastAPI: JSON serialisation, per-server bulkheads, timeouts, manifest pinning."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time

from ..errors import BulkheadFull, Code, McpError
from ..models import Envelope
from ..observability.tracing import Metrics, Tracer
from .base import McpServer


class McpHost:
    def __init__(self, auth_secret: str, tracer: Tracer, metrics: Metrics, timeout_ms: float = 150.0):
        self.secret = auth_secret
        self.tracer, self.metrics = tracer, metrics
        self.timeout_ms = timeout_ms
        self.servers: dict[str, McpServer] = {}
        self.pools: dict[str, asyncio.Semaphore] = {}
        self.pool_size: dict[str, int] = {}
        self._rid = 0

    def connect(self, alias: str, server: McpServer, pool_size: int, expected_sig: str | None = None) -> str:
        sm = server.signed_manifest()
        good = hmac.new(self.secret.encode(), json.dumps(sm["manifest"], sort_keys=True).encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(good, sm["signature"]) or (expected_sig and not hmac.compare_digest(expected_sig, sm["signature"])):
            raise McpError(Code.POLICY_BLOCKED, f"manifest signature mismatch for {alias}; refusing to connect")
        self.servers[alias] = server
        self.pools[alias] = asyncio.Semaphore(pool_size)
        self.pool_size[alias] = pool_size
        return sm["signature"]

    async def call(self, alias: str, tool: str, arguments: dict, env: Envelope, auth_token: str,
                   timeout_ms: float | None = None) -> dict:
        sem = self.pools[alias]
        if sem.locked():
            self.metrics.inc("bulkhead_rejected")
            raise BulkheadFull(alias)
        self._rid += 1
        t0 = time.perf_counter()
        raw = json.dumps({"jsonrpc": "2.0", "id": self._rid, "method": "tools/call", "params": {
            "name": tool, "arguments": arguments, "_meta": {**env.model_dump(), "auth_token": auth_token}}})
        ser_ms = (time.perf_counter() - t0) * 1000
        timeout = (timeout_ms or self.timeout_ms) / 1000
        async with sem:
            with self.tracer.span(f"mcp.{alias}.{tool}"):
                try:
                    resp_raw = await asyncio.wait_for(self.servers[alias].handle_raw(raw), timeout)
                except asyncio.TimeoutError as exc:
                    self.metrics.inc("mcp_timeouts")
                    raise McpError(Code.MCP_TIMEOUT, f"{alias}.{tool} exceeded {timeout * 1000:.0f} ms") from exc
        t1 = time.perf_counter()
        resp = json.loads(resp_raw)
        self.metrics.observe("mcp.serialize", ser_ms + (time.perf_counter() - t1) * 1000)
        if "error" in resp:
            e = resp["error"]
            raise McpError(e["code"], e["message"], e.get("data"))
        return resp["result"]["structuredContent"]

    async def read_resource(self, alias: str, uri: str, env: Envelope, auth_token: str) -> dict:
        raw = json.dumps({"jsonrpc": "2.0", "id": 0, "method": "resources/read", "params": {
            "uri": uri, "_meta": {**env.model_dump(), "auth_token": auth_token}}})
        resp = json.loads(await self.servers[alias].handle_raw(raw))
        if "error" in resp:
            raise McpError(resp["error"]["code"], resp["error"]["message"])
        return resp["result"]
