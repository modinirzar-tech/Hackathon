"""FastAPI gateway: authN, tenant bind, load shedding, REST surface for ask / approvals / audit / metrics."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from ..errors import BulkheadFull, Code, McpError
from ..models import AskRequest, AskResponse
from ..runtime import Runtime, build_runtime
from ..security.auth import AuthContext, issue_token, verify_token


class DecisionBody(BaseModel):
    decision: str  # approve | reject
    comment: str = ""


class TokenBody(BaseModel):
    tenant_id: str
    user_id: str
    role: str = "user"


def create_app(runtime: Runtime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rt = runtime or build_runtime()
        app.state.inflight = 0
        yield

    app = FastAPI(title="Enterprise Agentic Assistant", version="1.0.0", lifespan=lifespan)

    def rt_of(request: Request) -> Runtime:
        return request.app.state.rt

    def auth_of(request: Request, authorization: str = Header(default="")) -> tuple[AuthContext, str]:
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        token = authorization[7:]
        try:
            return verify_token(rt_of(request).settings.auth_secret, token), token
        except McpError:
            raise HTTPException(401, "invalid token")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request):
        rt = rt_of(request)
        return {"status": "ready", "chunks": len(rt.engine.chunks), "plugins": {n: p.enabled for n, p in rt.plugins.plugins.items()},
                "faults": sorted(rt.faults.active)}

    @app.post("/v1/dev/token")
    async def dev_token(body: TokenBody, request: Request):
        rt = rt_of(request)
        if not rt.settings.dev_mode:
            raise HTTPException(404)
        try:
            return {"token": issue_token(rt.settings.auth_secret, body.tenant_id, body.user_id, body.role)}
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.post("/v1/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request, a=Depends(auth_of)):
        rt = rt_of(request)
        if request.app.state.inflight >= rt.settings.max_inflight:  # shed load instead of queueing (protects p99)
            rt.metrics.inc("shed_503")
            return JSONResponse({"detail": "overloaded"}, status_code=503, headers={"Retry-After": "1"})
        request.app.state.inflight += 1
        try:
            return await rt.orch.ask(body, a[1])
        except BulkheadFull:
            return JSONResponse({"detail": "bulkhead full"}, status_code=503, headers={"Retry-After": "1"})
        finally:
            request.app.state.inflight -= 1

    def _env_call(rt: Runtime, a, tool: str, args: dict):
        from ..agent.harness import CallContext
        ctx = CallContext(a[0], a[1], str(uuid.uuid4()), rt.tracer.start_trace())
        return rt.host.call("state-store", tool, args, ctx.envelope(str(uuid.uuid4())), a[1])

    @app.get("/v1/approvals")
    async def approvals(request: Request, a=Depends(auth_of)):
        try:
            return await _env_call(rt_of(request), a, "approval_list", {})
        except McpError as e:
            raise HTTPException(403 if e.code == Code.RBAC_DENIED else 400, e.message)

    @app.post("/v1/approvals/{approval_id}/decision")
    async def decide(approval_id: str, body: DecisionBody, request: Request, a=Depends(auth_of)):
        try:
            return await _env_call(rt_of(request), a, "approval_decide",
                                   {"approval_id": approval_id, "decision": body.decision, "comment": body.comment})
        except McpError as e:
            raise HTTPException(403 if e.code == Code.RBAC_DENIED else 400, e.message)

    @app.get("/v1/audit/verify")
    async def audit_verify(request: Request, a=Depends(auth_of)):
        return await _env_call(rt_of(request), a, "audit_verify", {})

    @app.get("/v1/session/{session_id}/facts")
    async def facts(session_id: str, request: Request, a=Depends(auth_of)):
        rt = rt_of(request)
        return {"facts": rt.store.facts_get(a[0].tenant_id, session_id)}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics(request: Request):
        rt = rt_of(request)
        return rt.metrics.prometheus() + f"assistant_cache_hit_ratio {rt.cache.hit_ratio:.4f}\n"

    return app


app = create_app()
