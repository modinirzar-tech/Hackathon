"""mcp-state-store worker. `--http` serves Streamable-HTTP-style POST /mcp on :8081, otherwise stdio."""
import sys

from _common import Faults, Settings, StateStore
from assistant.mcp.servers import build_state_server

if __name__ == "__main__":
    s = Settings()
    srv = build_state_server(StateStore(s.db_path, s.approval_ttl_s), s.auth_secret, Faults())
    if "--http" in sys.argv:
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import Response

        app = FastAPI()

        @app.post("/mcp")
        async def mcp(request: Request):
            return Response(await srv.handle_raw((await request.body()).decode()), media_type="application/json")

        uvicorn.run(app, host="127.0.0.1", port=8081)
    else:
        srv.serve_stdio()
