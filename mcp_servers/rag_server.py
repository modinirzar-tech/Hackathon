"""stdio MCP worker: mcp-hybrid-rag.  Run: python mcp_servers/rag_server.py  (JSON-RPC lines on stdin/stdout)."""
from _common import Faults, RagEngine, Settings, load_vault
from assistant.mcp.servers import build_rag_server

if __name__ == "__main__":
    s = Settings()
    build_rag_server(RagEngine(load_vault(s.vault_dir), s), s.auth_secret, Faults()).serve_stdio()
