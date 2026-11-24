"""stdio MCP worker: mcp-graphify-vault."""
from _common import Faults, GraphStore, Settings, build_graph, load_vault
from assistant.mcp.servers import build_graph_server

if __name__ == "__main__":
    s = Settings()
    build_graph_server(GraphStore(build_graph(load_vault(s.vault_dir))), s.auth_secret, Faults()).serve_stdio()
