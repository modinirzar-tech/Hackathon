"""Shared bootstrap so each MCP server can run as its own stdio worker (blueprint `uv run mcp_servers/*.py`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistant.config import Settings  # noqa: E402
from assistant.faults import Faults  # noqa: E402
from assistant.graph.vault import GraphStore, build_graph, load_vault  # noqa: E402
from assistant.rag.engine import RagEngine  # noqa: E402
from assistant.state.store import StateStore  # noqa: E402
