#!/usr/bin/env python3
"""Equivalent of `graphify build ./knowledge_vault --output data/graph.json`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assistant.config import Settings
from assistant.graph.vault import load_vault, write_graph

s = Settings()
g = write_graph(load_vault(s.vault_dir), s.graph_path)
print(f"wrote {s.graph_path}: {len(g['nodes'])} nodes, {len(g['edges'])} edges")
