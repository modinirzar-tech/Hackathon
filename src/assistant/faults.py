"""Fault-injection switches used by chaos tests and the Streamlit Inspector (FMEA validation)."""
from __future__ import annotations

KNOWN = {"reranker_down", "rag_hang", "graph_down", "state_down", "llm_429", "pii_plugin_error", "injection_plugin_error"}


class Faults:
    def __init__(self):
        self.active: set[str] = set()

    def set(self, name: str, on: bool = True):
        if name not in KNOWN:
            raise ValueError(f"unknown fault {name}; known: {sorted(KNOWN)}")
        (self.active.add if on else self.active.discard)(name)

    def __contains__(self, name: str) -> bool:
        return name in self.active

    def clear(self):
        self.active.clear()
