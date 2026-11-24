"""Obsidian-style vault loader + Graphify-like graph build and bounded subgraph expansion (design §2.4)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..textutil import estimate_tokens, sentences

_FM = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.S)
_LINK = re.compile(r"\[\[([a-z0-9\-_.]+)\]\]")


@dataclass
class Note:
    id: str
    domain: str
    risk_tier: str
    version: int
    tenant: str
    body: str
    links: list[str] = field(default_factory=list)
    path: str = ""

    @property
    def summary(self) -> str:
        s = sentences(re.sub(r"^#.*\n", "", self.body).strip())
        return s[0] if s else self.id


def parse_note(path: Path) -> Note:
    m = _FM.match(path.read_text())
    if not m:
        raise ValueError(f"{path.name}: missing YAML frontmatter")
    meta = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    for req in ("id", "domain", "risk_tier", "version"):
        if req not in meta:
            raise ValueError(f"{path.name}: frontmatter missing '{req}'")
    body = m.group(2).strip()
    return Note(meta["id"], meta["domain"], meta["risk_tier"], int(meta["version"]),
                meta.get("tenant", "all"), body, sorted(set(_LINK.findall(body))), str(path))


def load_vault(vault_dir: Path) -> list[Note]:
    return [parse_note(p) for p in sorted(Path(vault_dir).glob("*.md"))]


def build_graph(notes: list[Note]) -> dict:
    ids = {n.id for n in notes}
    nodes = [{"id": n.id, "domain": n.domain, "risk_tier": n.risk_tier, "version": n.version,
              "tenant": n.tenant, "summary": n.summary} for n in notes]
    edges = [{"src": n.id, "dst": d, "kind": "wikilink"} for n in notes for d in n.links if d in ids]
    return {"nodes": nodes, "edges": edges}


def write_graph(notes: list[Note], out: Path) -> dict:
    g = build_graph(notes)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(g, indent=2))
    return g


class GraphStore:
    def __init__(self, graph: dict):
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        self.adj: dict[str, set[str]] = {i: set() for i in self.nodes}
        self.edges = graph["edges"]
        for e in self.edges:
            self.adj.setdefault(e["src"], set()).add(e["dst"])
            self.adj.setdefault(e["dst"], set()).add(e["src"])  # undirected traversal

    def visible(self, node_id: str, tenant_id: str) -> bool:
        n = self.nodes.get(node_id)
        return bool(n) and n["tenant"] in ("all", tenant_id)

    def expand(self, anchor: str, depth: int, token_cap: int, tenant_id: str) -> dict:
        if not self.visible(anchor, tenant_id):
            return {"nodes": [], "edges": [], "tokens": 0}
        seen, frontier, order = {anchor}, [anchor], [anchor]
        for _ in range(depth):
            nxt = []
            for cur in frontier:
                for nb in sorted(self.adj.get(cur, ())):
                    if nb not in seen and self.visible(nb, tenant_id):
                        seen.add(nb)
                        nxt.append(nb)
                        order.append(nb)
            frontier = nxt
        nodes, tokens = [], 0
        for nid in order:  # BFS order => closest nodes survive the token cap
            n = self.nodes[nid]
            cost = estimate_tokens(n["summary"]) + 6
            if tokens + cost > token_cap:
                break
            tokens += cost
            nodes.append({k: n[k] for k in ("id", "domain", "risk_tier", "version", "summary")})
        kept = {n["id"] for n in nodes}
        edges = [{"src": e["src"], "dst": e["dst"], "kind": e["kind"]} for e in self.edges
                 if e["src"] in kept and e["dst"] in kept]
        return {"nodes": nodes, "edges": edges, "tokens": tokens}
