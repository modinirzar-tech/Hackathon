"""Verification subagent: independent NLI-lite claim->chunk support check with veto power.
A sentence is supported iff the cited chunk contains >= SUPPORT of its content tokens AND all its numbers."""
from __future__ import annotations

import re

from ..textutil import content_tokens, sentences

SUPPORT = 0.8
_CITE = re.compile(r"\[(\d+)\]")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def verify(answer: str, context: dict[int, str]) -> dict:
    """context: n -> chunk text. Returns {faithfulness, supported, total, unsupported:[...]}"""
    total, ok, bad = 0, 0, []
    for sent in sentences(answer):
        cites = [int(c) for c in _CITE.findall(sent)]
        body = _CITE.sub("", sent).strip()
        if not content_tokens(body):
            continue
        total += 1
        evidence = " ".join(context.get(c, "") for c in cites)
        toks, ev = content_tokens(body), content_tokens(evidence)
        nums_ok = all(n in evidence for n in _NUM.findall(body))
        if cites and ev and len(toks & ev) / len(toks) >= SUPPORT and nums_ok:
            ok += 1
        else:
            bad.append(sent)
    return {"faithfulness": ok / total if total else 0.0, "supported": ok, "total": total, "unsupported": bad}
