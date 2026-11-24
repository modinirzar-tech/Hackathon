"""Shared tokenisation helpers (deterministic, dependency-free)."""
from __future__ import annotations

import math
import re

STOP = frozenset("""a an the of to in on for and or is are was were be been being do does did what which who whom how when where why
can could should would will shall may might must i you we they he she it its this that these those with by from at as about into than then
there their our your my me us please tell give any also not no if so such per via get got has have had""".split())
_WORD = re.compile(r"[a-z0-9]+")
_ID = re.compile(r"\b(?:sku|lot)-\d+\b")


def stem(w: str) -> str:
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    low = text.lower()
    out = [m for m in _ID.findall(low)]
    for w in _WORD.findall(low):
        if not keep_stop and w in STOP:
            continue
        out.append(stem(w))
    return out


def content_tokens(text: str) -> set[str]:
    return set(tokens(text))


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.split()) * 1.3)) if text.strip() else 0


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?!\[\d+\])|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


_WIKI = re.compile(r"\[\[([^\]]+)\]\]")


def unwiki(text: str) -> str:
    return _WIKI.sub(r"\1", text)
