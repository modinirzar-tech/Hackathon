"""Deterministic PII redaction (fail-closed callers). Handles plain and obfuscated forms:
name[at]domain[dot]com, 'name at domain dot com', fullwidth/homoglyph digits, spaced SSNs, card numbers."""
from __future__ import annotations

import re
import unicodedata

_ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_BAT = r"(?:\[\s*at\s*\]|\(\s*at\s*\)|\{\s*at\s*\})"
_BDOT = r"(?:\[\s*dot\s*\]|\(\s*dot\s*\)|\{\s*dot\s*\})"
_LABEL = r"[A-Za-z0-9\-]+"
_TLD = r"[A-Za-z]{2,}"

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(rf"[A-Za-z0-9._%+\-]+@{_LABEL}(?:\.{_LABEL})*\.{_TLD}")),                       # standard
    ("EMAIL", re.compile(rf"[A-Za-z0-9._%+\-]+\s*{_BAT}\s*{_LABEL}(?:\s*(?:\.|{_BDOT})\s*{_LABEL})*", re.I)),  # name[at]domain...
    ("EMAIL", re.compile(rf"\b[A-Za-z0-9._%+\-]+\s+at\s+{_LABEL}(?:\s+dot\s+{_LABEL})+\b", re.I)),          # name at domain dot com
    ("SSN", re.compile(r"\b\d{3}[\s.\-]?\d{2}[\s.\-]?\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ \-]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\w)")),
]
_WORD_DIGITS = re.compile(r"\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|oh)[\s\-]+){6,}(?:zero|one|two|three|four|five|six|seven|eight|nine|oh)\b", re.I)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_ZW)


def redact(text: str) -> tuple[str, list[str]]:
    """Returns (redacted_text, sorted unique PII types found). Input is NFKC-normalised first so
    fullwidth digits / zero-width joiners can't smuggle PII past the patterns."""
    out, found = normalize(text), set()
    for label, pat in PATTERNS:
        out, n = pat.subn(f"[{label}]", out)
        if n:
            found.add(label)
    out, n = _WORD_DIGITS.subn("[PHONE]", out)
    if n:
        found.add("PHONE")
    return out, sorted(found)
