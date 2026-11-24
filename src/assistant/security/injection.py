"""Heuristic prompt-injection / abuse classifier (direct, indirect, multi-turn, SQLi/arg-overflow).
Score = weighted rule hits; >= BLOCK_SCORE blocks. Same scanner is applied to retrieved chunks (indirect injection)."""
from __future__ import annotations

import re

from .pii import normalize

BLOCK_SCORE = 1.0
_HOMOGLYPH = str.maketrans({"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ѕ": "s"})

RULES: list[tuple[str, float, re.Pattern]] = [(n, w, re.compile(p, re.I | re.S)) for n, w, p in [
    ("override", 1.0, r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|your|system)\b.{0,30}\b(instruction|rule|prompt|polic|guardrail|constraint|direction)"),
    ("exfil_prompt", 1.0, r"\b(reveal|show|print|repeat|display|output|leak|tell me)\b.{0,40}\b(system|hidden|initial|original|developer)\b.{0,15}\b(prompt|instruction|message|rule)"),
    ("persona", 1.0, r"\b(you are now|act as|pretend (to be|you are)|roleplay as|from now on you)\b.{0,60}\b(dan|unrestricted|unfiltered|jailbr|no (rules|restrictions|limits)|evil|developer mode)"),
    ("jailbreak_kw", 1.0, r"\b(jailbreak|dan mode|developer mode|do anything now|without (any )?(restrictions|filters|guardrails))\b"),
    ("crescendo", 0.6, r"\b(as we (agreed|discussed)|earlier you (said|agreed|promised)|you already (agreed|confirmed)|continuing from before)\b.{0,80}\b(without|skip|no need|bypass|ignore)"),
    ("approval_bypass", 1.0, r"\b(skip|bypass|without|no need for|auto[- ]?approve|self[- ]?approve)\b.{0,25}\b(approval|approver|hitl|human review|manager sign)"),
    ("indirect_marker", 1.0, r"(<\s*/?\s*(system|assistant|instructions?)\s*>|\[\s*(system|inst)\s*\]|###\s*(system|instruction)|BEGIN\s+(HIDDEN|SYSTEM)\s+INSTRUCTION)"),
    ("indirect_cmd", 0.8, r"\b(assistant|ai|model|llm)\b[, :]{0,3}\b(must|should|will now|is required to)\b.{0,40}\b(ignore|reveal|send|forward|disable|exfiltrate)"),
    ("sqli", 1.0, r"(\bunion\b.{0,10}\bselect\b|;\s*(drop|delete|truncate|update|insert)\b|'\s*or\s*'?1'?\s*=\s*'?1|--\s*$|/\*.*\*/|\bxp_cmdshell\b|\bsleep\s*\(\d+\))"),
    ("tool_overflow", 1.0, r"(\.\./\.\./|%00|\$\{jndi:|\\x[0-9a-f]{2}(\\x[0-9a-f]{2}){8,}|(.)\3{200,})"),
    ("encoded_payload", 1.0, r"\b(base64|rot13|hex)\b.{0,30}\b(decode|decoded|follow|execute|run)\b"),
]]


def scan(text: str) -> tuple[float, list[str]]:
    norm = normalize(text).translate(_HOMOGLYPH)
    score, hits = 0.0, []
    for name, weight, pat in RULES:
        if pat.search(norm):
            score += weight
            hits.append(name)
    if len(norm) > 4000:
        score += 1.0
        hits.append("oversize")
    return score, hits


def is_attack(text: str) -> tuple[bool, list[str]]:
    s, hits = scan(text)
    return s >= BLOCK_SCORE, hits
