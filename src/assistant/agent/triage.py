"""Triage: (intent, domain, risk_tier, complexity) -> route. Deterministic rules, temp=0, <25 ms.
Safe defaults (FMEA #11): ambiguous => HYBRID_RAG read-only; never default to execution."""
from __future__ import annotations

import re
from dataclasses import dataclass

DOMAINS = {
    "finance": r"refund|expense|reimburs|discount|rebate|invoice|payment|receipt",
    "hr": r"pto|time off|vacation|leave|payroll|employee|carry ?over",
    "security": r"password|security|incident|sev\d|mfa|breach|retention|audit|pii|compromise",
    "quality": r"recall|quarantin|lot\b|workflow|release|state|transition|approval",
    "logistics": r"ship|delivery|rma|return|restocking|freight",
    "product": r"sku|warranty|catalog|printer|scanner|hardware|accessor",
}
OOS = re.compile(r"\b(weather|joke|poem|recipe|stock price|bitcoin|crypto price|capital of|movie|sports score|lyrics|horoscope|"
                 r"translate|write (me )?(a|an) (story|essay|song|poem|script)|who won|meaning of life|tell me a story)\b", re.I)
GREET = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening)|bye)\b[\s!.,?a-z]{0,25}$", re.I)
QUESTION_START = re.compile(r"^(what|how|why|when|where|who|which|is|are|does|do|did|should|am|will)\b", re.I)
POLITE = re.compile(r"^(please|kindly|can you|could you|would you|i want to|i need to|i'd like to|i would like to|go ahead and)[ ,]+", re.I)
ACTION_VERBS = {
    "transition_workflow": r"transition|move|promote|release|recall|quarantine|reject|revert|set|mark",
    "submit_form": r"submit|file|raise|open|request|create|lodge|send in",
}
TARGET = re.compile(r"\b(sku|lot)[\s_-]*(\d{1,5})\b", re.I)
AUDIT = re.compile(r"\b(verify|check|validate)\b.{0,25}\b(audit|chain|log integrity)\b", re.I)
MULTI = re.compile(r"\b(and also|compare|both|as well as|then|versus|vs\.?|difference between)\b|\?.*\?", re.I)


@dataclass(frozen=True)
class TriageResult:
    route: str  # DIRECT_ANSWER | HYBRID_RAG | TRANSACTION_EXEC | REJECT_OOS | AUDIT_VERIFY
    intent: str
    domain: str
    risk_tier: str
    complexity: str  # simple | multi_hop
    confidence: float
    action: str | None = None


def triage(query: str) -> TriageResult:
    q = query.strip()
    low = q.lower()
    domain = next((d for d, pat in DOMAINS.items() if re.search(pat, low)), "general")
    if GREET.match(q):
        return TriageResult("DIRECT_ANSWER", "greeting", "general", "TIER_1_READ", "simple", 0.95)
    if AUDIT.search(q):
        return TriageResult("AUDIT_VERIFY", "audit", "security", "TIER_1_READ", "simple", 0.9)
    stripped = POLITE.sub("", low)
    if not QUESTION_START.match(stripped) and TARGET.search(low):
        for action, verbs in ACTION_VERBS.items():
            if re.match(rf"^(?:{verbs})\b", stripped):
                tier = "TIER_3_MUTATE" if action == "transition_workflow" else "TIER_2_COMPUTE"
                return TriageResult("TRANSACTION_EXEC", action, domain, tier, "simple", 0.9, action)
    if OOS.search(low) and domain == "general":
        return TriageResult("REJECT_OOS", "out_of_scope", "general", "TIER_1_READ", "simple", 0.9)
    cx = "multi_hop" if MULTI.search(q) else "simple"
    return TriageResult("HYBRID_RAG", "question", domain, "TIER_1_READ", cx, 0.85 if domain != "general" else 0.6)


# ---- transaction slot filling (regex-only, deterministic) ---------------------------
STATE_WORDS = {"released": "RELEASED", "release": "RELEASED", "recall": "RECALLED", "recalled": "RECALLED",
               "quarantine": "QUARANTINED", "quarantined": "QUARANTINED", "review": "REVIEW", "draft": "DRAFT"}
FORMS = {"return": "return_request", "expense": "expense_report", "incident": "incident_report",
         "pto": "pto_request", "time off": "pto_request", "warranty": "warranty_claim"}


def parse_transaction(query: str, action: str) -> dict:
    low = query.lower()
    m = TARGET.search(low)
    raw_target = f"{m.group(1)} {m.group(2)}" if m else ""
    if action == "transition_workflow":
        state = next((v for k, v in STATE_WORDS.items() if re.search(rf"\b{k}\b", low)), None)
        payload = {"to_state": state, "reason": "requested via assistant"} if state else {}
    else:
        form = next((v for k, v in FORMS.items() if k in low), None)
        payload = {"form_type": form, "notes": "submitted via assistant"} if form else {}
    return {"target": raw_target, "payload": payload}
