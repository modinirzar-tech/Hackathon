"""HMAC-signed bearer tokens standing in for the enterprise IdP. Servers re-verify (defense in depth)."""
from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass

from ..errors import Code, McpError
from ..models import TIER_RANK

ROLE_MAX_TIER = {"user": "TIER_2_COMPUTE", "approver": "TIER_3_MUTATE", "readonly": "TIER_1_READ"}


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    user_id: str
    role: str

    @property
    def max_tier(self) -> str:
        return ROLE_MAX_TIER[self.role]

    def allows(self, tier: str) -> bool:
        return TIER_RANK[tier] <= TIER_RANK[self.max_tier]


def _sig(secret: str, body: str) -> str:
    return hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()[:32]


def issue_token(secret: str, tenant_id: str, user_id: str, role: str = "user") -> str:
    if role not in ROLE_MAX_TIER:
        raise ValueError(f"unknown role {role}")
    body = f"{tenant_id}|{user_id}|{role}"
    raw = f"{body}|{_sig(secret, body)}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def verify_token(secret: str, token: str) -> AuthContext:
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        tenant, user, role, sig = raw.split("|")
    except Exception as exc:  # malformed
        raise McpError(Code.RBAC_DENIED, "malformed token") from exc
    if not hmac.compare_digest(sig, _sig(secret, f"{tenant}|{user}|{role}")) or role not in ROLE_MAX_TIER:
        raise McpError(Code.RBAC_DENIED, "bad token signature")
    return AuthContext(tenant, user, role)
