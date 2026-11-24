"""JSON-RPC / MCP error codes (design §1.2.3)."""
from __future__ import annotations

from enum import IntEnum


class Code(IntEnum):
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    TENANT_MISMATCH = -32001
    RBAC_DENIED = -32002
    IDEMPOTENT_REPLAY = -32003
    MCP_TIMEOUT = -32004
    BREAKER_OPEN = -32005
    HITL_PENDING = -32006
    POLICY_BLOCKED = -32007
    ABSTAIN_LOW_CONFIDENCE = -32008
    RATE_LIMITED_UPSTREAM = -32009


class McpError(Exception):
    def __init__(self, code: Code | int, message: str = "", data: dict | None = None):
        self.code = int(code)
        self.name = Code(code).name if code in Code._value2member_map_ else "UNKNOWN"
        self.message = message or self.name
        self.data = data or {}
        super().__init__(f"{self.name}({self.code}): {self.message}")

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message, "data": {"name": self.name, **self.data}}


class PolicyBlocked(McpError):
    def __init__(self, message: str, data: dict | None = None):
        super().__init__(Code.POLICY_BLOCKED, message, data)


class BreakerOpen(McpError):
    def __init__(self, message: str, data: dict | None = None):
        super().__init__(Code.BREAKER_OPEN, message, data)


class BulkheadFull(Exception):
    """Raised when a bulkhead pool has no free slot (load shedding)."""
