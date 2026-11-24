"""State store: SQLite (dev/test) with the same schema as the Postgres design. tenant_id is in every WHERE.
Hash-chained audit log: hash_n = SHA256(hash_{n-1} || payload)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid

from ..errors import Code, McpError

SCHEMA = """
CREATE TABLE IF NOT EXISTS session_facts(id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, session_id TEXT NOT NULL,
  fact TEXT NOT NULL, source_chunk_ids TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS audit_log(seq INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, payload TEXT NOT NULL,
  prev_hash TEXT NOT NULL, hash TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS approvals(approval_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, requester TEXT NOT NULL, action TEXT NOT NULL,
  target TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL, approver TEXT, created_at REAL NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS transactions(transaction_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, action TEXT NOT NULL, target TEXT NOT NULL,
  payload TEXT NOT NULL, status TEXT NOT NULL, approval_id TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS idempotency(tenant_id TEXT NOT NULL, idem_key TEXT NOT NULL, method TEXT NOT NULL, result TEXT NOT NULL,
  PRIMARY KEY(tenant_id, idem_key, method));
"""
GENESIS = "0" * 64
TIER_FLOOR = {"submit_form": "TIER_2_COMPUTE", "transition_workflow": "TIER_3_MUTATE"}


class StateStore:
    def __init__(self, path: str = ":memory:", approval_ttl_s: float = 86400.0):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.approval_ttl_s = approval_ttl_s
        with self.lock:
            if path != ":memory:":
                self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)

    # ---- idempotency -----------------------------------------------------
    def idem_get(self, tenant: str, key: str, method: str) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT result FROM idempotency WHERE tenant_id=? AND idem_key=? AND method=?",
                                (tenant, key, method)).fetchone()
        return json.loads(r["result"]) if r else None

    def idem_put(self, tenant: str, key: str, method: str, result: dict) -> None:
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO idempotency VALUES(?,?,?,?)", (tenant, key, method, json.dumps(result)))

    # ---- audit -------------------------------------------------------------
    @staticmethod
    def _hash(prev: str, payload: str) -> str:
        return hashlib.sha256((prev + payload).encode()).hexdigest()

    def audit_append(self, tenant: str, event: dict) -> tuple[int, str]:
        payload = json.dumps(event, sort_keys=True, default=str)
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                r = self.db.execute("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
                prev = r["hash"] if r else GENESIS
                h = self._hash(prev, f"{tenant}|{payload}")
                cur = self.db.execute("INSERT INTO audit_log(tenant_id,payload,prev_hash,hash,created_at) VALUES(?,?,?,?,?)",
                                      (tenant, payload, prev, h, time.time()))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise
        return cur.lastrowid, h

    def audit_verify(self) -> dict:
        """The chain is global (single sequence); verification is read-only integrity metadata, not content."""
        with self.lock:
            rows = self.db.execute("SELECT seq,tenant_id,payload,prev_hash,hash FROM audit_log ORDER BY seq").fetchall()
        prev = GENESIS
        for r in rows:
            if r["prev_hash"] != prev or r["hash"] != self._hash(prev, f"{r['tenant_id']}|{r['payload']}"):
                return {"ok": False, "entries": len(rows), "first_bad_seq": r["seq"]}
            prev = r["hash"]
        return {"ok": True, "entries": len(rows), "first_bad_seq": None}

    def audit_events(self, tenant: str, limit: int = 100) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT seq,payload,hash,created_at FROM audit_log WHERE tenant_id=? ORDER BY seq DESC LIMIT ?",
                                   (tenant, limit)).fetchall()
        return [{"seq": r["seq"], "event": json.loads(r["payload"]), "hash": r["hash"], "ts": r["created_at"]} for r in rows]

    # ---- facts -------------------------------------------------------------
    def facts_put(self, tenant: str, session: str, fact: str, chunk_ids: list[str]) -> None:
        with self.lock:
            self.db.execute("INSERT INTO session_facts(tenant_id,session_id,fact,source_chunk_ids,created_at) VALUES(?,?,?,?,?)",
                            (tenant, session, fact, json.dumps(chunk_ids), time.time()))

    def facts_get(self, tenant: str, session: str) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT fact,source_chunk_ids,created_at FROM session_facts WHERE tenant_id=? AND session_id=? ORDER BY id",
                                   (tenant, session)).fetchall()
        return [{"fact": r["fact"], "source_chunk_ids": json.loads(r["source_chunk_ids"]), "created_at": r["created_at"]} for r in rows]

    # ---- transactions / HITL ---------------------------------------------------
    def _run(self, tenant: str, action: str, target: str, payload: dict, approval_id: str | None) -> str:
        tx = str(uuid.uuid4())
        self.db.execute("INSERT INTO transactions VALUES(?,?,?,?,?,?,?,?)",
                        (tx, tenant, action, target, json.dumps(payload), "EXECUTED", approval_id, time.time()))
        return tx

    def execute_transaction(self, tenant: str, user: str, action: str, tier: str, target: str, payload: dict) -> dict:
        from ..models import TIER_RANK
        effective = tier if TIER_RANK[tier] >= TIER_RANK[TIER_FLOOR[action]] else TIER_FLOOR[action]
        with self.lock:
            if effective == "TIER_3_MUTATE":  # never executes directly
                aid = str(uuid.uuid4())
                now = time.time()
                self.db.execute("INSERT INTO approvals VALUES(?,?,?,?,?,?,?,?,?,?)",
                                (aid, tenant, user, action, target, json.dumps(payload), "PENDING_APPROVAL", None, now, now + self.approval_ttl_s))
                return {"status": "PENDING_APPROVAL", "approval_id": aid, "transaction_id": None, "effective_tier": effective}
            tx = self._run(tenant, action, target, payload, None)
        return {"status": "EXECUTED", "approval_id": None, "transaction_id": tx, "effective_tier": effective}

    def expire_approvals(self, now: float | None = None) -> int:
        with self.lock:
            cur = self.db.execute("UPDATE approvals SET status='DENIED', approver='SYSTEM_TTL' WHERE status='PENDING_APPROVAL' AND expires_at<?",
                                  (now or time.time(),))
        return cur.rowcount

    def approvals_list(self, tenant: str) -> list[dict]:
        self.expire_approvals()
        with self.lock:
            rows = self.db.execute("SELECT * FROM approvals WHERE tenant_id=? AND status='PENDING_APPROVAL' ORDER BY created_at", (tenant,)).fetchall()
        return [{"approval_id": r["approval_id"], "requester": r["requester"], "action": r["action"], "target": r["target"],
                 "payload": json.loads(r["payload"]), "status": r["status"], "created_at": r["created_at"],
                 "expires_at": r["expires_at"]} for r in rows]

    def approval_decide(self, tenant: str, approver: str, approval_id: str, decision: str) -> dict:
        self.expire_approvals()
        with self.lock:
            r = self.db.execute("SELECT * FROM approvals WHERE approval_id=? AND tenant_id=?", (approval_id, tenant)).fetchone()
            if not r:
                raise McpError(Code.INVALID_PARAMS, "unknown approval")
            if r["status"] != "PENDING_APPROVAL":
                raise McpError(Code.INVALID_PARAMS, f"approval already {r['status']}")
            if r["requester"] == approver:
                raise McpError(Code.RBAC_DENIED, "approver must differ from requester")
            if decision == "reject":
                self.db.execute("UPDATE approvals SET status='DENIED', approver=? WHERE approval_id=?", (approver, approval_id))
                return {"status": "DENIED", "approval_id": approval_id, "transaction_id": None}
            tx = self._run(tenant, r["action"], r["target"], json.loads(r["payload"]), approval_id)
            self.db.execute("UPDATE approvals SET status='EXECUTED', approver=? WHERE approval_id=?", (approver, approval_id))
            return {"status": "EXECUTED", "approval_id": approval_id, "transaction_id": tx}

    def transactions_list(self, tenant: str) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM transactions WHERE tenant_id=? ORDER BY created_at", (tenant,)).fetchall()
        return [dict(r) for r in rows]

    def tamper_for_test(self, seq: int) -> None:  # used by tests/Streamlit to prove tamper detection
        with self.lock:
            self.db.execute("UPDATE audit_log SET payload='{\"tampered\": true}' WHERE seq=?", (seq,))
