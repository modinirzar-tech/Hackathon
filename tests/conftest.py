import asyncio
import os
import tempfile

import pytest

os.environ["DB_PATH"] = ":memory:"
os.environ["DEV_MODE"] = "1"


@pytest.fixture()
def rt(tmp_path, monkeypatch):
    monkeypatch.setenv("MANIFEST_LOCK", str(tmp_path / "lock.json"))
    monkeypatch.setenv("GRAPH_PATH", str(tmp_path / "graph.json"))
    from assistant.runtime import build_runtime
    return build_runtime()


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def ask(rt):
    from assistant.models import AskRequest

    def _ask(q, tenant="tenant_acme", user="user_alice", role="user", session=None):
        return run(rt.orch.ask(AskRequest(query=q, session_id=session), rt.token(tenant, user, role)))
    return _ask
