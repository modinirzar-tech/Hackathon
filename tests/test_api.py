from fastapi.testclient import TestClient

from assistant.api.main import create_app


def test_api_flow(rt):
    with TestClient(create_app(rt)) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.post("/v1/ask", json={"query": "hi"}).status_code == 401
        tok = c.post("/v1/dev/token", json={"tenant_id": "tenant_acme", "user_id": "user_alice", "role": "user"}).json()["token"]
        r = c.post("/v1/ask", json={"query": "How long is the hardware warranty?"}, headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200 and r.json()["status"] == "ANSWERED" and r.json()["citations"]
        assert c.post("/v1/ask", json={"query": "x", "tenant_id": "tenant_evil"}, headers={"Authorization": f"Bearer {tok}"}).status_code == 422
        assert "assistant_requests_total" in c.get("/metrics").text
        assert c.get("/readyz").json()["chunks"] > 0


def test_load_shedding(rt):
    rt.settings.max_inflight = 0
    with TestClient(create_app(rt)) as c:
        tok = rt.token()
        assert c.post("/v1/ask", json={"query": "hello"}, headers={"Authorization": f"Bearer {tok}"}).status_code == 503


def test_kpi_gates_have_no_measured_failures(rt):
    from tests.conftest import run
    from assistant.eval.kpi import evaluate
    kpis = run(evaluate(rt))
    assert len(kpis) == 36
    assert [k.name for k in kpis if k.status == "FAIL"] == []
