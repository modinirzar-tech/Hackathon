from assistant.eval import datasets as D


def test_golden_retrieval_top1(rt):
    misses = []
    for q, expect in D.GOLDEN:
        r = rt.engine.retrieve("tenant_x", q)
        if r["abstain"] or r["chunks"][0]["source_uri"] != f"vault://{expect}":
            misses.append(q)
    assert not misses, misses


def test_negatives_abstain(rt):
    for q in D.NEGATIVE:
        assert rt.engine.retrieve("tenant_x", q)["abstain"], q


def test_tenant_scoped_chunks(rt):
    assert not rt.engine.retrieve("tenant_x", "Acme partner volume discount units")["chunks"][0]["source_uri"].endswith("acme-discount-tiers")
    assert rt.engine.retrieve("tenant_acme", "Acme partner volume discount units")["chunks"][0]["source_uri"].endswith("acme-discount-tiers")


def test_rrf_dedupes_and_limits(rt):
    r = rt.engine.retrieve("tenant_x", "refund", rerank_k=8)
    assert len({c["chunk_hash"] for c in r["chunks"]}) == len(r["chunks"]) <= 8


def test_graph_expand_bounded(rt):
    g = rt.graph.expand("refund-policy", 2, 320, "tenant_x")
    assert g["tokens"] <= 320 and g["nodes"][0]["id"] == "refund-policy"
    assert rt.graph.expand("acme-discount-tiers", 1, 320, "tenant_x")["nodes"] == []  # foreign tenant node invisible
