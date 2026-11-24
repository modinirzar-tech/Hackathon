from assistant.eval import datasets as D
from assistant.security import injection, pii
from assistant.security.auth import issue_token, verify_token
from assistant.errors import McpError
import pytest


def test_pii_redaction_covers_dataset():
    for text, expected in D.PII:
        out, found = pii.redact(text)
        assert expected <= set(found), (text, found)
        assert "@" not in out and "123-45" not in out


def test_injection_blocks_all_redteam_categories():
    for cat, prompts in D.redteam_prompts().items():
        assert len(prompts) == 20
        for p in prompts:
            attack, _ = injection.is_attack(p)
            red, _ = pii.redact(p)
            assert attack or (cat == "pii_smuggling" and red != p), (cat, p[:60])


def test_benign_not_blocked():
    for q in D.BENIGN:
        assert not injection.is_attack(q)[0], q


def test_token_roundtrip_and_tamper():
    t = issue_token("s", "tenant_a", "user_b", "approver")
    assert verify_token("s", t).role == "approver"
    with pytest.raises(McpError):
        verify_token("other-secret", t)
    with pytest.raises(McpError):
        verify_token("s", t[:-4] + "AAAA")
