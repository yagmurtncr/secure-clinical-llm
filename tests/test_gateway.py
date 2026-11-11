import json

import pytest
from fastapi.testclient import TestClient

from scllm.api.engine import Engine, Tokenizer
from scllm.api.gateway import create_app
from scllm.api.security import AuditLog, RateLimiter, issue_token
from scllm.split.partition import split_model
from scllm.split.runtime import SplitPipeline

JWT = "j" * 40
REPORT = "Hasta Adı Soyadı: Ayşe Yılmaz\nT.C. Kimlik No: 10000000146\nHemoglobin  10.2  g/dL  12.0 - 16.0\n"


def _engine(model):
    gw, shards = split_model(model, 3)
    tok = Tokenizer(encode_chat=lambda msgs: [ord(c) % 128 for c in msgs[-1]["content"][:40]],
                    decode=lambda ids: "[AD_1] için hemoglobin düşük.", eos_token_ids=[])
    return Engine(SplitPipeline(gw, shards), tok, max_new_tokens=4)


@pytest.fixture
def client(tiny_model, tmp_path):
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    app = create_app(_engine(tiny_model), JWT, audit, RateLimiter(rate_per_min=0.5, burst=2))
    return TestClient(app), tmp_path / "audit.jsonl"


def test_requires_valid_token(client):
    c, _ = client
    assert c.post("/v1/analyze", json={"report": REPORT}).status_code == 401
    bad = issue_token("x" * 40, "u")
    assert c.post("/v1/analyze", json={"report": REPORT}, headers={"Authorization": f"Bearer {bad}"}).status_code == 401
    ro = issue_token(JWT, "u", role="readonly")
    assert c.post("/v1/analyze", json={"report": REPORT}, headers={"Authorization": f"Bearer {ro}"}).status_code == 403


def test_analyze_unmasks_answer_and_audit_has_no_pii(client):
    c, audit_path = client
    h = {"Authorization": f"Bearer {issue_token(JWT, 'dr-1')}"}
    r = c.post("/v1/analyze", json={"report": REPORT}, headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Ayşe Yılmaz için")
    assert body["labs"][0]["status"] == "düşük" and body["grounding"]["ok"]
    assert body["pii"] == {"AD": 1, "TCKN": 1} and "masked_report" not in body
    log = audit_path.read_text(encoding="utf-8")
    assert "Ayşe" not in log and "10000000146" not in log and json.loads(log)["user"] == "dr-1"


def test_rate_limit(client):
    c, _ = client
    h = {"Authorization": f"Bearer {issue_token(JWT, 'dr-2')}"}
    codes = [c.post("/v1/analyze", json={"report": REPORT}, headers=h).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_weak_secret_refused(tiny_model):
    with pytest.raises(ValueError):
        create_app(_engine(tiny_model), "short")
