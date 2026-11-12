"""Public HTTPS-facing gateway (put it behind TLS termination; nodes are reached over the encrypted channel)."""
from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .engine import Engine, PrivacyRefused
from .security import AuditLog, RateLimiter, digest, verify_token

MAX_REPORT_CHARS = 12_000


class AnalyzeRequest(BaseModel):
    report: str = Field(..., min_length=10, max_length=MAX_REPORT_CHARS)
    include_masked_report: bool = False


def create_app(engine: Engine, jwt_secret: str, audit: Optional[AuditLog] = None,
               limiter: Optional[RateLimiter] = None) -> FastAPI:
    if not jwt_secret or len(jwt_secret) < 32:
        raise ValueError("SCLLM_JWT_SECRET must be set and at least 32 characters")
    audit = audit or AuditLog()
    limiter = limiter or RateLimiter()
    app = FastAPI(title="secure-clinical-llm", version="0.1.0")

    def user(authorization: str = Header(default="")) -> dict:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "missing bearer token")
        try:
            claims = verify_token(jwt_secret, token)
        except Exception:
            raise HTTPException(401, "invalid token")
        if claims.get("role") not in {"clinician", "admin"}:
            raise HTTPException(403, "insufficient role")
        return claims

    @app.get("/health")
    def health():
        return {"status": "ok", "stages": len(engine.pipeline.stages)}

    @app.post("/v1/analyze")
    def analyze(req: AnalyzeRequest, claims: dict = Depends(user)):
        if not limiter.allow(claims["sub"]):
            raise HTTPException(429, "rate limit exceeded")
        rid = uuid.uuid4().hex
        try:
            out = engine.analyze(req.report)
        except PrivacyRefused as e:
            audit.write(request_id=rid, user=claims["sub"], event="privacy_refused", report_sha=digest(req.report),
                        findings=[f["label"] for f in e.report.findings])
            raise HTTPException(422, "residual PII detected after masking; the report was not sent for inference")
        except ValueError as e:
            audit.write(request_id=rid, user=claims["sub"], event="rejected", reason=str(e))
            raise HTTPException(413, str(e))
        audit.write(request_id=rid, user=claims["sub"], event="analyze", report_sha=digest(req.report),
                    pii=out["pii"], n_labs=len(out["labs"]), grounding_ok=out["grounding"]["ok"],
                    timing=out["timing"])
        if not req.include_masked_report:
            out.pop("masked_report")
        return {"request_id": rid, **out}

    return app


def build_from_env() -> FastAPI:  # pragma: no cover - wiring for docker-compose
    import torch
    from transformers import AutoTokenizer

    from ..split.loading import load_gateway
    from ..split.runtime import SplitPipeline
    from ..transport.remote_stage import RemoteStage
    from .engine import Tokenizer

    model_id = os.environ["SCLLM_MODEL"]
    secret = bytes.fromhex(os.environ["SCLLM_CLUSTER_SECRET"])
    dtype = getattr(torch, os.environ.get("SCLLM_DTYPE", "bfloat16"))
    gw = load_gateway(model_id, dtype=dtype, device=os.environ.get("SCLLM_DEVICE", "cpu"))
    stages = [RemoteStage(u, secret) for u in os.environ["SCLLM_NODES"].split(",")]
    ner = gate = None
    if os.environ.get("SCLLM_NER", "1") == "1":
        from ..pii.detectors import HFNer
        from ..pii.gate import PrivacyGate
        ner = HFNer(os.environ.get("SCLLM_NER_MODEL", "akdeniz27/bert-base-turkish-cased-ner"))
        gate = PrivacyGate(ner)
    engine = Engine(SplitPipeline(gw, stages), Tokenizer.from_hf(AutoTokenizer.from_pretrained(model_id)), ner=ner,
                    max_new_tokens=int(os.environ.get("SCLLM_MAX_NEW_TOKENS", 512)), gate=gate,
                    structured=os.environ.get("SCLLM_STRUCTURED", "1") == "1")
    return create_app(engine, os.environ["SCLLM_JWT_SECRET"], AuditLog(os.environ.get("SCLLM_AUDIT_LOG")))
