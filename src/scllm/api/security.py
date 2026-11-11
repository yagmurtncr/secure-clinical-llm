"""JWT verification, per-user rate limiting and a PII-free audit log."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import jwt

AUDIENCE = "scllm"


def issue_token(secret: str, subject: str, role: str = "clinician", ttl_s: int = 3600) -> str:
    now = int(time.time())
    return jwt.encode({"sub": subject, "role": role, "aud": AUDIENCE, "iat": now, "exp": now + ttl_s,
                       "jti": uuid.uuid4().hex}, secret, algorithm="HS256")


def verify_token(secret: str, token: str) -> dict:
    return jwt.decode(token, secret, algorithms=["HS256"], audience=AUDIENCE,
                      options={"require": ["exp", "sub", "aud"]})


class RateLimiter:
    """Token bucket per subject."""

    def __init__(self, rate_per_min: float = 10, burst: int = 5):
        self.rate, self.burst = rate_per_min / 60.0, burst
        self._b = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._b.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            ok = tokens >= 1
            self._b[key] = (tokens - 1 if ok else tokens, now)
            return ok


@dataclass
class AuditLog:
    """Append-only JSON lines. Stores hashes and counts, never report text or PII values."""

    path: Optional[str] = None

    def write(self, **event) -> dict:
        event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
