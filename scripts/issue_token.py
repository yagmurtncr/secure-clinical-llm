"""Issue a short-lived API token:  python scripts/issue_token.py --sub dr.yilmaz --ttl 3600"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scllm.api.security import issue_token  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sub", required=True)
ap.add_argument("--role", default="clinician", choices=["clinician", "admin"])
ap.add_argument("--ttl", type=int, default=3600)
a = ap.parse_args()
print(issue_token(os.environ["SCLLM_JWT_SECRET"], a.sub, a.role, a.ttl))
