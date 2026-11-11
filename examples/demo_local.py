"""Single-machine demo: the model is split into shards in-process and the full gateway pipeline runs.

    python examples/demo_local.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
    python examples/demo_local.py --model Qwen/Qwen2.5-1.5B-Instruct --dtype bfloat16   # CPU
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scllm.api.engine import Engine, Tokenizer  # noqa: E402
from scllm.split.partition import split_model  # noqa: E402
from scllm.split.runtime import SplitPipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--shards", type=int, default=3)
ap.add_argument("--report", default=str(Path(__file__).with_name("sample_report.txt")))
ap.add_argument("--ner", action="store_true")
a = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
gw, shards = split_model(model, a.shards)
ner = None
if a.ner:
    from scllm.pii.detectors import HFNer
    ner = HFNer()
engine = Engine(SplitPipeline(gw, shards), Tokenizer.from_hf(AutoTokenizer.from_pretrained(a.model)), ner=ner,
                max_new_tokens=400)
out = engine.analyze(Path(a.report).read_text(encoding="utf-8"))
print("=== what the shards' input was built from (masked) ===\n" + out.pop("masked_report"))
print("\n=== answer (unmasked at the gateway) ===\n" + out.pop("answer"))
print("\n=== checks ===\n" + json.dumps(out, ensure_ascii=False, indent=2))
