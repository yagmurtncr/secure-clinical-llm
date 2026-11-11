"""Ablation: does giving the model deterministic lab flags reduce hallucinated numbers and wrong high/low claims?

    python eval/grounding_eval.py --model Qwen/Qwen2.5-7B-Instruct --n 50 --device cuda --dtype bfloat16

Runs the full gateway pipeline (mask -> prompt -> split inference -> grounding check) on synthetic reports, once
with the computed-flags table in the prompt and once without, and reports:
  * answers_with_unsupported_numbers: share of answers containing a number that is not in the report
  * contradiction_rate: wrong high/low/normal claims / checkable claims
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pii_benchmark import make_report  # noqa: E402
from scllm.clinical.grounding import check  # noqa: E402
from scllm.clinical.labs import parse_labs  # noqa: E402
from scllm.clinical.prompt import build_messages  # noqa: E402
from scllm.pii.masker import mask  # noqa: E402
from scllm.split.partition import split_model  # noqa: E402
from scllm.split.runtime import SplitPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16" if torch.cuda.is_available() else "float32")
    ap.add_argument("--out", default="results/grounding.json")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scllm.api.engine import Tokenizer

    tok = Tokenizer.from_hf(AutoTokenizer.from_pretrained(a.model))
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    gw, shards = split_model(model, a.shards)
    pipe = SplitPipeline(gw, shards)

    r = random.Random(a.seed)
    reports = [make_report(r).text for _ in range(a.n)]
    conds = {"with_flags": True, "without_flags": False}
    agg = {c: {"answers": 0, "with_unsupported": 0, "unsupported_total": 0, "claims": 0, "contradictions": 0,
               "with_contradiction": 0, "tokens": 0, "decode_s": 0.0} for c in conds}
    samples = []
    for i, rep in enumerate(reports):
        m = mask(rep)
        labs = parse_labs(m.text)
        for c, flags in conds.items():
            ids = tok.encode_chat(build_messages(m.text, labs, include_flags=flags))
            new, stats = pipe.generate(torch.tensor([ids], device=a.device), a.max_new_tokens, tok.eos_token_ids)
            ans = tok.decode(new)
            g = check(ans, m.text, labs)
            s = agg[c]
            s["answers"] += 1
            s["with_unsupported"] += bool(g.unsupported_numbers)
            s["unsupported_total"] += len(g.unsupported_numbers)
            s["claims"] += g.checked_statements
            s["contradictions"] += len(g.contradictions)
            s["with_contradiction"] += bool(g.contradictions)
            s["tokens"] += stats.new_tokens
            s["decode_s"] += stats.decode_s
            if i < 3:
                samples.append({"condition": c, "answer": ans, "grounding": g.to_dict()})
        print(f"[{i + 1}/{a.n}] " + "  ".join(
            f"{c}: unsup={agg[c]['with_unsupported']} contra={agg[c]['contradictions']}/{agg[c]['claims']}"
            for c in conds), flush=True)

    summary = {}
    for c, s in agg.items():
        summary[c] = {
            "answers_with_unsupported_numbers": round(s["with_unsupported"] / s["answers"], 3),
            "unsupported_numbers_per_answer": round(s["unsupported_total"] / s["answers"], 3),
            "contradiction_rate": round(s["contradictions"] / s["claims"], 3) if s["claims"] else None,
            "answers_with_contradiction": round(s["with_contradiction"] / s["answers"], 3),
            "checkable_claims": s["claims"],
            "tokens_per_s": round(s["tokens"] / s["decode_s"], 2) if s["decode_s"] else None,
        }
    out = {"model": a.model, "n_reports": a.n, "seed": a.seed, "device": a.device, "dtype": a.dtype,
           "shards": a.shards, "summary": summary, "samples": samples,
           "date": time.strftime("%Y-%m-%d")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
