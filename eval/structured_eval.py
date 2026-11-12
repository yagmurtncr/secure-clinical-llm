"""Free-text answers vs structured answers (model explains, code renders statuses), same reports, same model.

    python eval/structured_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --n 40 --max-new-tokens 320

Per answer: contradictions and invented numbers the checker finds in what the patient would *see*, coverage of
the abnormal tests (is every out-of-range test mentioned?), and for structured mode the JSON parse rate and the
number of explanation sentences the gateway removed. Paired bootstrap CIs over reports.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pii_benchmark import make_report  # noqa: E402
from scllm.clinical.grounding import check  # noqa: E402
from scllm.clinical.labs import parse_labs  # noqa: E402
from scllm.clinical.prompt import build_messages  # noqa: E402
from scllm.clinical.structured import build_structured_messages, render  # noqa: E402
from scllm.pii.masker import mask  # noqa: E402
from scllm.split.partition import split_model  # noqa: E402
from scllm.split.runtime import SplitPipeline  # noqa: E402


def coverage(answer: str, labs) -> float:
    abn = [l for l in labs if l.status != "normal"]
    if not abn:
        return 1.0
    low = answer.lower()
    return sum(l.name.lower().split(" (")[0] in low for l in abn) / len(abn)


def ci(vals, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return {"mean": round(float(v.mean()), 3), "lo": round(float(np.percentile(b, 2.5)), 3),
            "hi": round(float(np.percentile(b, 97.5)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results/structured_vs_free.json")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scllm.api.engine import Tokenizer

    tok = Tokenizer.from_hf(AutoTokenizer.from_pretrained(a.model))
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).eval()
    for p in model.parameters():  # copy out of the safetensors mmap so page-cache eviction cannot slow decoding
        p.data = p.data.clone()
    pipe = SplitPipeline(*split_model(model, a.shards))
    r = random.Random(a.seed)
    reports = [make_report(r).text for _ in range(a.n)]
    rows, samples = [], []
    for i, rep in enumerate(reports):
        m = mask(rep)
        labs = parse_labs(m.text)
        for cond in ("free_text", "structured"):
            msgs = build_structured_messages(m.text, labs) if cond == "structured" else build_messages(m.text, labs)
            new, st = pipe.generate(torch.tensor([tok.encode_chat(msgs)]), a.max_new_tokens, tok.eos_token_ids)
            raw = tok.decode(new)
            row = {"report": i, "condition": cond, "new_tokens": st.new_tokens, "raw": raw,
                   "truncated": st.new_tokens >= a.max_new_tokens}
            if cond == "structured":
                sa = render(raw, m.text, labs)
                shown = sa.text
                row.update(parsed=sa.parsed, explained=sa.explained, total=sa.total, removed=len(sa.removed_sentences))
            else:
                shown = raw
            g = check(shown, m.text, labs)
            row.update(contradictions=len(g.contradictions), unsupported=len(g.unsupported_numbers),
                       claims=g.checked_statements, coverage=round(coverage(shown, labs), 3))
            rows.append(row)
            if i < 2:
                samples.append({"condition": cond, "raw": raw, "shown": shown})
        print(i, [{k: v for k, v in x.items() if k not in ("report", "raw")} for x in rows[-2:]], flush=True)

    summary = {}
    for cond in ("free_text", "structured"):
        rr = [x for x in rows if x["condition"] == cond]
        s = {"answers_with_contradiction": ci([x["contradictions"] > 0 for x in rr]),
             "answers_with_unsupported_numbers": ci([x["unsupported"] > 0 for x in rr]),
             "abnormal_test_coverage": ci([x["coverage"] for x in rr]),
             "contradictions_total": sum(x["contradictions"] for x in rr),
             "checkable_claims_total": sum(x["claims"] for x in rr),
             "truncated_answers": sum(x["truncated"] for x in rr)}
        if cond == "structured":
            s["json_parse_rate"] = ci([x["parsed"] for x in rr])
            s["tests_with_explanation"] = round(sum(x["explained"] for x in rr) / max(1, sum(x["total"] for x in rr)), 3)
            s["removed_sentences_total"] = sum(x["removed"] for x in rr)
        summary[cond] = s
    out = {"model": a.model, "n_reports": a.n, "seed": a.seed, "max_new_tokens": a.max_new_tokens,
           "shards": a.shards, "dtype": a.dtype, "summary": summary, "rows": rows, "samples": samples,
           "date": time.strftime("%Y-%m-%d")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    with torch.no_grad():
        main()
