"""PII masking on held-out template families and unseen names: what actually leaks?

    python eval/pii_heldout.py --n 400 --configs regex,regex+ner,regex+ner_ft,regex+ner_ft+gate \
        --ft-model models/ner-clinical-tr

Two views per configuration:
  * exact-span P / R / F1 for names (AD) — the usual NER metric;
  * privacy view — character-level recall of *all* PII and the share of documents where at least one PII
    character reaches the nodes unmasked ("leaky documents"). With the gate, documents the gate refuses are
    counted separately: they never leave the gateway, so they cannot leak, but they cost availability.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scllm.pii.detectors import HFNer  # noqa: E402
from scllm.pii.gate import PrivacyGate  # noqa: E402
from scllm.pii.masker import mask  # noqa: E402
from scllm.pii.synthetic import generate  # noqa: E402


def evaluate(docs, ner=None, gate=None):
    tp = fp = fn = 0
    pii_chars = leaked_chars = 0
    leaky = refused = 0
    examples = []
    for d in docs:
        m = mask(d.text, ner)
        if gate is not None and not gate.check(m).ok:
            refused += 1
            continue
        pred = {(s.start, s.end) for s in m.spans if s.label == "AD"}
        gold = {(a, b) for a, b, lab in d.spans if lab == "AD"}
        tp, fp, fn = tp + len(pred & gold), fp + len(pred - gold), fn + len(gold - pred)
        covered = set()
        for s in m.spans:
            covered.update(range(s.start, s.end))
        doc_leak = 0
        for a, b, lab in d.spans:
            miss = sum(1 for i in range(a, b) if i not in covered and not d.text[i].isspace())
            pii_chars += sum(1 for i in range(a, b) if not d.text[i].isspace())
            leaked_chars += miss
            doc_leak += miss
            if miss and len(examples) < 12:
                examples.append({"family": d.family, "label": lab, "value": d.text[a:b]})
        leaky += doc_leak > 0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    sent = len(docs) - refused
    return {"name_precision": round(p, 3), "name_recall": round(r, 3),
            "name_f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "pii_char_recall": round(1 - leaked_chars / pii_chars, 4) if pii_chars else None,
            "leaky_docs": leaky, "leaky_doc_rate": round(leaky / sent, 4) if sent else 0.0,
            "refused_docs": refused, "refusal_rate": round(refused / len(docs), 4), "n": len(docs),
            "leak_examples": examples}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--split", default="test", choices=["test", "test_seen_templates"])
    ap.add_argument("--configs", default="regex,regex+ner")
    ap.add_argument("--base-model", default="akdeniz27/bert-base-turkish-cased-ner")
    ap.add_argument("--ft-model", default="models/ner-clinical-tr")
    ap.add_argument("--out", default="results/pii_heldout.json")
    a = ap.parse_args()
    docs = generate(a.n, a.split, a.seed)
    cache = {}

    def ner(kind):
        if kind not in cache:
            cache[kind] = HFNer(a.base_model if kind == "base" else a.ft_model)
        return cache[kind]

    out = {"n": a.n, "seed": a.seed, "split": a.split, "families": sorted({d.family for d in docs}), "results": {}}
    for cfg in a.configs.split(","):
        parts = set(cfg.split("+"))
        backend = ner("ft") if "ner_ft" in parts else (ner("base") if "ner" in parts else None)
        gate = PrivacyGate(ner("ft") if "ner_ft" in parts else ner("base")) if "gate" in parts else None
        res = evaluate(docs, backend, gate)
        res["leaky_doc_rate_ci95"] = wilson(res["leaky_docs"], res["n"] - res["refused_docs"])
        out["results"][cfg] = res
        print(cfg, {k: v for k, v in res.items() if k != "leak_examples"}, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    random.seed(0)
    main()
