"""Does fine-tuning on synthetic clinical text break name detection on real Turkish text?

    python eval/ner_real_text.py --n 1000 --models akdeniz27/bert-base-turkish-cased-ner models/ner-clinical-tr

Person-name (PER) exact-span precision / recall on the WikiANN Turkish test split (real Wikipedia sentences, not
written by this project). It checks that the clinical fine-tune generalizes beyond its synthetic generator and
did not forget what the base model knew.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scllm.pii.detectors import HFNer  # noqa: E402

PER_B, PER_I = 1, 2  # WikiANN tags: 0 O, 1 B-PER, 2 I-PER, 3 B-ORG, 4 I-ORG, 5 B-LOC, 6 I-LOC


def load(n):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    df = pd.read_parquet(hf_hub_download("unimelb-nlp/wikiann", "tr/test-00000-of-00001.parquet", repo_type="dataset"))
    out = []
    for toks, tags in zip(df["tokens"][:n], df["ner_tags"][:n]):
        text, offs = "", []
        for t in toks:
            if text:
                text += " "
            offs.append((len(text), len(text) + len(t)))
            text += t
        gold, cur = set(), None
        for (a, b), tag in zip(offs, tags):
            if tag == PER_B:
                if cur:
                    gold.add(cur)
                cur = (a, b)
            elif tag == PER_I and cur:
                cur = (cur[0], b)
            else:
                if cur:
                    gold.add(cur)
                cur = None
        if cur:
            gold.add(cur)
        out.append((text, gold))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--models", nargs="+", default=["akdeniz27/bert-base-turkish-cased-ner", "models/ner-clinical-tr"])
    ap.add_argument("--out", default="results/ner_real_text.json")
    a = ap.parse_args()
    data = load(a.n)
    res = {}
    for m in a.models:
        ner = HFNer(m)
        tp = fp = fn = 0
        for text, gold in data:
            pred = {(s.start, s.end) for s in ner(text)}
            tp, fp, fn = tp + len(pred & gold), fp + len(pred - gold), fn + len(gold - pred)
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        res[m] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3),
                  "gold_names": tp + fn}
        print(m, res[m], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": "unimelb-nlp/wikiann (tr, test)", "sentences": len(data), "results": res},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
