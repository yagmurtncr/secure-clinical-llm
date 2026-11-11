"""How reliable is the grounding checker itself? Perturbation test with known ground truth.

    python eval/checker_validation.py --n 400

For synthetic reports we write correct answers from templates, then inject known errors: a flipped
high/low/normal claim, or an invented number. The checker should flag every perturbed answer (recall) and
none of the correct ones (false-positive rate). Templates are split into phrasings the checker's keyword list
covers and paraphrases it was not written for, so the gap between the two is visible.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pii_benchmark import make_report  # noqa: E402
from scllm.clinical.grounding import check  # noqa: E402
from scllm.clinical.labs import parse_labs  # noqa: E402

WORD = {"yüksek": "yüksek", "düşük": "düşük", "normal": "normal"}
COVERED = ["{name} değeriniz {val} {unit} ile {status}.", "{name} {status} çıkmış ({val} {unit}).",
           "Sonuçlarınızda {name} {status} görünüyor."]
PARAPHRASE = {"yüksek": ["{name} referans aralığının üstünde ({val} {unit}).", "{name} beklenenden fazla çıkmış."],
              "düşük": ["{name} referans aralığının altında ({val} {unit}).", "{name} olması gerekenden az."],
              "normal": ["{name} aralık içinde ({val} {unit}).", "{name} değerinizde sorun yok."]}
# written before the checker's lexicon was extended and never used to tune it
HELD_OUT = {"yüksek": ["{name} sınırı aşmış durumda.", "{name} değerinde artış var ({val} {unit})."],
            "düşük": ["{name} yetersiz düzeyde ({val} {unit}).", "{name} değerinde azalma var."],
            "normal": ["{name} olağan düzeyde.", "{name} beklenen aralıkta ({val} {unit})."]}
FLIP = {"yüksek": "düşük", "düşük": "yüksek", "normal": "yüksek"}
FLIP_PARA = {"yüksek": "düşük", "düşük": "yüksek", "normal": "düşük"}


def sentence(lab, status, template):
    return template.format(name=lab.name, val=lab.raw_value, unit=lab.unit or "", status=WORD.get(status, status))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="results/checker_validation.json")
    a = ap.parse_args()
    r = random.Random(a.seed)
    res = {k: {"n": 0, "flagged": 0} for k in
           ("correct_covered", "correct_paraphrase", "correct_held_out", "flip_covered", "flip_paraphrase",
            "flip_held_out", "invented_number")}
    for _ in range(a.n):
        text = make_report(r).text
        labs = parse_labs(text)
        lab = r.choice(labs)
        others = " ".join(sentence(o, o.status, r.choice(COVERED)) for o in labs if o is not lab)
        cases = {
            "correct_covered": sentence(lab, lab.status, r.choice(COVERED)),
            "correct_paraphrase": sentence(lab, lab.status, r.choice(PARAPHRASE[lab.status])),
            "flip_covered": sentence(lab, FLIP[lab.status], r.choice(COVERED)),
            "flip_paraphrase": sentence(lab, lab.status, r.choice(PARAPHRASE[FLIP_PARA[lab.status]])),
            "correct_held_out": sentence(lab, lab.status, r.choice(HELD_OUT[lab.status])),
            "flip_held_out": sentence(lab, lab.status, r.choice(HELD_OUT[FLIP_PARA[lab.status]])),
            "invented_number": sentence(lab, lab.status, COVERED[0]).replace(lab.raw_value,
                                                                            f"{lab.value * 1.37 + 3.1:.1f}", 1),
        }
        for k, s in cases.items():
            g = check(f"{s} {others}", text, labs)
            res[k]["n"] += 1
            res[k]["flagged"] += not g.ok
    summary = {k: round(v["flagged"] / v["n"], 3) for k, v in res.items()}
    out = {"n": a.n, "seed": a.seed, "flag_rate": summary,
           "false_positive_rate_covered": summary["correct_covered"],
           "false_positive_rate_paraphrase": summary["correct_paraphrase"],
           "recall_flip_covered": summary["flip_covered"], "recall_flip_paraphrase": summary["flip_paraphrase"],
           "recall_invented_number": summary["invented_number"],
           "false_positive_rate_held_out": summary["correct_held_out"], "recall_flip_held_out": summary["flip_held_out"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
