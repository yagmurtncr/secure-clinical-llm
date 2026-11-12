"""Fine-tune the Turkish NER model on synthetic clinical documents (train families + train names only), on CPU.

    python scripts/train_ner.py --n 2000 --epochs 1 --out models/ner-clinical-tr

Keeps the base model's label set (O, B/I-PER, B/I-ORG, B/I-LOC) so ``HFNer`` loads the result unchanged.
Only PER is supervised; ORG/LOC tokens are trained as O, which is fine because the gateway keeps only names.
Augmentation: whole-document upper/lower casing, because the base cased model misses lower-case names.
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
from scllm.pii.synthetic import generate  # noqa: E402


def tr_upper(s: str) -> str:
    return s.replace("i", "İ").replace("ı", "I").upper()


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def encode(tok, text, spans, max_len):
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=max_len)
    names = [(a, b) for a, b, lab in spans if lab == "AD"]
    labels = []
    for s, e in enc["offset_mapping"]:
        if s == e:
            labels.append(-100)  # special tokens
            continue
        lab = 0
        for a, b in names:
            if s >= a and e <= b:
                lab = 1 if s == a else 2
                break
        labels.append(lab)
    enc.pop("offset_mapping")
    enc["labels"] = labels
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="akdeniz27/bert-base-turkish-cased-ner")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--case-aug", type=float, default=0.15, help="share of docs fully upper- or lower-cased")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/ner-clinical-tr")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    r = random.Random(a.seed)
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForTokenClassification.from_pretrained(a.base)
    docs = generate(a.n, "train", seed=a.seed)
    feats = []
    for d in docs:
        text = d.text
        u = r.random()
        if u < a.case_aug / 2:
            text = tr_lower(text)  # Turkish casing is length-preserving, so gold offsets stay valid
        elif u < a.case_aug:
            text = tr_upper(text)
        assert len(text) == len(d.text)
        feats.append(encode(tok, text, d.spans, a.max_len))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(feats) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))
    model.train()
    log, t0, step = [], time.time(), 0
    for ep in range(a.epochs):
        r.shuffle(feats)
        for i in range(0, len(feats), a.bs):
            batch = tok.pad([{k: f[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in f}
                             for f in feats[i:i + a.bs]], return_tensors="pt")
            L = batch["input_ids"].shape[1]
            labels = torch.tensor([f["labels"] + [-100] * (L - len(f["labels"])) for f in feats[i:i + a.bs]])
            loss = model(**batch, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 20 == 0 or step == steps:
                log.append({"step": step, "loss": round(loss.item(), 4), "elapsed_s": round(time.time() - t0)})
                print(log[-1], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    json.dump({"base": a.base, "n_docs": a.n, "epochs": a.epochs, "lr": a.lr, "batch_size": a.bs,
               "max_len": a.max_len, "case_aug": a.case_aug, "seed": a.seed, "families": "train only",
               "names": "train pool only", "log": log}, open(Path(a.out) / "training.json", "w"), indent=1)


if __name__ == "__main__":
    main()
