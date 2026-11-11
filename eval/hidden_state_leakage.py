"""How much of the input can a curious node read back from the hidden states it receives?

    python eval/hidden_state_leakage.py --model Qwen/Qwen2.5-0.5B-Instruct --train 150 --test 50

For every layer boundary we measure *token recovery accuracy* of two attackers that only see the hidden states:
  * nearest-embedding: cosine nearest neighbour in the (public) input embedding matrix, no training;
  * linear probe: ridge regression from hidden state to input embedding, trained on reports the attacker
    generated itself, then nearest neighbour. This is the realistic attacker: the model weights are public.
Accuracy is reported for all tokens and for tokens inside patient names (on *unmasked* reports, i.e. what
would leak without the gateway's PII masking). Train and test reports use disjoint first/last-name lists, so
name recovery is measured on names the attacker never saw.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pii_benchmark import make_report  # noqa: E402


@torch.inference_mode()
def collect(model, tok, docs, layers):
    """hidden[L] (N_tokens x H), token ids, and a mask of tokens that fall inside a name span."""
    H = {L: [] for L in layers}
    ids_all, name_all = [], []
    for d in docs:
        enc = tok(d.text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=384)
        out = model(input_ids=enc["input_ids"], output_hidden_states=True)
        names = [(a, b) for a, b, lab in d.spans if lab == "AD"]
        offs = enc["offset_mapping"][0].tolist()
        name_all += [any(s < b and e > a for a, b in names) and e > s for s, e in offs]
        ids_all.append(enc["input_ids"][0])
        for L in layers:
            H[L].append(out.hidden_states[L][0].to(torch.float16))  # memory: cast back per layer when used
    return {L: torch.cat(v) for L, v in H.items()}, torch.cat(ids_all), torch.tensor(name_all)


def nearest(vectors, emb_n, chunk=2048):
    v = torch.nn.functional.normalize(vectors, dim=-1)
    return torch.cat([(v[i:i + chunk] @ emb_n.T).argmax(-1) for i in range(0, len(v), chunk)])


def ridge(X, Y, lam=1.0):
    Xb = torch.cat([X, torch.ones(len(X), 1)], 1)
    A = Xb.T @ Xb + lam * torch.eye(Xb.shape[1])
    return torch.linalg.solve(A, Xb.T @ Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--test", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/hidden_state_leakage.json")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    n_layers = model.config.num_hidden_layers
    layers = sorted({0, 1, 2, 4, n_layers // 3, n_layers // 2, 2 * n_layers // 3, n_layers - 1})
    # the attacker trains on reports with one half of the name lists and is tested on names it never saw
    import pii_benchmark as pb
    first, last = list(pb.FIRST), list(pb.LAST)
    r = random.Random(a.seed)
    pb.FIRST, pb.LAST = first[::2], last[::2]
    train_docs = [make_report(r) for _ in range(a.train)]
    pb.FIRST, pb.LAST = first[1::2], last[1::2]
    test_docs = [make_report(r) for _ in range(a.test)]
    pb.FIRST, pb.LAST = first, last
    Htr, ytr, _ = collect(model, tok, train_docs, layers)
    Hte, yte, name_te = collect(model, tok, test_docs, layers)
    emb_n = torch.nn.functional.normalize(model.get_input_embeddings().weight.detach().float(), dim=-1)
    del model

    rows = []
    for L in layers:
        Xtr, Xte = Htr.pop(L).float(), Hte.pop(L).float()
        nn_pred = nearest(Xte, emb_n)
        W = ridge(Xtr, emb_n[ytr])
        lp_pred = nearest(torch.cat([Xte, torch.ones(len(Xte), 1)], 1) @ W, emb_n)
        del Xtr
        row = {"layer": L,
               "nearest_embedding_all": round((nn_pred == yte).float().mean().item(), 3),
               "linear_probe_all": round((lp_pred == yte).float().mean().item(), 3),
               "linear_probe_names": round((lp_pred[name_te] == yte[name_te]).float().mean().item(), 3)}
        rows.append(row)
        print(row, flush=True)
    out = {"model": a.model, "num_layers": n_layers, "train_reports": a.train, "test_reports": a.test,
           "test_tokens": int(len(yte)), "name_tokens": int(name_te.sum()), "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
