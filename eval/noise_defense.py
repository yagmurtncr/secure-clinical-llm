"""Can Gaussian noise on the hidden states sent to nodes stop inversion without breaking the model?

    python eval/noise_defense.py --model Qwen/Qwen2.5-0.5B-Instruct --boundary 0 2 --sigmas 0 0.1 0.25 0.5 1 2

For each boundary layer k (the gateway runs layers < k and sends the input of layer k) and noise scale sigma
(relative to the RMS of the clean hidden state), the gateway adds N(0, (sigma*rms)^2) to what it sends.
  * privacy: token recovery by a ridge-regression probe trained on *noisy* states (the attacker knows sigma and
    the public weights) and by a training-free nearest-input-embedding lookup, overall and on unseen
    patient-name tokens;
  * utility: teacher-forced top-1 agreement of the noisy model with the clean model, and the perplexity ratio.
Train and test documents use disjoint template families and disjoint name pools (scllm.pii.synthetic).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hidden_state_leakage import nearest, ridge  # noqa: E402
from scllm.pii.synthetic import generate  # noqa: E402


class Noise:
    """Forward pre-hook on decoder layer k: adds scaled Gaussian noise to its input and records what was sent."""

    def __init__(self, sigma: float, seed: int = 0):
        self.sigma, self.g = sigma, torch.Generator().manual_seed(seed)
        self.sent = None

    def __call__(self, module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        if self.sigma > 0:
            rms = h.pow(2).mean(-1, keepdim=True).sqrt()
            h = h + self.sigma * rms * torch.randn(h.shape, generator=self.g, dtype=h.dtype)
        self.sent = h[0].detach().to(torch.float16)
        if args:
            return (h,) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = h
        return args, kwargs


def run(model, tok, docs, layer, sigma, max_len=256):
    """Returns what the node received, token ids, name mask, and per-document (top-1 ids, summed NLL).
    Full logits are not kept (vocab 152k x tokens would not fit in memory)."""
    hook = Noise(sigma)
    handle = model.model.layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
    H, ids, names, util = [], [], [], []
    try:
        for d in docs:
            enc = tok(d.text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=max_len)
            x = enc["input_ids"]
            lg = model(input_ids=x).logits[0, :-1].float()
            gold = [(a, b) for a, b, lab in d.spans if lab == "AD"]
            names += [any(s < b and e > a for a, b in gold) and e > s for s, e in enc["offset_mapping"][0].tolist()]
            H.append(hook.sent)
            ids.append(x[0])
            util.append((lg.argmax(-1), torch.nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum").item()))
            del lg
    finally:
        handle.remove()
    return torch.cat(H), torch.cat(ids), torch.tensor(names), util


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train", type=int, default=80)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--boundary", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--out", default="results/noise_defense.json")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    emb_n = torch.nn.functional.normalize(model.get_input_embeddings().weight.detach().float(), dim=-1)
    train, test = generate(a.train, "train", seed=5), generate(a.test, "test", seed=6)
    rows = []
    for k in a.boundary:
        clean = None
        for sigma in a.sigmas:
            Htr, ytr, _, _ = run(model, tok, train, k, sigma)
            Hte, yte, nte, util = run(model, tok, test, k, sigma)
            if sigma == 0:
                clean = util
            W = ridge(Htr.float(), emb_n[ytr], lam=10.0)
            pred = nearest(torch.cat([Hte.float(), torch.ones(len(Hte), 1)], 1) @ W, emb_n)
            nn_pred = nearest(Hte.float(), emb_n)  # no training at all: nearest input embedding
            agree, nll_noisy, nll_clean, n_tok = 0, 0.0, 0.0, 0
            for (top_c, nll_c), (top_n, nll_n) in zip(clean, util):
                agree += (top_c == top_n).sum().item()
                nll_clean, nll_noisy, n_tok = nll_clean + nll_c, nll_noisy + nll_n, n_tok + len(top_c)
            row = {"boundary_layer": k, "sigma": sigma,
                   "probe_token_recovery": round((pred == yte).float().mean().item(), 3),
                   "probe_name_token_recovery": round((pred[nte] == yte[nte]).float().mean().item(), 3),
                   "nearest_embedding_recovery": round((nn_pred == yte).float().mean().item(), 3),
                   "nearest_embedding_name_recovery": round((nn_pred[nte] == yte[nte]).float().mean().item(), 3),
                   "top1_agreement_with_clean": round(agree / n_tok, 3),
                   "perplexity_ratio": round(math.exp((nll_noisy - nll_clean) / n_tok), 3)}
            rows.append(row)
            print(row, flush=True)
            del Htr, Hte
    out = {"model": a.model, "train_docs": a.train, "test_docs": a.test, "test_split": "held-out families + names",
           "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
