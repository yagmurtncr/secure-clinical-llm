"""Cost of splitting + encrypting: crypto micro-benchmark and end-to-end decode throughput.

    python eval/split_overhead.py --model Qwen/Qwen2.5-0.5B-Instruct --new-tokens 64 --out results/overhead.json

End-to-end compares (a) monolithic HF generate, (b) split into N local shards, (c) N shards behind real HTTP
servers on localhost with the encrypted channel. (c) - (b) isolates serialization + encryption + HTTP cost.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scllm.split.partition import split_model  # noqa: E402
from scllm.split.runtime import SplitPipeline  # noqa: E402
from scllm.transport.crypto import GatewayHandshake, node_accept, pack, unpack  # noqa: E402

SECRET = bytes(range(32))


def crypto_bench(hidden_sizes=(896, 3584, 4096), reps=200):
    hs = GatewayHandshake(SECRET)
    resp, node = node_accept(SECRET, hs.start())
    gw = hs.finish(resp)
    out = {}
    for h in hidden_sizes:
        for label, shape in (("decode_step", (1, 1, h)), ("prefill_512", (1, 512, h))):
            t = torch.randn(*shape).to(torch.bfloat16)
            n = reps if label == "decode_step" else max(10, reps // 20)
            times = []
            for _ in range(n):
                s = time.perf_counter()
                unpack(node.open(gw.seal(pack({"op": "forward"}, t))))
                times.append(time.perf_counter() - s)
            out[f"h{h}_{label}"] = {"median_ms": round(statistics.median(times) * 1e3, 4),
                                    "bytes": t.numel() * 2}
    return out


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _serve(app, port):
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return server


def e2e(model_id, n_shards, new_tokens, prompt_len, reps):
    from transformers import AutoModelForCausalLM

    from scllm.transport.node_server import create_node_app
    from scllm.transport.remote_stage import RemoteStage

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    torch.manual_seed(0)
    ids = torch.randint(100, 5000, (1, prompt_len))

    def timed(fn):
        fn()  # warm-up
        vals = []
        for _ in range(reps):
            s = time.perf_counter()
            fn()
            vals.append(time.perf_counter() - s)
        return statistics.median(vals)

    res = {}
    t = timed(lambda: model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False))
    res["monolithic_hf"] = {"s": round(t, 3), "tok_per_s": round(new_tokens / t, 2)}

    gw, shards = split_model(model, n_shards)
    local = SplitPipeline(gw, shards)
    ref, _ = local.generate(ids, new_tokens)
    t = timed(lambda: local.generate(ids, new_tokens))
    res["split_local"] = {"s": round(t, 3), "tok_per_s": round(new_tokens / t, 2)}

    servers, stages = [], []
    for s in shards:
        port = _free_port()
        servers.append(_serve(create_node_app(s, SECRET), port))
        stages.append(RemoteStage(f"http://127.0.0.1:{port}", SECRET))
    remote = SplitPipeline(gw, stages)
    out, _ = remote.generate(ids, new_tokens)
    assert out == ref, "remote pipeline must reproduce the local pipeline exactly"
    t = timed(lambda: remote.generate(ids, new_tokens))
    res["split_remote_encrypted"] = {"s": round(t, 3), "tok_per_s": round(new_tokens / t, 2)}
    for s in servers:
        s.should_exit = True
    base = res["split_local"]["s"]
    res["remote_overhead_ms_per_token"] = round((res["split_remote_encrypted"]["s"] - base) / new_tokens * 1e3, 2)
    res["identical_output"] = True
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--out", default="results/overhead.json")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    result = {"hardware": {"machine": platform.machine(), "processor": platform.processor() or "unknown",
                           "torch_threads": torch.get_num_threads(), "cuda": torch.cuda.is_available()},
              "crypto": crypto_bench()}
    if not a.skip_e2e:
        result["e2e"] = {"model": a.model, "shards": a.shards, "new_tokens": a.new_tokens,
                         "prompt_len": a.prompt_len, **e2e(a.model, a.shards, a.new_tokens, a.prompt_len, a.reps)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
