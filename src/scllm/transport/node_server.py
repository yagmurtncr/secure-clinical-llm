"""HTTP service that hosts one layer shard. Every request body after the handshake is an encrypted frame."""
from __future__ import annotations

import os
import threading
import time
from typing import Dict

import torch
from fastapi import FastAPI, HTTPException, Request, Response

from ..split.partition import LayerShard
from .crypto import Channel, ChannelError, node_accept, pack, unpack

MAX_CHANNELS = 64
CHANNEL_TTL_S = 3600


def create_node_app(shard: LayerShard, cluster_secret: bytes) -> FastAPI:
    app = FastAPI(title=f"scllm shard {shard.spec.index}", docs_url=None, redoc_url=None)
    channels: Dict[str, tuple] = {}
    lock = threading.Lock()
    device = next(shard.parameters()).device
    dtype = next(shard.parameters()).dtype

    @app.get("/health")
    def health():
        return {"shard": shard.spec.index, "layers": [shard.spec.start, shard.spec.end]}

    @app.post("/handshake")
    def handshake(req: dict):
        try:
            resp, ch = node_accept(cluster_secret, req)
        except (ChannelError, KeyError, ValueError):
            raise HTTPException(403, "handshake rejected")
        with lock:
            now = time.time()
            for k in [k for k, (_, t) in channels.items() if now - t > CHANNEL_TTL_S]:
                channels.pop(k)
            if len(channels) >= MAX_CHANNELS:
                channels.pop(min(channels, key=lambda k: channels[k][1]))
            channels[req["channel_id"]] = (ch, now)
        return resp

    @app.post("/call/{channel_id}")
    async def call(channel_id: str, request: Request):
        entry = channels.get(channel_id)
        if entry is None:
            raise HTTPException(404, "unknown channel")
        ch: Channel = entry[0]
        body = await request.body()
        with lock:  # one frame at a time per node keeps counters and caches consistent
            try:
                meta, hidden = unpack(ch.open(body))
            except ChannelError:
                raise HTTPException(403, "bad frame")
            op = meta.get("op")
            if op == "forward":
                pos = torch.tensor(meta["position_ids"], device=device)
                out = shard.forward(hidden.to(device=device, dtype=dtype), pos, meta["session"])
                reply = pack({"ok": True}, out)
            elif op == "reset":
                shard.reset(meta.get("session"))
                reply = pack({"ok": True})
            else:
                reply = pack({"ok": False, "error": "unknown op"})
            return Response(ch.seal(reply), media_type="application/octet-stream")

    return app


def main():  # pragma: no cover - entry point for containers
    import uvicorn

    from ..split.loading import load_shard
    from ..split.partition import plan_partition
    from transformers import AutoConfig

    model_id = os.environ["SCLLM_MODEL"]
    index, total = int(os.environ["SCLLM_SHARD_INDEX"]), int(os.environ["SCLLM_NUM_SHARDS"])
    secret = bytes.fromhex(os.environ["SCLLM_CLUSTER_SECRET"])
    dtype = getattr(torch, os.environ.get("SCLLM_DTYPE", "bfloat16"))
    device = os.environ.get("SCLLM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    n = AutoConfig.from_pretrained(model_id).num_hidden_layers
    spec = plan_partition(n, total)[index]
    app = create_node_app(load_shard(model_id, spec, dtype=dtype, device=device), secret)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":  # pragma: no cover
    main()
