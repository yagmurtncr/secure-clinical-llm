"""Gateway-side proxy that makes a remote shard look like a local stage."""
from __future__ import annotations

from typing import Optional

import httpx
import torch

from .crypto import GatewayHandshake, pack, unpack


class RemoteStage:
    def __init__(self, base_url: str, cluster_secret: bytes, client: Optional[httpx.Client] = None,
                 timeout: float = 120.0, wire_dtype: torch.dtype = torch.float32):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)
        self.wire_dtype = wire_dtype
        hs = GatewayHandshake(cluster_secret)
        r = self.client.post(f"{self.base_url}/handshake", json=hs.start())
        r.raise_for_status()
        self.channel = hs.finish(r.json())
        self._url = f"{self.base_url}/call/{self.channel.channel_id.hex()}"

    def _call(self, meta, tensor=None):
        r = self.client.post(self._url, content=self.channel.seal(pack(meta, tensor)),
                             headers={"content-type": "application/octet-stream"})
        r.raise_for_status()
        return unpack(self.channel.open(r.content))

    def forward(self, hidden: torch.Tensor, position_ids: torch.LongTensor, session: str) -> torch.Tensor:
        meta, out = self._call({"op": "forward", "session": session, "position_ids": position_ids.tolist()},
                               hidden.to(self.wire_dtype))
        return out.to(hidden.dtype)

    def reset(self, session: Optional[str] = None) -> None:
        self._call({"op": "reset", "session": session})
