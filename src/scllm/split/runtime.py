"""Pipeline runtime: the gateway drives tokens through a chain of (local or remote) stages."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

import torch

from .partition import Gateway


class Stage(Protocol):
    def forward(self, hidden: torch.Tensor, position_ids: torch.LongTensor, session: str) -> torch.Tensor: ...
    def reset(self, session: Optional[str] = None) -> None: ...


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    new_tokens: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    stage_s: List[float] = field(default_factory=list)  # cumulative wall time spent in each stage call

    @property
    def tokens_per_s(self) -> float:
        return self.new_tokens / self.decode_s if self.decode_s else 0.0


class SplitPipeline:
    def __init__(self, gateway: Gateway, stages: Sequence[Stage]):
        self.gateway = gateway
        self.stages = list(stages)

    def _run_stages(self, hidden, position_ids, session, stats):
        for i, stage in enumerate(self.stages):
            t = time.perf_counter()
            hidden = stage.forward(hidden, position_ids, session)
            stats.stage_s[i] += time.perf_counter() - t
        return hidden

    @torch.inference_mode()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int = 128,
                 eos_token_id: Optional[Sequence[int] | int] = None):
        """Greedy decoding with per-stage KV caches. Returns (new token ids, stats)."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise ValueError("batch size 1 only")
        eos = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
        session = uuid.uuid4().hex
        stats = GenerationStats(prompt_tokens=input_ids.shape[1], stage_s=[0.0] * len(self.stages))
        out: List[int] = []
        try:
            t0 = time.perf_counter()
            n = input_ids.shape[1]
            pos = torch.arange(n).unsqueeze(0)
            hidden = self._run_stages(self.gateway.embed_ids(input_ids), pos, session, stats)
            nxt = int(self.gateway.logits(hidden[:, -1:, :]).argmax(-1))
            stats.prefill_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            while True:
                out.append(nxt)
                if nxt in eos or len(out) >= max_new_tokens:
                    break
                pos = torch.tensor([[n + len(out) - 1]])
                hidden = self._run_stages(self.gateway.embed_ids(torch.tensor([[nxt]])), pos, session, stats)
                nxt = int(self.gateway.logits(hidden).argmax(-1))
            stats.decode_s = time.perf_counter() - t1
            stats.new_tokens = len(out)
        finally:
            for s in self.stages:
                s.reset(session)
        return out, stats
