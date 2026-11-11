"""Partition a decoder-only transformer into a trusted *gateway* part and N untrusted *layer shards*.

Design choice: the gateway keeps the token embedding, the final norm and the LM head. Shards only ever see
intermediate hidden states, never token ids or output logits, so no shard can read the prompt or the answer
directly. (Hidden states can still leak information through inversion attacks, which is why PII is masked
before the gateway embeds the prompt; see docs/THREAT_MODEL.md.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class ShardSpec:
    index: int
    start: int  # first decoder layer (inclusive)
    end: int    # last decoder layer (exclusive)

    @property
    def num_layers(self) -> int:
        return self.end - self.start


def plan_partition(num_layers: int, num_shards: int) -> List[ShardSpec]:
    """Split ``num_layers`` contiguous decoder layers into ``num_shards`` balanced shards."""
    if not 1 <= num_shards <= num_layers:
        raise ValueError(f"need 1 <= num_shards <= num_layers, got {num_shards} shards for {num_layers} layers")
    base, extra = divmod(num_layers, num_shards)
    specs, start = [], 0
    for i in range(num_shards):
        size = base + (1 if i < extra else 0)
        specs.append(ShardSpec(i, start, start + size))
        start += size
    return specs


def _decoder(model: nn.Module) -> nn.Module:
    """The inner decoder (``model.model`` for Llama / Qwen2 / Mistral style models)."""
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "layers"):
        raise TypeError(f"unsupported architecture {type(model).__name__}: expected model.model.layers")
    return inner


class LayerShard(nn.Module):
    """A contiguous slice of decoder layers plus its own rotary embedding and KV caches (one per session)."""

    def __init__(self, layers: nn.ModuleList, rotary_emb: nn.Module, spec: ShardSpec, config):
        super().__init__()
        self.layers = layers
        self.rotary_emb = rotary_emb
        self.spec = spec
        self.config = config
        self._caches = {}

    @classmethod
    def from_model(cls, model: nn.Module, spec: ShardSpec) -> "LayerShard":
        dec = _decoder(model)
        return cls(nn.ModuleList(dec.layers[spec.start:spec.end]), dec.rotary_emb, spec, model.config)

    def reset(self, session: Optional[str] = None) -> None:
        if session is None:
            self._caches.clear()
        else:
            self._caches.pop(session, None)

    @torch.inference_mode()
    def forward(self, hidden: torch.Tensor, position_ids: torch.LongTensor, session: str = "default") -> torch.Tensor:
        from transformers import DynamicCache

        cache = self._caches.get(session)
        if cache is None:
            cache = self._caches[session] = DynamicCache(config=self.config)
        pos_emb = self.rotary_emb(hidden, position_ids)
        for layer in self.layers:
            out = layer(hidden, attention_mask=None, position_ids=position_ids, past_key_values=cache,
                        use_cache=True, position_embeddings=pos_emb)
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden


class Gateway(nn.Module):
    """Trusted part of the model: embedding in, final norm + LM head out."""

    def __init__(self, embed: nn.Module, norm: nn.Module, lm_head: nn.Module):
        super().__init__()
        self.embed = embed
        self.norm = norm
        self.lm_head = lm_head

    @classmethod
    def from_model(cls, model: nn.Module) -> "Gateway":
        dec = _decoder(model)
        return cls(dec.embed_tokens, dec.norm, model.lm_head)

    @torch.inference_mode()
    def embed_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embed(input_ids)

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden))


def split_model(model: nn.Module, num_shards: int) -> Tuple[Gateway, List[LayerShard]]:
    """Split an already-loaded model (useful for tests and single-machine demos)."""
    model.eval()
    n = len(_decoder(model).layers)
    return Gateway.from_model(model), [LayerShard.from_model(model, s) for s in plan_partition(n, num_shards)]
