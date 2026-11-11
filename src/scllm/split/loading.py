"""Load only the weights a node needs, straight from safetensors.

Each shard node reads the tensors of its own decoder layers and nothing else, so a 3-node deployment of an
8B model needs roughly a third of the GPU memory per node. The gateway reads only embedding, final norm and
LM head. Weights are never unpickled (safetensors only).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Union

import torch
from torch import nn

from .partition import Gateway, LayerShard, ShardSpec


def _resolve(model_id: str, filename: str, revision: Optional[str]) -> str:
    if os.path.isdir(model_id):
        return os.path.join(model_id, filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_id, filename, revision=revision)


def _weight_files(model_id: str, revision: Optional[str]) -> Dict[str, str]:
    """Map tensor name -> safetensors file name."""
    try:
        index = json.load(open(_resolve(model_id, "model.safetensors.index.json", revision), encoding="utf-8"))
        return index["weight_map"]
    except (FileNotFoundError, OSError, ValueError):
        from safetensors import safe_open

        path = _resolve(model_id, "model.safetensors", revision)
        with safe_open(path, framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}


def _load_tensors(model_id: str, prefixes: Iterable[str], revision: Optional[str], dtype) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open

    prefixes = tuple(prefixes)
    wmap = _weight_files(model_id, revision)
    by_file: Dict[str, List[str]] = {}
    for name, fn in wmap.items():
        if name.startswith(prefixes):
            by_file.setdefault(fn, []).append(name)
    out = {}
    for fn, names in by_file.items():
        with safe_open(_resolve(model_id, fn, revision), framework="pt") as f:
            for n in names:
                t = f.get_tensor(n)
                out[n] = t.to(dtype) if t.is_floating_point() else t
    return out


def _meta_model(config, dtype):
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config, dtype=dtype)


def _strip(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_gateway(model_id: str, dtype=torch.float32, device: Union[str, torch.device] = "cpu",
                 revision: Optional[str] = None) -> Gateway:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    meta = _meta_model(config, dtype)
    sd = _load_tensors(model_id, ("model.embed_tokens.", "model.norm.", "lm_head."), revision, dtype)
    gw = Gateway(meta.model.embed_tokens, meta.model.norm, meta.lm_head)
    gw.embed.load_state_dict(_strip(sd, "model.embed_tokens."), assign=True)
    gw.norm.load_state_dict(_strip(sd, "model.norm."), assign=True)
    head = _strip(sd, "lm_head.")
    if not head:  # tied word embeddings
        if not getattr(config, "tie_word_embeddings", False):
            raise KeyError("lm_head weights not found and embeddings are not tied")
        gw.lm_head.weight = gw.embed.weight
    else:
        gw.lm_head.load_state_dict(head, assign=True)
    return gw.to(device).eval()


def load_shard(model_id: str, spec: ShardSpec, dtype=torch.float32, device: Union[str, torch.device] = "cpu",
               revision: Optional[str] = None) -> LayerShard:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    meta = _meta_model(config, dtype)
    layers = nn.ModuleList(meta.model.layers[spec.start:spec.end])
    sd = _load_tensors(model_id, tuple(f"model.layers.{i}." for i in range(spec.start, spec.end)), revision, dtype)
    for j, i in enumerate(range(spec.start, spec.end)):
        layers[j].load_state_dict(_strip(sd, f"model.layers.{i}."), assign=True)
    # rotary inv_freq is a non-persistent buffer: rebuild it on a real device instead of loading it
    with torch.device(device):
        rotary = type(meta.model.rotary_emb)(config=config)
    return LayerShard(layers, rotary, spec, config).to(device).eval()
