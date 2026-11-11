import pytest
import torch

from scllm.split.loading import load_gateway, load_shard
from scllm.split.partition import plan_partition, split_model


@pytest.mark.parametrize("max_shard_size", [None, "60KB"])  # single file and multi-file (index.json) checkpoints
def test_loaded_parts_match_in_memory_split(tiny_model, tmp_path, max_shard_size):
    kw = {"max_shard_size": max_shard_size} if max_shard_size else {}
    tiny_model.save_pretrained(tmp_path, **kw)
    ids = torch.randint(0, 128, (1, 9))
    ref = tiny_model(ids).logits

    specs = plan_partition(tiny_model.config.num_hidden_layers, 3)
    gw = load_gateway(str(tmp_path))
    shards = [load_shard(str(tmp_path), s) for s in specs]
    h, pos = gw.embed_ids(ids), torch.arange(9).unsqueeze(0)
    for s in shards:
        h = s.forward(h, pos, "x")
    assert torch.allclose(gw.logits(h), ref, atol=1e-5)
    # a shard holds only its own layers
    assert sum(p.numel() for p in shards[0].parameters()) < sum(p.numel() for p in tiny_model.parameters()) / 2
