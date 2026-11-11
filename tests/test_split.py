import pytest
import torch

from scllm.split.partition import plan_partition, split_model
from scllm.split.runtime import SplitPipeline


def test_plan_partition_is_balanced_and_contiguous():
    specs = plan_partition(32, 3)
    assert [(s.start, s.end) for s in specs] == [(0, 11), (11, 22), (22, 32)]
    with pytest.raises(ValueError):
        plan_partition(2, 3)


@pytest.mark.parametrize("shards", [1, 2, 3])
def test_split_forward_matches_monolithic_logits(tiny_model, shards):
    ids = torch.randint(0, 128, (1, 12))
    ref = tiny_model(ids).logits
    gw, stages = split_model(tiny_model, shards)
    h = gw.embed_ids(ids)
    pos = torch.arange(12).unsqueeze(0)
    for s in stages:
        h = s.forward(h, pos, "t")
    assert torch.allclose(gw.logits(h), ref, atol=1e-5)


def _greedy_reference(model, ids, n):
    """Monolithic greedy decoding without any KV cache: the ground truth the split pipeline must reproduce."""
    seq = ids
    for _ in range(n):
        nxt = model(seq).logits[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
    return seq[0, ids.shape[1]:].tolist()


def test_split_greedy_generation_matches_monolithic(tiny_model):
    ids = torch.randint(0, 128, (1, 10))
    ref = _greedy_reference(tiny_model, ids, 15)
    gw, stages = split_model(tiny_model, 3)
    out, stats = SplitPipeline(gw, stages).generate(ids, max_new_tokens=15)
    assert out == ref
    assert stats.new_tokens == 15 and len(stats.stage_s) == 3
    assert all(not s._caches for s in stages)  # caches released after the request
