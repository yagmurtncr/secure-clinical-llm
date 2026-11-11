"""Property-based tests: invariants that must hold for *any* input, not just hand-picked examples."""
import random

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from scllm.pii.detectors import detect, tckn_valid
from scllm.pii.masker import mask, unmask
from scllm.split.partition import plan_partition
from scllm.transport.crypto import GatewayHandshake, node_accept, pack, unpack

SECRET = b"s" * 32
text_chars = st.characters(blacklist_categories=("Cs",), blacklist_characters="[]")


def _tckn(seed: int) -> str:
    r = random.Random(seed)
    d = [r.randint(1, 9)] + [r.randint(0, 9) for _ in range(8)]
    d.append(((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10)
    d.append(sum(d) % 10)
    return "".join(map(str, d))


@given(st.integers(1, 200), st.integers(1, 64))
def test_partition_covers_every_layer_exactly_once(layers, shards):
    if shards > layers:
        return
    specs = plan_partition(layers, shards)
    covered = [i for s in specs for i in range(s.start, s.end)]
    assert covered == list(range(layers))
    sizes = [s.num_layers for s in specs]
    assert max(sizes) - min(sizes) <= 1


@given(st.integers(0, 10**6))
def test_generated_tckn_always_valid_and_single_digit_change_invalid(seed):
    t = _tckn(seed)
    assert tckn_valid(t)
    i = seed % 11
    bad = t[:i] + str((int(t[i]) + 1) % 10) + t[i + 1:]
    assert not tckn_valid(bad)


@settings(max_examples=150, deadline=None)
@given(st.text(text_chars, max_size=200), st.integers(0, 10**6))
def test_mask_then_unmask_is_identity(noise, seed):
    """Whatever surrounds the identifiers, masking is lossless at the gateway and hides the values."""
    tc = _tckn(seed)
    doc = f"{noise}\nHasta Adı Soyadı: Elif Kara\nT.C. Kimlik No: {tc}\nTelefon: 0532 111 22 33\n{noise}"
    r = mask(doc)
    assert unmask(r.text, r.mapping) == doc
    assert tc not in r.text and "Elif Kara" not in r.text


@settings(max_examples=100, deadline=None)
@given(st.text(text_chars, max_size=300))
def test_detector_spans_are_well_formed(text):
    spans = detect(text)
    for s in spans:
        assert 0 <= s.start < s.end <= len(text) and text[s.start:s.end] == s.text
    for a, b in zip(spans, spans[1:]):
        assert a.end <= b.start  # no overlaps


@settings(max_examples=60, deadline=None)
@given(st.lists(st.integers(1, 6), min_size=1, max_size=3), st.sampled_from([torch.float32, torch.bfloat16]),
       st.dictionaries(st.text(min_size=1, max_size=8), st.integers(), max_size=4))
def test_encrypted_frame_round_trip(shape, dtype, meta):
    hs = GatewayHandshake(SECRET)
    resp, node = node_accept(SECRET, hs.start())
    gw = hs.finish(resp)
    t = torch.randn(*shape).to(dtype)
    meta2, out = unpack(node.open(gw.seal(pack(meta, t))))
    assert meta2 == meta and torch.equal(out, t)
