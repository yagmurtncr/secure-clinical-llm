import pytest
import torch
from fastapi.testclient import TestClient

from scllm.split.partition import split_model
from scllm.split.runtime import SplitPipeline
from scllm.transport.crypto import ChannelError, GatewayHandshake, node_accept, pack, unpack
from scllm.transport.node_server import create_node_app
from scllm.transport.remote_stage import RemoteStage

SECRET = bytes(range(32))


def _pair(secret_gw=SECRET, secret_node=SECRET):
    hs = GatewayHandshake(secret_gw)
    resp, node = node_accept(secret_node, hs.start())
    return hs.finish(resp), node


def test_round_trip_both_directions():
    gw, node = _pair()
    assert node.open(gw.seal(b"hello")) == b"hello"
    assert gw.open(node.seal(b"world")) == b"world"


def test_tampering_is_detected():
    gw, node = _pair()
    msg = bytearray(gw.seal(b"hidden states"))
    msg[-1] ^= 1
    with pytest.raises(ChannelError):
        node.open(bytes(msg))


def test_replay_is_rejected():
    gw, node = _pair()
    m = gw.seal(b"a")
    node.open(m)
    with pytest.raises(ChannelError):
        node.open(m)


def test_reflection_is_rejected():
    gw, node = _pair()
    with pytest.raises(ChannelError):
        gw.open(gw.seal(b"echo"))


def test_wrong_cluster_secret_cannot_handshake():
    hs = GatewayHandshake(SECRET)
    with pytest.raises(ChannelError):
        node_accept(b"x" * 32, hs.start())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tensor_framing_is_lossless(dtype):
    t = torch.randn(1, 3, 8).to(dtype)
    meta, out = unpack(pack({"k": 1}, t))
    assert meta == {"k": 1} and out.dtype == dtype and torch.equal(out, t)


def test_remote_pipeline_matches_local(tiny_model):
    ids = torch.randint(0, 128, (1, 8))
    gw, shards = split_model(tiny_model, 3)
    local, _ = SplitPipeline(gw, shards).generate(ids, max_new_tokens=10)
    remotes = [RemoteStage("http://node", SECRET, client=TestClient(create_node_app(s, SECRET))) for s in shards]
    remote, stats = SplitPipeline(gw, remotes).generate(ids, max_new_tokens=10)
    assert remote == local
    assert all(not s._caches for s in shards)


def test_node_rejects_unauthenticated_gateway(tiny_model):
    _, shards = split_model(tiny_model, 2)
    client = TestClient(create_node_app(shards[0], SECRET))
    with pytest.raises(Exception):
        RemoteStage("http://node", b"y" * 32, client=client)
    assert client.post("/call/" + "00" * 16, content=b"x" * 40).status_code == 404
