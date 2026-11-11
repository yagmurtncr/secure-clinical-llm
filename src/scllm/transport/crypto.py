"""Authenticated, encrypted gateway <-> node channel.

Handshake: ephemeral X25519 key agreement, mutually authenticated with an HMAC over the transcript keyed by a
pre-shared cluster secret (a node that does not know the secret cannot complete it, and neither can a
man-in-the-middle). Session keys come from HKDF-SHA256 and are fresh for every channel (forward secrecy).

Messages: AES-256-GCM. The 96-bit nonce is (direction || 64-bit counter), so it is never reused under a key,
and the same bytes are bound as associated data. The receiver only accepts strictly increasing counters, which
rejects replayed or reordered messages.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import struct
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PROTOCOL = b"scllm/1"
TO_NODE, TO_GATEWAY = 0, 1


class ChannelError(Exception):
    """Raised for any authentication, integrity or replay failure."""


def _raw(pub: X25519PublicKey) -> bytes:
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _mac(secret: bytes, label: bytes, *parts: bytes) -> bytes:
    return hmac.new(secret, PROTOCOL + label + b"".join(parts), hashlib.sha256).digest()


def _derive(shared: bytes, secret: bytes, transcript: bytes) -> Tuple[bytes, bytes]:
    okm = HKDF(algorithm=hashes.SHA256(), length=64, salt=hashlib.sha256(secret).digest(),
               info=PROTOCOL + transcript).derive(shared)
    return okm[:32], okm[32:]  # (gateway->node key, node->gateway key)


@dataclass
class Channel:
    channel_id: bytes
    send_key: bytes
    recv_key: bytes
    send_dir: int
    _send_ctr: int = 0
    _recv_ctr: int = -1
    _aead_send: AESGCM = field(init=False, repr=False)
    _aead_recv: AESGCM = field(init=False, repr=False)

    def __post_init__(self):
        self._aead_send, self._aead_recv = AESGCM(self.send_key), AESGCM(self.recv_key)

    def seal(self, plaintext: bytes) -> bytes:
        header = struct.pack(">BQ", self.send_dir, self._send_ctr)
        self._send_ctr += 1
        nonce = b"\x00\x00\x00" + header  # 3 + 1 + 8 = 12 bytes
        return header + self._aead_send.encrypt(nonce, plaintext, self.channel_id + header)

    def open(self, message: bytes) -> bytes:
        if len(message) < 9 + 16:
            raise ChannelError("message too short")
        header, body = message[:9], message[9:]
        direction, ctr = struct.unpack(">BQ", header)
        if direction != 1 - self.send_dir:
            raise ChannelError("wrong direction")
        if ctr <= self._recv_ctr:
            raise ChannelError("replayed or reordered message")
        try:
            pt = self._aead_recv.decrypt(b"\x00\x00\x00" + header, body, self.channel_id + header)
        except Exception as e:  # cryptography raises InvalidTag
            raise ChannelError("authentication failed") from e
        self._recv_ctr = ctr
        return pt


class GatewayHandshake:
    """Client side. ``start()`` -> request dict; ``finish(response)`` -> Channel."""

    def __init__(self, cluster_secret: bytes):
        self.secret = cluster_secret
        self.priv = X25519PrivateKey.generate()
        self.channel_id = os.urandom(16)

    def start(self) -> dict:
        pub = _raw(self.priv.public_key())
        return {"channel_id": self.channel_id.hex(), "pub": pub.hex(),
                "mac": _mac(self.secret, b"hello", self.channel_id, pub).hex()}

    def finish(self, resp: dict) -> Channel:
        node_pub = bytes.fromhex(resp["pub"])
        gw_pub = _raw(self.priv.public_key())
        expect = _mac(self.secret, b"reply", self.channel_id, gw_pub, node_pub)
        if not hmac.compare_digest(expect, bytes.fromhex(resp["mac"])):
            raise ChannelError("node failed to authenticate")
        shared = self.priv.exchange(X25519PublicKey.from_public_bytes(node_pub))
        k_out, k_in = _derive(shared, self.secret, self.channel_id + gw_pub + node_pub)
        return Channel(self.channel_id, k_out, k_in, TO_NODE)


def node_accept(cluster_secret: bytes, req: dict) -> Tuple[dict, Channel]:
    """Server side of the handshake."""
    cid, gw_pub = bytes.fromhex(req["channel_id"]), bytes.fromhex(req["pub"])
    if len(cid) != 16 or len(gw_pub) != 32:
        raise ChannelError("malformed handshake")
    if not hmac.compare_digest(_mac(cluster_secret, b"hello", cid, gw_pub), bytes.fromhex(req["mac"])):
        raise ChannelError("gateway failed to authenticate")
    priv = X25519PrivateKey.generate()
    node_pub = _raw(priv.public_key())
    shared = priv.exchange(X25519PublicKey.from_public_bytes(gw_pub))
    k_in, k_out = _derive(shared, cluster_secret, cid + gw_pub + node_pub)
    resp = {"pub": node_pub.hex(), "mac": _mac(cluster_secret, b"reply", cid, gw_pub, node_pub).hex()}
    return resp, Channel(cid, k_out, k_in, TO_GATEWAY)


# ---- message framing (no pickle anywhere) ---------------------------------------------------------------------

_DTYPES = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16", torch.int64: "int64"}
_NP = {"float32": np.float32, "float16": np.float16, "bfloat16": np.int16, "int64": np.int64}


def pack(meta: dict, tensor: torch.Tensor | None = None) -> bytes:
    if tensor is not None:
        t = tensor.detach().contiguous().cpu()
        name = _DTYPES[t.dtype]
        raw = (t.view(torch.int16) if name == "bfloat16" else t).numpy().tobytes()
        meta = {**meta, "_t": {"dtype": name, "shape": list(t.shape)}}
    else:
        raw = b""
    head = json.dumps(meta, separators=(",", ":")).encode()
    return struct.pack(">I", len(head)) + head + raw


def unpack(data: bytes) -> Tuple[dict, torch.Tensor | None]:
    (n,) = struct.unpack(">I", data[:4])
    meta = json.loads(data[4:4 + n])
    spec = meta.pop("_t", None)
    if spec is None:
        return meta, None
    arr = np.frombuffer(data[4 + n:], dtype=_NP[spec["dtype"]]).reshape(spec["shape"])
    t = torch.from_numpy(arr.copy())
    return meta, (t.view(torch.bfloat16) if spec["dtype"] == "bfloat16" else t)
