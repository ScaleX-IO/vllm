from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

OBJECT_ID_BYTES = 24


def make_namespace(config: dict[str, Any]) -> bytes:
    """Return a stable namespace for model/layout incompatible KV objects."""
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).digest()


def make_object_id(namespace: bytes, block_hash: bytes) -> bytes:
    if not isinstance(block_hash, bytes):
        raise TypeError("vLLM block hashes must be bytes")
    return hashlib.blake2b(namespace + block_hash, digest_size=OBJECT_ID_BYTES).digest()


def make_object_ids(namespace: bytes, block_hashes: Iterable[bytes]) -> list[bytes]:
    return [make_object_id(namespace, block_hash) for block_hash in block_hashes]


def split_object_id(object_id: bytes) -> tuple[int, int, int]:
    """Split a 192-bit ID into the GPU-LSM key and verification tag."""
    if len(object_id) != OBJECT_ID_BYTES:
        raise ValueError(f"object ID must contain {OBJECT_ID_BYTES} bytes")
    return (
        int.from_bytes(object_id[0:8], "little", signed=False),
        int.from_bytes(object_id[8:16], "little", signed=False),
        int.from_bytes(object_id[16:24], "little", signed=False),
    )
