from __future__ import annotations

OBJECT_ID_BYTES = 32


def split_object_id(object_id: bytes) -> tuple[int, int, int, int]:
    """Split vLLM's SHA-256 block hash into an index key and tag."""
    if len(object_id) != OBJECT_ID_BYTES:
        raise ValueError(f"object ID must contain {OBJECT_ID_BYTES} bytes")
    return tuple(
        int.from_bytes(object_id[offset : offset + 8], "little", signed=False)
        for offset in range(0, OBJECT_ID_BYTES, 8)
    )
