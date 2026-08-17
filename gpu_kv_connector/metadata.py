from __future__ import annotations

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)


@dataclass(frozen=True)
class GPUKVTransfer:
    object_ids: tuple[bytes, ...]
    block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.object_ids) != len(self.block_ids):
            raise ValueError("object_ids and block_ids must have equal length")


@dataclass
class GPUKVConnectorMetadata(KVConnectorMetadata):
    loads: dict[str, GPUKVTransfer] = field(default_factory=dict)
    stores: dict[str, GPUKVTransfer] = field(default_factory=dict)
