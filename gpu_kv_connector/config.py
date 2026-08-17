from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class GPUKVConfig:
    device_path: str
    catalog_path: str
    disk_start_page: int
    capacity_pages: int
    max_objects: int
    max_batch: int
    queue_depth: int
    num_queues: int
    max_request_pages: int
    max_superrequest_objects: int
    min_superrequest_objects: int
    superrequest_target_bytes: int
    min_superrequest_bytes: int
    read_executor_blocks: int
    write_executor_blocks: int
    prefetch_layers: int
    ready_cache_entries: int
    fuse_kv_planes: bool
    reset_catalog: bool
    tp_size: int

    @property
    def full_rank_mask(self) -> int:
        return (1 << self.tp_size) - 1

    def disk_start_for_rank(self, rank: int) -> int:
        if not 0 <= rank < self.tp_size:
            raise ValueError(f"invalid tensor-parallel rank {rank}")
        if self.tp_size > 1 and self.capacity_pages == 0:
            raise ValueError(
                "capacity_pages must be set for tensor parallelism so ranks "
                "receive disjoint SSD ranges"
            )
        return self.disk_start_page + rank * self.capacity_pages

    @classmethod
    def from_vllm(cls, vllm_config: VllmConfig) -> GPUKVConfig:
        transfer = vllm_config.kv_transfer_config
        if transfer is None:
            raise ValueError("GPUKVConnector requires kv_transfer_config")
        extra: dict[str, Any] = dict(transfer.kv_connector_extra_config or {})

        def integer(name: str, default: int, minimum: int = 0) -> int:
            value = int(extra.get(name, default))
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
            return value

        def boolean(name: str, default: bool) -> bool:
            value = extra.get(name, default)
            if isinstance(value, bool):
                return value
            if isinstance(value, int) and value in (0, 1):
                return bool(value)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    return True
                if normalized in {"0", "false", "no", "off"}:
                    return False
            raise ValueError(f"{name} must be a boolean")

        tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
        if not 1 <= tp_size < 63:
            raise ValueError("GPUKVConnector supports tensor parallel sizes 1..62")
        return cls(
            device_path=str(extra.get("device_path", "/dev/libnvm0")),
            catalog_path=str(
                extra.get("catalog_path", "/tmp/vllm-gpu-kv/catalog.sqlite3")
            ),
            disk_start_page=integer("disk_start_page", 1_048_576),
            capacity_pages=integer("capacity_pages", 0),
            max_objects=integer("max_objects", 1 << 20, 1),
            max_batch=integer("max_batch", 8192, 1),
            queue_depth=integer("queue_depth", 64, 2),
            num_queues=integer("num_queues", 16, 1),
            max_request_pages=integer("max_request_pages", 0),
            max_superrequest_objects=integer("max_superrequest_objects", 512),
            min_superrequest_objects=integer("min_superrequest_objects", 2, 1),
            superrequest_target_bytes=integer(
                "superrequest_target_bytes", 512 * 1024, 4096
            ),
            min_superrequest_bytes=integer("min_superrequest_bytes", 64 * 1024),
            read_executor_blocks=integer("read_executor_blocks", 16, 1),
            write_executor_blocks=integer("write_executor_blocks", 8, 1),
            prefetch_layers=integer("prefetch_layers", 2, 1),
            ready_cache_entries=integer("ready_cache_entries", 65_536),
            fuse_kv_planes=boolean("fuse_kv_planes", True),
            reset_catalog=boolean("reset_catalog", True),
            tp_size=tp_size,
        )
