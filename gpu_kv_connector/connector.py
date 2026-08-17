from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from gpu_kv_connector.catalog import SQLiteObjectCatalog
from gpu_kv_connector.config import GPUKVConfig
from gpu_kv_connector.hashing import split_object_id
from gpu_kv_connector.lifecycle import StoreCompletionTracker
from gpu_kv_connector.metadata import GPUKVConnectorMetadata, GPUKVTransfer
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _as_signed_64(value: int) -> int:
    return value if value < (1 << 63) else value - (1 << 64)


class GPUKVConnectorScheduler:
    def __init__(self, vllm_config: VllmConfig, config: GPUKVConfig) -> None:
        self.config = config
        self.block_size = int(vllm_config.cache_config.block_size)
        self.catalog = SQLiteObjectCatalog(
            config.catalog_path, reset=config.reset_catalog
        )
        self._requests: dict[str, Request] = {}
        self._request_block_ids: dict[str, list[int]] = {}
        self._next_store_index: dict[str, int] = {}
        self._loads: dict[str, GPUKVTransfer] = {}
        self._pending_store_ids: set[bytes] = set()
        self._store_ids_by_request: dict[str, set[bytes]] = defaultdict(set)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        num_full_blocks = min(
            len(request.block_hashes), request.num_tokens // self.block_size
        )
        full_block_tokens = num_full_blocks * self.block_size
        if full_block_tokens - num_computed_tokens < self.block_size:
            return 0, False
        start = num_computed_tokens // self.block_size
        object_ids = list(request.block_hashes[start:num_full_blocks])
        hits = self.catalog.longest_ready_prefix(object_ids, self.config.full_rank_mask)
        if hits == 0:
            return 0, False
        hit_tokens = (start + hits) * self.block_size - num_computed_tokens
        if hit_tokens < self.block_size:
            return 0, False
        return hit_tokens, True

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        request_id = request.request_id
        self._requests[request_id] = request
        self._request_block_ids[request_id] = []
        if num_external_tokens == 0:
            return

        block_ids = blocks.get_block_ids()[0]
        num_local_blocks = sum(
            block.block_hash is not None for block in blocks.blocks[0]
        )
        num_local_tokens = num_local_blocks * self.block_size
        total_full_tokens = num_local_tokens + num_external_tokens
        if total_full_tokens % self.block_size != 0:
            raise RuntimeError("external KV token count is not block aligned")
        num_external_blocks = num_external_tokens // self.block_size
        if num_external_blocks != len(block_ids) - num_local_blocks:
            raise RuntimeError("external KV allocation does not match GPU blocks")

        start = num_local_blocks
        end = start + num_external_blocks
        object_ids = list(request.block_hashes[start:end])
        self._loads[request_id] = GPUKVTransfer(
            tuple(object_ids), tuple(block_ids[num_local_blocks:])
        )
        self._next_store_index[request_id] = end

    def _build_stores(
        self, scheduler_output: SchedulerOutput
    ) -> dict[str, GPUKVTransfer]:
        stores: dict[str, GPUKVTransfer] = {}
        for request_id, new_block_groups, preempted in yield_req_data(scheduler_output):
            if request_id not in self._requests:
                continue
            if preempted:
                self._request_block_ids[request_id] = []
            if new_block_groups:
                self._request_block_ids[request_id].extend(new_block_groups[0])

            request = self._requests[request_id]
            total_tokens = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request_id]
            )
            end = min(total_tokens // self.block_size, len(request.block_hashes))
            start = self._next_store_index.get(request_id, 0)
            if end <= start:
                continue
            block_hashes = list(request.block_hashes[start:end])
            object_ids = block_hashes
            ready = self.catalog.ready_set(object_ids, self.config.full_rank_mask)
            candidates: list[tuple[bytes, int]] = []
            block_ids = self._request_block_ids[request_id]
            if len(block_ids) < end:
                raise RuntimeError(
                    f"request {request_id} has {len(block_ids)} GPU blocks but "
                    f"needs {end} full blocks"
                )
            for offset, object_id in enumerate(object_ids):
                if object_id in ready or object_id in self._pending_store_ids:
                    continue
                candidates.append((object_id, block_ids[start + offset]))

            self._next_store_index[request_id] = end
            if not candidates:
                continue
            ids, source_blocks = zip(*candidates)
            stores[request_id] = GPUKVTransfer(tuple(ids), tuple(source_blocks))
            self._pending_store_ids.update(ids)
            self._store_ids_by_request[request_id].update(ids)
        return stores

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> GPUKVConnectorMetadata:
        metadata = GPUKVConnectorMetadata(
            loads=self._loads,
            stores=self._build_stores(scheduler_output),
        )
        self._loads = {}
        return metadata

    def update_connector_output(self, connector_output: Any) -> None:
        for request_id in connector_output.finished_sending or ():
            object_ids = self._store_ids_by_request.pop(request_id, set())
            self._pending_store_ids.difference_update(object_ids)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        request_id = request.request_id
        self._requests.pop(request_id, None)
        self._request_block_ids.pop(request_id, None)
        self._next_store_index.pop(request_id, None)
        return request_id in self._store_ids_by_request, None

    def close(self) -> None:
        self.catalog.close()


@dataclass
class _NativeBatch:
    object_ids: tuple[bytes, ...]
    block_ids: tuple[int, ...]
    descriptors: torch.Tensor
    status: torch.Tensor
    base_offsets: torch.Tensor


@dataclass
class _IOOperation:
    request_id: str
    object_ids: tuple[bytes, ...]
    batches: list[_NativeBatch]
    layer_events: dict[int, torch.cuda.Event] = field(default_factory=dict)
    compute_ready: dict[int, torch.cuda.Event] = field(default_factory=dict)
    metadata_ready: torch.cuda.Event | None = None
    submitted: bool = False
    final_event: torch.cuda.Event | None = None


class GPUKVConnectorWorker:
    def __init__(self, config: GPUKVConfig) -> None:
        self.config = config
        self.rank = self._get_tp_rank(config.tp_size)
        self.catalog = SQLiteObjectCatalog(config.catalog_path)
        self._native: Any | None = None
        self._region_id: int | None = None
        self._kv_cache: torch.Tensor | None = None
        self._num_layers = 0
        self._plane_bytes = 0
        self._object_bytes = 0
        self._layer_to_index: dict[str, int] = {}
        self._read_stream: torch.cuda.Stream | None = None
        self._write_stream: torch.cuda.Stream | None = None
        self._active_loads: list[_IOOperation] = []
        self._active_stores: list[_IOOperation] = []
        self._pending_loads: list[_IOOperation] = []
        self._unsubmitted_stores: list[_IOOperation] = []
        self._pending_stores: list[_IOOperation] = []
        self._store_tracker = StoreCompletionTracker()
        self._invalid_block_ids: set[int] = set()

    @staticmethod
    def _get_tp_rank(tp_size: int) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() % tp_size
        return 0

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ) -> None:
        del attn_backend
        if not kv_cache.is_cuda or not kv_cache.is_contiguous():
            raise ValueError("GPUKVConnector requires a contiguous CUDA KV cache")
        if kv_cache.ndim != 6 or kv_cache.shape[2] != 2:
            raise ValueError(
                "cross-layer NHD cache must have shape "
                "[block, layer, 2, token, kv_head, head_dim]"
            )
        num_blocks, num_layers = int(kv_cache.shape[0]), int(kv_cache.shape[1])
        plane_elements = 1
        for dimension in kv_cache.shape[3:]:
            plane_elements *= int(dimension)
        plane_bytes = plane_elements * kv_cache.element_size()
        object_bytes = num_layers * 2 * plane_bytes
        if plane_bytes == 0 or plane_bytes % 4096 != 0:
            raise ValueError("each K/V plane must be a non-zero multiple of 4 KiB")
        if kv_cache.numel() * kv_cache.element_size() != num_blocks * object_bytes:
            raise ValueError("cross-layer KV cache has unexpected padding or strides")
        address = kv_cache.data_ptr()
        region_bytes = kv_cache.numel() * kv_cache.element_size()
        if address % (64 * 1024) != 0 or region_bytes % (64 * 1024) != 0:
            raise ValueError(
                "cross-layer KV allocation address and size must be 64 KiB aligned"
            )

        from gpu_kv_connector.native import ObjectStore

        device = kv_cache.device.index
        if device is None:
            raise ValueError("KV cache does not have a concrete CUDA device")
        self._native = ObjectStore(
            self.config.device_path,
            self.config.disk_start_for_rank(self.rank),
            self.config.capacity_pages,
            num_layers * 2,
            plane_bytes,
            self.config.max_objects,
            self.config.max_batch,
            self.config.queue_depth,
            self.config.num_queues,
            device,
        )
        self._region_id = int(self._native.register_region(kv_cache))
        self._kv_cache = kv_cache
        self._num_layers = num_layers
        self._plane_bytes = plane_bytes
        self._object_bytes = object_bytes
        self._read_stream = torch.cuda.Stream(device=device, priority=-1)
        self._write_stream = torch.cuda.Stream(device=device, priority=0)
        logger.info(
            "Registered GPU-KV cross-layer cache: blocks=%d layers=%d "
            "plane_bytes=%d rank=%d",
            num_blocks,
            num_layers,
            plane_bytes,
            self.rank,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        del kv_caches
        raise RuntimeError("GPUKVConnector requires vLLM cross-layer uniform KV blocks")

    def _require_registered(self) -> None:
        if self._native is None or self._kv_cache is None:
            raise RuntimeError("GPUKVConnector KV cache was not registered")

    def _identity_tensors(
        self, object_ids: tuple[bytes, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._kv_cache is not None
        identities = [split_object_id(object_id) for object_id in object_ids]
        keys = torch.tensor(
            [_as_signed_64(identity[0]) for identity in identities],
            dtype=torch.int64,
            device=self._kv_cache.device,
        )
        tags = torch.tensor(
            [
                [_as_signed_64(value) for value in identity[1:]]
                for identity in identities
            ],
            dtype=torch.int64,
            device=self._kv_cache.device,
        )
        return keys, tags

    def _make_batches(
        self, transfer: GPUKVTransfer, *, reserve: bool
    ) -> tuple[list[_NativeBatch], list[tuple[bytes, int, int]]]:
        assert self._native is not None and self._kv_cache is not None
        batches: list[_NativeBatch] = []
        invalid: list[tuple[bytes, int, int]] = []
        for start in range(0, len(transfer.object_ids), self.config.max_batch):
            object_ids = transfer.object_ids[start : start + self.config.max_batch]
            block_ids = transfer.block_ids[start : start + self.config.max_batch]
            keys, tags = self._identity_tensors(object_ids)
            descriptors, status = (
                self._native.reserve(keys, tags)
                if reserve
                else self._native.resolve(keys, tags)
            )
            host_status = status.cpu().tolist()
            if reserve and any(value != 1 for value in host_status):
                raise RuntimeError(
                    f"GPU-KV reserve returned invalid statuses: {host_status}"
                )
            if not reserve:
                invalid.extend(
                    (object_id, block_id, int(value))
                    for object_id, block_id, value in zip(
                        object_ids, block_ids, host_status
                    )
                    if value != 1
                )
            base_offsets = torch.tensor(
                block_ids, dtype=torch.int64, device=self._kv_cache.device
            ).mul_(self._object_bytes)
            batches.append(
                _NativeBatch(
                    tuple(object_ids),
                    tuple(block_ids),
                    descriptors,
                    status,
                    base_offsets,
                )
            )
        return batches, invalid

    def _make_operation(
        self, request_id: str, transfer: GPUKVTransfer, *, reserve: bool
    ) -> _IOOperation:
        stream = self._write_stream if reserve else self._read_stream
        assert stream is not None
        with torch.cuda.stream(stream):
            batches, invalid = self._make_batches(transfer, reserve=reserve)
            metadata_ready = torch.cuda.Event(blocking=False)
            metadata_ready.record(stream)
        if invalid:
            invalid_ids = [object_id for object_id, _, _ in invalid]
            self.catalog.clear_rank(invalid_ids, self.rank)
            self._invalid_block_ids.update(block_id for _, block_id, _ in invalid)
            logger.error(
                "GPU-KV resolve failed for request %s: %s",
                request_id,
                [(block_id, status) for _, block_id, status in invalid],
            )
        return _IOOperation(
            request_id,
            transfer.object_ids,
            batches,
            metadata_ready=metadata_ready,
        )

    def _initialize_layer_order(self, forward_context: ForwardContext) -> None:
        if self._layer_to_index:
            return
        names = [
            name
            for name, layer in forward_context.no_compile_layers.items()
            if getattr(layer, "kv_cache", None) is not None
        ]
        if len(names) != self._num_layers:
            raise RuntimeError(
                f"vLLM exposed {len(names)} KV layers, expected {self._num_layers}"
            )
        self._layer_to_index = {name: index for index, name in enumerate(names)}

    def _issue_layer(
        self, operation: _IOOperation, layer: int, *, write: bool
    ) -> torch.cuda.Event:
        if layer in operation.layer_events:
            return operation.layer_events[layer]
        stream = self._write_stream if write else self._read_stream
        assert stream is not None and self._native is not None
        assert self._region_id is not None
        with torch.cuda.stream(stream):
            if operation.metadata_ready is not None:
                stream.wait_event(operation.metadata_ready)
            for plane in (layer * 2, layer * 2 + 1):
                for batch in operation.batches:
                    offsets = batch.base_offsets + plane * self._plane_bytes
                    if write:
                        self._native.write_plane(
                            batch.descriptors,
                            batch.status,
                            offsets,
                            plane,
                            self._region_id,
                        )
                    else:
                        self._native.read_plane(
                            batch.descriptors,
                            batch.status,
                            offsets,
                            plane,
                            self._region_id,
                        )
            event = torch.cuda.Event(blocking=False)
            event.record(stream)
        operation.layer_events[layer] = event
        return event

    def _submit_deferred_stores(self, operations: list[_IOOperation]) -> None:
        if not operations:
            return
        assert self._write_stream is not None
        for operation in operations:
            if operation.submitted:
                continue
            with torch.cuda.stream(self._write_stream):
                for layer in range(self._num_layers):
                    ready = operation.compute_ready.get(layer)
                    if ready is None:
                        raise RuntimeError(
                            f"store request {operation.request_id} did not see "
                            f"layer {layer}"
                        )
                    self._write_stream.wait_event(ready)
                    self._issue_layer(operation, layer, write=True)
                operation.final_event = operation.layer_events[self._num_layers - 1]
            operation.submitted = True
            self._pending_stores.append(operation)

    def start_load_kv(self, forward_context: ForwardContext) -> None:
        self._require_registered()
        self._initialize_layer_order(forward_context)
        metadata = self._metadata()

        # Do not mix SSD writes with critical-path reads. Writes prepared by a
        # previous step are launched only when this step has no external load.
        if not metadata.loads and self._unsubmitted_stores:
            deferred, self._unsubmitted_stores = self._unsubmitted_stores, []
            self._submit_deferred_stores(deferred)

        self._active_stores = [
            self._make_operation(request_id, transfer, reserve=True)
            for request_id, transfer in metadata.stores.items()
        ]
        for operation in self._active_stores:
            self._store_tracker.started(operation.request_id)

        self._active_loads = [
            self._make_operation(request_id, transfer, reserve=False)
            for request_id, transfer in metadata.loads.items()
        ]
        self._pending_loads.extend(self._active_loads)
        initial = min(self.config.prefetch_layers, self._num_layers)
        for operation in self._active_loads:
            for layer in range(initial):
                self._issue_layer(operation, layer, write=False)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._active_loads:
            return
        layer = self._layer_to_index[layer_name]
        compute_stream = torch.cuda.current_stream()
        for operation in self._active_loads:
            event = self._issue_layer(operation, layer, write=False)
            compute_stream.wait_event(event)
            end = min(layer + self.config.prefetch_layers + 1, self._num_layers)
            for future in range(layer + 1, end):
                self._issue_layer(operation, future, write=False)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del kv_layer, attn_metadata, kwargs
        if not self._active_stores:
            return
        layer = self._layer_to_index[layer_name]
        compute_stream = torch.cuda.current_stream()
        for operation in self._active_stores:
            if layer in operation.compute_ready:
                continue
            event = torch.cuda.Event(blocking=False)
            event.record(compute_stream)
            operation.compute_ready[layer] = event

    def wait_for_save(self) -> None:
        if not self._active_stores:
            return
        self._unsubmitted_stores.extend(self._active_stores)
        self._active_stores = []

    def _metadata(self) -> GPUKVConnectorMetadata:
        metadata = getattr(self, "_connector_metadata", None)
        if not isinstance(metadata, GPUKVConnectorMetadata):
            raise RuntimeError("GPUKVConnector metadata is not bound")
        return metadata

    def bind_metadata(self, metadata: GPUKVConnectorMetadata) -> None:
        self._connector_metadata = metadata

    def handle_preemptions(self, preempted_request_ids: set[str]) -> None:
        to_submit = [
            operation
            for operation in self._unsubmitted_stores
            if operation.request_id in preempted_request_ids
        ]
        if to_submit:
            self._unsubmitted_stores = [
                operation
                for operation in self._unsubmitted_stores
                if operation.request_id not in preempted_request_ids
            ]
            self._submit_deferred_stores(to_submit)
        for operation in list(self._pending_stores):
            if operation.request_id in preempted_request_ids:
                assert operation.final_event is not None
                operation.final_event.synchronize()
        self._reap_stores()

    def _reap_stores(self) -> None:
        remaining: list[_IOOperation] = []
        for operation in self._pending_stores:
            assert operation.final_event is not None
            if not operation.final_event.query():
                remaining.append(operation)
                continue
            self.catalog.mark_rank_ready(operation.object_ids, self.rank)
            self._store_tracker.completed(operation.request_id)
        self._pending_stores = remaining

    def get_finished(self, finished_request_ids: set[str]) -> tuple[set[str], set[str]]:
        self._store_tracker.mark_requests_finished(finished_request_ids)
        finish_submit = [
            operation
            for operation in self._unsubmitted_stores
            if operation.request_id in finished_request_ids
        ]
        if finish_submit:
            self._unsubmitted_stores = [
                operation
                for operation in self._unsubmitted_stores
                if operation.request_id not in finished_request_ids
            ]
            self._submit_deferred_stores(finish_submit)
        self._reap_stores()

        finished_recving: set[str] = set()
        pending_loads: list[_IOOperation] = []
        for operation in self._pending_loads:
            final = operation.layer_events.get(self._num_layers - 1)
            if final is not None and final.query():
                finished_recving.add(operation.request_id)
            else:
                pending_loads.append(operation)
        self._pending_loads = pending_loads

        finished_sending = self._store_tracker.take_ready()
        return finished_sending, finished_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    def shutdown(self) -> None:
        self._submit_deferred_stores(self._unsubmitted_stores)
        self._unsubmitted_stores = []
        for operation in self._pending_stores:
            assert operation.final_event is not None
            operation.final_event.synchronize()
        if self._read_stream is not None:
            self._read_stream.synchronize()
        if self._write_stream is not None:
            self._write_stream.synchronize()
        self._reap_stores()
        # Tear down the libnvm DMA mapping while its CUDA allocation is still
        # alive. The native store owns the mapping; _kv_cache owns the memory.
        self._native = None
        self._kv_cache = None
        self.catalog.close()


class GPUKVConnector(KVConnectorBase_V1):
    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return True

    @property
    def required_kv_cache_alignment(self) -> int:
        return 64 * 1024

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self.config = GPUKVConfig.from_vllm(vllm_config)
        self.scheduler: GPUKVConnectorScheduler | None = None
        self.worker: GPUKVConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.scheduler = GPUKVConnectorScheduler(vllm_config, self.config)
        elif role == KVConnectorRole.WORKER:
            self.worker = GPUKVConnectorWorker(self.config)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str:
        del vllm_config
        return "NHD"

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        del extra_config
        return True

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ) -> None:
        assert self.worker is not None
        self.worker.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self.worker is not None
        self.worker.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: Any) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self.worker is not None:
            if not isinstance(connector_metadata, GPUKVConnectorMetadata):
                raise TypeError("unexpected GPUKVConnector metadata type")
            self.worker.bind_metadata(connector_metadata)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        del kwargs
        assert self.worker is not None
        self.worker.start_load_kv(forward_context)

    def wait_for_layer_load(self, layer_name: str) -> None:
        assert self.worker is not None
        self.worker.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        assert self.worker is not None
        self.worker.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        assert self.worker is not None
        self.worker.wait_for_save()

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        assert self.worker is not None
        self.worker.handle_preemptions(preempted_req_ids)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.worker is not None
        return self.worker.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.worker is not None
        return self.worker.get_block_ids_with_load_errors()

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.scheduler is not None
        return self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        assert self.scheduler is not None
        self.scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> GPUKVConnectorMetadata:
        assert self.scheduler is not None
        return self.scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: Any) -> None:
        assert self.scheduler is not None
        self.scheduler.update_connector_output(connector_output)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.scheduler is not None
        return self.scheduler.request_finished(request, block_ids)

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()
        if self.scheduler is not None:
            self.scheduler.close()
