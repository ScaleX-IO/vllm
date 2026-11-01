# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalPageMapping,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import HybridLoadStoreSpec
from vllm.v1.kv_offload.cpu.host_register import host_register, host_unregister
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.swap_blocks_triton import (
    THRESHOLD_BYTES,
    swap_blocks_batch,
)

logger = init_logger(__name__)


def _clear_pending_device_error() -> None:
    """Clear the error a failed copy left pending for the next kernel launch."""
    if not current_platform.is_cuda_alike():
        return
    try:
        CudaRTLibrary().cudaGetLastError()
    except (AssertionError, AttributeError, OSError):
        logger.warning("Could not clear the pending device error", exc_info=True)


def _select_swap_blocks_fn(
    layer_refs_per_group: list[list[CanonicalKVCacheRef]],
    gpu_to_cpu: bool,
    host_memory_is_pinned: bool = True,
):
    """Resolve the swap_blocks function for a handler at init time."""
    # GPU->CPU is bandwidth-bound; the dedicated copy engine beats Triton.
    # The Triton kernel dereferences CPU pointers on the GPU, which is only
    # valid for pinned host memory.
    if gpu_to_cpu or not host_memory_is_pinned:
        return ops.swap_blocks_batch
    # Fall back to the C++ DMA path on platforms where Triton isn't usable
    # (e.g. ROCm host mappings) or where GPU kernels cannot directly
    # dereference CPU pointers (XPU lacks CUDA's unified virtual address space,
    # so the Triton kernel's tl.load(cpu_ptr) is invalid on XPU).
    if not HAS_TRITON or current_platform.is_xpu() or current_platform.is_rocm():
        return ops.swap_blocks_batch
    page_sizes = [r.page_size_bytes for g in layer_refs_per_group for r in g]
    # Triton wins only on small, 8-byte-aligned payloads.
    if (
        not page_sizes
        or max(page_sizes) >= THRESHOLD_BYTES
        or any(s % 8 for s in page_sizes)
    ):
        return ops.swap_blocks_batch
    chunk = min(triton.next_power_of_2(max(page_sizes)), 8192)
    return functools.partial(swap_blocks_batch, bytes_per_chunk=chunk)


@dataclass
class Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    batch_src: torch.Tensor
    batch_dst: torch.Tensor
    batch_sizes: torch.Tensor
    failed: bool = False


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    blocks_per_chunk: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
):
    """Compute byte pointers for sub-blocks of the given block IDs.

    Each block in block_ids contains blocks_per_chunk sub-blocks.
    The pointer for sub-block j of block b is:
        base_ptr + b * row_stride + j * block_page_size

    where block_page_size = tensor.shape[1] // blocks_per_chunk (gpu page size).

    This handles tensors where row_stride != blocks_per_chunk * block_page_size
    (e.g. non-contiguous CPU tensors).

    Args:
        block_ids: array of block IDs at the tensor's native granularity.
        blocks_per_chunk: number of sub-blocks per block.
        output: pre-allocated pointer array to write pointers into.
        tensor: the source or destination tensor.
        skip_count: sub-blocks to skip in the first block.

    """
    assert skip_count < blocks_per_chunk

    num_sub_blocks = len(output)
    base_ptr = tensor.data_ptr()
    row_stride = tensor.stride(0)

    if blocks_per_chunk == 1:
        # Fast path: 1:1 mapping, no sub-block expansion needed.
        output[:] = base_ptr + block_ids.astype(np.uint64)[:num_sub_blocks] * row_stride
        return

    # Vectorized expansion for blocks_per_chunk > 1.
    assert tensor.shape[1] % blocks_per_chunk == 0
    block_page_size = tensor.shape[1] // blocks_per_chunk
    sub_offsets = np.arange(blocks_per_chunk, dtype=np.uint64) * block_page_size
    # (num_blocks, 1) + (1, blocks_per_chunk) -> (num_blocks, blocks_per_chunk)
    all_ptrs = (
        base_ptr + block_ids.astype(np.uint64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    # Flatten and apply skip_count / truncation
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


class CopyPlan(NamedTuple):
    """Precomputed fragment-copy template for one data ref under the canonical
    CPU layout, unrolled from the ref's mapped runs. Offsets are relative to
    the per-block base pointers on each side."""

    frag_offsets_src: np.ndarray
    frag_offsets_dst: np.ndarray
    frag_sizes: np.ndarray
    total_bytes: int

    @property
    def num_frags(self) -> int:
        return len(self.frag_sizes)


def _build_copy_plan(ref: CanonicalKVCacheRef, gpu_to_cpu: bool) -> CopyPlan:
    """Unroll one data ref's mapped runs into a per-fragment CopyPlan."""
    mapping = ref.mapping
    assert mapping is not None
    local: list[int] = []
    canonical: list[int] = []
    sizes: list[int] = []
    for run in mapping.runs:
        for i in range(run.num_fragments):
            local.append(run.local_offset + i * run.local_stride)
            canonical.append(run.canonical_offset + i * run.canonical_stride)
            sizes.append(run.fragment_size)
    src, dst = (local, canonical) if gpu_to_cpu else (canonical, local)
    return CopyPlan(
        frag_offsets_src=np.asarray(src, dtype=np.uint64),
        frag_offsets_dst=np.asarray(dst, dtype=np.uint64),
        frag_sizes=np.asarray(sizes, dtype=np.int64),
        total_bytes=sum(sizes),
    )


def _canonical_page_ids(
    block_ids: np.ndarray, blocks_per_chunk: int, count: int, skip_count: int
) -> np.ndarray:
    """Global canonical page ids matching compute_sub_block_ptrs' enumeration.
    These identify canonical pages consistently across ranks, so they key
    CanonicalPageMapping.is_writer rotation."""
    if blocks_per_chunk == 1:
        return block_ids[:count]
    flat = (
        block_ids[:, np.newaxis] * blocks_per_chunk + np.arange(blocks_per_chunk)
    ).ravel()
    return flat[skip_count : skip_count + count]


def _canonical_block_sizes(
    layer_refs_per_group: list[list[CanonicalKVCacheRef]], num_tensors: int
) -> list[int]:
    """Canonical CPU bytes per GPU block for each tensor, taken from the refs'
    mappings. Requires every ref to carry a mapping."""
    canonical_bytes_per_block = [0] * num_tensors
    for layer_refs in layer_refs_per_group:
        for ref in layer_refs:
            assert ref.mapping is not None
            canonical_bytes_per_block[ref.tensor_idx] = max(
                canonical_bytes_per_block[ref.tensor_idx],
                ref.mapping.canonical_page_size_bytes,
            )
    assert all(size > 0 for size in canonical_bytes_per_block)
    return canonical_bytes_per_block


# Bound registration size to avoid driver limits on large host allocations.
MAX_HOST_REGISTER_CHUNK_BYTES = 64 * 1024**3
# Amortize per-call overhead with the bytes not registered in each row.
MIN_HOST_REGISTER_SAVED_BYTES = 3 * 1024**2


def pin_mmap_region(
    region: SharedOffloadRegion, worker_slot_only: bool = False
) -> None:
    """Register row-aligned chunks or this worker's slots, rolling back on failure."""
    rank = region.rank
    base_ptr = region._base.data_ptr()
    total_size = region.total_size_bytes
    # Chunks end on block-row boundaries, which are page aligned, so neither the
    # driver's page rounding nor any single block transfer straddles two
    # registrations.
    rows_per_chunk = max(MAX_HOST_REGISTER_CHUNK_BYTES // region._row_stride, 1)
    chunk_size = rows_per_chunk * region._row_stride
    ranges: Iterator[tuple[int, int]] = (
        (offset, min(chunk_size, total_size - offset))
        for offset in range(0, total_size, chunk_size)
    )
    stride, slot, page = region._row_stride, region.cpu_page_size, region.page_size
    min_savings = max(page, MIN_HOST_REGISTER_SAVED_BYTES)
    if worker_slot_only and region.rank is not None and stride - slot >= min_savings:
        # Round each slot out to pages; the worker's next slot is a page away.
        ranges = region.iter_worker_slot_ranges()
    else:
        reason = (
            "whole-region layout requested"
            if not worker_slot_only
            else "no worker rank"
            if region.rank is None
            else f"less than {min_savings} bytes saved per row"
        )
        logger.info(
            "Registering full CPU offload region rank=%s (row=%d, slot=%d bytes): %s",
            rank,
            stride,
            slot,
            reason,
        )

    # Register, drain and roll back through the same helper, so a failed
    # chunk leaves neither a pending error nor a partly pinned region.
    addresses: list[int] = []
    registered_bytes = 0
    for offset, size in ranges:
        address = base_ptr + offset
        if host_register(address, size):
            addresses.append(address)
            registered_bytes += size
            continue
        logger.warning(
            "host_register failed for rank=%d at %.2f of %.2f GB; "
            "the offload region stays pageable",
            rank,
            offset / 1e9,
            total_size / 1e9,
        )
        for registered in reversed(addresses):
            host_unregister(registered)
        return

    region.pinned_addresses.extend(addresses)
    region.is_pinned = True
    logger.debug(
        "Host-registered mmap region rank=%d %.2f GB in %d chunk(s)",
        rank,
        registered_bytes / 1e9,
        len(addresses),
    )


def _new_descriptor_buffers(
    num_copy_ops: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pin = PIN_MEMORY
    # CUDA cache_kernels.cu requires int64; XPU DMA engine requires uint64.
    ptr_dtype = torch.uint64 if current_platform.is_xpu() else torch.int64
    return (
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
    )


class SingleDirectionOffloadingHandler:
    """Handles transfers for a single direction, either CPU->GPU or GPU->CPU.
    Transfers are guaranteed to be executed in order of their submission.
    Each transfer uses a unique CUDA stream, and its stream will start
    executing only after the streams of previous transfers have finished.
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        blocks_per_chunk: int,
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        canonical_layout: bool = False,
        host_memory_is_pinned: bool = True,
    ):
        """Initialize a SingleDirectionOffloadingHandler.

        Args:
            gpu_tensors: list of GPU KV cache tensors.
                Each of shape (num_gpu_blocks, gpu_page_size_bytes) with dtype int8.
            cpu_tensors: list of CPU KV cache tensors.
                Each of shape (num_cpu_chunks, cpu_page_size_bytes) with dtype int8.
                Order should match gpu_tensors.
            blocks_per_chunk: number of blocks transferred per chunk.
            layer_refs_per_group: list of CanonicalKVCacheRef per group.
            gpu_to_cpu: if True, transfer from GPU to CPU; otherwise CPU to GPU.
            canonical_layout: if True, CPU pages use the canonical layout
                described by the refs' mappings.
            host_memory_is_pinned: whether the CPU tensors are pinned, so GPU
                kernels may dereference them directly.

        """
        assert len(gpu_tensors) == len(cpu_tensors)
        assert len(gpu_tensors) > 0

        canonical_bytes_per_block = (
            _canonical_block_sizes(layer_refs_per_group, len(gpu_tensors))
            if canonical_layout
            else None
        )

        # assert input tensors are as expected
        for t_idx, (gpu_tensor, cpu_tensor) in enumerate(zip(gpu_tensors, cpu_tensors)):
            assert gpu_tensor.dtype == torch.int8
            assert gpu_tensor.ndim == 2
            assert gpu_tensor.is_cuda or gpu_tensor.is_xpu
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            _, gpu_page_size = gpu_tensor.shape
            _, cpu_page_size = cpu_tensor.shape
            if canonical_bytes_per_block is not None:
                assert (
                    cpu_page_size == canonical_bytes_per_block[t_idx] * blocks_per_chunk
                )
            else:
                assert cpu_page_size == gpu_page_size * blocks_per_chunk

        self.src_tensors: list[torch.Tensor] = (
            gpu_tensors if gpu_to_cpu else cpu_tensors
        )
        self.dst_tensors: list[torch.Tensor] = (
            cpu_tensors if gpu_to_cpu else gpu_tensors
        )
        self.gpu_to_cpu: bool = gpu_to_cpu
        self.layer_refs_per_group = layer_refs_per_group
        self._swap_blocks_batch = _select_swap_blocks_fn(
            layer_refs_per_group, gpu_to_cpu, host_memory_is_pinned
        )

        # GPU blocks may be smaller
        # cpu_page_size = gpu_page_size * blocks_per_chunk.
        self.src_blocks_per_chunk = 1 if self.gpu_to_cpu else blocks_per_chunk
        self.dst_blocks_per_chunk = blocks_per_chunk if self.gpu_to_cpu else 1

        # Per (group, ref) static copy plans for the canonical layout
        self._canonical_copy_plans: list[list[CopyPlan]] | None = (
            [
                [_build_copy_plan(ref, gpu_to_cpu) for ref in layer_refs]
                for layer_refs in layer_refs_per_group
            ]
            if canonical_layout
            else None
        )
        self._fill_group_ops = (
            self._fill_canonical_ops if canonical_layout else self._fill_direct_ops
        )
        # Reusable per-block base-pointer scratch for the canonical fill,
        # sized to the largest possible group (grown on demand)
        num_scratch_blocks = gpu_tensors[0].shape[0] if canonical_layout else 0
        self._scratch_bases_src = np.empty(num_scratch_blocks, dtype=np.uint64)
        self._scratch_bases_dst = np.empty(num_scratch_blocks, dtype=np.uint64)

        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        # queue of transfers (job_id, stream, event)
        self._transfers: deque[Transfer] = deque()
        # list of CUDA streams available for re-use
        self._stream_pool: list[torch.cuda.Stream] = []
        # list of CUDA events available for re-use
        self._event_pool: list[torch.Event] = []
        # list of pinned descriptor buffer sets available for re-use
        self._buffer_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def _estimate_max_copy_ops(self, group_sizes: Sequence[int]) -> int:
        """Upper bound on the number of copy descriptors for a transfer.

        Exact for the direct layout. The canonical path may fill fewer:
        writer rotation later drops the blocks this rank does not write."""
        num_copy_ops = 0
        for g_idx, (group_size, layer_refs) in enumerate(
            zip(group_sizes, self.layer_refs_per_group)
        ):
            if self._canonical_copy_plans is None:
                num_copy_ops += group_size * len(layer_refs)
            else:
                num_copy_ops += group_size * sum(
                    plan.num_frags for plan in self._canonical_copy_plans[g_idx]
                )
        return num_copy_ops

    def _fill_direct_ops(
        self,
        g_idx: int,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        src_skip_count: int,
        dst_skip_count: int,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        all_sizes: np.ndarray,
        op_idx: int,
    ) -> tuple[int, int]:
        """Fill one group's copy descriptors for the direct (worker-private)
        layout: one whole-page copy per (block, ref).

        Returns (op_idx past the filled descriptors, bytes added)."""
        num_bytes = 0
        for data_ref in self.layer_refs_per_group[g_idx]:
            t_idx = data_ref.tensor_idx
            end_idx = op_idx + group_size

            compute_sub_block_ptrs(
                group_src,
                self.src_blocks_per_chunk,
                all_src[op_idx:end_idx],
                self.src_tensors[t_idx],
                skip_count=src_skip_count,
            )
            compute_sub_block_ptrs(
                group_dst,
                self.dst_blocks_per_chunk,
                all_dst[op_idx:end_idx],
                self.dst_tensors[t_idx],
                skip_count=dst_skip_count,
            )

            all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
            num_bytes += group_size * data_ref.page_size_bytes
            op_idx = end_idx
        return op_idx, num_bytes

    def _fill_canonical_ops(
        self,
        g_idx: int,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        src_skip_count: int,
        dst_skip_count: int,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        all_sizes: np.ndarray,
        op_idx: int,
    ) -> tuple[int, int]:
        """Fill one group's copy descriptors for the canonical layout:
        scatter each block through the ref's precomputed CopyPlan, keeping
        only the blocks this rank writes.

        Returns (op_idx past the filled descriptors, bytes added)."""
        assert self._canonical_copy_plans is not None
        # Zero-copy reinterpretation for pointer arithmetic: uint64 and the
        # buffers' int64 are bit-equivalent for addresses
        all_src_u64 = all_src.view(np.uint64)
        all_dst_u64 = all_dst.view(np.uint64)
        if group_size > len(self._scratch_bases_src):
            self._scratch_bases_src = np.empty(group_size, dtype=np.uint64)
            self._scratch_bases_dst = np.empty(group_size, dtype=np.uint64)

        num_bytes = 0
        for plan, data_ref in zip(
            self._canonical_copy_plans[g_idx], self.layer_refs_per_group[g_idx]
        ):
            if plan.num_frags == 0:
                continue
            t_idx = data_ref.tensor_idx

            # 1. Base byte pointer of every block on each side
            block_bases_src = self._scratch_bases_src[:group_size]
            block_bases_dst = self._scratch_bases_dst[:group_size]
            compute_sub_block_ptrs(
                group_src,
                self.src_blocks_per_chunk,
                block_bases_src,
                self.src_tensors[t_idx],
                skip_count=src_skip_count,
            )
            compute_sub_block_ptrs(
                group_dst,
                self.dst_blocks_per_chunk,
                block_bases_dst,
                self.dst_tensors[t_idx],
                skip_count=dst_skip_count,
            )

            # 2. On store, keep only the blocks this rank is elected to write
            mapping = data_ref.mapping
            assert mapping is not None
            if self.gpu_to_cpu and mapping.num_writers > 1:
                block_bases_src, block_bases_dst = self._filter_writer_blocks(
                    block_bases_src,
                    block_bases_dst,
                    mapping,
                    group_dst,
                    group_size,
                    dst_skip_count,
                )
            num_active_blocks = len(block_bases_src)

            # 3. Expand (block base + fragment offset) into one descriptor
            #    per (block, fragment), writing straight into the descriptor
            #    buffers: reshaping a contiguous 1D slice is a view, so the
            #    broadcasts below allocate nothing
            end_idx = op_idx + num_active_blocks * plan.num_frags
            np.add(
                block_bases_src[:, None],
                plan.frag_offsets_src[None, :],
                out=all_src_u64[op_idx:end_idx].reshape(
                    num_active_blocks, plan.num_frags
                ),
            )
            np.add(
                block_bases_dst[:, None],
                plan.frag_offsets_dst[None, :],
                out=all_dst_u64[op_idx:end_idx].reshape(
                    num_active_blocks, plan.num_frags
                ),
            )
            all_sizes[op_idx:end_idx].reshape(num_active_blocks, plan.num_frags)[:] = (
                plan.frag_sizes
            )
            num_bytes += num_active_blocks * plan.total_bytes
            op_idx = end_idx
        return op_idx, num_bytes

    def _filter_writer_blocks(
        self,
        block_bases_src: np.ndarray,
        block_bases_dst: np.ndarray,
        mapping: CanonicalPageMapping,
        group_dst: np.ndarray,
        group_size: int,
        dst_skip_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep only the blocks this rank writes: replicated ranks take turns
        writing shared canonical pages, keyed by the rank-consistent CPU-side
        canonical page id."""
        cpu_page_ids = _canonical_page_ids(
            group_dst,
            self.dst_blocks_per_chunk,
            group_size,
            dst_skip_count,
        )
        writer_mask = cpu_page_ids % mapping.num_writers == mapping.writer_index
        return block_bases_src[writer_mask], block_bases_dst[writer_mask]

    def transfer_async(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        num_src_blocks = len(src_blocks)
        num_dst_blocks = len(dst_blocks)

        # There are 2 types of transfers:
        # 1. GPU -> CPU
        # 2. CPU -> GPU
        #
        # transfers are also to CPU chunks, EXCEPT MAYBE for the first and last chunk.
        # i.e. the first and last CPU chunks in src_blocks can match against
        # a smaller (byte-wise) set of GPU blocks in dst_blocks.
        # In such cases, we may need to skip some gpu-sized sub-blocks,
        # and start reading/writing from the middle of the first CPU chunk.
        # If we have multiple KV cache groups (when using HMA with hybrid models),
        # we may have a partial first/last CPU chunk per each group.
        # The group_sizes parameter encodes the size of each group of blocks
        # in the GPU dst_blocks.
        # If group_sizes is None, we assume all blocks belong to a single group.
        # The logical_offset parameter maps each group of blocks to its logical
        # offset inside the request, counting in GPU blocks.
        # This allows us to find the correct starting position
        # in the matching first CPU chunk.

        # extract group_sizes from the GPU spec
        gpu_spec = src_spec if self.gpu_to_cpu else dst_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        group_sizes = gpu_spec.group_sizes
        assert len(group_sizes) == len(self.layer_refs_per_group)

        # extract block indices from the GPU spec
        block_indices = gpu_spec.block_indices
        assert len(block_indices) == len(self.layer_refs_per_group)

        num_copy_ops = self._estimate_max_copy_ops(group_sizes)

        # reuse a pooled buffer set, growing it if this transfer needs more room
        batch_src, batch_dst, batch_sizes = (
            self._buffer_pool.pop()
            if self._buffer_pool
            else _new_descriptor_buffers(num_copy_ops)
        )
        if batch_src.numel() < num_copy_ops:
            batch_src, batch_dst, batch_sizes = _new_descriptor_buffers(num_copy_ops)

        src = batch_src[:num_copy_ops]
        dst = batch_dst[:num_copy_ops]
        sizes = batch_sizes[:num_copy_ops]
        all_src = src.numpy()
        all_dst = dst.numpy()
        all_sizes = sizes.numpy()

        src_offset = 0
        dst_offset = 0
        op_idx = 0
        # count total number of bytes copied
        num_transfer_bytes = 0
        for g_idx, (group_size, block_idx) in enumerate(
            zip(group_sizes, block_indices)
        ):
            if group_size == 0:
                continue

            src_logical_blocks_to_skip = block_idx % self.src_blocks_per_chunk
            dst_logical_blocks_to_skip = block_idx % self.dst_blocks_per_chunk
            src_logical_blocks_count = group_size + src_logical_blocks_to_skip
            dst_logical_blocks_count = group_size + dst_logical_blocks_to_skip

            dst_blocks_count = cdiv(dst_logical_blocks_count, self.dst_blocks_per_chunk)
            dst_end_offset = dst_offset + dst_blocks_count
            assert dst_end_offset <= num_dst_blocks

            src_blocks_count = cdiv(src_logical_blocks_count, self.src_blocks_per_chunk)
            src_end_offset = src_offset + src_blocks_count
            assert src_end_offset <= num_src_blocks

            op_idx, group_bytes = self._fill_group_ops(
                g_idx,
                group_src=src_blocks[src_offset:src_end_offset],
                group_dst=dst_blocks[dst_offset:dst_end_offset],
                group_size=group_size,
                src_skip_count=src_logical_blocks_to_skip,
                dst_skip_count=dst_logical_blocks_to_skip,
                all_src=all_src,
                all_dst=all_dst,
                all_sizes=all_sizes,
                op_idx=op_idx,
            )
            num_transfer_bytes += group_bytes

            src_offset = src_end_offset
            dst_offset = dst_end_offset

        assert src_offset == num_src_blocks
        assert dst_offset == num_dst_blocks
        # Writer rotation may skip non-writer blocks, leaving op_idx below
        # the sized upper bound
        assert op_idx <= num_copy_ops
        src = src[:op_idx]
        dst = dst[:op_idx]
        sizes = sizes[:op_idx]

        stream = (
            self._stream_pool.pop() if self._stream_pool else current_platform.Stream()
        )
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

        # Stores must wait for the model to finish writing the KV they read.
        # Loads must wait for pending writes (including zeroing) to their
        # destination blocks: the scheduler leaves a block unzeroed only in
        # the step that allocates it for the load and cannot retract zeroing
        # shipped in an earlier step for a since-reallocated block; with async
        # scheduling nothing else orders that zeroing against this copy.
        stream.wait_stream(current_platform.current_stream())
        if self._transfers:
            last_transfer: Transfer = self._transfers[-1]
            last_event = last_transfer.end_event
            # assure job will start only after the previous one completes
            stream.wait_event(last_event)
        # CPU->GPU reads from host memory, which is never written
        # by a concurrent GPU stream, so CU_MEMCPY_SRC_ACCESS_ORDER_ANY is
        # safe and lets the driver pipeline source reads. GPU->CPU reads
        # from the live GPU KV cache, which the compute stream keeps
        # writing; we must keep STREAM ordering so source reads are gated
        # by the transfer stream's wait_stream(compute) barrier.
        is_src_access_order_any = not self.gpu_to_cpu
        try:
            with current_platform.stream(stream):
                start_event.record(stream)
                if op_idx > 0:
                    self._swap_blocks_batch(
                        src,
                        dst,
                        sizes,
                        is_src_access_order_any=is_src_access_order_any,
                    )
                end_event.record(stream)
        except RuntimeError:
            _clear_pending_device_error()
            logger.exception(
                "KV offload %s transfer failed for job %d",
                "GPU->CPU" if self.gpu_to_cpu else "CPU->GPU",
                job_id,
            )
            end_event.record(stream)
            self._transfer_events[job_id] = end_event
            self._transfers.append(
                Transfer(
                    job_id=job_id,
                    stream=stream,
                    start_event=start_event,
                    end_event=end_event,
                    num_bytes=0,
                    batch_src=batch_src,
                    batch_dst=batch_dst,
                    batch_sizes=batch_sizes,
                    failed=True,
                )
            )
            return True

        self._transfer_events[job_id] = end_event
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_transfer_bytes,
                batch_src=batch_src,
                batch_dst=batch_dst,
                batch_sizes=batch_sizes,
            )
        )

        # success
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers:
            transfer = self._transfers[0]
            try:
                if not transfer.end_event.query():
                    break
                transfer_time = (
                    None
                    if transfer.failed
                    # elapsed_time is in milliseconds
                    else transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
                )
            except RuntimeError:
                logger.exception(
                    "KV offload %s transfer failed for job %d while polling "
                    "for completion",
                    "GPU->CPU" if self.gpu_to_cpu else "CPU->GPU",
                    transfer.job_id,
                )
                self._transfers.popleft()
                self._transfer_events.pop(transfer.job_id, None)
                results.append(TransferResult(job_id=transfer.job_id, success=False))
                continue

            self._transfers.popleft()
            self._transfer_events.pop(transfer.job_id, None)
            if transfer.failed:
                results.append(TransferResult(job_id=transfer.job_id, success=False))
                continue

            results.append(
                TransferResult(
                    job_id=transfer.job_id,
                    success=True,
                    transfer_size=transfer.num_bytes,
                    transfer_time=transfer_time,
                )
            )
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            self._buffer_pool.append(
                (transfer.batch_src, transfer.batch_dst, transfer.batch_sizes)
            )
        return results

    def wait(self, job_ids: set[int]):
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                try:
                    event.synchronize()
                except RuntimeError:
                    logger.exception(
                        "KV offload transfer for job %d failed while waiting "
                        "for it to complete",
                        job_id,
                    )

    def shutdown(self) -> None:
        """Drain this direction and release its transfer-side resources."""
        sync_error: Exception | None = None
        while self._transfers:
            transfer = self._transfers[0]
            try:
                transfer.end_event.synchronize()
            except Exception as e:
                logger.exception(
                    "Failed to synchronize transfer end event; "
                    "skipping %d remaining transfers",
                    len(self._transfers) - 1,
                )
                self._transfers.clear()
                sync_error = e
                break
            self._transfers.popleft()

        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self._buffer_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()
        if sync_error is not None:
            raise sync_error


class CPUOffloadingWorker(OffloadingWorker):
    """OffloadingWorker for CPU offloading.

    Composes two SingleDirectionOffloadingHandler instances (one for each
    direction) and exposes them through the explicit submit_store /
    submit_load API.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_chunks: int,
        mmap_region: SharedOffloadRegion | None = None,
        canonical_layout: bool = False,
    ):
        assert not canonical_layout or mmap_region is not None
        # The caller owns mmap_region until this constructor returns. After a
        # successful construction, the worker is the sole owner and releases
        # it after both transfer directions have stopped.
        self._mmap_region = mmap_region
        pin_memory = PIN_MEMORY
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        if mmap_region is not None and pin_memory:
            pin_mmap_region(mmap_region, worker_slot_only=not canonical_layout)
        host_memory_is_pinned = pin_memory and (
            mmap_region is None or mmap_region.is_pinned
        )

        canonical_bytes_per_block = (
            _canonical_block_sizes(kv_caches.group_data_refs, len(kv_caches.tensors))
            if canonical_layout
            else None
        )

        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for t_idx, kv_cache_tensor in enumerate(kv_caches.tensors):
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * blocks_per_chunk

            if canonical_bytes_per_block is not None:
                assert mmap_region is not None
                cpu_tensor = mmap_region.create_next_canonical_view(
                    canonical_bytes_per_block[t_idx] * blocks_per_chunk
                )
            elif mmap_region is not None:
                cpu_tensor = mmap_region.create_next_worker_view(cpu_page_size_bytes)
            else:
                t0 = time.monotonic()
                cpu_tensor = torch.zeros(
                    (num_cpu_chunks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                logger.debug(
                    "torch.zeros pinned tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_chunks,
                    cpu_page_size_bytes,
                    num_cpu_chunks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)

        make_handler = functools.partial(
            SingleDirectionOffloadingHandler,
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            canonical_layout=canonical_layout,
            host_memory_is_pinned=host_memory_is_pinned,
        )
        self._store_handler = make_handler(gpu_to_cpu=True)
        self._load_handler = make_handler(gpu_to_cpu=False)
        # Own stream, so expedited loads do not queue behind regular loads.
        self._expedited_load_handler = make_handler(gpu_to_cpu=False)

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """Async GPU -> CPU."""
        return self._store_handler.transfer_async(job_id, src_spec, dst_spec)

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
        expedited: bool = False,
    ) -> bool:
        """Async CPU -> GPU."""
        handler = self._expedited_load_handler if expedited else self._load_handler
        return handler.transfer_async(job_id, src_spec, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        return (
            self._store_handler.get_finished()
            + self._load_handler.get_finished()
            + self._expedited_load_handler.get_finished()
        )

    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)
        self._expedited_load_handler.wait(job_ids)

    def shutdown(self) -> None:
        handler_failed = False
        try:
            self._store_handler.shutdown()
        except Exception:
            logger.exception("Failed to shut down store offloading handler")
            handler_failed = True

        for handler in (self._load_handler, self._expedited_load_handler):
            try:
                handler.shutdown()
            except Exception:
                logger.exception("Failed to shut down load offloading handler")
                handler_failed = True

        if self._mmap_region is not None:
            if handler_failed:
                try:
                    torch.accelerator.synchronize()
                except Exception:
                    logger.warning(
                        "Device sync before mmap cleanup failed; "
                        "proceeding with cleanup anyway",
                        exc_info=True,
                    )
            self._mmap_region.cleanup()
            self._mmap_region = None


@dataclass
class _HybridTransfer:
    done: threading.Event
    success: bool = True


class HybridOffloadingWorker(OffloadingWorker):
    """Stream disk blocks through SimpleCPU slots and fill read gaps with hot DMA."""

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        num_cpu_blocks: int,
        num_disk_blocks: int,
        disk_path: str,
        num_buffer_slots: int,
        hot_batch_blocks: int,
        cpu_budget: int,
        block_bytes: int,
    ):
        from vllm.v1.simple_kv_offload.cuda_mem_ops import (
            CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
            CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
            build_params,
            pin_tensor,
        )
        from vllm.v1.simple_kv_offload.disk_backend import DiskBackend, _alloc_aligned

        if len(kv_caches.group_data_refs) != 1:
            raise ValueError("Hybrid KV Backend requires one full-attention group")
        refs = kv_caches.group_data_refs[0]
        if {ref.tensor_idx for ref in refs} != set(range(len(kv_caches.tensors))):
            raise ValueError("Hybrid KV Backend requires worker-local whole pages")
        self.gpu = {
            str(i): t.tensor.view(torch.int8).view(-1, t.page_size_bytes)
            for i, t in enumerate(kv_caches.tensors)
        }
        if not self.gpu or any(
            not t.is_cuda or not t.is_contiguous() or t.shape[1] % 4096
            for t in self.gpu.values()
        ):
            raise ValueError(
                "Hybrid KV Backend requires contiguous, 4096-aligned CUDA pages"
            )
        actual_bytes = sum(t.shape[1] for t in self.gpu.values())
        if actual_bytes != block_bytes:
            raise ValueError(
                "Hybrid KV Backend block size disagrees with worker layout"
            )
        allocation = (num_cpu_blocks + 2 * num_buffer_slots) * actual_bytes
        allocation += (3 if num_cpu_blocks else 2) * 4096 * len(self.gpu)
        if allocation > cpu_budget:
            raise ValueError("Hybrid KV Backend buffers exceed cpu_bytes_to_use")
        logger.info(
            "Hybrid KV Backend: hot=%d blocks, staging=%d slots per direction, "
            "hot batch=%d blocks, host allocation=%d bytes",
            num_cpu_blocks,
            num_buffer_slots,
            hot_batch_blocks,
            allocation,
        )
        device = next(iter(self.gpu.values())).device
        self.load_stream = torch.cuda.Stream(device=device, priority=0)
        self.store_stream = torch.cuda.Stream(device=device, priority=0)
        self.hot_batch_blocks = hot_batch_blocks
        self.hot: dict[str, torch.Tensor] = {}
        self._jobs: dict[int, _HybridTransfer] = {}
        self._events: list[tuple[int, torch.Event]] = []
        self._fatal_error: Exception | None = None
        self._closed = False
        self.disk = DiskBackend()
        try:
            for name, tensor in self.gpu.items():
                buf = (
                    _alloc_aligned(num_cpu_blocks, tensor.shape[1])
                    if num_cpu_blocks
                    else torch.empty((0, tensor.shape[1]), dtype=torch.int8)
                )
                if num_cpu_blocks:
                    pin_tensor(buf)
                self.hot[name] = buf
            self._hot_load_params = build_params(
                self.hot,
                self.gpu,
                self.load_stream,
                src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
            )
            self._hot_store_params = build_params(
                self.gpu,
                self.hot,
                self.store_stream,
                src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
            )
            self.disk.init(
                self.gpu,
                device,
                self.load_stream,
                self.store_stream,
                disk_path,
                num_disk_blocks,
                actual_bytes,
                num_buffer_slots=num_buffer_slots,
                use_page_cache=False,
            )
        except Exception:
            self.shutdown()
            raise

    def _submit(
        self,
        job_id: int,
        spec: HybridLoadStoreSpec,
        gpu_spec: GPULoadStoreSpec,
        is_store: bool,
    ) -> bool:
        from vllm.v1.simple_kv_offload.cuda_mem_ops import copy_blocks

        if self._closed:
            raise RuntimeError("Hybrid KV Backend is shut down")
        if job_id in self._jobs:
            raise ValueError(f"Duplicate transfer ID: {job_id}")
        gpu_ids = gpu_spec.block_ids.tolist()
        if len(gpu_spec.group_sizes) != 1 or not (
            len(spec.cpu_ids) == len(spec.disk_ids) == len(gpu_ids)
        ):
            raise ValueError("Hybrid KV Backend requires one chunk per GPU block")
        hot = [(c, g) for c, g in zip(spec.cpu_ids, gpu_ids) if c >= 0]
        cold = [(d, g) for d, g in zip(spec.disk_ids, gpu_ids) if d >= 0]
        stream = self.store_stream if is_store else self.load_stream
        fence = torch.Event()
        fence.record(torch.cuda.current_stream(stream.device))
        job = _HybridTransfer(threading.Event())
        self._jobs[job_id] = job
        hot_iter = iter(hot)
        previous_hot: torch.Event | None = None

        def hot_batch() -> bool:
            nonlocal previous_hot
            # Bound queued hot DMA, including the tail after the last disk read.
            if previous_hot is not None:
                previous_hot.synchronize()
            batch = []
            for _ in range(self.hot_batch_blocks):
                pair = next(hot_iter, None)
                if pair is None:
                    break
                batch.append(pair)
            if not batch:
                return False
            cpu_ids = [c for c, _ in batch]
            gpu_ids = [g for _, g in batch]
            copy_blocks(
                gpu_ids if is_store else cpu_ids,
                cpu_ids if is_store else gpu_ids,
                self._hot_store_params if is_store else self._hot_load_params,
            )
            previous_hot = torch.Event()
            previous_hot.record(stream)
            return True

        def store_hot() -> None:
            while hot_batch():
                pass

        def failed(exc: Exception) -> None:
            logger.error("Hybrid KV transfer %d failed: %s", job_id, exc)
            if not isinstance(exc, OSError):
                self._fatal_error = exc
            job.success = False
            job.done.set()

        disk_ids = [d for d, _ in cold]
        cold_gpu_ids = [g for _, g in cold]
        self.disk.launch_copy(
            cold_gpu_ids if is_store else disk_ids,
            disk_ids if is_store else cold_gpu_ids,
            is_store,
            job_id,
            self._events,
            fence,
            before_read=None if is_store else hot_batch,
            before_store=store_hot if is_store else None,
            on_error=failed,
        )
        return True

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(dst_spec, HybridLoadStoreSpec)
        return self._submit(job_id, dst_spec, src_spec, True)

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
        expedited: bool = False,
    ) -> bool:
        assert isinstance(src_spec, HybridLoadStoreSpec)
        return self._submit(job_id, src_spec, dst_spec, False)

    def _poll(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(
                "Hybrid KV Backend CUDA transfer failed"
            ) from self._fatal_error
        pending = []
        while self._events:
            job_id, event = self._events.pop()
            if event.query():
                self._jobs[job_id].done.set()
            else:
                pending.append((job_id, event))
        self._events.extend(pending)

    def get_finished(self) -> list[TransferResult]:
        self._poll()
        finished = [
            TransferResult(job_id, job.success)
            for job_id, job in self._jobs.items()
            if job.done.is_set()
        ]
        for result in finished:
            del self._jobs[result.job_id]
        return finished

    def wait(self, job_ids: set[int]) -> None:
        self._poll()
        while any(j in self._jobs and not self._jobs[j].done.is_set() for j in job_ids):
            self._poll()
            time.sleep(0.001)
        self._poll()

    def shutdown(self) -> None:
        if self._closed:
            return
        from vllm.v1.simple_kv_offload.cuda_mem_ops import unpin_tensor

        self._closed = True
        self.wait(set(self._jobs))
        self.disk.shutdown()
        for tensor in self.hot.values():
            if tensor.numel():
                unpin_tensor(tensor)
        self.hot.clear()
