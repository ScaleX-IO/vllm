# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk I/O backend for GPU<->NVMe block transfers via pinned staging buffers.

Uses separate IO threads for store and load so that loads (latency-critical)
never block behind stores (background work). Each thread owns its own pinned
staging buffers to avoid contention.
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from typing import NamedTuple

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
    pin_tensor,
    unpin_tensor,
)

logger = init_logger(__name__)

O_DIRECT = getattr(os, "O_DIRECT", 0)
_ALIGNMENT = 4096


class DiskReadSlot(NamedTuple):
    """An already pinned row and its DMA mapping; ownership stays with the caller."""

    views: list[memoryview]
    params: BatchMemcpyParams
    block_id: int


def _alloc_aligned(num_slots: int, bpb: int) -> torch.Tensor:
    """Allocate a staging buffer whose base address is O_DIRECT aligned.

    The CPU allocator only guarantees 64-byte alignment, so over-allocate by
    one alignment unit and return an aligned view. The view keeps the backing
    storage alive.
    """
    nbytes = num_slots * bpb
    raw = torch.zeros(nbytes + _ALIGNMENT, dtype=torch.int8, device="cpu")
    offset = -raw.data_ptr() % _ALIGNMENT
    return raw[offset : offset + nbytes].view(num_slots, bpb)


class DiskBackend:
    """Async disk offload backend with pipelined GPU DMA and interleaved IO.

    Architecture:
    - Separate coordinator threads for store and load (never block each other)
    - Interleaved pipeline: DMA slot N while preadv/pwritev slot N-1
    - O_DIRECT by default; page cache is opt-in via use_page_cache

    Same launch_copy interface as DmaCopyBackend so the worker can swap
    backends without changing calling code.
    """

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._store_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._load_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._store_thread: threading.Thread | None = None
        self._load_thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._fd: int = -1
        self._disk_path: str = ""
        self._total_block_bytes: int = 0
        self._store_buffer_caches: dict[str, torch.Tensor] = {}
        self._load_buffer_caches: dict[str, torch.Tensor] = {}
        self._store_slot_views: list[list[memoryview]] = []
        self._load_slot_views: list[list[memoryview]] = []
        self._per_tensor_bpb: list[int] = []
        self._tensor_names: list[str] = []
        self._read_executor: ThreadPoolExecutor | None = None

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        disk_path: str,
        num_disk_slots: int,
        total_block_bytes: int,
        num_buffer_slots: int = 2,
        use_page_cache: bool = False,
        max_load_slots: int | None = None,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._total_block_bytes = total_block_bytes
        self._num_buffer_slots = num_buffer_slots
        self._tensor_names = list(gpu_caches.keys())
        self._per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in gpu_caches.values()
        ]

        # The stride alignment is an O_DIRECT kernel-DMA requirement; buffered
        # I/O through the page cache has no alignment constraint.
        if not use_page_cache:
            assert total_block_bytes % _ALIGNMENT == 0, (
                f"total_block_bytes={total_block_bytes} not aligned to {_ALIGNMENT}"
            )

        # Separate buffer pools for store and load threads
        self._store_buffer_caches = {}
        self._load_buffer_caches = {}
        for name, gpu_t in gpu_caches.items():
            bpb = gpu_t.stride(0) * gpu_t.element_size()
            store_buf = _alloc_aligned(num_buffer_slots, bpb)
            pin_tensor(store_buf)
            self._store_buffer_caches[name] = store_buf
            load_buf = _alloc_aligned(num_buffer_slots, bpb)
            pin_tensor(load_buf)
            self._load_buffer_caches[name] = load_buf

        # Pre-built iovec views per slot (avoid per-transfer .numpy() calls)
        self._store_slot_views = [
            [
                memoryview(self._store_buffer_caches[name][slot].numpy())
                for name in self._tensor_names
            ]
            for slot in range(num_buffer_slots)
        ]
        self._load_slot_views = [
            [
                memoryview(self._load_buffer_caches[name][slot].numpy())
                for name in self._tensor_names
            ]
            for slot in range(num_buffer_slots)
        ]

        self._store_params = build_params(
            gpu_caches,
            self._store_buffer_caches,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_params(
            self._load_buffer_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )
        if max_load_slots is not None and max_load_slots > num_buffer_slots:
            self._read_executor = ThreadPoolExecutor(
                max_workers=max_load_slots, thread_name_prefix="kv_read"
            )

        os.makedirs(os.path.dirname(disk_path) or ".", exist_ok=True)
        # Slot contents never outlive the process, so unlink then O_EXCL rather
        # than reopening: a pre-existing file would otherwise keep its own
        # (possibly world-readable) mode, and blocks may encode user prompts.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(disk_path)
        # O_DIRECT by default: page cache would consume the very host DRAM this
        # backend exists to conserve, and doubles the copy on the store path.
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if not use_page_cache:
            flags |= O_DIRECT
        self._fd = os.open(disk_path, flags, 0o600)
        self._disk_path = disk_path
        os.ftruncate(self._fd, num_disk_slots * total_block_bytes)

        logger.info(
            "DiskBackend: path=%s, slots=%d, total=%.2f GB, buf=%dx%d bytes"
            " (page_cache=%s)",
            disk_path,
            num_disk_slots,
            (num_disk_slots * total_block_bytes) / (1024**3),
            num_buffer_slots,
            total_block_bytes,
            use_page_cache,
        )

        self._store_thread = threading.Thread(
            target=self._store_loop,
            args=(device, store_stream),
            daemon=True,
        )
        self._load_thread = threading.Thread(
            target=self._load_loop,
            args=(device, load_stream),
            daemon=True,
        )
        self._store_thread.start()
        self._load_thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
        before_read: Callable[[], bool] | None = None,
        before_store: Callable[[], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        extra_load_slots: list[DiskReadSlot] | None = None,
    ) -> None:
        q = self._store_queue if is_store else self._load_queue
        q.put(
            (
                src_blocks,
                dst_blocks,
                event_idx,
                events_list,
                wait_event,
                before_read,
                before_store,
                on_error,
                extra_load_slots,
            )
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._store_queue.put(None)
        self._load_queue.put(None)
        if self._store_thread is not None:
            self._store_thread.join(timeout=10.0)
        if self._load_thread is not None:
            self._load_thread.join(timeout=10.0)
        # Slot contents can encode user prompts, so drop the name now rather
        # than leaving them readable until the next run overwrites the file.
        # Unlinking only removes the directory entry: any thread still holding
        # the fd keeps writing to the (now anonymous) inode, which the kernel
        # frees once the last fd goes away.
        with contextlib.suppress(OSError):
            os.unlink(self._disk_path)
        # Closing under a still-running IO thread would let the fd number be
        # reused by an unrelated open(), turning its next pwritev into a write
        # into that file. Leaking one fd for the remaining process lifetime is
        # the cheaper failure.
        if any(
            t is not None and t.is_alive()
            for t in (self._store_thread, self._load_thread)
        ):
            logger.warning(
                "IO thread still running after shutdown timeout; leaking fd %d",
                self._fd,
            )
            return
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        if self._read_executor is not None:
            self._read_executor.shutdown(wait=True)
        for pool in (self._store_buffer_caches, self._load_buffer_caches):
            for tensor in pool.values():
                unpin_tensor(tensor)
            pool.clear()
        self._store_slot_views.clear()
        self._load_slot_views.clear()

    def _store_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._store_queue.get()
            if item is None:
                return
            self._run_job(item, stream, is_store=True)

    def _run_job(self, item: tuple, stream: torch.cuda.Stream, is_store: bool) -> None:
        (src, dst, idx, events, fence, before_read, before_store, on_error, slots) = (
            item
        )
        try:
            if fence is not None:
                stream.wait_event(fence)
            if is_store:
                if before_store is not None:
                    before_store()
                self._do_store(src, dst, stream)
            else:
                self._do_load(src, dst, stream, before_read, slots)
            event = torch.Event()
            event.record(stream)
            events.append((idx, event))
        except Exception as exc:
            if on_error is None:
                raise
            # A failed read/write can leave earlier DMA using a staging slot.
            # Do not release the parent job's source/destination pins early.
            try:
                stream.synchronize()
            except Exception as drain_error:
                logger.exception("Could not drain failed disk transfer")
                on_error(RuntimeError(f"Could not drain disk transfer: {drain_error}"))
                return
            on_error(exc)

    def _writev_slot(self, buf_slot: int, file_offset: int) -> None:
        written = os.pwritev(self._fd, self._store_slot_views[buf_slot], file_offset)
        if written < self._total_block_bytes:
            raise OSError(
                f"Short write: expected {self._total_block_bytes} bytes, "
                f"wrote {written}"
            )

    def _readv_slot(self, buf_slot: int, file_offset: int) -> None:
        self._read_into(self._load_slot_views[buf_slot], file_offset)

    def _read_into(self, views: list[memoryview], file_offset: int) -> None:
        bytes_read = os.preadv(self._fd, views, file_offset)
        if bytes_read < self._total_block_bytes:
            raise OSError(
                f"Short read: expected {self._total_block_bytes} bytes, "
                f"read {bytes_read}"
            )

    def _load_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._load_queue.get()
            if item is None:
                return
            self._run_job(item, stream, is_store=False)

    def _do_store(
        self,
        gpu_blocks: list[int],
        disk_slots: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """GPU -> buffer (DMA) -> disk (pwritev), interleaved double-buffer."""
        assert self._store_params is not None
        n = self._num_buffer_slots
        # (DMA event, file offset) of the block already staged in each slot.
        pending: list[tuple[torch.Event, int] | None] = [None] * n

        for i, (gpu_blk, disk_slot) in enumerate(zip(gpu_blocks, disk_slots)):
            buf_slot = i % n
            prev = pending[buf_slot]
            if prev is not None:
                prev[0].synchronize()
                self._writev_slot(buf_slot, prev[1])

            copy_blocks([gpu_blk], [buf_slot], self._store_params)
            ev = torch.Event()
            ev.record(stream)
            pending[buf_slot] = (ev, disk_slot * self._total_block_bytes)

        for slot, last in enumerate(pending):
            if last is not None:
                last[0].synchronize()
                self._writev_slot(slot, last[1])

    def _do_load(
        self,
        disk_slots: list[int],
        gpu_blocks: list[int],
        stream: torch.cuda.Stream,
        before_read: Callable[[], bool] | None = None,
        extra_load_slots: list[DiskReadSlot] | None = None,
    ) -> None:
        """Disk (preadv) -> buffer -> GPU (DMA), interleaved double-buffer."""
        assert self._load_params is not None
        if extra_load_slots:
            self._do_load_parallel(
                disk_slots, gpu_blocks, stream, before_read, extra_load_slots
            )
            return
        n = self._num_buffer_slots
        prev_dma_events: list[torch.Event | None] = [None] * n

        for i, (disk_slot, gpu_blk) in enumerate(zip(disk_slots, gpu_blocks)):
            buf_slot = i % n
            prev = prev_dma_events[buf_slot]
            if prev is not None:
                prev.synchronize()

            if before_read is not None:
                before_read()
            self._readv_slot(buf_slot, disk_slot * self._total_block_bytes)

            copy_blocks([buf_slot], [gpu_blk], self._load_params)
            ev = torch.Event()
            ev.record(stream)
            prev_dma_events[buf_slot] = ev

        if before_read is not None:
            while before_read():
                pass
        for ev in prev_dma_events:
            if ev is not None:
                ev.synchronize()

    def _do_load_parallel(
        self,
        disk_blocks: list[int],
        gpu_blocks: list[int],
        stream: torch.cuda.Stream,
        before_read: Callable[[], bool] | None,
        extra_slots: list[DiskReadSlot],
    ) -> None:
        assert self._load_params is not None and self._read_executor is not None
        executor = self._read_executor
        slots = [
            DiskReadSlot(views, self._load_params, i)
            for i, views in enumerate(self._load_slot_views)
        ] + extra_slots
        slots = slots[: len(disk_blocks)]
        pending = []

        def read(index: int):
            if before_read is not None:
                before_read()
            return executor.submit(
                self._read_into,
                slots[index % len(slots)].views,
                disk_blocks[index] * self._total_block_bytes,
            )

        try:
            for i in range(len(slots)):
                pending.append(read(i))
            for i, gpu_block in enumerate(gpu_blocks):
                slot_index = i % len(slots)
                pending[slot_index].result()
                slot = slots[slot_index]
                copy_blocks([slot.block_id], [gpu_block], slot.params)
                if i + len(slots) < len(disk_blocks):
                    # A slot may be reused only after its previous DMA drains.
                    event = torch.Event()
                    event.record(stream)
                    event.synchronize()
                    pending[slot_index] = read(i + len(slots))
            if before_read is not None:
                while before_read():
                    pass
        finally:
            # A failed read must not return leases while sibling reads use them.
            for future in pending:
                future.cancel()
            wait(pending)
            stream.synchronize()
