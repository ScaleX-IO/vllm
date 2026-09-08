# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PR 52516 at d61e44b85b: sender-path payload probes for non-uniform KV pages.

The actual connector registration and send path runs on host memory. The
Mooncake transport is replaced by bounds-checked ``memmove`` copies, so no
GPU, RDMA device, or model kernel is required. Base controls execute the
exact planner, validator, and ``send_kv_to_decode`` from the PR's base commit.
"""

import ast
import asyncio
import ctypes
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType, SimpleNamespace

# isort: off
# Imports below must follow the platform override.
import msgspec  # noqa: E402
import torch  # noqa: E402
import vllm.platforms as platforms  # noqa: E402
from vllm.platforms.cpu import CpuPlatform  # noqa: E402

platforms._current_platform = CpuPlatform()
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (  # noqa: E402
    mooncake_connector as mc,
)
from vllm.model_executor.layers.attention.attention import (  # noqa: E402
    Attention,
    AttentionType,
)
from vllm.platforms.interface import Platform  # noqa: E402
from vllm.v1.attention.backends.flashinfer import FlashInferBackend  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    create_kv_cache_views,
    get_kv_quant_mode,
)
# isort: on

HEAD_REF = "d61e44b85b04cb9b3c90938e57a1ba72a98cab5d"
BASE_REF = "5db652225f00b55783823ae6606d36925e3e3efe"
CONNECTOR_PATH = (
    "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
)
ATTN = "model.layers.0.self_attn"
HKV = 4
SENTINEL = 253


def base_namespace():
    """Base-commit planner, validator, and sender methods bound to the head module."""
    source = subprocess.run(
        ["git", "show", f"{BASE_REF}:{CONNECTOR_PATH}"],
        cwd=os.environ["VLLM_REPO"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tree = ast.parse(source)
    wanted = {"_compute_sender_transfer_plan", "_validate_asymmetric_region_lengths"}
    nodes = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted
    ]
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MooncakeConnectorWorker"
    )
    nodes += [
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name in {"send_kv_to_decode", "_get_sender_transfer_plan"}
    ]
    assert len(nodes) == 4
    ns = dict(mc.__dict__)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<base>", "exec"), ns)
    return ns


class MemoryEngine:
    """Stands in for TransferEngine; copies must stay inside registered memory."""

    def __init__(self):
        self.registrations = []
        self.copies = []

    def batch_register_memory(self, ptrs, lengths):
        self.registrations.extend(zip(ptrs, lengths))
        return 0

    def batch_transfer_sync_write(self, session, sources, targets, sizes):
        for src, dst, size in zip(sources, targets, sizes):
            for ptr in (src, dst):
                assert any(
                    lo <= ptr and ptr + size <= lo + n for lo, n in self.registrations
                ), "copy outside registered memory"
            self.copies.append((src, dst, size))
            ctypes.memmove(dst, src, size)
        return 0


class CaptureSocket:
    def __init__(self):
        self.responses = []

    async def send_multipart(self, parts):
        self.responses.append(msgspec.msgpack.decode(parts[1]))


def spec_for(kind, tp):
    heads = max(HKV // tp, 1)
    if kind == "padded":
        # A BF16 sliding-window layer excluded from NVFP4 quantization shares
        # the NVFP4 full-attention block pool, so its page gets tail padding.
        # Produced by the normal alignment and spec-construction code paths.
        model = SimpleNamespace(
            dtype=torch.bfloat16,
            get_num_kv_heads=lambda pc: heads,
            get_num_attention_heads=lambda pc: 32 // tp,
            get_head_size=lambda: 128,
        )
        cache = SimpleNamespace(
            cache_dtype="nvfp4",
            block_size=16,
            kv_cache_dtype_skip_layers=["sliding_window"],
            mamba_page_size_padded=None,
            skip_page_size_padded=None,
        )
        config = SimpleNamespace(
            model_config=model, cache_config=cache, parallel_config=SimpleNamespace()
        )
        Platform._align_heterogeneous_kv_block_size(config, FlashInferBackend)
        layer = SimpleNamespace(
            attn_type=AttentionType.DECODER,
            kv_cache_dtype="auto",
            sliding_window=4096,
            attn_backend=FlashInferBackend,
            num_kv_heads=heads,
            head_size=128,
            head_size_v=128,
            kv_cache_torch_dtype=torch.bfloat16,
        )
        spec = Attention.get_kv_cache_spec(layer, config)
        assert spec.page_size_bytes > spec.unpadded_page_size_bytes
        return spec
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=heads,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=get_kv_quant_mode("nvfp4" if kind == "nvfp4" else "auto"),
    )
    spec = FlashInferBackend.customize_spec(spec)
    if kind == "nvfp4":
        assert spec.num_head_slots == 2 * heads
    return spec


def cache_view(spec):
    """Two-block LBHNC cache; block 0 is a guard, block 1 carries the payload."""
    size = 2 * spec.page_size_bytes
    raw = torch.full((size,), SENTINEL, dtype=torch.uint8)
    placement = KVCacheTensor(
        size=size, layers=[ATTN], layer_stride=size, block_stride=spec.page_size_bytes
    )
    view = create_kv_cache_views(raw, spec, 2, KVCacheLayout.LBHNC, placement)[0]
    return raw, view


def worker(engine, tp, rank, spec, cache):
    value = object.__new__(mc.MooncakeConnectorWorker)
    value.shutdown = lambda: None
    value.engine = engine
    value.tp_size, value.tp_rank = tp, rank
    value.use_mla = False
    value.is_kv_consumer = True  # Register without starting the bootstrap server.
    value.is_kv_producer = True
    value._physical_blocks_per_logical_kv_block = 1
    value._layer_specs = {ATTN: spec}
    value._layer_group_indices = {ATTN: 0}
    value.kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([ATTN], spec)],
    )
    value.transfer_topo = mc.TransferTopology(
        tp_rank=rank,
        tp_size=tp,
        block_size=16,
        engine_id=f"review-{tp}-{rank}",
        is_mla=False,
        is_mamba=False,
        total_num_kv_heads=HKV,
        attn_backends=[],
    )
    value._encoder = msgspec.msgpack.Encoder()
    value.xfer_stats = mc.MooncakeKVConnectorStats()
    value.reqs_need_send = {}
    value.finished_sending_reqs = set()
    value.register_kv_caches({ATTN: cache})
    return value


def slot_value(first_head, heads, slot):
    # Encodes the global KV head; V slots (NVFP4) are offset by 10.
    return 10 + first_head + slot % heads + (10 if slot >= heads else 0)


async def exercise(kind, p, d, dr, baseline=None):
    """Send block 1 from every producer rank paired with consumer rank ``dr``."""
    engine = MemoryEngine()
    d_spec = spec_for(kind, d)
    d_raw, d_view = cache_view(d_spec)
    dw = worker(engine, d, dr, d_spec, d_view)
    fields = dict(
        kv_caches_base_addr=dw.kv_caches_base_addr,
        block_lens=dw.block_len_per_layer,
        kv_block_lens=dw.kv_block_len_per_layer,
        registered_layer_names=dw.registered_layer_names,
        registered_layer_indices=dw.registered_layer_indices,
        registered_group_indices=dw.registered_group_indices,
    )
    p_spec = spec_for(kind, p)
    responses, copies = [], []
    for pr in range(p):
        topo = mc.TransferTopology(
            tp_rank=pr,
            tp_size=p,
            block_size=16,
            engine_id="pairing",
            is_mla=False,
            is_mamba=False,
            total_num_kv_heads=HKV,
            attn_backends=[],
        )
        if dr not in topo.handshake_target_ranks(d):
            continue
        _, src = cache_view(p_spec)
        p_heads = max(HKV // p, 1)
        for slot in range(src.shape[1]):
            src[1, slot].view(torch.uint8).fill_(
                slot_value(pr * HKV // p, p_heads, slot)
            )
        pw = worker(engine, p, pr, p_spec, src)
        if baseline:
            for name in ("send_kv_to_decode", "_get_sender_transfer_plan"):
                setattr(pw, name, MethodType(baseline[name], pw))
        ready = asyncio.Event()
        ready.set()
        pw.reqs_need_send["x"] = mc.SendBlockMeta(
            p_req_id="p", transfer_id="x", local_block_ids=[[1]], ready=ready
        )
        meta = mc.MooncakeXferMetadata(
            remote_hostname="host",
            remote_port=1,
            remote_tp_size=d,
            remote_tp_rank=dr,
            req_blocks={"d": ("x", [[1]])},
            **fields,
        )
        sock = CaptureSocket()
        pw.sender_loop = asyncio.get_running_loop()
        before = len(engine.copies)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pw._sender_executor = pool
            await pw.send_kv_to_decode(b"review", sock, meta)
        responses.extend(sock.responses)
        copies.extend(
            {
                "producer_rank": pr,
                "src_offset_in_page": s - src.data_ptr() - p_spec.page_size_bytes,
                "dst_offset_in_page": t - d_view.data_ptr() - d_spec.page_size_bytes,
                "length": n,
            }
            for s, t, n in engine.copies[before:]
        )
    dst = d_view[1].view(torch.uint8)
    d_heads = max(HKV // d, 1)
    expected = torch.empty_like(dst)
    for slot in range(dst.shape[0]):
        expected[slot].fill_(slot_value(dr * HKV // d, d_heads, slot))
    assert bool((d_raw[: d_spec.page_size_bytes] == SENTINEL).all()), "guard block"
    return dict(
        kind=kind,
        p_tp=p,
        d_tp=d,
        reader_rank=dr,
        version="base" if baseline else "head",
        responses=responses,
        copies=copies,
        producer_page_bytes=p_spec.page_size_bytes,
        producer_content_bytes=p_spec.unpadded_page_size_bytes,
        producer_head_slots=p_spec.num_heads,
        consumer_page_bytes=d_spec.page_size_bytes,
        consumer_content_bytes=d_spec.unpadded_page_size_bytes,
        consumer_head_slots=d_spec.num_heads,
        content_bytes=dst.numel(),
        mismatched_content_bytes=int((dst != expected).sum()),
        untouched_content_bytes=int((dst == SENTINEL).sum()),
        expected_slot_values=[int(expected[i, 0, 0]) for i in range(dst.shape[0])],
        actual_slot_values=[int(dst[i, 0, 0]) for i in range(dst.shape[0])],
    )


def check(entry, expect):
    responses = entry["responses"]
    assert responses
    if expect == "reject":
        assert all(r.get("status") == 2 and r.get("err_msg") for r in responses)
        assert not entry["copies"]
        assert entry["untouched_content_bytes"] == entry["content_bytes"]
        return
    assert all(r.get("ok_reqs") and not r.get("err_reqs") for r in responses)
    if expect == "correct":
        assert entry["mismatched_content_bytes"] == 0
    else:
        assert expect == "corrupt"
        assert entry["mismatched_content_bytes"] > 0


def run(cases, output_name):
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=os.environ["VLLM_REPO"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == HEAD_REF, f"expected PR head {HEAD_REF}, got {head}"
    base = base_namespace()
    print(json.dumps({"module": mc.__file__, "torch": torch.__version__}), flush=True)
    results = []
    for label, kind, p, d, dr, version, expect in cases:
        entry = asyncio.run(
            exercise(kind, p, d, dr, base if version == "base" else None)
        )
        entry["case"] = label
        entry["expect"] = expect
        results.append(entry)
        print(json.dumps(entry), flush=True)
    output = Path(os.environ.get("REVIEW_OUTPUT", output_name))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    for entry in results:
        check(entry, entry["expect"])
    print(
        "REPRODUCED: GQA controls correct; base rejected the probe before "
        "transfer; head reported success with incorrect destination contents",
        flush=True,
    )
