# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PR 52516: real connector control path and GPU TransferEngine payload checks.

No model kernels are executed. CpuPlatform avoids importing unavailable model
extensions; tensors and the native Mooncake data plane still use CUDA GPUs.
Logical TP ranks are exercised sequentially on two physical GPUs.
"""

import ast
import asyncio
import json
import multiprocessing as mp
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# isort: off
# Imports below must follow the environment and platform overrides.
os.environ.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
import torch  # noqa: E402
import msgspec  # noqa: E402
import vllm.platforms as platforms  # noqa: E402
from vllm.platforms.cpu import CpuPlatform  # noqa: E402

platforms._current_platform = CpuPlatform()
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (  # noqa: E402
    mooncake_connector as mc,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec  # noqa: E402
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum  # noqa: E402
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator  # noqa: E402
# isort: on

IP = os.environ.get("REVIEW_RDMA_IP", "200.9.0.18")
DEVICE = os.environ.get("REVIEW_RDMA_DEVICE", "mlx5_bond_0")
BASE_REF = "e862c2f45bc81c72bb84f3807d88d60cfb93d694"
CONNECTOR_PATH = (
    "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
)
ATTN = "model.layers.0.self_attn"
GDN = "model.layers.1.linear_attn"


def engine():
    value = mc.TransferEngine()
    assert value.initialize(IP, "P2PHANDSHAKE", "rdma", DEVICE) == 0
    return value


def attn_spec(heads, size=65536):
    return mc.FullAttentionSpec(
        block_size=16, num_kv_heads=heads, head_size=size // 32, dtype=torch.uint8
    )


def gdn_spec(tp):
    shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp, 16, 48, 128, 128, 4
    )
    return MambaSpec(
        block_size=16,
        shapes=shapes,
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )


def worker(native, tp, rank, heads, specs, caches):
    value = object.__new__(mc.MooncakeConnectorWorker)
    value.shutdown = lambda: None
    value.engine = native
    value.tp_size, value.tp_rank = tp, rank
    value.use_mla = False
    value.is_kv_consumer = True  # Register without starting the bootstrap server.
    value.is_kv_producer = True
    value._physical_blocks_per_logical_kv_block = 1
    value._layer_specs = specs
    value._layer_group_indices = {name: i for i, name in enumerate(specs)}
    value.kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([name], spec) for name, spec in specs.items()
        ],
    )
    value.transfer_topo = mc.TransferTopology(
        tp_rank=rank,
        tp_size=tp,
        block_size=16,
        engine_id=f"review-{tp}-{rank}",
        is_mla=False,
        is_mamba=GDN in specs,
        total_num_kv_heads=heads,
        attn_backends=[],
    )
    value._encoder = msgspec.msgpack.Encoder()
    value.xfer_stats = mc.MooncakeKVConnectorStats()
    value.reqs_need_send = {}
    value.finished_sending_reqs = set()
    value.register_kv_caches(caches)
    return value


def page(payload, device):
    result = torch.full((2, payload.numel()), 253, dtype=torch.uint8, device=device)
    result[1].copy_(payload)
    return result


def make_payloads(name):
    name = name.removeprefix("base_")
    # All payloads are deterministic and independent of the sender's planner.
    if name == "gqa_fan_in":
        p, d, h = 8, 2, 4
        src_specs = {ATTN: attn_spec(1)}
        dst_specs = {ATTN: attn_spec(2)}
        sources = [
            (r, {ATTN: torch.full((65536,), 10 + r // 2, dtype=torch.uint8)})
            for r in range(4)
        ]
        readers = [0]
        expected = {
            0: {
                ATTN: torch.cat(
                    [torch.full((65536,), i, dtype=torch.uint8) for i in (10, 11)]
                )
            }
        }
    elif name in ("gqa_fan_out", "mismatched_region"):
        p, d, h = (2, 8, 4) if name == "gqa_fan_out" else (1, 8, 4)
        local_heads = h // p
        src_specs = {ATTN: attn_spec(local_heads)}
        dst_specs = {ATTN: attn_spec(1, 65536 if name == "gqa_fan_out" else 131072)}
        sources = [
            (
                0,
                {
                    ATTN: torch.cat(
                        [
                            torch.full((65536,), 10 + i, dtype=torch.uint8)
                            for i in range(local_heads)
                        ]
                    )
                },
            )
        ]
        readers = list(range(4)) if name == "gqa_fan_out" else [0]
        expected = {
            r: {
                ATTN: torch.full(
                    (dst_specs[ATTN].page_size_bytes,), 10 + r // 2, dtype=torch.uint8
                )
            }
            for r in readers
        }
    else:
        # Qwen3.6-27B's actual GDN dimensions, P TP4 -> D TP8, Hkv=4.
        p, d, h = 4, 8, 4
        src_specs = {ATTN: attn_spec(1), GDN: gdn_spec(4)}
        dst_specs = {ATTN: attn_spec(1), GDN: gdn_spec(8)}
        assert src_specs[GDN].shapes == ((2560, 3), (12, 128, 128))
        conv = (torch.arange(2560 * 3).reshape(2560, 3) % 97 + 1).to(torch.bfloat16)
        state = (torch.arange(12 * 128 * 128).reshape(12, 128, 128) % 101 + 100).float()
        raw = torch.cat(
            [
                conv.contiguous().view(torch.uint8).flatten(),
                state.contiguous().view(torch.uint8).flatten(),
            ]
        )
        sources = [(0, {ATTN: torch.full((65536,), 10, dtype=torch.uint8), GDN: raw})]
        readers = [0, 1]
        expected = {}
        for r in readers:
            parts = [
                conv[r * 256 : (r + 1) * 256],
                conv[512 + r * 256 : 512 + (r + 1) * 256],
                conv[1024 + r * 768 : 1024 + (r + 1) * 768],
            ]
            expected[r] = {
                ATTN: torch.full((65536,), 10, dtype=torch.uint8),
                GDN: torch.cat(
                    [
                        torch.cat(parts).contiguous().view(torch.uint8).flatten(),
                        state[r * 6 : (r + 1) * 6]
                        .contiguous()
                        .view(torch.uint8)
                        .flatten(),
                    ]
                ),
            }
        assert raw.numel() == src_specs[GDN].page_size_bytes
    return p, d, h, src_specs, dst_specs, sources, readers, expected


def receiver(pipe):
    torch.cuda.set_device(1)
    native = engine()
    pipe.send({"port": native.get_rpc_port(), "device": torch.cuda.get_device_name(1)})
    while True:
        item = pipe.recv()
        if item == "stop":
            break
        name, rank = item
        p, d, h, ss, ds, sources, readers, expected = make_payloads(name)
        caches = {
            key: torch.full(
                (2, spec.page_size_bytes), 253, dtype=torch.uint8, device="cuda:1"
            )
            for key, spec in ds.items()
        }
        value = worker(native, d, rank, h, ds, caches)
        torch.cuda.synchronize()
        pipe.send(
            {
                "kv_caches_base_addr": value.kv_caches_base_addr,
                "block_lens": value.block_len_per_layer,
                "kv_block_lens": value.kv_block_len_per_layer,
                "registered_layer_names": value.registered_layer_names,
                "registered_layer_indices": value.registered_layer_indices,
                "registered_group_indices": value.registered_group_indices,
            }
        )
        assert pipe.recv() == "check"
        torch.cuda.synchronize()
        results = {}
        for key, cache in caches.items():
            actual = cache[1].cpu()
            ref = expected[rank][key]
            results[key] = {
                "bytes": actual.numel(),
                "mismatched_bytes": int((actual != ref).sum()),
                "untouched_sentinel_bytes": int((actual == 253).sum()),
                "guard_block_intact": bool((cache[0] == 253).all()),
                "first_16_bytes": actual[:16].tolist(),
            }
            if key == GDN:
                conv_bytes = 1280 * 3 * 2
                results[key]["conv_mismatched_elements"] = int(
                    (
                        actual[:conv_bytes].view(torch.bfloat16)
                        != ref[:conv_bytes].view(torch.bfloat16)
                    ).sum()
                )
                results[key]["ssm_mismatched_elements"] = int(
                    (
                        actual[conv_bytes:].view(torch.float32)
                        != ref[conv_bytes:].view(torch.float32)
                    ).sum()
                )
        pipe.send(results)
        for cache in caches.values():
            assert native.unregister_memory(cache.data_ptr()) == 0
        del value, caches


class CaptureSocket:
    def __init__(self):
        self.responses = []

    async def send_multipart(self, parts):
        self.responses.append(msgspec.msgpack.decode(parts[1]))


async def send(value, rank, d, port, fields, name):
    transfer_id = f"{name}-{value.tp_rank}-{rank}"
    req_id = f"d-{transfer_id}"
    groups = [[1] for _ in value.kv_cache_config.kv_cache_groups]
    ready = asyncio.Event()
    ready.set()
    value.reqs_need_send[transfer_id] = mc.SendBlockMeta(
        p_req_id=f"p-{transfer_id}",
        transfer_id=transfer_id,
        local_block_ids=groups,
        ready=ready,
    )
    meta = mc.MooncakeXferMetadata(
        remote_hostname=IP,
        remote_port=port,
        remote_tp_size=d,
        remote_tp_rank=rank,
        req_blocks={req_id: (transfer_id, groups)},
        **fields,
    )
    value.sender_loop = asyncio.get_running_loop()
    value._sender_executor = ThreadPoolExecutor(
        max_workers=1, initializer=lambda: torch.cuda.set_device(0)
    )
    sock = CaptureSocket()
    current_validator = mc._validate_asymmetric_region_lengths
    if name.startswith("base_"):
        baseline = subprocess.run(
            ["git", "show", f"{BASE_REF}:{CONNECTOR_PATH}"],
            cwd=os.environ["VLLM_REPO"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        tree = ast.parse(baseline)
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_validate_asymmetric_region_lengths"
        )
        ns = dict(mc.__dict__)
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]), "<base-validator>", "exec"
            ),
            ns,
        )

        def base_validator(**kwargs):
            kwargs.pop("consumer_cache_replicated")
            return ns["_validate_asymmetric_region_lengths"](**kwargs)

        mc._validate_asymmetric_region_lengths = base_validator
    try:
        await value.send_kv_to_decode(b"review", sock, meta)
    finally:
        mc._validate_asymmetric_region_lengths = current_validator
        value._sender_executor.shutdown(wait=True)
    return {
        "producer_rank": value.tp_rank,
        "responses": sock.responses,
        "stats": value.xfer_stats.data,
    }


def run(cases, output_name):
    torch.cuda.set_device(0)
    print(
        json.dumps(
            {
                "module": mc.__file__,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
            }
        ),
        flush=True,
    )
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=receiver, args=(child,))
    proc.start()
    assert parent.poll(180), "receiver initialization timed out"
    info = parent.recv()
    native = engine()
    results = []
    try:
        for name in cases:
            p, d, h, ss, ds, sources, readers, expected = make_payloads(name)
            for rank in readers:
                parent.send((name, rank))
                assert parent.poll(60)
                fields = parent.recv()
                transfers = []
                for pr, payloads in sources:
                    caches = {
                        key: page(data, "cuda:0") for key, data in payloads.items()
                    }
                    value = worker(native, p, pr, h, ss, caches)
                    torch.cuda.synchronize()
                    transfers.append(
                        asyncio.run(send(value, rank, d, info["port"], fields, name))
                    )
                    for cache in caches.values():
                        assert native.unregister_memory(cache.data_ptr()) == 0
                    del value, caches
                parent.send("check")
                assert parent.poll(60)
                entry = {
                    "case": name,
                    "p_tp": p,
                    "d_tp": d,
                    "reader_rank": rank,
                    "transfers": transfers,
                    "payload": parent.recv(),
                }
                results.append(entry)
                print(json.dumps(entry), flush=True)
    finally:
        if proc.is_alive():
            parent.send("stop")
        proc.join(20)
        if proc.is_alive():
            proc.terminate()
            proc.join()
    output = Path(os.environ.get("REVIEW_OUTPUT", output_name))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    assert proc.exitcode == 0
    for entry in results:
        if entry["case"].startswith("base_"):
            assert all(
                not r.get("ok_reqs") and r.get("err_msg")
                for t in entry["transfers"]
                for r in t["responses"]
            )
            assert all(
                p["untouched_sentinel_bytes"] == p["bytes"]
                for p in entry["payload"].values()
            )
            continue
        for transfer in entry["transfers"]:
            assert transfer["responses"]
            assert all(
                r.get("ok_reqs") and not r.get("err_reqs")
                for r in transfer["responses"]
            )
        for key, result in entry["payload"].items():
            assert result["guard_block_intact"]
            if entry["case"] in ("gqa_fan_in", "gqa_fan_out") or (
                entry["case"] == "hybrid_gdn" and key == ATTN
            ):
                assert result["mismatched_bytes"] == 0
            else:
                assert result["mismatched_bytes"] > 0
    print(
        "PASS: GQA controls correct; selected defect reproduced "
        "with successful native GPU transfers",
        flush=True,
    )
