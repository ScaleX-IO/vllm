#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role", choices=("producer", "consumer", "reference"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--store-tp-size", type=int)
    parser.add_argument("--cache-prefix", default="")
    parser.add_argument("--lookup-rpc-port", type=int)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-layout", choices=("NHD", "HND"))
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-flashinfer-autotune", action="store_true")
    args = parser.parse_args()

    if args.role != "reference" and args.store_tp_size is None:
        parser.error("--store-tp-size is required for Store roles")

    os.environ.setdefault("PYTHONHASHSEED", "0")
    if args.kv_cache_layout:
        os.environ["VLLM_KV_CACHE_LAYOUT"] = args.kv_cache_layout
    if args.role == "producer":
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"

    command = [
        os.environ.get("VLLM_BIN", "vllm"),
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(args.pipeline_parallel_size),
        "--block-size",
        str(args.block_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.disable_flashinfer_autotune:
        command.extend(
            ("--kernel-config", json.dumps({"enable_flashinfer_autotune": False}))
        )

    if args.role != "reference":
        extra_config: dict[str, object] = {
            "cache_prefix": args.cache_prefix,
            "store_tp_size": args.store_tp_size,
        }
        if args.lookup_rpc_port is not None:
            extra_config["lookup_rpc_port"] = args.lookup_rpc_port
        config = {
            "kv_connector": "MooncakeStoreConnector",
            "kv_role": "kv_both" if args.role == "producer" else "kv_consumer",
            "kv_connector_extra_config": extra_config,
        }
        command.extend(("--kv-transfer-config", json.dumps(config)))

    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
