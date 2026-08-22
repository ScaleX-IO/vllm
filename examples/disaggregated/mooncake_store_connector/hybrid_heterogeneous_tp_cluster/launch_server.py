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
    parser.add_argument("--store-tp-size", type=int, default=4)
    parser.add_argument("--cache-prefix", default="")
    parser.add_argument("--lookup-rpc-port", type=int)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--save-decode-cache", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ["VLLM_SSM_CONV_STATE_LAYOUT"] = "DS"
    if args.role != "reference":
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
        "--block-size",
        str(args.block_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--mamba-cache-mode",
        "align",
        "--language-model-only",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--enforce-eager",
    ]
    if args.role != "reference":
        extra_config: dict[str, object] = {
            "cache_prefix": args.cache_prefix,
            "store_tp_size": args.store_tp_size,
        }
        if args.lookup_rpc_port is not None:
            extra_config["lookup_rpc_port"] = args.lookup_rpc_port
        if args.save_decode_cache:
            extra_config["save_decode_cache"] = True
        config = {
            "kv_connector": "MooncakeStoreConnector",
            "kv_role": "kv_both" if args.role == "producer" else "kv_consumer",
            "kv_connector_extra_config": extra_config,
        }
        command.extend(("--kv-transfer-config", json.dumps(config)))

    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
