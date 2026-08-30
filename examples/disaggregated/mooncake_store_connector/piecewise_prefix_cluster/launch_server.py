#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--lookup-rpc-port", type=int, required=True)
    parser.add_argument("--cache-prefix", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--transfer-protocol", default="rdma")
    parser.add_argument("--transfer-device", default="mlx5_bond_0")
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--disable-store-lookup", action="store_true")
    args = parser.parse_args()

    role = "kv_producer" if args.role == "prefill" else "kv_consumer"
    store_extra = {
        "cache_prefix": args.cache_prefix,
        "lookup_rpc_port": args.lookup_rpc_port,
        "store_tp_size": args.tensor_parallel_size,
    }
    if args.role == "decode":
        store_extra["save_decode_cache"] = True
    if args.disable_store_lookup:
        store_extra["enable_lookup"] = False

    config = {
        "kv_connector": "MultiConnector",
        "kv_role": role,
        "kv_connector_extra_config": {
            "load_policy": "range_aware",
            "connectors": [
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both" if args.role == "prefill" else role,
                    "kv_connector_extra_config": store_extra,
                },
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": role,
                    "kv_connector_extra_config": {
                        "mooncake_protocol": args.transfer_protocol,
                        "device_name": args.transfer_device,
                    },
                },
            ],
        },
    }

    command = [
        args.vllm_bin,
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
        "16",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--disable-log-stats",
        "--enforce-eager",
        "--kv-transfer-config",
        json.dumps(config),
    ]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
