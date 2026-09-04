#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import shlex


def _store_config(args: argparse.Namespace, role: str) -> dict:
    return {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": role,
        "kv_connector_extra_config": {
            "cache_prefix": args.cache_prefix,
            "lookup_rpc_port": args.lookup_rpc_port,
            "store_tp_size": args.store_tp_size,
        },
    }


def _piecewise_config(args: argparse.Namespace) -> dict:
    role = "kv_producer" if args.role == "prefill" else "kv_consumer"
    nixl = "NixlConnector" if args.mode == "pull" else "NixlPushConnector"
    return {
        "kv_connector": "MultiConnector",
        "kv_role": role,
        "kv_connector_extra_config": {
            "load_policy": "range_aware",
            "connectors": [
                _store_config(args, "kv_consumer"),
                {
                    "kv_connector": nixl,
                    "kv_role": role,
                    "kv_load_failure_policy": "fail",
                },
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role", choices=("reference", "seeder", "prefill", "decode"), required=True
    )
    parser.add_argument("--mode", choices=("pull", "push"), default="pull")
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--lookup-rpc-port", type=int, default=0)
    parser.add_argument("--cache-prefix", default="unused")
    parser.add_argument("--store-tp-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        os.environ.get("PIECEWISE_VLLM_BIN", "vllm"),
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--block-size",
        str(args.block_size),
        "--prefix-match-unit",
        str(args.block_size),
        "--max-model-len",
        str(args.max_model_len),
        "--kv-cache-memory-bytes",
        str(args.kv_cache_memory_bytes),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--enforce-eager",
    ]
    if args.role == "seeder":
        command.extend(
            ("--kv-transfer-config", json.dumps(_store_config(args, "kv_producer")))
        )
    elif args.role in ("prefill", "decode"):
        command.extend(("--kv-transfer-config", json.dumps(_piecewise_config(args))))
    command.extend(shlex.split(os.environ.get("VLLM_SERVE_EXTRA_ARGS", "")))
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
