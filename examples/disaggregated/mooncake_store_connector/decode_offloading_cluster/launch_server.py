#!/usr/bin/env python3
import argparse
import json
import os


def store_connector(role: str, args: argparse.Namespace) -> dict:
    extra = {"cache_prefix": args.cache_prefix}
    if args.save_decode_cache:
        extra["save_decode_cache"] = True
    return {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": role,
        "kv_connector_extra_config": extra,
    }


def direct_connector(role: str, args: argparse.Namespace) -> dict:
    extra = {"mooncake_protocol": args.transfer_protocol}
    if args.transfer_device:
        extra["device_name"] = args.transfer_device
    return {
        "kv_connector": "MooncakeConnector",
        "kv_role": role,
        "kv_connector_extra_config": extra,
    }


def connector_config(args: argparse.Namespace) -> dict | None:
    if args.role == "reference":
        return None
    if args.role in ("mixed", "verifier"):
        return store_connector("kv_both", args)
    role = "kv_producer" if args.role == "prefill" else "kv_consumer"
    return store_connector(role, args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        choices=("prefill", "decode", "mixed", "verifier", "reference"),
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cache-prefix", required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--transfer-protocol", default="tcp")
    parser.add_argument("--transfer-device", default="")
    parser.add_argument("--save-decode-cache", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

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
        "--block-size",
        str(args.block_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--disable-log-stats",
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    config = connector_config(args)
    if config is not None:
        command.extend(("--kv-transfer-config", json.dumps(config)))
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
