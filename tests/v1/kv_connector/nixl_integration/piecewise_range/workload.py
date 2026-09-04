#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer

PREFIX_TEXT = "External KV cache prefixes should be loaded once and reused. "
SUFFIX_TEXT = "The prefiller computes only this request-specific suffix. "


def _tokens(tokenizer: Any, text: str, count: int) -> list[int]:
    source = tokenizer.encode(text, add_special_tokens=False)
    if not source:
        raise ValueError("Tokenizer returned an empty prompt fragment")
    return (source * (count // len(source) + 1))[:count]


def _payload(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt = _tokens(tokenizer, PREFIX_TEXT, args.base_tokens)
    prompt.extend(_tokens(tokenizer, SUFFIX_TEXT, args.suffix_tokens))
    return (
        {
            "model": args.served_model,
            "prompt": prompt,
            "max_tokens": args.output_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
            "stream": False,
        },
        len(prompt),
    )


def _post(url: str, payload: dict[str, Any], request_id: str) -> dict[str, Any]:
    response = httpx.post(
        f"{url.rstrip('/')}/v1/completions",
        json=payload,
        headers={"X-Request-Id": request_id},
        timeout=900,
    )
    response.raise_for_status()
    return response.json()


def _cached_tokens(response: dict[str, Any]) -> int:
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0)


def reset_cache(url: str) -> None:
    response = httpx.post(
        f"{url.rstrip('/')}/reset_prefix_cache?reset_external=false",
        timeout=60,
    )
    response.raise_for_status()


def request(args: argparse.Namespace) -> None:
    payload, prompt_tokens = _payload(args)
    started = time.perf_counter()
    response = _post(args.url, payload, args.request_id)
    record = {
        "request_id": args.request_id,
        "base_tokens": args.base_tokens,
        "suffix_tokens": args.suffix_tokens,
        "prompt_tokens": prompt_tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "usage": response.get("usage"),
        "output_token_ids": response["choices"][0].get("token_ids"),
        "output_text": response["choices"][0].get("text"),
    }
    Path(args.output).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


def probe(args: argparse.Namespace) -> None:
    payload, _ = _payload(args)
    payload["max_tokens"] = 1
    last_cached = 0
    for attempt in range(1, args.attempts + 1):
        reset_cache(args.url)
        response = _post(args.url, payload, f"{args.request_id}-{attempt}")
        last_cached = _cached_tokens(response)
        if last_cached >= args.expected_cached_tokens:
            print(
                json.dumps(
                    {"attempt": attempt, "cached_tokens": last_cached}, sort_keys=True
                )
            )
            return
        time.sleep(min(args.initial_delay * 2 ** (attempt - 1), args.max_delay))
    raise RuntimeError(
        f"Store probe expected {args.expected_cached_tokens} cached tokens, "
        f"last response reported {last_cached}"
    )


def write_store_config(args: argparse.Namespace) -> None:
    config = {
        "metadata_server": f"http://{args.host}:{args.metadata_port}/metadata",
        "master_server_address": f"{args.host}:{args.master_port}",
        "protocol": "rdma",
        "device_name": args.device,
        "mode": "standalone-store",
        "global_segment_size": "0 B",
        "local_buffer_size": "64 MB",
        "enable_offload": False,
    }
    Path(args.output).write_text(json.dumps(config, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", required=True)
    common.add_argument("--model-path", required=True)
    common.add_argument("--served-model", required=True)
    common.add_argument("--base-tokens", type=int, required=True)
    common.add_argument("--suffix-tokens", type=int, default=0)
    common.add_argument("--output-tokens", type=int, default=8)
    common.add_argument("--request-id", required=True)

    request_parser = subparsers.add_parser("request", parents=[common])
    request_parser.add_argument("--output", required=True)

    probe_parser = subparsers.add_parser("probe", parents=[common])
    probe_parser.add_argument("--expected-cached-tokens", type=int, required=True)
    probe_parser.add_argument("--attempts", type=int, default=8)
    probe_parser.add_argument("--initial-delay", type=float, default=0.5)
    probe_parser.add_argument("--max-delay", type=float, default=8)

    reset_parser = subparsers.add_parser("reset")
    reset_parser.add_argument("--url", required=True)

    config_parser = subparsers.add_parser("store-config")
    config_parser.add_argument("--output", required=True)
    config_parser.add_argument("--host", required=True)
    config_parser.add_argument("--master-port", type=int, required=True)
    config_parser.add_argument("--metadata-port", type=int, required=True)
    config_parser.add_argument("--device", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "request":
        request(args)
    elif args.command == "probe":
        probe(args)
    elif args.command == "reset":
        reset_cache(args.url)
    else:
        write_store_config(args)


if __name__ == "__main__":
    main()
