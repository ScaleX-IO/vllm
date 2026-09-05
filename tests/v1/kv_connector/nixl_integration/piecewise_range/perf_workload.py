#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer

from .workload import PREFIX_TEXT, SUFFIX_TEXT, _tokens, reset_cache


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def _request(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_token = None
    usage = None
    token_ids: list[int] = []
    with client.stream(
        "POST",
        f"{url.rstrip('/')}/v1/completions",
        json=payload,
        headers={"X-Request-Id": request_id},
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            ids = choice.get("token_ids") or []
            token_ids.extend(ids)
            if first_token is None and (ids or choice.get("text")):
                first_token = time.perf_counter() - started
    if first_token is None:
        raise RuntimeError(f"No streamed token for {request_id}")
    return {
        "request_id": request_id,
        "ttft_seconds": first_token,
        "total_seconds": time.perf_counter() - started,
        "token_ids": token_ids,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--base-tokens", type=int, default=1024)
    parser.add_argument("--suffix-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--request-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt = _tokens(tokenizer, PREFIX_TEXT, args.base_tokens)
    prompt += _tokens(tokenizer, SUFFIX_TEXT, args.suffix_tokens)
    payload = {
        "model": args.served_model,
        "prompt": prompt,
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    samples = []
    with httpx.Client(timeout=900) as client:
        for index in range(args.iterations):
            reset_cache(args.prefill_url)
            reset_cache(args.decode_url)
            samples.append(
                _request(client, args.url, payload, f"{args.request_prefix}-{index}")
            )

    ttft = [sample["ttft_seconds"] for sample in samples]
    result = {
        "iterations": args.iterations,
        "prompt_tokens": len(prompt),
        "ttft_p50_seconds": _percentile(ttft, 0.5),
        "ttft_p90_seconds": _percentile(ttft, 0.9),
        "samples": samples,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
