from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model", default="gpukv-benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--target-prompt-tokens", type=int, default=2592)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--fill-concurrency", type=int, default=1)
    parser.add_argument("--reload-concurrency", type=int, default=4)
    parser.add_argument("--require-external-hits", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def main() -> None:
    args = parse_args()
    if args.requests <= 0 or args.block_size <= 0:
        raise ValueError("requests and block-size must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    repeated_text = " storage benchmark token"
    repeated_tokens = len(
        tokenizer.encode(repeated_text, add_special_tokens=False)
    )

    def make_prompt(index: int) -> tuple[str, int]:
        prefix = f"GPU-KV prefix reload benchmark request {index}:"
        initial = max(1, args.target_prompt_tokens // max(1, repeated_tokens))
        for repetitions in range(initial, initial + args.target_prompt_tokens):
            prompt = prefix + repeated_text * repetitions
            tokens = len(tokenizer.encode(prompt))
            if tokens >= args.target_prompt_tokens and tokens % args.block_size == 0:
                return prompt, tokens
        raise RuntimeError("could not construct a block-aligned prompt")

    prompt_rows = [make_prompt(index) for index in range(args.requests)]
    prompts = [row[0] for row in prompt_rows]
    prompt_tokens = [row[1] for row in prompt_rows]

    def request(prompt: str) -> dict:
        payload = json.dumps(
            {
                "model": args.served_model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
            }
        ).encode()
        http_request = urllib.request.Request(
            f"{args.url}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(http_request, timeout=120) as response:
            return json.loads(response.read())

    def metric(name: str) -> float:
        with urllib.request.urlopen(f"{args.url}/metrics", timeout=30) as response:
            body = response.read().decode()
        total = 0.0
        for line in body.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                total += float(line.rsplit(" ", 1)[1])
        return total

    def run_batch(concurrency: int) -> tuple[list[float], list[dict], float]:
        def timed(prompt: str) -> tuple[float, dict]:
            started = time.perf_counter()
            response = request(prompt)
            return time.perf_counter() - started, response

        wall_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(timed, prompts))
        wall_seconds = time.perf_counter() - wall_started
        return (
            [result[0] for result in results],
            [result[1] for result in results],
            wall_seconds,
        )

    fill_times, fill_responses, fill_wall = run_batch(args.fill_concurrency)
    external_before = metric("vllm:external_prefix_cache_hits_total")
    reload_times, reload_responses, reload_wall = run_batch(args.reload_concurrency)
    external_hits = metric("vllm:external_prefix_cache_hits_total") - external_before
    same_output = all(
        first["choices"][0]["text"] == second["choices"][0]["text"]
        for first, second in zip(fill_responses, reload_responses)
    )
    if not same_output:
        raise RuntimeError("reloaded output differs from the fill output")
    if args.require_external_hits and external_hits <= 0:
        raise RuntimeError("the reload phase did not use the external cache")

    print(
        json.dumps(
            {
                "requests": args.requests,
                "prompt_tokens_min": min(prompt_tokens),
                "prompt_tokens_max": max(prompt_tokens),
                "fill_concurrency": args.fill_concurrency,
                "fill_wall_seconds": fill_wall,
                "fill_p50_seconds": statistics.median(fill_times),
                "fill_p95_seconds": percentile(fill_times, 0.95),
                "reload_concurrency": args.reload_concurrency,
                "reload_wall_seconds": reload_wall,
                "reload_requests_per_second": args.requests / reload_wall,
                "reload_p50_seconds": statistics.median(reload_times),
                "reload_p95_seconds": percentile(reload_times, 0.95),
                "external_hit_tokens": external_hits,
                "same_output": same_output,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
