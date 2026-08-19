#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


async def completion(url: str, payload: dict, timeout: float) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{url}/v1/completions", json=payload)
        response.raise_for_status()
        return response.json()


def base_payload(model: str, prompt: str | list[int], max_tokens: int) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "return_token_ids": True,
    }


async def probe(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    prompt = (
        "Decode offloading must preserve block hashes and KV contents under "
        "concurrent requests. " * args.prompt_repetitions
    )
    result = await completion(
        args.url,
        base_payload(args.model, prompt, args.decode_tokens),
        args.timeout,
    )
    choice = result["choices"][0]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt_token_ids = tokenizer.encode(prompt)
    prompt_floor = len(prompt_token_ids) // args.block_size * args.block_size
    cache_hit = cached_tokens(result)
    if args.require_prompt_cache_hit and cache_hit < prompt_floor:
        raise AssertionError(
            f"D did not load the P-side prompt KV from Store: "
            f"expected at least {prompt_floor} cached tokens, got {cache_hit}"
        )
    payload = {
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": choice["token_ids"],
        "text": choice["text"],
        "cached_tokens": cache_hit,
    }
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps({"probe": "passed", **payload}, ensure_ascii=False))


async def stress(args: argparse.Namespace) -> None:
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    failures: list[str] = []

    async def one(index: int) -> None:
        prompt = (
            f"request={index} "
            + "Concurrent decode offloading should keep every cache key isolated. "
            * args.prompt_repetitions
        )
        url = args.urls[index % len(args.urls)]
        started = time.perf_counter()
        try:
            async with semaphore:
                await completion(
                    url,
                    base_payload(args.model, prompt, args.output_tokens),
                    args.timeout,
                )
            latencies.append(time.perf_counter() - started)
        except Exception as exc:
            failures.append(f"request {index}: {exc}")

    await asyncio.gather(*(one(i) for i in range(args.requests)))
    if failures:
        raise RuntimeError("\n".join(failures[:10]))
    ordered = sorted(latencies)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    print(
        json.dumps(
            {
                "requests": len(latencies),
                "failures": 0,
                "mean_seconds": statistics.mean(latencies),
                "p95_seconds": p95,
            }
        )
    )


def cached_tokens(result: dict) -> int:
    details = result.get("usage", {}).get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0)


async def verify(args: argparse.Namespace) -> None:
    probe_result = json.loads(args.probe.read_text(encoding="utf-8"))
    prompt_ids = probe_result["prompt_token_ids"]
    full_ids = prompt_ids + probe_result["output_token_ids"]
    request = base_payload(args.model, full_ids, 1)
    cached, reference = await asyncio.gather(
        completion(args.cached_url, request, args.timeout),
        completion(args.reference_url, request, args.timeout),
    )
    cached_count = cached_tokens(cached)
    prompt_floor = len(prompt_ids) // args.block_size * args.block_size
    if cached_count < prompt_floor + args.block_size:
        raise AssertionError(
            f"expected decode KV hit beyond {prompt_floor}, got {cached_count}"
        )
    cached_ids = cached["choices"][0]["token_ids"]
    reference_ids = reference["choices"][0]["token_ids"]
    if cached_ids != reference_ids:
        raise AssertionError(
            f"cached continuation {cached_ids} != reference {reference_ids}; "
            f"cached_tokens={cached_count}, prompt_tokens={len(prompt_ids)}, "
            f"decode_tokens={len(probe_result['output_token_ids'])}"
        )
    print(
        json.dumps(
            {
                "verification": "passed",
                "prompt_tokens": len(prompt_ids),
                "decode_tokens": len(probe_result["output_token_ids"]),
                "cached_tokens": cached_count,
                "continuation_token_ids": cached_ids,
            }
        )
    )


async def verify_recorded(args: argparse.Namespace) -> None:
    """Verify Store KV using the last recorded decode token as reference."""
    probe_result = json.loads(args.probe.read_text(encoding="utf-8"))
    prompt_ids = probe_result["prompt_token_ids"]
    output_ids = probe_result["output_token_ids"]
    if len(output_ids) < 2:
        raise AssertionError("probe must record at least two decode tokens")

    # The final sampled token has not itself gone through a model forward. Use
    # the preceding decode tokens as input and the final one as the reference.
    request = base_payload(args.model, prompt_ids + output_ids[:-1], 1)
    cached = await completion(args.cached_url, request, args.timeout)
    cached_count = cached_tokens(cached)
    prompt_floor = len(prompt_ids) // args.block_size * args.block_size
    if cached_count < prompt_floor + args.block_size:
        raise AssertionError(
            f"expected decode KV hit beyond {prompt_floor}, got {cached_count}"
        )
    actual_ids = cached["choices"][0]["token_ids"]
    expected_ids = output_ids[-1:]
    if actual_ids != expected_ids:
        raise AssertionError(
            f"cached continuation {actual_ids} != recorded {expected_ids}; "
            f"cached_tokens={cached_count}"
        )
    print(
        json.dumps(
            {
                "verification": "passed",
                "prompt_tokens": len(prompt_ids),
                "decode_tokens": len(output_ids),
                "cached_tokens": cached_count,
                "continuation_token_ids": actual_ids,
            }
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True)
    common.add_argument("--timeout", type=float, default=300)

    probe_parser = subparsers.add_parser("probe", parents=[common])
    probe_parser.add_argument("--url", required=True)
    probe_parser.add_argument("--output", type=Path, required=True)
    probe_parser.add_argument("--tokenizer", required=True)
    probe_parser.add_argument("--decode-tokens", type=int, default=65)
    probe_parser.add_argument("--prompt-repetitions", type=int, default=80)
    probe_parser.add_argument("--block-size", type=int, default=16)
    probe_parser.add_argument("--require-prompt-cache-hit", action="store_true")
    probe_parser.set_defaults(handler=probe)

    stress_parser = subparsers.add_parser("stress", parents=[common])
    stress_parser.add_argument("--urls", nargs="+", required=True)
    stress_parser.add_argument("--requests", type=int, default=64)
    stress_parser.add_argument("--concurrency", type=int, default=16)
    stress_parser.add_argument("--prompt-repetitions", type=int, default=80)
    stress_parser.add_argument("--output-tokens", type=int, default=128)
    stress_parser.set_defaults(handler=stress)

    verify_parser = subparsers.add_parser("verify", parents=[common])
    verify_parser.add_argument("--cached-url", required=True)
    verify_parser.add_argument("--reference-url", required=True)
    verify_parser.add_argument("--probe", type=Path, required=True)
    verify_parser.add_argument("--block-size", type=int, default=16)
    verify_parser.set_defaults(handler=verify)

    recorded_parser = subparsers.add_parser("verify-recorded", parents=[common])
    recorded_parser.add_argument("--cached-url", required=True)
    recorded_parser.add_argument("--probe", type=Path, required=True)
    recorded_parser.add_argument("--block-size", type=int, default=16)
    recorded_parser.set_defaults(handler=verify_recorded)
    return root


if __name__ == "__main__":
    parsed = parser().parse_args()
    asyncio.run(parsed.handler(parsed))
