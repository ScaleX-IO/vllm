#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import httpx
import regex as re

MOONCAKE_METRICS = {
    "bytes": "vllm:mooncake_store_operation_bytes_total",
    "seconds": "vllm:mooncake_store_operation_time_seconds_sum",
    "calls": "vllm:mooncake_store_operation_total",
    "keys": "vllm:mooncake_store_operation_keys_total",
    "failed_keys": "vllm:mooncake_store_operation_failed_keys_total",
}
LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def prompt_for(index: int, repetitions: int, group: str) -> str:
    facts = " ".join(
        f"Record {index}-{item} belongs to validation group {group}."
        for item in range(repetitions)
    )
    return (
        f"Validation group {group}, request {index}. "
        "Read the following numbered records carefully, retain their order, "
        f"and continue with one concise observation. {facts}"
    )


def payload(model: str, prompt: str | list[int], output_tokens: int) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "return_token_ids": True,
    }


def cached_tokens(result: dict) -> int:
    details = result.get("usage", {}).get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0)


def cacheable_prompt_tokens(prompt_tokens: int, block_size: int) -> int:
    return max(0, (prompt_tokens - 1) // block_size * block_size)


async def completion(client: httpx.AsyncClient, url: str, body: dict) -> dict:
    response = await client.post(f"{url}/v1/completions", json=body)
    response.raise_for_status()
    return response.json()


async def reset_local_cache(client: httpx.AsyncClient, url: str) -> None:
    response = await client.post(f"{url}/reset_prefix_cache")
    response.raise_for_status()
    if not response.json().get("success"):
        raise RuntimeError("producer local prefix cache is still busy")


async def token_lengths(tokenizer_name: str, prompts: list[str]) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return [len(tokenizer.encode(prompt)) for prompt in prompts]


async def encode_prompts(tokenizer_name: str, prompts: list[str]) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return [tokenizer.encode(prompt) for prompt in prompts]


async def wait_until_store_jobs_finish(
    client: httpx.AsyncClient,
    producer_url: str,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            await reset_local_cache(client, producer_url)
            return
        except RuntimeError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out waiting for producer cache reset"
                ) from error
            await asyncio.sleep(0.25)


async def seed_prompts(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompts: list[str],
    concurrency: int,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async def seed(prompt: str) -> None:
        async with semaphore:
            await completion(client, url, payload(model, prompt, 1))

    await asyncio.gather(*(seed(prompt) for prompt in prompts))


async def functional(args: argparse.Namespace) -> None:
    prompts = [prompt_for(0, args.prompt_repetitions, "functional")]
    lengths = await token_lengths(args.tokenizer, prompts)
    limits = httpx.Limits(max_connections=max(8, args.concurrency * 2))
    async with httpx.AsyncClient(timeout=args.request_timeout, limits=limits) as client:
        await seed_prompts(client, args.producer_url, args.model, prompts, 1)
        await wait_until_store_jobs_finish(
            client,
            args.producer_url,
            args.visibility_timeout,
        )
        body = payload(args.model, prompts[0], args.output_tokens)
        cached, reference = await asyncio.gather(
            completion(client, args.consumer_url, body),
            completion(client, args.reference_url, body),
        )

    expected_hit = cacheable_prompt_tokens(lengths[0], args.block_size)
    actual_hit = cached_tokens(cached)
    if actual_hit < expected_hit:
        raise AssertionError(
            f"heterogeneous-TP Store miss: expected {expected_hit}, got {actual_hit}"
        )
    cached_ids = cached["choices"][0]["token_ids"]
    reference_ids = reference["choices"][0]["token_ids"]
    tokens_match = cached_ids == reference_ids
    if not tokens_match and not args.allow_token_mismatch:
        raise AssertionError(
            "cached and same-TP reference continuations differ; rerun with "
            "--allow-token-mismatch only when low-precision TP arithmetic is known "
            "not to be token invariant"
        )
    print(
        json.dumps(
            {
                "verification": "passed",
                "prompt_tokens": lengths[0],
                "cached_tokens": actual_hit,
                "output_tokens": len(cached_ids),
                "reference_tokens_match": tokens_match,
            }
        )
    )


async def lcm_multi_prefill(args: argparse.Namespace) -> None:
    prompts = [
        prompt_for(0, args.prompt_repetitions, "lcm-first"),
        prompt_for(1, args.prompt_repetitions, "lcm-second"),
    ]
    prompt_ids = await encode_prompts(args.tokenizer, prompts)
    limits = httpx.Limits(max_connections=8)

    async with httpx.AsyncClient(timeout=args.request_timeout, limits=limits) as client:

        async def verify_direction(
            seed_url: str,
            verify_url: str,
            prompt: str,
            ids: list[int],
            label: str,
        ) -> dict[str, int]:
            await completion(client, seed_url, payload(args.model, prompt, 1))
            await wait_until_store_jobs_finish(
                client, seed_url, args.visibility_timeout
            )

            decoded = await completion(
                client,
                args.decode_url,
                payload(args.model, ids, args.decode_tokens),
            )
            base_expected = cacheable_prompt_tokens(len(ids), args.block_size)
            decode_hit = cached_tokens(decoded)
            if decode_hit < base_expected:
                raise AssertionError(
                    f"{label}: decode missed the prefill prefix: "
                    f"expected {base_expected}, got {decode_hit}"
                )

            decoded_ids = decoded["choices"][0]["token_ids"]
            if not isinstance(decoded_ids, list) or not decoded_ids:
                raise AssertionError(f"{label}: decode returned no token ids")
            await wait_until_store_jobs_finish(
                client, args.decode_url, args.visibility_timeout
            )

            extended_ids = ids + decoded_ids
            verified = await completion(
                client, verify_url, payload(args.model, extended_ids, 1)
            )
            extended_cacheable = cacheable_prompt_tokens(
                len(extended_ids), args.block_size
            )
            extended_hit = cached_tokens(verified)
            if extended_hit <= base_expected:
                raise AssertionError(
                    f"{label}: cache hit did not extend beyond the original "
                    f"prefill prefix ({extended_hit} <= {base_expected})"
                )
            return {
                "prompt_tokens": len(ids),
                "decode_cached_tokens": decode_hit,
                "decode_output_tokens": len(decoded_ids),
                "extended_cached_tokens": extended_hit,
                "extended_cacheable_tokens": extended_cacheable,
            }

        first_to_second = await verify_direction(
            args.prefill_first_url,
            args.prefill_second_url,
            prompts[0],
            prompt_ids[0],
            "first-prefill-to-second-prefill",
        )
        second_to_first = await verify_direction(
            args.prefill_second_url,
            args.prefill_first_url,
            prompts[1],
            prompt_ids[1],
            "second-prefill-to-first-prefill",
        )

    print(
        json.dumps(
            {
                "verification": "passed",
                "first_prefill_to_second_prefill": first_to_second,
                "second_prefill_to_first_prefill": second_to_first,
            }
        )
    )


@dataclass
class Sample:
    latency: float
    cached_tokens: int
    output_tokens: int


async def mooncake_load_metrics(
    client: httpx.AsyncClient, url: str
) -> dict[str, float]:
    response = await client.get(f"{url}/metrics")
    response.raise_for_status()
    totals = dict.fromkeys(MOONCAKE_METRICS, 0.0)
    names = {metric: field for field, metric in MOONCAKE_METRICS.items()}
    for line in response.text.splitlines():
        metric = line.split("{", 1)[0]
        field = names.get(metric)
        if field is None:
            continue
        labels = dict(LABEL_RE.findall(line))
        if labels.get("operation") != "load_get":
            continue
        value = float(line.rsplit(None, 1)[-1])
        if field == "failed_keys" or labels.get("status") == "ok":
            totals[field] += value
    return totals


def mooncake_throughput(
    before: dict[str, float], after: dict[str, float], wall_time: float
) -> dict[str, float]:
    delta = {key: after[key] - before[key] for key in before}
    gib = delta["bytes"] / 2**30
    return {
        "bytes": int(delta["bytes"]),
        "keys": int(delta["keys"]),
        "rpc_calls": int(delta["calls"]),
        "failed_keys": int(delta["failed_keys"]),
        "rpc_time_seconds": delta["seconds"],
        "rpc_throughput_gib_s": gib / delta["seconds"] if delta["seconds"] > 0 else 0.0,
        "phase_effective_throughput_gib_s": gib / wall_time,
    }


async def run_group(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompts: list[str],
    output_tokens: int,
    concurrency: int,
) -> tuple[list[Sample], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(prompt: str) -> Sample:
        async with semaphore:
            started = time.perf_counter()
            result = await completion(
                client, url, payload(model, prompt, output_tokens)
            )
            latency = time.perf_counter() - started
        return Sample(
            latency=latency,
            cached_tokens=cached_tokens(result),
            output_tokens=len(result["choices"][0]["token_ids"]),
        )

    started = time.perf_counter()
    samples = await asyncio.gather(*(one(prompt) for prompt in prompts))
    return samples, time.perf_counter() - started


def summarize(samples: list[Sample], wall_time: float) -> dict:
    latencies = sorted(sample.latency for sample in samples)
    p95_index = max(0, int(len(latencies) * 0.95) - 1)
    return {
        "requests": len(samples),
        "mean_latency_seconds": statistics.mean(latencies),
        "p50_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": latencies[p95_index],
        "wall_seconds": wall_time,
        "request_throughput": len(samples) / wall_time,
        "output_token_throughput": sum(s.output_tokens for s in samples) / wall_time,
        "mean_cached_tokens": statistics.mean(s.cached_tokens for s in samples),
    }


async def performance(args: argparse.Namespace) -> None:
    cached_prompts = [
        prompt_for(index, args.prompt_repetitions, "cached")
        for index in range(args.requests)
    ]
    cold_prompts = [
        prompt_for(index, args.prompt_repetitions, "cold")
        for index in range(args.requests)
    ]
    warmup_prompts = [
        prompt_for(index, args.prompt_repetitions, "warmup")
        for index in range(args.warmup_requests)
    ]
    lengths = await token_lengths(args.tokenizer, cached_prompts)
    limits = httpx.Limits(max_connections=max(8, args.concurrency * 2))
    async with httpx.AsyncClient(timeout=args.request_timeout, limits=limits) as client:
        await seed_prompts(
            client,
            args.producer_url,
            args.model,
            cached_prompts,
            args.seed_concurrency,
        )
        await wait_until_store_jobs_finish(
            client,
            args.producer_url,
            args.visibility_timeout,
        )
        if warmup_prompts:
            await run_group(
                client,
                args.consumer_url,
                args.model,
                warmup_prompts,
                1,
                min(args.concurrency, len(warmup_prompts)),
            )

        groups = {
            "cached": cached_prompts,
            "cold": cold_prompts,
        }
        order = args.order.split("-")
        measurements: dict[str, tuple[list[Sample], float]] = {}
        await asyncio.sleep(args.metrics_settle_seconds)
        metrics_before_cached = await mooncake_load_metrics(client, args.consumer_url)
        for group in order:
            measurements[group] = await run_group(
                client,
                args.consumer_url,
                args.model,
                groups[group],
                args.output_tokens,
                args.concurrency,
            )
            if group == "cached":
                await asyncio.sleep(args.metrics_settle_seconds)
                metrics_after_cached = await mooncake_load_metrics(
                    client, args.consumer_url
                )

    cached_samples, cached_wall = measurements["cached"]
    cold_samples, cold_wall = measurements["cold"]

    missing = [
        index
        for index, sample in enumerate(cached_samples)
        if sample.cached_tokens
        < cacheable_prompt_tokens(lengths[index], args.block_size)
    ]
    cold_hits = [
        sample.cached_tokens for sample in cold_samples if sample.cached_tokens
    ]

    cached_summary = summarize(cached_samples, cached_wall)
    cold_summary = summarize(cold_samples, cold_wall)
    result = {
        "validation": {
            "passed": not missing and not cold_hits,
            "cached_miss_indices": missing,
            "cold_hit_tokens": cold_hits,
        },
        "order": args.order,
        "warmup_requests": len(warmup_prompts),
        "cached": cached_summary,
        "mooncake_load": mooncake_throughput(
            metrics_before_cached, metrics_after_cached, cached_wall
        ),
        "cold": cold_summary,
        "mean_latency_speedup": (
            cold_summary["mean_latency_seconds"]
            / cached_summary["mean_latency_seconds"]
        ),
        "request_throughput_speedup": (
            cached_summary["request_throughput"] / cold_summary["request_throughput"]
        ),
    }
    print(json.dumps(result))
    if missing:
        raise AssertionError(f"cached benchmark requests missed Store: {missing[:10]}")
    if cold_hits:
        raise AssertionError(f"cold benchmark unexpectedly hit cache: {cold_hits[:10]}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--producer-url", required=True)
    common.add_argument("--consumer-url", required=True)
    common.add_argument("--model", required=True)
    common.add_argument("--tokenizer", required=True)
    common.add_argument("--block-size", type=int, default=16)
    common.add_argument("--prompt-repetitions", type=int, default=80)
    common.add_argument("--output-tokens", type=int, default=64)
    common.add_argument("--concurrency", type=int, default=16)
    common.add_argument("--visibility-timeout", type=float, default=300)
    common.add_argument("--request-timeout", type=float, default=300)
    commands = root.add_subparsers(dest="command", required=True)

    functional_parser = commands.add_parser("functional", parents=[common])
    functional_parser.add_argument("--reference-url", required=True)
    functional_parser.add_argument("--allow-token-mismatch", action="store_true")
    functional_parser.set_defaults(handler=functional)

    lcm_parser = commands.add_parser("lcm-multi-prefill")
    lcm_parser.add_argument("--prefill-first-url", required=True)
    lcm_parser.add_argument("--prefill-second-url", required=True)
    lcm_parser.add_argument("--decode-url", required=True)
    lcm_parser.add_argument("--model", required=True)
    lcm_parser.add_argument("--tokenizer", required=True)
    lcm_parser.add_argument("--block-size", type=int, default=16)
    lcm_parser.add_argument("--prompt-repetitions", type=int, default=80)
    lcm_parser.add_argument("--decode-tokens", type=int, default=64)
    lcm_parser.add_argument("--visibility-timeout", type=float, default=300)
    lcm_parser.add_argument("--request-timeout", type=float, default=300)
    lcm_parser.set_defaults(handler=lcm_multi_prefill)

    performance_parser = commands.add_parser("performance", parents=[common])
    performance_parser.add_argument("--requests", type=int, default=64)
    performance_parser.add_argument("--seed-concurrency", type=int, default=8)
    performance_parser.add_argument("--warmup-requests", type=int, default=2)
    performance_parser.add_argument("--metrics-settle-seconds", type=float, default=2)
    performance_parser.add_argument(
        "--order", choices=("cold-cached", "cached-cold"), default="cold-cached"
    )
    performance_parser.set_defaults(handler=performance)
    return root


if __name__ == "__main__":
    parsed = parser().parse_args()
    asyncio.run(parsed.handler(parsed))
