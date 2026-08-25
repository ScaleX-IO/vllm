#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


def payload(model: str, prompt: list[int], output_tokens: int) -> dict:
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


async def completion(client: httpx.AsyncClient, url: str, body: dict) -> dict:
    response = await client.post(f"{url}/v1/completions", json=body)
    response.raise_for_status()
    return response.json()


async def save_put_count(client: httpx.AsyncClient, url: str) -> float:
    response = await client.get(f"{url}/metrics")
    response.raise_for_status()
    return sum(
        float(line.rsplit(" ", 1)[-1])
        for line in response.text.splitlines()
        if line.startswith("vllm:mooncake_store_operation_total{")
        and 'operation="save_put"' in line
        and 'status="ok"' in line
    )


async def wait_for_store(
    client: httpx.AsyncClient,
    url: str,
    initial_save_puts: float,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    saw_put = False
    while True:
        saw_put = saw_put or await save_put_count(client, url) > initial_save_puts
        if saw_put:
            response = await client.post(f"{url}/reset_prefix_cache")
            response.raise_for_status()
            if response.json().get("success"):
                return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for Store jobs at {url}")
        await asyncio.sleep(0.25)


def make_prompt_ids(
    tokenizer_path: str, repetitions: int, prompt_tokens: int
) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    records = " ".join(
        f"Hybrid validation record {index} has checksum {index * 17 + 3}."
        for index in range(repetitions)
    )
    prompt = (
        "Retain the following records in order and continue with a concise "
        f"observation. {records}"
    )
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) < prompt_tokens:
        raise ValueError(
            f"prompt has {len(prompt_ids)} tokens; increase --prompt-repetitions "
            f"to reach --prompt-tokens={prompt_tokens}"
        )
    return prompt_ids[:prompt_tokens]


async def save_reference(args: argparse.Namespace) -> None:
    prompt_ids = make_prompt_ids(
        args.tokenizer, args.prompt_repetitions, args.prompt_tokens
    )
    async with httpx.AsyncClient(timeout=args.request_timeout) as client:
        result = await completion(
            client,
            args.reference_url,
            payload(args.model, prompt_ids, args.output_tokens),
        )
    output_ids = result["choices"][0]["token_ids"]
    Path(args.reference_json).write_text(
        json.dumps({"prompt_ids": prompt_ids, "output_ids": output_ids}),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "reference_prompt_tokens": len(prompt_ids),
                "reference_output_tokens": len(output_ids),
            }
        )
    )


async def verify_checkpoint(args: argparse.Namespace) -> None:
    reference = json.loads(Path(args.reference_json).read_text(encoding="utf-8"))
    prompt_ids = reference["prompt_ids"]
    reference_ids = reference["output_ids"]
    async with httpx.AsyncClient(timeout=args.request_timeout) as client:
        producer_puts = await save_put_count(client, args.producer_url)
        await completion(client, args.producer_url, payload(args.model, prompt_ids, 1))
        await wait_for_store(
            client, args.producer_url, producer_puts, args.visibility_timeout
        )

        consumer_puts = await save_put_count(client, args.consumer_url)
        decoded = await completion(
            client,
            args.consumer_url,
            payload(args.model, prompt_ids, args.output_tokens),
        )
        decode_ids = decoded["choices"][0]["token_ids"]
        decode_hit = cached_tokens(decoded)
        if decode_hit != args.expected_decode_cached_tokens:
            raise AssertionError(
                f"unexpected initial hit: {decode_hit} != "
                f"{args.expected_decode_cached_tokens}"
            )
        if decode_ids != reference_ids:
            raise AssertionError("cached continuation differs from reference")

        await wait_for_store(
            client, args.consumer_url, consumer_puts, args.visibility_timeout
        )
        verified = await completion(
            client,
            args.producer_url,
            payload(args.model, prompt_ids + decode_ids, 1),
        )
        extended_hit = cached_tokens(verified)
        if extended_hit != args.expected_extended_cached_tokens:
            raise AssertionError(
                f"unexpected extended hit: {extended_hit} != "
                f"{args.expected_extended_cached_tokens}"
            )

    print(
        json.dumps(
            {
                "verification": "passed",
                "prompt_tokens": len(prompt_ids),
                "decode_cached_tokens": decode_hit,
                "decode_output_tokens": len(decode_ids),
                "extended_cached_tokens": extended_hit,
                "reference_tokens_match": True,
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("reference", "checkpoint"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--model", required=True)
        subparser.add_argument("--tokenizer", required=True)
        subparser.add_argument("--reference-json", required=True)
        subparser.add_argument("--output-tokens", type=int, required=True)
        subparser.add_argument("--prompt-repetitions", type=int, default=256)
        subparser.add_argument("--prompt-tokens", type=int, default=1601)
        subparser.add_argument("--request-timeout", type=float, default=600)
    subparsers.choices["reference"].add_argument("--reference-url", required=True)
    checkpoint = subparsers.choices["checkpoint"]
    checkpoint.add_argument("--producer-url", required=True)
    checkpoint.add_argument("--consumer-url", required=True)
    checkpoint.add_argument("--visibility-timeout", type=float, default=300)
    checkpoint.add_argument("--expected-decode-cached-tokens", type=int, required=True)
    checkpoint.add_argument(
        "--expected-extended-cached-tokens", type=int, required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        save_reference(args) if args.command == "reference" else verify_checkpoint(args)
    )


if __name__ == "__main__":
    main()
