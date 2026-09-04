#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from .metrics import total

_PLAN_RE = re.compile(r"KV_LOAD_PLAN\s+(\{.*\})")


def parse_plans(log: str, request_id: str) -> list[dict[str, Any]]:
    plans = []
    for line in log.splitlines():
        match = _PLAN_RE.search(line)
        if match is None:
            continue
        plan = json.loads(match.group(1))
        actual_id = plan.get("request_id")
        if actual_id == request_id or (
            isinstance(actual_id, str) and actual_id.startswith(f"cmpl-{request_id}-")
        ):
            plans.append(plan)
    return plans


def read_plans(
    path: Path, request_id: str, wait_seconds: float = 0
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + wait_seconds
    while True:
        plans = parse_plans(path.read_text(errors="replace"), request_id)
        if plans or time.monotonic() >= deadline:
            return plans
        time.sleep(0.2)


def _delta(
    before: str, after: str, name: str, labels: dict[str, str] | None = None
) -> float:
    return total(after, name, labels) - total(before, name, labels)


def _cached_tokens(response: dict[str, Any]) -> int:
    details = response["usage"]["prompt_tokens_details"]
    return int(details["cached_tokens"])


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    response = json.loads(args.response.read_text())
    reference = json.loads(args.reference.read_text())
    p_before = args.prefill_metrics_before.read_text()
    p_after = args.prefill_metrics_after.read_text()
    d_before = args.decode_metrics_before.read_text()
    d_after = args.decode_metrics_after.read_text()
    plans = read_plans(
        args.decode_log,
        args.request_id,
        getattr(args, "plan_wait_seconds", 0) if args.case != "miss" else 0,
    )

    nixl_before, nixl_after = (
        (d_before, d_after) if args.mode == "pull" else (p_before, p_after)
    )
    store_bytes = _delta(
        d_before,
        d_after,
        "vllm:mooncake_store_operation_bytes_total",
        {"operation": "load_get", "status": "ok"},
    )
    store_failed_keys = _delta(
        d_before,
        d_after,
        "vllm:mooncake_store_operation_failed_keys_total",
    )
    nixl_bytes = _delta(nixl_before, nixl_after, "vllm:nixl_bytes_transferred_sum")
    nixl_failures = sum(
        _delta(before, after, metric)
        for before, after in ((p_before, p_after), (d_before, d_after))
        for metric in (
            "vllm:nixl_num_failed_transfers_total",
            "vllm:nixl_num_failed_notifications_total",
        )
    )
    cached_tokens = _cached_tokens(response)

    assert response["output_token_ids"]
    assert response["output_token_ids"] == reference["output_token_ids"]
    assert cached_tokens == response["prompt_tokens"]
    assert store_failed_keys == 0
    assert nixl_failures == 0

    if args.case in ("piecewise", "full"):
        assert store_bytes > 0
        assert nixl_bytes > 0
        assert plans, "Decode did not emit a piecewise load plan"
        entries = plans[-1]["entries"]
        assert len(entries) == 2
        store, nixl = entries
        assert store["connector"] == "MooncakeStoreConnector"
        boundary = (
            args.base_tokens
            if args.case == "piecewise"
            else cached_tokens - args.block_size
        )
        assert [store["start_token"], store["end_token"]] == [0, boundary]
        assert store["is_terminal"] is False
        assert nixl["connector"] in ("NixlPullConnector", "NixlPushConnector")
        assert [nixl["start_token"], nixl["end_token"]] == [
            boundary,
            cached_tokens,
        ]
        assert nixl["is_terminal"] is True
    elif args.case == "miss":
        assert store_bytes == 0
        assert nixl_bytes > 0
        assert not plans
    return {
        "case": args.case,
        "mode": args.mode,
        "request_id": args.request_id,
        "cached_tokens": cached_tokens,
        "store_load_bytes": store_bytes,
        "nixl_bytes": nixl_bytes,
        "store_failed_keys": store_failed_keys,
        "nixl_failures": nixl_failures,
        "load_plan": plans[-1] if plans else None,
        "output_token_ids": response["output_token_ids"],
    }


def compare_cases(args: argparse.Namespace) -> None:
    piecewise = json.loads(args.piecewise.read_text())
    miss = json.loads(args.miss.read_text())
    full = json.loads(args.full.read_text())
    assert 0 < full["nixl_bytes"] < piecewise["nixl_bytes"] < miss["nixl_bytes"]
    assert piecewise["store_load_bytes"] < full["store_load_bytes"]
    Path(args.output).write_text(
        json.dumps(
            {
                "piecewise_nixl_bytes": piecewise["nixl_bytes"],
                "miss_nixl_bytes": miss["nixl_bytes"],
                "full_nixl_bytes": full["nixl_bytes"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--case", choices=("piecewise", "miss", "full"), required=True)
    check.add_argument("--mode", choices=("pull", "push"), required=True)
    check.add_argument("--request-id", required=True)
    check.add_argument("--base-tokens", type=int, required=True)
    check.add_argument("--block-size", type=int, required=True)
    check.add_argument("--response", type=Path, required=True)
    check.add_argument("--reference", type=Path, required=True)
    check.add_argument("--decode-log", type=Path, required=True)
    check.add_argument("--plan-wait-seconds", type=float, default=0)
    check.add_argument("--prefill-metrics-before", type=Path, required=True)
    check.add_argument("--prefill-metrics-after", type=Path, required=True)
    check.add_argument("--decode-metrics-before", type=Path, required=True)
    check.add_argument("--decode-metrics-after", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--piecewise", type=Path, required=True)
    compare.add_argument("--miss", type=Path, required=True)
    compare.add_argument("--full", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "compare":
        compare_cases(args)
        return
    evidence = build_evidence(args)
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
