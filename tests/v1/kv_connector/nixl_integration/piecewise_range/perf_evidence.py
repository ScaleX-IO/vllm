#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from pathlib import Path
from typing import Any

from .metrics import total

STORE_BYTES = "vllm:mooncake_store_operation_bytes_total"
STORE_TIME = "vllm:mooncake_store_operation_time_seconds"
NIXL_BYTES = "vllm:nixl_bytes_transferred_sum"
NIXL_TIME = "vllm:nixl_xfer_time_seconds"


def _delta(root: Path, role: str, metric: str, labels: dict[str, str] | None = None):
    before = (root / f"{role}-metrics-before.prom").read_text()
    after = (root / f"{role}-metrics-after.prom").read_text()
    return total(after, metric, labels) - total(before, metric, labels)


def _mean_histogram(
    root: Path, role: str, metric: str, labels: dict[str, str] | None = None
) -> float:
    count = _delta(root, role, f"{metric}_count", labels)
    value = _delta(root, role, f"{metric}_sum", labels)
    return value / count if count else 0


def _network(path: Path, threshold: int, counter: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    samples = payload["samples"]
    unit = payload["counter_unit_bytes"]
    devices = tuple(samples[0]["counters"])
    totals = {device: 0 for device in devices}
    active = {device: 0 for device in devices}
    peaks = {device: 0.0 for device in devices}
    overlap_intervals = 0
    overlap_seconds = 0.0
    union_intervals = 0
    for previous, current in zip(samples, samples[1:]):
        seconds = (current["time_ns"] - previous["time_ns"]) / 1e9
        deltas = {}
        for device in devices:
            delta = unit * (
                current["counters"][device][counter]
                - previous["counters"][device][counter]
            )
            delta = max(delta, 0)
            deltas[device] = delta
            totals[device] += delta
            if delta >= threshold:
                active[device] += 1
            if seconds:
                peaks[device] = max(peaks[device], delta / seconds)
        active_now = [deltas[device] >= threshold for device in devices]
        if any(active_now):
            union_intervals += 1
        if all(active_now):
            overlap_intervals += 1
            overlap_seconds += seconds
    return {
        "samples": len(samples),
        "counter": counter,
        "bytes": totals,
        "active_intervals": active,
        "peak_bytes_per_second": peaks,
        "overlap_intervals": overlap_intervals,
        "overlap_seconds": overlap_seconds,
        "overlap_fraction_of_active": (
            overlap_intervals / union_intervals if union_intervals else 0
        ),
    }


def _scenario(root: Path, threshold: int) -> dict[str, Any]:
    benchmark = json.loads((root / "benchmark.json").read_text())
    correctness = json.loads((root / "correctness.json").read_text())
    store_labels = {"operation": "load_get", "status": "ok"}
    failures = sum(
        _delta(root, role, metric)
        for role in ("prefill", "decode")
        for metric in (
            "vllm:nixl_num_failed_transfers_total",
            "vllm:nixl_num_failed_notifications_total",
            "vllm:mooncake_store_operation_failed_keys_total",
        )
    )
    return {
        "ttft_p50_seconds": benchmark["ttft_p50_seconds"],
        "ttft_p90_seconds": benchmark["ttft_p90_seconds"],
        "samples": benchmark["samples"],
        "correctness": correctness,
        "prefill_store_bytes": _delta(root, "prefill", STORE_BYTES, store_labels),
        "decode_store_bytes": _delta(root, "decode", STORE_BYTES, store_labels),
        "decode_nixl_bytes": _delta(root, "decode", NIXL_BYTES),
        "prefill_store_mean_seconds": _mean_histogram(
            root, "prefill", STORE_TIME, store_labels
        ),
        "decode_store_mean_seconds": _mean_histogram(
            root, "decode", STORE_TIME, store_labels
        ),
        "decode_nixl_mean_seconds": _mean_histogram(root, "decode", NIXL_TIME),
        "failures": failures,
        "node1_network": _network(
            root / "net-node1.json", threshold, "port_xmit_data"
        ),
        "node2_network": _network(
            root / "net-node2.json", threshold, "port_rcv_data"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--threshold-bytes", type=int, default=1048576)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads((args.root / "reference.json").read_text())
    legacy = _scenario(args.root / "legacy", args.threshold_bytes)
    piecewise = _scenario(args.root / "piecewise", args.threshold_bytes)
    for result in (legacy, piecewise):
        assert (
            result["correctness"]["output_token_ids"] == reference["output_token_ids"]
        )
        details = result["correctness"]["usage"]["prompt_tokens_details"]
        assert details["cached_tokens"] == result["correctness"]["prompt_tokens"]
        assert result["failures"] == 0
    assert legacy["decode_store_bytes"] == 0
    assert legacy["prefill_store_bytes"] > 0
    assert piecewise["prefill_store_bytes"] > 0
    assert piecewise["decode_store_bytes"] > 0
    assert 0 < piecewise["decode_nixl_bytes"] < legacy["decode_nixl_bytes"]

    report = {
        "legacy": legacy,
        "piecewise": piecewise,
        "p50_improvement_fraction": 1
        - piecewise["ttft_p50_seconds"] / legacy["ttft_p50_seconds"],
        "p90_improvement_fraction": 1
        - piecewise["ttft_p90_seconds"] / legacy["ttft_p90_seconds"],
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
