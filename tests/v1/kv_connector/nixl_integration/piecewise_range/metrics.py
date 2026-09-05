#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import re
import time
import urllib.request
from pathlib import Path

_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)(?:\s+\d+)?$"
)
_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"])*)"')


def parse_samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples = []
    for line in text.splitlines():
        match = _SAMPLE_RE.match(line)
        if match is None:
            continue
        labels = {
            key: bytes(value, "utf-8").decode("unicode_escape")
            for key, value in _LABEL_RE.findall(match.group("labels") or "")
        }
        samples.append((match.group("name"), labels, float(match.group("value"))))
    return samples


def total(text: str, name: str, labels: dict[str, str] | None = None) -> float:
    required = labels or {}
    return sum(
        value
        for sample_name, sample_labels, value in parse_samples(text)
        if sample_name == name
        and all(
            sample_labels.get(key) == expected for key, expected in required.items()
        )
    )


def fetch(url: str) -> str:
    with urllib.request.urlopen(f"{url.rstrip('/')}/metrics", timeout=30) as response:
        return response.read().decode()


def snapshot(url: str, output: Path, settle_attempts: int) -> None:
    names = (
        "vllm:mooncake_store_operation_bytes_total",
        "vllm:mooncake_store_operation_failed_keys_total",
        "vllm:nixl_bytes_transferred_sum",
        "vllm:nixl_num_failed_transfers_total",
        "vllm:nixl_num_failed_notifications_total",
    )

    def counters(text: str) -> tuple[float, ...]:
        return tuple(total(text, name) for name in names)

    current = fetch(url)
    for _ in range(settle_attempts - 1):
        time.sleep(1)
        updated = fetch(url)
        if counters(updated) == counters(current):
            current = updated
            break
        current = updated
    output.write_text(current, encoding="utf-8")


def wait_for_total(
    url: str,
    output: Path,
    name: str,
    labels: dict[str, str],
    minimum: float,
    attempts: int,
) -> None:
    for _ in range(attempts):
        current = fetch(url)
        if total(current, name, labels) >= minimum:
            output.write_text(current, encoding="utf-8")
            return
        time.sleep(1)
    raise RuntimeError(f"{name} did not reach {minimum}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-attempts", type=int, default=1)
    parser.add_argument("--wait-name")
    parser.add_argument("--wait-label", action="append", default=[])
    parser.add_argument("--wait-minimum", type=float, default=1)
    parser.add_argument("--wait-attempts", type=int, default=60)
    args = parser.parse_args()
    if args.wait_name:
        labels = dict(label.split("=", 1) for label in args.wait_label)
        wait_for_total(
            args.url,
            args.output,
            args.wait_name,
            labels,
            args.wait_minimum,
            args.wait_attempts,
        )
        return
    snapshot(args.url, args.output, args.settle_attempts)


if __name__ == "__main__":
    main()
