#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import time
from pathlib import Path


def _read(device: str, counter: str) -> int:
    path = Path("/sys/class/infiniband") / device / "ports/1/counters" / counter
    return int(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--interval", type=float, default=0.02)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    counters = ("port_xmit_data", "port_rcv_data")
    deadline = time.monotonic() + args.max_seconds
    samples = []
    while not args.stop_file.exists() and time.monotonic() < deadline:
        samples.append(
            {
                "time_ns": time.time_ns(),
                "counters": {
                    device: {counter: _read(device, counter) for counter in counters}
                    for device in args.devices
                },
            }
        )
        if len(samples) == 1:
            args.ready_file.touch()
        time.sleep(args.interval)
    samples.append(
        {
            "time_ns": time.time_ns(),
            "counters": {
                device: {counter: _read(device, counter) for counter in counters}
                for device in args.devices
            },
        }
    )
    args.output.write_text(
        json.dumps({"counter_unit_bytes": 4, "samples": samples}), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
