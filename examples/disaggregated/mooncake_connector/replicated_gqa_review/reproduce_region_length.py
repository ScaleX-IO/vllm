# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce an incompatible half-written KV region reported as successful."""

from transfer_probe import run

if __name__ == "__main__":
    run(
        ("gqa_fan_out", "base_mismatched_region", "mismatched_region"),
        "region_length_results.json",
    )
