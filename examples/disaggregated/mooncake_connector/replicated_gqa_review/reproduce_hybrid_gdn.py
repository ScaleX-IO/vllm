# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce successful transfers containing incorrectly sharded GDN state."""

from transfer_probe import run

if __name__ == "__main__":
    run(
        ("gqa_fan_in", "gqa_fan_out", "base_hybrid_gdn", "hybrid_gdn"),
        "hybrid_gdn_results.json",
    )
