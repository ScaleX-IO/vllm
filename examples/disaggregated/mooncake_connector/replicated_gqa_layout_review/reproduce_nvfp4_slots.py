# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce wrong K/V slot selection for FlashInfer NVFP4 head slots."""

from layout_probe import run

if __name__ == "__main__":
    run(
        (
            ("gqa_fan_out_control", "dense", 1, 8, 2, "head", "correct"),
            ("base_nvfp4_slots", "nvfp4", 1, 8, 2, "base", "reject"),
            ("nvfp4_slots", "nvfp4", 1, 8, 2, "head", "corrupt"),
        ),
        "nvfp4_slots_results.json",
    )
