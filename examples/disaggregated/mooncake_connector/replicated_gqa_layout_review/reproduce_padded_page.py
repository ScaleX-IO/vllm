# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce head misplacement when the KV page carries tail padding."""

from layout_probe import run

if __name__ == "__main__":
    run(
        (
            ("gqa_fan_out_control", "dense", 1, 8, 2, "head", "correct"),
            ("gqa_fan_in_control", "dense", 8, 2, 0, "head", "correct"),
            ("base_padded_page", "padded", 1, 8, 2, "base", "reject"),
            ("padded_page", "padded", 1, 8, 2, "head", "corrupt"),
        ),
        "padded_page_results.json",
    )
