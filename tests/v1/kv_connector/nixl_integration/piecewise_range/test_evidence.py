# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from argparse import Namespace
from pathlib import Path

import pytest

from .evidence import build_evidence, parse_plans


def _metric(name: str, value: float, labels: str = "") -> str:
    suffix = f"{{{labels}}}" if labels else ""
    return f"{name}{suffix} {value}\n"


def _args(tmp_path: Path, case: str, mode: str = "pull") -> Namespace:
    response = {
        "request_id": "req-1",
        "base_tokens": 64,
        "suffix_tokens": 32,
        "prompt_tokens": 96,
        "usage": {"prompt_tokens_details": {"cached_tokens": 96}},
        "output_token_ids": [1, 2],
    }
    boundary = 64 if case == "piecewise" else 80
    plan = {
        "request_id": "req-1",
        "entries": [
            {
                "connector": "MooncakeStoreConnector",
                "start_token": 0,
                "end_token": boundary,
                "is_terminal": False,
            },
            {
                "connector": (
                    "NixlPullConnector" if mode == "pull" else "NixlPushConnector"
                ),
                "start_token": boundary,
                "end_token": 96,
                "is_terminal": True,
            },
        ],
    }
    nixl_metric = _metric(
        "vllm:nixl_bytes_transferred_sum",
        {"piecewise": 40, "miss": 100, "full": 10}[case],
    )
    files = {}
    for name, content in {
        "response": json.dumps(response),
        "reference": json.dumps(response),
        "decode_log": f"DEBUG KV_LOAD_PLAN {json.dumps(plan)}\n"
        if case != "miss"
        else "",
        "prefill_metrics_before": "",
        "decode_metrics_before": "",
        "prefill_metrics_after": nixl_metric if mode == "push" else "",
        "decode_metrics_after": (
            _metric(
                "vllm:mooncake_store_operation_bytes_total",
                100 if case != "miss" else 0,
                'operation="load_get",status="ok"',
            )
            + (nixl_metric if mode == "pull" else "")
        ),
    }.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        files[name] = path
    return Namespace(
        case=case,
        mode=mode,
        request_id="req-1",
        base_tokens=64,
        block_size=16,
        output=tmp_path / "evidence.json",
        **files,
    )


@pytest.mark.parametrize("case", ["piecewise", "miss", "full"])
@pytest.mark.parametrize("mode", ["pull", "push"])
def test_evidence_contract(tmp_path: Path, case: str, mode: str):
    evidence = build_evidence(_args(tmp_path, case, mode))
    assert evidence["case"] == case


def test_piecewise_plan_is_request_scoped():
    log = "\n".join(
        [
            'KV_LOAD_PLAN {"request_id":"other","entries":[]}',
            'KV_LOAD_PLAN {"request_id":"cmpl-wanted-0-a1b2","entries":[1]}',
        ]
    )
    assert parse_plans(log, "wanted") == [
        {"request_id": "cmpl-wanted-0-a1b2", "entries": [1]}
    ]
