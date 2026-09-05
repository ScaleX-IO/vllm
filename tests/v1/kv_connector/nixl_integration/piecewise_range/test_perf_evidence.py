# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from argparse import Namespace
from pathlib import Path

from .launch_server import _pd_config
from .perf_evidence import _network
from .perf_workload import _percentile


def test_percentile_uses_nearest_rank():
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.9) == 4.0


def test_legacy_decode_disables_store_lookup():
    config = _pd_config(
        Namespace(
            role="decode",
            mode="pull",
            scenario="legacy",
            cache_prefix="test",
            lookup_rpc_port=1234,
            store_tp_size=1,
        )
    )
    extra = config["kv_connector_extra_config"]
    assert "load_policy" not in extra
    assert extra["connectors"][0]["kv_connector_extra_config"]["enable_lookup"] is False


def test_network_detects_request_window_overlap(tmp_path: Path):
    def sample(time_ns: int, bond0: int, bond1: int) -> dict:
        return {
            "time_ns": time_ns,
            "counters": {
                "mlx5_bond_0": {
                    "port_xmit_data": bond0,
                    "port_rcv_data": bond0 * 2,
                },
                "mlx5_bond_1": {
                    "port_xmit_data": bond1,
                    "port_rcv_data": bond1 * 2,
                },
            },
        }

    path = tmp_path / "network.json"
    path.write_text(
        json.dumps(
            {
                "counter_unit_bytes": 4,
                "samples": [
                    sample(0, 0, 0),
                    sample(20_000_000, 300_000, 400_000),
                    sample(40_000_000, 300_000, 700_000),
                ],
            }
        )
    )
    result = _network(path, threshold=1_000_000, counter="port_xmit_data")
    assert result["overlap_intervals"] == 1
    assert result["overlap_fraction_of_active"] == 0.5
    assert result["counter"] == "port_xmit_data"
    assert result["bytes"] == {
        "mlx5_bond_0": 1_200_000,
        "mlx5_bond_1": 2_800_000,
    }
    received = _network(path, threshold=2_000_000, counter="port_rcv_data")
    assert received["counter"] == "port_rcv_data"
    assert received["bytes"] == {
        "mlx5_bond_0": 2_400_000,
        "mlx5_bond_1": 5_600_000,
    }
