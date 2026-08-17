from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_kv_connector.catalog import SQLiteObjectCatalog
from gpu_kv_connector.config import GPUKVConfig
from gpu_kv_connector.hashing import (
    OBJECT_ID_BYTES,
    split_object_id,
)
from gpu_kv_connector.lifecycle import StoreCompletionTracker


def _vllm_config(extra: dict[str, object]) -> SimpleNamespace:
    model_config = SimpleNamespace(
        model="test/model",
        dtype="float16",
        hf_config=SimpleNamespace(rope_scaling=None, rope_theta=500000.0),
    )
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        cache_config=SimpleNamespace(block_size=16, cache_dtype="auto"),
        model_config=model_config,
    )


def test_native_vllm_hash_round_trips() -> None:
    block_hash = bytes(range(32))
    identity = split_object_id(block_hash)

    assert len(block_hash) == OBJECT_ID_BYTES
    assert block_hash == b"".join(value.to_bytes(8, "little") for value in identity)


def test_catalog_requires_every_tensor_parallel_rank(tmp_path) -> None:
    path = tmp_path / "catalog.sqlite3"
    first = SQLiteObjectCatalog(str(path), reset=True)
    second = SQLiteObjectCatalog(str(path))
    ids = [bytes([index]) * OBJECT_ID_BYTES for index in range(1, 4)]

    first.mark_rank_ready(ids, rank=0)
    assert first.longest_ready_prefix(ids, full_mask=0b11) == 0
    second.mark_rank_ready(ids[:2], rank=1)
    assert first.longest_ready_prefix(ids, full_mask=0b11) == 2
    assert first.ready_set(ids, full_mask=0b11) == set(ids[:2])

    second.clear_rank([ids[0]], rank=1)
    assert first.longest_ready_prefix(ids, full_mask=0b11) == 0
    first.close()
    second.close()


def test_config_parses_false_and_missing_optional_model_fields(tmp_path) -> None:
    config = GPUKVConfig.from_vllm(
        _vllm_config(
            {
                "catalog_path": str(tmp_path / "catalog.sqlite3"),
                "capacity_pages": "4096",
                "reset_catalog": "false",
            }
        )
    )

    assert config.reset_catalog is False
    assert config.capacity_pages == 4096
    assert config.tp_size == 2
    assert config.disk_start_for_rank(1) == config.disk_start_page + 4096


def test_config_rejects_ambiguous_boolean() -> None:
    with pytest.raises(ValueError, match="reset_catalog must be a boolean"):
        GPUKVConfig.from_vllm(_vllm_config({"reset_catalog": "sometimes"}))


def test_store_completion_before_request_finish_is_retained() -> None:
    tracker = StoreCompletionTracker()
    tracker.started("request")
    tracker.completed("request")
    assert tracker.take_ready() == set()

    tracker.mark_requests_finished({"request"})
    assert tracker.take_ready() == {"request"}


def test_request_finish_waits_for_every_store() -> None:
    tracker = StoreCompletionTracker()
    tracker.started("request")
    tracker.started("request")
    tracker.mark_requests_finished({"request", "no-store"})
    tracker.completed("request")
    assert tracker.pending_count("request") == 1
    assert tracker.take_ready() == set()

    tracker.completed("request")
    assert tracker.take_ready() == {"request"}
    with pytest.raises(RuntimeError, match="no matching start"):
        tracker.completed("request")
