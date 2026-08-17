from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_kv_connector.catalog import SQLiteObjectCatalog
from gpu_kv_connector.config import GPUKVConfig
from gpu_kv_connector.connector import (
    GPUKVConnectorScheduler,
    GPUKVConnectorWorker,
    _physical_superrequest_limits,
)
from gpu_kv_connector.hashing import (
    OBJECT_ID_BYTES,
    split_object_id,
)
from gpu_kv_connector.lifecycle import StoreCompletionTracker
from gpu_kv_connector.metadata import GPUKVTransfer


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


def test_catalog_full_hit_recomputes_last_token_and_loads_its_block(tmp_path) -> None:
    config = GPUKVConfig.from_vllm(
        _vllm_config({"catalog_path": str(tmp_path / "catalog.sqlite3")})
    )
    scheduler = GPUKVConnectorScheduler(_vllm_config({}), config)
    block_hashes = [bytes([index]) * OBJECT_ID_BYTES for index in range(1, 3)]
    scheduler.catalog.mark_rank_ready(block_hashes, rank=0)
    scheduler.catalog.mark_rank_ready(block_hashes, rank=1)
    request = SimpleNamespace(
        request_id="request", block_hashes=block_hashes, num_tokens=32
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (31, False)
    blocks = SimpleNamespace(get_block_ids=lambda: [[10, 11]])
    scheduler.update_state_after_alloc(request, blocks, 31)
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[], new_block_ids=[], resumed_req_ids=set()
            ),
        )
    )

    assert metadata.loads["request"].object_ids == tuple(block_hashes)
    assert metadata.loads["request"].block_ids == (10, 11)
    scheduler.close()


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
    assert config.prefetch_layers == 2
    assert config.ready_cache_entries == 65_536
    assert config.fuse_kv_planes is True
    assert _physical_superrequest_limits(config, 4096) == (128, 16)
    assert _physical_superrequest_limits(config, 32 * 1024) == (16, 2)
    assert config.disk_start_for_rank(1) == config.disk_start_page + 4096


def test_config_rejects_ambiguous_boolean() -> None:
    with pytest.raises(ValueError, match="reset_catalog must be a boolean"):
        GPUKVConfig.from_vllm(_vllm_config({"reset_catalog": "sometimes"}))


def test_ready_cache_invalidates_failed_gpu_blocks(tmp_path) -> None:
    config = GPUKVConfig.from_vllm(
        _vllm_config({"catalog_path": str(tmp_path / "catalog.sqlite3")})
    )
    scheduler = GPUKVConnectorScheduler(_vllm_config({}), config)
    block_hashes = [bytes([index]) * OBJECT_ID_BYTES for index in range(1, 3)]
    scheduler.catalog.mark_rank_ready(block_hashes, rank=0)
    scheduler.catalog.mark_rank_ready(block_hashes, rank=1)
    request = SimpleNamespace(
        request_id="request", block_hashes=block_hashes, num_tokens=33
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, False)
    scheduler.catalog.clear_rank(block_hashes, rank=0)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, False)

    scheduler._loaded_objects_by_request[request.request_id] = {
        10: block_hashes[0],
        11: block_hashes[1],
    }
    scheduler.update_connector_output(
        SimpleNamespace(finished_sending=None, invalid_block_ids={10})
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.close()


def test_store_filter_skips_catalog_for_known_objects(tmp_path) -> None:
    config = GPUKVConfig.from_vllm(
        _vllm_config(
            {
                "catalog_path": str(tmp_path / "catalog.sqlite3"),
                "ready_cache_entries": 2,
            }
        )
    )
    scheduler = GPUKVConnectorScheduler(_vllm_config({}), config)
    object_ids = [bytes([index]) * OBJECT_ID_BYTES for index in range(1, 4)]
    scheduler._cache_ready(object_ids)
    assert list(scheduler._ready_cache) == object_ids[-2:]

    request = SimpleNamespace(
        request_id="request",
        block_hashes=object_ids[-2:],
        num_tokens=32,
        num_computed_tokens=0,
    )
    scheduler._requests[request.request_id] = request
    scheduler._request_block_ids[request.request_id] = [10, 11]
    scheduler._next_store_index[request.request_id] = 0

    def unexpected_query(*args, **kwargs):
        raise AssertionError("ready objects must not query SQLite")

    scheduler.catalog.ready_set = unexpected_query
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[request.request_id],
            new_block_ids=[()],
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens={request.request_id: 32},
    )
    assert scheduler._build_stores(output) == {}
    scheduler.close()


def test_store_operation_defers_native_reservation() -> None:
    object_id = bytes([1]) * OBJECT_ID_BYTES
    transfer = GPUKVTransfer((object_id,), (10,))

    operation = GPUKVConnectorWorker._make_store_operation("request", transfer)

    assert operation.object_ids == (object_id,)
    assert operation.batches == []
    assert operation.store_transfer is transfer
    assert operation.metadata_ready is None


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
