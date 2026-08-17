from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.kv_connector_model_runner_mixin as runner_mixin
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheTensor,
)
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
)
from vllm.v1.worker.utils import AttentionGroup


class _NHDBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: object = None,
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (2, 0, 1, 3, 4, 5)
        return (1, 0, 2, 3, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cross_layer_allocation_honors_connector_alignment(monkeypatch) -> None:
    alignment = 64 * 1024
    monkeypatch.setattr(runner_mixin, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(
        runner_mixin,
        "get_kv_transfer_group",
        lambda: SimpleNamespace(required_kv_cache_alignment=alignment),
    )

    num_blocks = 3
    num_layers = 2
    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * num_blocks,
                shared_by=[f"layer.{layer}"],
            )
            for layer in range(num_layers)
        ],
        kv_cache_groups=[],
    )
    kv_caches, cross_layer, backend = (
        KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
            kv_cache_config=config,
            attn_groups=[
                [
                    AttentionGroup(
                        backend=_NHDBackend,
                        layer_names=[],
                        kv_cache_spec=spec,
                        kv_cache_group_id=0,
                    )
                ]
            ],
            cache_dtype=torch.float16,
            device=torch.device("cuda"),
            kernel_block_sizes=[spec.block_size],
        )
    )

    assert backend is _NHDBackend
    assert cross_layer.data_ptr() % alignment == 0
    assert cross_layer.shape == (
        num_blocks,
        num_layers,
        2,
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
    )
    assert cross_layer.numel() * cross_layer.element_size() == (
        spec.page_size_bytes * num_blocks * num_layers
    )
    assert set(kv_caches) == {"layer.0", "layer.1"}
