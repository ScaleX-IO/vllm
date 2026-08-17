"""External vLLM connector backed by the GPU-KV object store."""

from typing import Any

__all__ = ["GPUKVConnector"]


def __getattr__(name: str) -> Any:
    # Keep hashing, catalog, and configuration tests usable without importing
    # torch/vLLM or loading the native extension.
    if name == "GPUKVConnector":
        from gpu_kv_connector.connector import GPUKVConnector

        return GPUKVConnector
    raise AttributeError(name)
