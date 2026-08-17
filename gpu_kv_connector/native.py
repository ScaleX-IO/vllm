from __future__ import annotations

import torch  # noqa: F401  # Load libtorch/libc10 before the CUDA extension.

try:
    from gpu_kv_connector import _gpu_kv_native
except ImportError as error:
    raise ImportError(
        "GPU-KV native extension is not built. Run "
        "`python -m pip install -e ./gpu_kv_connector --no-build-isolation` "
        "after building GPU-KV and BaM."
    ) from error

ObjectStore = _gpu_kv_native.ObjectStore

__all__ = ["ObjectStore"]
