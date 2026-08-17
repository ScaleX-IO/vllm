from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent
GPU_KV_ROOT = Path(
    os.environ.get("GPU_KV_ROOT", Path.home() / "GPU-KV-vllm-core")
).resolve()
BAM_ROOT = Path(os.environ.get("BAM_ROOT", Path.home() / "bam-master")).resolve()

for required in (
    GPU_KV_ROOT / "include" / "gpu_lsm_kvstore" / "object_store.cuh",
    BAM_ROOT / "include" / "nvm_types.h",
    BAM_ROOT / "build" / "lib" / "libnvm.so",
):
    if not required.exists():
        raise FileNotFoundError(
            f"required GPU-KV/BaM build artifact is missing: {required}"
        )

setup(
    name="vllm-gpu-kv-connector",
    version="0.1.0",
    packages=["gpu_kv_connector"],
    package_dir={"gpu_kv_connector": "."},
    ext_modules=[
        CUDAExtension(
            name="gpu_kv_connector._gpu_kv_native",
            sources=[str(HERE / "csrc" / "bindings.cu")],
            include_dirs=[
                str(GPU_KV_ROOT / "include"),
                str(BAM_ROOT / "include"),
                str(BAM_ROOT / "include" / "freestanding" / "include"),
            ],
            library_dirs=[str(BAM_ROOT / "build" / "lib")],
            libraries=["nvm"],
            runtime_library_dirs=[str(BAM_ROOT / "build" / "lib")],
            define_macros=[("GPU_LSM_KVSTORE_ENABLE_BAM", "1")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-lineinfo",
                    "-D__is_convertible=__simt_is_convertible",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=False)},
)
