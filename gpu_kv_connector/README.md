# GPU-KV connector for vLLM

This external vLLM 0.17 connector stores reusable prefix KV blocks in the
GPU-KV object layer. vLLM computes chained SHA-256 block hashes on the CPU
scheduler; GPU-KV uses each native 256-bit hash unchanged as the object identity
and maps it to a physical SSD descriptor with its GPU-resident LSM index. BaM
transfers K/V planes directly between an
aligned cross-layer HBM allocation and NVMe. Host DRAM is not a data tier.

One immutable object contains every layer's K and V plane for one vLLM token
block. Objects reserved together use a plane-major SSD stripe, so physical I/O
can be grouped after logical lookup without changing prefix-cache granularity.
The SQLite catalog records only whether every tensor-parallel rank completed
an object; it never stores SSD locations.

The read path resolves native vLLM SHA-256 identities on the GPU, forms bounded
physical runs only after resolution, and pipelines a fixed two-layer SSD read
window with model execution. K and V use one layer-level native call. Independent vLLM
requests remain independent I/O operations: combining unrelated requests was
measured to reduce four-request throughput because their SSD stripes were not
physically contiguous.

Writes are deferred until a step without external reads, request completion,
or preemption. Descriptor reservation is deferred with the write, so its
device-wide metadata synchronization cannot delay a mixed read/compute step.

## Build

Build GPU-KV and BaM first, then install the connector and native extension in
the vLLM environment. The editable install makes the external connector
importable regardless of the directory from which `vllm serve` is launched.

```bash
cd $HOME/vllm-gpukv
GPU_KV_ROOT=$HOME/GPU-KV-vllm-core \
BAM_ROOT=$HOME/bam-master \
python -m pip install -e ./gpu_kv_connector --no-build-isolation
```

BaM requires the target NVMe IOMMU group to use identity mapping. NVIDIA must
also allow non-root I/O-memory peer mappings. Configure the NVIDIA module for a
maintenance reboot with:

```text
options nvidia NVreg_RegistryDwords="PeerMappingOverride=1;"
```

Without that module option, BaM applications must run as root.

## Serve

Bind the dedicated NVMe to `/dev/libnvm0`, then start vLLM with prefix caching:

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector":"GPUKVConnector",
    "kv_role":"kv_both",
    "kv_connector_module_path":"gpu_kv_connector.connector",
    "kv_connector_extra_config":{
      "device_path":"/dev/libnvm0",
      "disk_start_page":1048576,
      "capacity_pages":400000000,
      "max_objects":1048576,
      "max_batch":8192,
      "queue_depth":64,
      "num_queues":16,
      "prefetch_layers":2,
      "reset_catalog":true
    }
  }'
```

The catalog and SSD extent scope the cache to one compatible model and KV
layout. Use a distinct `catalog_path` and a disjoint
`disk_start_page`/`capacity_pages` range for concurrent or incompatible
servers. `capacity_pages` is the capacity of each
tensor-parallel rank; rank `r` uses the consecutive range beginning at
`disk_start_page + r * capacity_pages`.

The primary evaluation target is repeated long-prefix serving with
Llama-3-8B-Instruct on LEval and LooGLE. Compare against vLLM recomputation and
LMCache SSD/GDS under the same prefix hit trace, request arrival process, TTFT,
ITL, throughput, and SSD configuration.

## Component benchmark

The included benchmark fills distinct block-aligned prefixes and reloads them
after the configured HBM cache has evicted them:

```bash
export GPUKV_MODEL=/path/to/local/model
sudo -E ./gpu_kv_connector/benchmarks/serve_prefix_benchmark.sh

python gpu_kv_connector/benchmarks/benchmark_prefix_reload.py \
  --model "$GPUKV_MODEL" --require-external-hits

# Run the fixed one-factor component matrix (requires BaM privileges).
sudo -E ./gpu_kv_connector/benchmarks/run_component_ablation.sh
```

Use `GPUKV_MAX_SUPERREQUEST_OBJECTS=0`, `GPUKV_FUSE_KV_PLANES=false`,
`GPUKV_PREFETCH_LAYERS=1`, or `GPUKV_READY_CACHE_ENTRIES=0` for one-factor
ablations. `GPUKV_BENCH_MODE=recompute` starts the same vLLM configuration
without the external connector. Stop one server before starting the next so
only one process owns the BaM controller.

The small Qwen2.5-0.5B smoke workload is suitable for correctness and component
overhead checks, but its prompt recomputation is faster than SSD reload. It is
not evidence that GPU-KV improves end-to-end performance for larger models;
that claim requires the long-prefix target workloads above.
