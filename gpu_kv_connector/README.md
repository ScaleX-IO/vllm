# GPU-KV connector for vLLM

This external vLLM 0.17 connector stores reusable prefix KV blocks in the
GPU-KV object layer. vLLM computes chained block hashes on the CPU scheduler;
GPU-KV maps a namespaced 192-bit object identity to a physical SSD descriptor
with its GPU-resident LSM index. BaM transfers K/V planes directly between an
aligned cross-layer HBM allocation and NVMe. Host DRAM is not a data tier.

One immutable object contains every layer's K and V plane for one vLLM token
block. Objects reserved together use a plane-major SSD stripe, so physical I/O
can be grouped after logical lookup without changing prefix-cache granularity.
The SQLite catalog records only whether every tensor-parallel rank completed
an object; it never stores SSD locations.

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

Use a distinct `catalog_path` and a disjoint `disk_start_page`/`capacity_pages`
range for concurrent servers. `capacity_pages` is the capacity of each
tensor-parallel rank; rank `r` uses the consecutive range beginning at
`disk_start_page + r * capacity_pages`.

The primary evaluation target is repeated long-prefix serving with
Llama-3-8B-Instruct on LEval and LooGLE. Compare against vLLM recomputation and
LMCache SSD/GDS under the same prefix hit trace, request arrival process, TTFT,
ITL, throughput, and SSD configuration.
