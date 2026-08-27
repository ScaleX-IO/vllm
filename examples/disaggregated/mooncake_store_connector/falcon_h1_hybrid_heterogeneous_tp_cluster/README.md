# Falcon-H1 hybrid heterogeneous-TP boundary test

This test exercises divisible heterogeneous TP sharing for a Falcon-H1 model
through Mooncake Store. Falcon-H1 combines Attention KV cache with Mamba
convolution and recurrent state. The runner uses Store TP 2 and covers:

- TP2 producer to TP1 consumer across a complete decode checkpoint;
- TP1 producer to TP2 consumer across the same checkpoint;
- a short decode that stores Attention KV without advancing the reusable
  Hybrid prefix past the latest Mamba checkpoint;
- a prompt immediately below a local Hybrid page boundary.

With the tested Falcon-H1 configuration, vLLM aligns the local Attention and
Mamba pages to 800 tokens. The default Store match unit is 16 tokens. The
boundary workload therefore verifies a 799-token prompt hits 784 Store tokens
and advances to the 800-token Hybrid checkpoint after decode.

Each case starts a connector-free reference at the consumer TP size. It then
starts a producer and a decode-offloading consumer against a standalone
Mooncake Store. The test checks the exact cached-token boundaries, compares the
first 64 generated token IDs with the same-TP reference, waits for asynchronous
Store jobs, and verifies decode writeback from a fresh lookup.

The runner uses RDMA and starts all services directly. Resource allocation is
left to the surrounding environment. It uses at most three GPUs concurrently
and executes all cases serially.

Required variables:

- `FEATURE_ROOT`: vLLM worktree containing the feature;
- `MODEL_PATH`: local Falcon-H1 model path;
- `VENV_ROOT`: Python environment containing vLLM and test dependencies;
- `MOONCAKE_CLIENT_BIN`: standalone `mooncake_client` executable;
- `RDMA_DEVICE`: Mooncake RDMA device name.

Optional variables include `TP2_GPUS`, `TP1_GPUS`, `HOST_IP`, `PORT_BASE`,
`RESULT_ROOT`, `STORE_SIZE`, `GPU_MEMORY_UTILIZATION`, `PREFIX_MATCH_UNIT`,
`STORE_TP_SIZE`, and the workload-size variables documented in the runner.

```bash
FEATURE_ROOT=/path/to/vllm-feature \
MODEL_PATH=/path/to/Falcon-H1-0.5B-Instruct \
VENV_ROOT=/path/to/venv \
MOONCAKE_CLIENT_BIN=/path/to/mooncake_client \
RDMA_DEVICE=mlx5_0 \
./run_single_node_rdma_boundaries.sh
```

The result directory contains the tested feature revision, configuration,
per-case JSON results, service logs, RDMA evidence, and a `PASS` marker.
