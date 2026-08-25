# Hybrid heterogeneous-TP Mooncake Store test

This test verifies that a hybrid Full Attention and Mamba/GDN model can reuse
one Mooncake Store namespace across divisible TP sizes in both directions.

For each TP4/TP2 direction, the runner first starts a connector-free reference
at the consumer TP size. It then starts a producer and a consumer with Store TP
4. The consumer must load the producer's prompt, match the same-TP reference,
and write its decode KV and recurrent state back. After its asynchronous Store
jobs finish, the producer must hit beyond the original prompt when given the
extended token sequence.

The default workload uses a 1601-token prompt and generates 816 tokens. With
the tested hybrid model's `align` cache mode, the initial reusable checkpoint is
at token 1600. Decode crosses the next checkpoint at token 2400, so the final
lookup proves that both the Full Attention KV block and the aligned GDN state
were written back. `PROMPT_TOKENS` and `OUTPUT_TOKENS` may be adjusted for a
model with a different aligned checkpoint interval.

All Store endpoints use `--prefix-match-unit 16` by default. Heterogeneous TP
endpoints must use the same value, and it must divide the normalized Attention
Store chunk size. Otherwise the connector intentionally falls back to an
isolated rank-local namespace.

The first 64 generated token IDs are compared with a connector-free reference
at the consumer TP size. The longer generation is used to cross a checkpoint,
not to require bit-identical decoding across TP sizes; the tested TP2 and TP4
reference runs naturally diverge after that prefix.

The runner uses RDMA and starts all services directly; resource allocation is
left to the surrounding environment. At most six GPUs are used concurrently.
It requires these variables:

- `FEATURE_ROOT`: vLLM worktree containing the feature;
- `MODEL_PATH`: local hybrid model path;
- `VENV_ROOT`: Python environment containing vLLM and test dependencies;
- `MOONCAKE_CLIENT_BIN`: standalone `mooncake_client` executable;
- `RDMA_DEVICE`: Mooncake RDMA device name.

Optional variables include `TP4_GPUS`, `TP2_GPUS`, `HOST_IP`, `PORT_BASE`,
`RESULT_ROOT`, `STORE_SIZE`, `GPU_MEMORY_UTILIZATION`, `OUTPUT_TOKENS`,
`REFERENCE_OUTPUT_TOKENS`, `PROMPT_TOKENS`, `PROMPT_REPETITIONS`,
`PREFIX_MATCH_UNIT`, `STORE_TP_SIZE`, `EXPECTED_DECODE_CACHED_TOKENS`, and
`EXPECTED_EXTENDED_CACHED_TOKENS`.

The result directory records the exact feature revision and test configuration
in `test-config.txt`, per-direction JSON results, service logs, and RDMA
transfer evidence.

`run_hybrid_attention_early_store.sh` is a focused workload for already-running
reference, producer, and consumer endpoints. Its default 401-token Decode stops
before the next complete Mamba/GDN checkpoint: Attention KV is written early,
but the externally reusable prefix correctly remains at 1600 tokens. The main
matrix runner crosses the next checkpoint and verifies full Decode writeback.

```bash
FEATURE_ROOT=/path/to/vllm-feature \
MODEL_PATH=/path/to/model \
VENV_ROOT=/path/to/venv \
MOONCAKE_CLIENT_BIN=/path/to/mooncake_client \
RDMA_DEVICE=mlx5_0 \
./run_single_node_rdma.sh
```
