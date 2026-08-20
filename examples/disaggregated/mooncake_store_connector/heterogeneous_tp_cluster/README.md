# Mooncake Store heterogeneous-TP cluster tests

These scripts test a Mooncake Store written with a larger prefill tensor
parallel size and read with a smaller, divisible decode tensor parallel size.
The test environment starts Mooncake and then launches the endpoints described
below. Allocate the required GPUs and other resources before running a script.

## Endpoints

Start three vLLM instances with the same model revision and
`PYTHONHASHSEED`:

- producer: `MooncakeStoreConnector`, `kv_both`, larger TP;
- consumer: `MooncakeStoreConnector`, `kv_consumer`, smaller TP;
- reference: smaller TP without a KV connector.

Set `--store-tp-size` on producer and consumer to the producer TP size. Their
pipeline-parallel sizes must match. Use the same Mooncake configuration, model
name, cache prefix, and block size. The producer enables the development cache
API so the workload can clear only its local prefix cache while preserving the
external Store.

For example, after allocating the required GPUs, launch TP4 -> TP2 as follows:

```bash
python launch_server.py --role producer --model "$MODEL_PATH" \
  --served-model-name hetero-tp-test --port 8100 \
  --tensor-parallel-size 4 --store-tp-size 4 \
  --cache-prefix "$CACHE_PREFIX"

python launch_server.py --role consumer --model "$MODEL_PATH" \
  --served-model-name hetero-tp-test --port 8200 \
  --tensor-parallel-size 2 --store-tp-size 4 \
  --cache-prefix "$CACHE_PREFIX"

python launch_server.py --role reference --model "$MODEL_PATH" \
  --served-model-name hetero-tp-test --port 8300 \
  --tensor-parallel-size 2
```

The surrounding runner is responsible for process placement, GPU visibility,
RDMA configuration, Mooncake capacity, unique ports, memory-lock limits, and
checking that each service remains alive while waiting for `/health`.

For multiple prefill TP sizes, replace `--store-tp-size` on every Store
endpoint with the same opt-in configuration:

```text
--enable-store-tp-lcm --prefill-tp-sizes 4 2
```

This selects Store TP `lcm(4, 2) = 4`; it does not change either server's
runtime TP. Add `--save-decode-cache` to the decode endpoint so newly completed
decode blocks are written back in the same Store layout.

## Functional test

```bash
PRODUCER_URL=http://producer:8100 \
CONSUMER_URL=http://consumer:8200 \
REFERENCE_URL=http://reference:8300 \
MODEL=hetero-tp-test TOKENIZER="$MODEL_PATH" \
./run_functional.sh
```

The producer first writes the prompt. The workload waits until the producer can
clear its local prefix cache, which establishes that pinned asynchronous Store
jobs have finished without clearing the external Store. A fresh consumer then
has one opportunity to load the prompt; its `cached_tokens` value is the final
Store-visibility check and cannot be a local-cache retry. The test requires a
block-aligned external hit and, by default, the same continuation as the
no-connector reference. For models
whose low-precision tensor-parallel arithmetic is not token invariant, set
`ALLOW_TOKEN_MISMATCH=1`; cache-hit correctness remains mandatory and the
result records whether tokens matched.

The expected reusable prefix is
`floor((prompt_tokens - 1) / block_size) * block_size`: the final prompt token
has not yet produced a reusable successor KV position. This distinction matters
when the prompt length is exactly block aligned.

## Performance test

```bash
PRODUCER_URL=http://producer:8100 \
CONSUMER_URL=http://consumer:8200 \
MODEL=hetero-tp-test TOKENIZER="$MODEL_PATH" \
REQUESTS=64 CONCURRENCY=16 ./run_performance.sh
```

The producer seeds a set of unique prompts and the local-cache reset procedure
waits for their pinned Store jobs to finish. After independent warmup requests,
the consumer processes every cold and cached prompt exactly once. The groups
use different first blocks, preventing local or external prefix hits from
contaminating the cold arm. The default order is cold then cached and can be
changed with `PERFORMANCE_ORDER=cached-cold` for an order-sensitivity run.

The JSON result is printed before any cache-hit assertion, so a failed run still
records missing cached indices, unexpected cold hits, latency percentiles, wall
time, request and output-token throughput, and cached/cold speedups. It also
reports the cached phase's Mooncake `load_get` bytes, keys, RPC count, failed
keys, cumulative RPC time, RPC throughput, and whole-phase effective
throughput. `rpc_throughput_gib_s` excludes vLLM scheduling and KV placement;
`phase_effective_throughput_gib_s` includes all E2E overhead. Set
`METRICS_SETTLE_SECONDS` if the server needs longer to publish connector
metrics. Run
correctness before performance and provision enough Store capacity so eviction
is not part of this test.

## Single-node RDMA runners

`run_single_node_rdma_layout_matrix.sh` is the portable eight-case layout
matrix for P4 -> D2 and P2 -> D1. It accepts `FEATURE_ROOT`, `TEST_ROOT`,
`VENV_ROOT`, `MODEL_PATH`, `RESULT_ROOT`, `RUN_ID`, `PORT_BASE`, and
`ONLY_CASE` overrides.

`run_single_node_lcm_tp4_tp2_d2.sh` is the focused common-Store-TP test. It
starts P4/HND, P2/NHD, and D2/NHD together on eight GPUs. The test first writes
a prefix at P4, extends it through decode offloading at D2, and reads the
extended prefix at P2. It then repeats the reverse P2 -> D2 -> P4 direction
with a distinct prompt. Both final reads must hit beyond the original prefill
prefix. It accepts the same path, result, run ID, and port overrides as the
matrix runner, plus endpoint GPU-list and GPU-memory-utilization overrides.
