# Mooncake Store decode-offloading cluster checks

These scripts validate decode KV persistence independently of a cluster
scheduler. Allocate nodes and GPUs with your site's normal tooling, then start
the required vLLM and Mooncake processes on those compute nodes.

The tests contain no scheduler-specific assumptions, fixed usernames, model
paths, GPU models, hostnames, RDMA devices, or ports.

## Required endpoints

Start the following endpoints with the same Mooncake Store and cache prefix:

- `PROBE_URL`: generates the prompt and decode KV under test.
- `CACHED_URL`: a separate vLLM instance that has not served the probe and can
  load its KV from the Store.
- `REFERENCE_URL`: vLLM without a KV connector, used to compare continuation.
- `TARGET_URLS`: one or more endpoints used by the pressure phase.

For 1P1D, put `store_pd_proxy.py` in front of the prefill producer and decode
consumer and use that proxy as `PROBE_URL`. For the non-PD test, use one mixed
instance as `PROBE_URL` and another fresh mixed instance as `CACHED_URL`.

`launch_server.py` builds the connector configuration for these roles:
`prefill`, `decode`, `mixed`, `verifier`, and `reference`. Process placement and
GPU selection remain the responsibility of the surrounding cluster runner.

## Run

```bash
export MODEL=decode-offload-test-model
export TOKENIZER=/models/Qwen3-32B-FP8
export PROBE_URL=http://prefill-proxy:8000
export CACHED_URL=http://fresh-store-reader:8001
export REFERENCE_URL=http://reference:8002
export TARGET_URLS="http://prefill-proxy:8000"

./run_scenario.sh
```

The defaults issue 64 requests at concurrency 16 with 128 output tokens. The
runner first proves that the fresh Store reader hits at least one decode block
and produces the same continuation as the reference, then runs the pressure
phase. Override `REQUESTS`, `CONCURRENCY`, `OUTPUT_TOKENS`, or
`PROMPT_REPETITIONS` as needed.

The same runner covers all three placements:

1. single-node 1P1D;
2. two-node 1P1D;
3. non-PD mixed serving.

Only endpoint placement changes; the correctness and pressure checks stay
identical.
