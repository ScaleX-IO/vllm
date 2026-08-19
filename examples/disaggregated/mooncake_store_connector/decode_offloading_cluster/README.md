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

All vLLM processes that share Store keys must use the same `PYTHONHASHSEED`.
`launch_server.py` defaults it to `0` before starting vLLM. An OpenAI response
can arrive before an asynchronous Store PUT is visible, so Store-only 1P1D
setups should give the proxy a short `--prefill-store-wait` and set
`REQUIRE_PROMPT_CACHE_HIT=1`. The probe then verifies that D actually loaded
the block-aligned prompt from the Store.

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
export REQUIRE_PROMPT_CACHE_HIT=1  # Store-only 1P1D

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

## Full P-D-Store-P loop

Launch the prefiller and decoder with `--transfer-mode direct`. This wraps
`MooncakeConnector` and `MooncakeStoreConnector` in `MultiConnector`: P writes
the prompt to the Store while transferring it directly to D, and D writes the
completed decode blocks to the same Store. Launch the fresh `verifier` with the
same KV-cache layout as P and D. Store entries contain raw KV bytes and must be
read with the same layout that wrote them. Then run:

```bash
export MODEL=decode-offload-test-model
export TOKENIZER=/models/Qwen3-32B-FP8
export PD_PROXY_URL=http://prefill-proxy:8000
export NEXT_PREFILL_URL=http://fresh-prefill-reader:8001

./run_full_loop.sh
```

The final request contains the original prompt plus D's output. The test
requires the fresh P-side reader to hit at least one full decode block beyond
the prompt boundary and to generate the last token recorded from D. This
verifies both Store persistence and continuation correctness without a separate
reference model.

The fresh reader must not have processed the probe locally. It may be started
after the producer is stopped to reuse that GPU, but keep the standalone Store
and decoder alive until verification completes. Store completion is established
by the fresh reader's actual prefix hit, not by `done_sending`, which is unused
by the pinned store-job lifecycle.
