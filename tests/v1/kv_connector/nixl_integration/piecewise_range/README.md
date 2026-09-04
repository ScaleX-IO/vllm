# NIXL piecewise range-load E2E

This suite verifies that a decode request can load `[0, S)` from Mooncake
Store while NIXL transfers only `[S, N)`. Request success alone is not a pass:
each case records the connector load plan, cached-token count, Store bytes,
NIXL bytes, failure counters, and reference output in `evidence.json`.
`response.json` also records one request duration for diagnostics. This
single-node suite does not benchmark TTFT or network overlap.

## PR-gate matrix

| Case | Store | Expected NIXL |
|---|---|---|
| `piecewise-pull` | `[0, S)` | pull `[S, N)` |
| `piecewise-push` | `[0, S)` | push `[S, N)` |
| `store-miss` | empty | full `[0, N)` pull |
| `store-full` | fully populated | terminal block `[N-B, N)` |

The matrix runs serially. Each case has a fresh Store namespace and a distinct
160-port job window. A cold reference server exits before the persistent Store seeder,
prefiller, and decoder occupy three GPUs. Every server is an independent
`srun --exclusive --exact` step; do not set `CUDA_VISIBLE_DEVICES` manually.
The Seeder extends the prompt beyond the desired Store boundary because its last
block is not saved as a reusable prefix. `piecewise` needs one extra base token;
`store-full` needs one scheduler block appended to the suffix so the last target
block uses its regular key instead of a tail key.

Mooncake Store intentionally caps lookup at `num_tokens - 1`, then aligns down
to its lookup unit. Therefore even a fully populated Store leaves one scheduler
block (`B`) to the terminal NIXL connector; the comparison checks
`full < piecewise < miss` NIXL bytes instead of expecting zero.

## Run

Use a single-node allocation with at least three GPUs and enough CPU cores:

```bash
salloc --no-shell --partition=h20 --gres=gpu:h20:3 \
  --cpus-per-task=48 --time=12:00:00

JOB_ID=<job-id> \
VLLM_ROOT=/path/to/the/integration-worktree \
bash tests/v1/kv_connector/nixl_integration/piecewise_range/run_matrix.sh
```

The worktree must contain the #54240 framework, #54421 Mooncake range support,
and the NIXL range-load commit. Before running, link compatible editable-build
artifacts into a fresh worktree and verify `vllm._custom_ops` on the allocated
compute node. `run_case.sh` also verifies `vllm`, `_custom_ops`, and `nixl`
before starting services.

Useful overrides include `MODEL_PATH`, `BASE_TOKENS`, `SUFFIX_TOKENS`,
`KV_CACHE_MEMORY_BYTES`, `BLOCK_SIZE`, `RESULT_ROOT`, and
`VLLM_SERVE_EXTRA_ARGS`. The H20 defaults use the user-space OpenSSL 3 runtime
only inside compute-node steps. Do not add the broader user-local library path;
it can shadow the system TLS libraries.

Each case writes `manifest.json`, request/reference JSON, before/after metrics,
service logs, `evidence.json`, and `PASS`. On failure, the exit trap terminates
only the recorded Slurm step process groups. The script removes only its three
known lookup IPC sockets after checking that its ports and namespace are unique.
