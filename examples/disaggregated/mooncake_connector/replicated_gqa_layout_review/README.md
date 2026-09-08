# Replicated-GQA layout probes for PR #52516

Two independently runnable probes reproduce the review findings at
[`d61e44b85b`](https://github.com/vllm-project/vllm/pull/52516):

- `reproduce_padded_page.py`: when the KV page carries tail padding
  (`page_size_padded`), the new per-head plan derives the head stride from the
  padded page, misplaces heads, and reports success.
- `reproduce_nvfp4_slots.py`: FlashInfer NVFP4 stores K slots and V slots
  separately (`num_head_slots = 2 * num_kv_heads`), so one head's K/V is not a
  contiguous run; the new per-head plan copies the wrong slots and reports
  success.

Both cases are `Hkv=4, TP1 -> TP8`, consumer rank 2. The base commit
(`5db652225f`) rejects both before any transfer because the region lengths do
not satisfy the raw TP ratio; the PR's effective-TP validation accepts them.
Both use `layout_probe.py`, which runs the actual `register_kv_caches`,
`send_kv_to_decode`, `_build_transfer_params`, and `_send_blocks` on host
memory with the Mooncake transport replaced by bounds-checked `memmove`.
No GPU, RDMA device, or model kernel is needed. Each script includes ordinary
GQA controls and a base control that executes the exact base planner,
validator, and `send_kv_to_decode`. **Exit status zero with a final `REPRODUCED:` line
means the defect was reproduced and the controls passed; these are reproduction
scripts, not tests expecting a corrected implementation.**

## Run

Set `VLLM_REPO` to a checkout of the PR's exact head, with its base commit
available in Git:

```bash
git fetch https://github.com/vllm-project/vllm.git refs/pull/52516/head
git worktree add --detach /path/to/pr52516 d61e44b85b04cb9b3c90938e57a1ba72a98cab5d
git -C /path/to/pr52516 fetch --depth=1 https://github.com/vllm-project/vllm.git \
  5db652225f00b55783823ae6606d36925e3e3efe
```

Use a Python environment containing the PR's dependencies and torch. The
scripts select vLLM's CPU platform; a compiled `vllm._C` is not required.

```bash
export VLLM_REPO=/path/to/pr52516
export VENV=/path/to/venv
probe_dir=/path/to/e2e/examples/disaggregated/mooncake_connector/replicated_gqa_layout_review

REVIEW_OUTPUT=/path/to/results/padded.json bash "$probe_dir/run.sh" padded
REVIEW_OUTPUT=/path/to/results/nvfp4.json bash "$probe_dir/run.sh" nvfp4
```

`run.sh` activates the venv, checks the PR commit and base commit, and uses
the specified source checkout.

## Validation scope

- Actual PR methods: `register_kv_caches`, `send_kv_to_decode`,
  `_build_transfer_params`, and `_send_blocks`; pairing via
  `TransferTopology.handshake_target_ranks`.
- The padded spec is produced by the normal alignment and spec-construction
  code (`Platform._align_heterogeneous_kv_block_size` and
  `Attention.get_kv_cache_spec`): an NVFP4 full-attention block pool with a
  BF16 sliding-window layer excluded via `--kv-cache-dtype-skip-layers`.
  No Mamba/GDN layer is involved.
- The NVFP4 spec is produced by `FlashInferBackend.customize_spec`. The
  FlashInfer NVFP4 serving path itself requires Blackwell; this probe checks
  the connector's transfer plan against that spec's real cache view, not
  NVFP4 inference.
- Cache views come from `create_kv_cache_views(..., LBHNC)`, so head strides
  and padding are the real ones. Block 0 is an untouched guard block.
- Base controls substitute only the four functions from `5db652225f`
  (`_compute_sender_transfer_plan`, `_validate_asymmetric_region_lengths`,
  `send_kv_to_decode`, `_get_sender_transfer_plan`); this is not a separate
  full-base deployment.
- Destination contents are checked per head slot against an independent
  reference (byte value encodes the global KV head; V slots are offset by 10),
  not against the planner's own offsets.

## Recorded results

Run on an H20 node CPU allocation (4 CPUs, 8 GiB, no GPU) on 2026-09-09 with
torch `2.13.0+cu130`; outputs are preserved in `padded_page_results.json` and
`nvfp4_slots_results.json`.

| Case | Base (`5db652225f`) | Head (`d61e44b85b`) |
| --- | --- | --- |
| Dense GQA TP1 → TP8, D2 (control) | — | `FINISH`, 0 mismatched bytes |
| Dense GQA TP8 → TP2, D0 (control) | — | `FINISH` from P0/P2, 0 mismatched bytes |
| Padded SWA page TP1 → TP8, D2 | `ERROR`, 0 bytes copied | `FINISH`; copies 9216 B from src offset 9216; 1024 of 8192 content bytes wrong |
| FlashInfer NVFP4 TP1 → TP8, D2 | `ERROR`, 0 bytes copied | `FINISH`; copies 2304 B from src offset 2304; all 2304 content bytes wrong (`[K2,K3]` instead of `[K1,V1]`) |

Padded case geometry: P page 36864 B (32768 B content, 4 heads, real head
stride 8192 B); D page 9216 B (8192 B content). NVFP4 case geometry: P page
9216 B with 8 slots of 1152 B (`[K0..K3, V0..V3]`); D page 2304 B with 2 slots.
