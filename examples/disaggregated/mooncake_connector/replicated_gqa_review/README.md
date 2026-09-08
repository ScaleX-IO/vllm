# Replicated-GQA review probes for PR #52516

Two independently runnable probes reproduce the review findings at
[`3d52c8329d`](https://github.com/vllm-project/vllm/pull/52516):

- `reproduce_hybrid_gdn.py`: crossing the Attention KV-head replication
  boundary permits a Hybrid transfer, but whole-page slicing corrupts the
  packed GDN convolution/recurrent state.
- `reproduce_region_length.py`: incompatible per-head region lengths are
  accepted, and the sender reports success after writing half the destination.

Both use `transfer_probe.py` for real CUDA buffers and native Mooncake RDMA
transfers. Each includes ordinary GQA controls and the original base commit's
region validator as a rejection control. **Exit status zero means the defect
was reproduced and the controls passed; these are reproduction scripts, not
tests expecting a corrected implementation.**

## Run

Keep this script directory in the E2E checkout. Set `VLLM_REPO` to a separate
checkout of the PR's exact head, with its base commit available in Git:

```bash
git fetch https://github.com/vllm-project/vllm.git refs/pull/52516/head
git worktree add --detach /path/to/pr52516 3d52c8329ded27d62e043c960cc3e2250a5c02de
```

Use a Python environment containing the PR's dependencies, CUDA-enabled torch,
and Mooncake. The scripts select vLLM's CPU platform to avoid importing model
CUDA extensions; torch GPU allocations and the native Mooncake transfers still
execute on two real GPUs. No model extension rebuild is required for this probe.

On H20, allocate **two GPUs and 24 CPUs** on node02, then run the two probes
sequentially. Substitute the actual allocation ID and absolute paths:

```bash
export VLLM_REPO=/path/to/pr52516
export VENV=/home/felixlinker/.venv
export REVIEW_RDMA_IP=200.9.0.18
export REVIEW_RDMA_DEVICE=mlx5_bond_0
probe_dir=/path/to/e2e/examples/disaggregated/mooncake_connector/replicated_gqa_review

export REVIEW_OUTPUT=/path/to/results/hybrid.json
srun --jobid=<JOBID> --ntasks=1 --exclusive --exact \
  --gres=gpu:h20:2 --cpus-per-task=24 bash "$probe_dir/run.sh" hybrid

export REVIEW_OUTPUT=/path/to/results/region.json
srun --jobid=<JOBID> --ntasks=1 --exclusive --exact \
  --gres=gpu:h20:2 --cpus-per-task=24 bash "$probe_dir/run.sh" region
```

`run.sh` sets memlock, activates the venv, checks the PR commit, and uses the
specified source checkout. It does not change `CUDA_VISIBLE_DEVICES`.
Use `200.9.0.34` instead if the allocation is on node01. GPU indices 0 and 1
are interpreted within the Slurm step's assigned device namespace.

## Validation scope

- Actual PR methods: `register_kv_caches`, `send_kv_to_decode`,
  `_build_transfer_params`, and `_send_blocks`.
- Independent GPU sender and receiver processes, with logical TP ranks
  exercised sequentially on two physical GPUs.
- A capture socket records encoded sender responses; ZMQ control-plane
  transport and a live TP4/TP8 model-serving deployment are not exercised.
- Base controls substitute only the exact base region validator from
  `e862c2f45bc81c72bb84f3807d88d60cfb93d694`. Both defects are rejected before
  transfer by that validator. This is not a separate full-base deployment.
- GDN uses the Qwen3.6-27B-FP8 configuration's dimensions and synthetic state
  contents, not states produced by a model forward pass. In DS layout, TP4
  has BF16 conv `(2560, 3)` plus FP32 SSM `(12, 128, 128)`; TP8 has
  `(1280, 3)` plus `(6, 128, 128)`. The reference separately shards Q/K/V conv
  channels and SSM heads.
- The incompatible-length probe deliberately advertises 64 KiB per head on
  P and 128 KiB per head on D. It validates rejection of incompatible byte
  geometry, not a live model pair with differing KV dtypes.

## Recorded H20 results

The initial combined run used node02, two H20-3e GPUs, torch `2.13.0+cu130`,
`mlx5_bond_0` RDMA with GID 3, and allocation 2565 on 2026-09-08.
The original related upstream unit tests passed: **42 tests, zero failures**.
The combined payload results are preserved in `h20_results.json`.

| Case | Observation |
| --- | --- |
| Hkv4 TP8 → TP2, D0 | Canonical P0/P2 fill the whole destination; zero mismatched bytes |
| Hkv4 TP2 → TP8, D0–D3 | Correct head contents on all four readers |
| Hybrid Hkv4 TP4 → TP8, D0 | Attention correct; 3072 conv and 98304 SSM elements incorrect |
| Hybrid Hkv4 TP4 → TP8, D1 | Attention correct; all 3840 conv elements incorrect; SSM correct |
| Incompatible TP1 → TP8 region | 65536/131072 bytes written; remaining bytes retain the sentinel |

The defective PR transfers return `FINISH` with `ok_reqs`, no error requests,
and no native transfer failures. Guard blocks remain intact. The baseline
validator controls return `ERROR` and leave the destination entirely unchanged.

Both independent entry points passed again on node02 in allocation 2566 before
publishing this directory. Both allocations have been released. Ruff lint,
format, Python compilation, and shell syntax checks passed. This validation
does not make model accuracy or throughput claims.
