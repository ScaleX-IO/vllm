#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
job_id=${JOB_ID:-${SLURM_JOB_ID:-}}
if [[ -z $job_id ]]; then
  echo "Set JOB_ID to a single-node allocation with at least 3 GPUs" >&2
  exit 2
fi
result_root=${RESULT_ROOT:-/home/felixlinker/piecewise-nixl-e2e-$job_id}
base_port=${PORT_BASE:-$((32000 + (job_id % 200) * 160))}
mkdir -p "$result_root"

run_case() {
  local name=$1
  local case_name=$2
  local mode=$3
  JOB_ID="$job_id" CASE="$case_name" MODE="$mode" RUN_ID="$name" \
    RESULT_ROOT="$result_root/$name" PORT_BASE="$4" \
    bash "$script_dir/run_case.sh"
}

run_case piecewise-pull piecewise pull "$base_port"
run_case piecewise-push piecewise push "$((base_port + 40))"
run_case store-miss miss pull "$((base_port + 80))"
run_case store-full full pull "$((base_port + 120))"

python_bin=${PYTHON_BIN:-/home/felixlinker/.venv/bin/python}
vllm_root=$(cd -- "$script_dir/../../../../.." && pwd -P)
node=$(squeue -h -j "$job_id" -o %N)
srun --jobid="$job_id" --overlap --nodes=1 --nodelist="$node" --ntasks=1 \
  --gres=none --cpus-per-task=2 \
  bash -lc 'cd "$1"; shift; exec "$@"' _ "$vllm_root" \
  env PYTHONPATH="$vllm_root" "$python_bin" -m \
  tests.v1.kv_connector.nixl_integration.piecewise_range.evidence compare \
  --piecewise "$result_root/piecewise-pull/evidence.json" \
  --miss "$result_root/store-miss/evidence.json" \
  --full "$result_root/store-full/evidence.json" \
  --output "$result_root/comparison.json"
touch "$result_root/PASS"
echo "PASS: $result_root"
