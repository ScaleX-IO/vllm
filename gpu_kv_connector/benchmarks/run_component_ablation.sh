#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run this benchmark as root so BaM can register CUDA I/O memory" >&2
  exit 2
fi
: "${GPUKV_MODEL:?Set GPUKV_MODEL to a local Hugging Face model path}"

run_user=${SUDO_USER:-${USER:-}}
run_home=$(getent passwd "$run_user" 2>/dev/null | cut -d: -f6)
run_home=${run_home:-$HOME}
python_bin=${GPUKV_PYTHON:-$run_home/miniconda3/envs/vllm-gpukv/bin/python}
benchmark_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
results_dir=${GPUKV_RESULTS_DIR:-/tmp/gpukv-component-ablation}
mkdir -p "$results_dir"

server_pid=
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid"
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=
}
trap stop_server EXIT

run_case() {
  local name=$1
  local mode=$2
  shift 2
  echo "case=$name"
  env \
    GPUKV_MODEL="$GPUKV_MODEL" \
    GPUKV_BENCH_MODE="$mode" \
    GPUKV_MAX_OBJECTS=8192 \
    GPUKV_MAX_BATCH=8192 \
    "$@" \
    "$benchmark_dir/serve_prefix_benchmark.sh" \
    >"$results_dir/$name.server.log" 2>&1 &
  server_pid=$!

  local ready=false
  for _ in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null; then
      ready=true
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -80 "$results_dir/$name.server.log" >&2
      return 1
    fi
    sleep 1
  done
  if [[ "$ready" != true ]]; then
    echo "server startup timed out for $name" >&2
    tail -80 "$results_dir/$name.server.log" >&2
    return 1
  fi

  local external=()
  if [[ "$mode" == gpukv ]]; then
    external=(--require-external-hits)
  fi
  "$python_bin" "$benchmark_dir/benchmark_prefix_reload.py" \
    --model "$GPUKV_MODEL" \
    --reload-concurrency 1 \
    "${external[@]}" \
    >"$results_dir/$name.json"
  cat "$results_dir/$name.json"
  stop_server

  for _ in $(seq 1 30); do
    if ! curl -sf http://127.0.0.1:8000/health >/dev/null; then
      break
    fi
    sleep 1
  done
}

run_case final gpukv
run_case no_physical_superrequests gpukv GPUKV_MAX_SUPERREQUEST_OBJECTS=0
run_case unfused_kv gpukv GPUKV_FUSE_KV_PLANES=false
run_case prefetch_one gpukv GPUKV_PREFETCH_LAYERS=1
run_case no_ready_cache gpukv GPUKV_READY_CACHE_ENTRIES=0
run_case recompute recompute
