#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail
ulimit -l unlimited

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tests=${TEST_ROOT:-$(cd "$script_dir/../../../.." && pwd)}
feature=${FEATURE_ROOT:-$tests}
python_bin=${PYTHON_BIN:-python}
vllm_bin=${VLLM_BIN:-vllm}
unset VLLM_BIN
if [[ $python_bin == */* ]]; then
  export PATH=$(dirname "$python_bin"):$PATH
fi
model=${MODEL_PATH:-/mnt/nvme/shared/felixlinker/models/Qwen3-32B-FP8}
served=${SERVED_MODEL_NAME:-pd-store-piecewise-probe}
rdma_device=${RDMA_DEVICE:-mlx5_bond_0}
mooncake_master_bin=${MOONCAKE_MASTER_BIN:-mooncake_master}
mooncake_client_bin=${MOONCAKE_CLIENT_BIN:-mooncake_client}
store_gpu=${STORE_GPU:-0}
prefill_gpu=${PREFILL_GPU:-1}
decode_gpu=${DECODE_GPU:-2}
run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}
run_root=${RESULT_ROOT:-/tmp/pd-store-piecewise-$run_id}
proxy_script=$feature/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py
mkdir -p "$run_root"

host_ip=${HOST_IP:-$(hostname -I | awk '{print $1}')}
if [[ -n ${PORT_BASE:-} ]]; then
  base=$PORT_BASE
else
  base=$($python_bin - <<'PY'
import socket

for candidate in range(54000, 62000, 10):
    sockets = []
    try:
        for port in range(candidate, candidate + 8):
            sock = socket.socket()
            sock.bind(("0.0.0.0", port))
            sockets.append(sock)
    except OSError:
        pass
    else:
        print(candidate)
        break
    finally:
        for sock in sockets:
            sock.close()
else:
    raise RuntimeError("No free consecutive port range found")
PY
  )
fi
master_port=$base
metadata_port=$((base + 1))
store_port=$((base + 2))
prefill_port=$((base + 3))
decode_port=$((base + 4))
proxy_port=$((base + 5))
bootstrap_port=$((base + 6))
metrics_port=$((base + 7))
cache_prefix=pd-store-piecewise-$run_id

export PYTHONPATH=$feature${PYTHONPATH:+:$PYTHONPATH}
export PYTHONHASHSEED=0
export VLLM_KV_CACHE_LAYOUT=LBHNC
export VLLM_SERVER_DEV_MODE=1
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_MOONCAKE_LOAD_RECV_THREADS=1
export MOONCAKE_CONFIG_PATH=$run_root/mooncake.json

HOST_IP=$host_ip METADATA_PORT=$metadata_port MASTER_PORT=$master_port \
RDMA_DEVICE=$rdma_device MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH \
  "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

config = {
    "metadata_server": (
        f"http://{os.environ['HOST_IP']}:{os.environ['METADATA_PORT']}/metadata"
    ),
    "master_server_address": (
        f"{os.environ['HOST_IP']}:{os.environ['MASTER_PORT']}"
    ),
    "protocol": "rdma",
    "device_name": os.environ["RDMA_DEVICE"],
    "mode": "standalone-store",
    "global_segment_size": "0 B",
    "local_buffer_size": "64 MB",
    "enable_offload": False,
}
Path(os.environ["MOONCAKE_CONFIG_PATH"]).write_text(
    json.dumps(config), encoding="utf-8"
)
PY

store_pids=()
service_pids=()

start_store() {
  local log=$1
  shift
  setsid "$@" >"$log" 2>&1 &
  store_pids+=("$!")
}

start_service() {
  local log=$1
  shift
  setsid "$@" >"$log" 2>&1 &
  service_pids+=("$!")
}

stop_group() {
  local -n pids=$1
  local pid
  for pid in "${pids[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  sleep 4
  for pid in "${pids[@]:-}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  pids=()
}

cleanup() {
  stop_group service_pids
  stop_group store_pids
}
trap cleanup EXIT

wait_url() {
  local pid=$1
  local url=$2
  local log=$3
  local attempt
  for attempt in $(seq 1 360); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -160 "$log" || true
      return 1
    fi
    sleep 2
  done
  tail -160 "$log" || true
  return 1
}

launch_pd() {
  local phase=$1
  local phase_dir=$run_root/$phase
  local store_lookup_args=()
  if [[ $phase == reference ]]; then
    store_lookup_args+=(--disable-store-lookup)
  fi
  mkdir -p "$phase_dir"

  start_service "$phase_dir/prefill.log" \
    env CUDA_VISIBLE_DEVICES="$prefill_gpu" VLLM_HOST_IP="$host_ip" \
    VLLM_MOONCAKE_BOOTSTRAP_PORT="$bootstrap_port" \
    "$python_bin" "$script_dir/launch_server.py" --role prefill \
    --model "$model" --served-model-name "$served" --port "$prefill_port" \
    --lookup-rpc-port "$prefill_port" --cache-prefix "$cache_prefix" \
    --transfer-device "$rdma_device" --vllm-bin "$vllm_bin" \
    "${store_lookup_args[@]}"
  local prefill_pid=${service_pids[-1]}

  start_service "$phase_dir/decode.log" \
    env CUDA_VISIBLE_DEVICES="$decode_gpu" VLLM_HOST_IP="$host_ip" \
    "$python_bin" "$script_dir/launch_server.py" --role decode \
    --model "$model" --served-model-name "$served" --port "$decode_port" \
    --lookup-rpc-port "$decode_port" --cache-prefix "$cache_prefix" \
    --transfer-device "$rdma_device" --vllm-bin "$vllm_bin" \
    "${store_lookup_args[@]}"
  local decode_pid=${service_pids[-1]}

  wait_url "$prefill_pid" "http://$host_ip:$prefill_port/health" \
    "$phase_dir/prefill.log"
  wait_url "$decode_pid" "http://$host_ip:$decode_port/health" \
    "$phase_dir/decode.log"

  start_service "$phase_dir/proxy.log" \
    "$python_bin" "$proxy_script" --host "$host_ip" --port "$proxy_port" \
    --prefill "http://$host_ip:$prefill_port" "$bootstrap_port" \
    --decode "http://$host_ip:$decode_port"
  local proxy_pid=${service_pids[-1]}
  wait_url "$proxy_pid" "http://$host_ip:$proxy_port/docs" "$phase_dir/proxy.log"
}

run_request() {
  local phase=$1
  local delta_tokens=$2
  PHASE=$phase DELTA_TOKENS=$delta_tokens RUN_ROOT=$run_root \
  PROXY_URL="http://$host_ip:$proxy_port" MODEL_PATH=$model SERVED=$served \
    "$python_bin" - <<'PY'
import json
import os
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"])
source = tokenizer.encode(
    "A shared external KV prefix should be reused without retransmission. ",
    add_special_tokens=False,
)
base = (source * (1024 // len(source) + 1))[:1024]
delta_source = tokenizer.encode(
    "This suffix is computed only by the prefiller. ", add_special_tokens=False
)
delta_count = int(os.environ["DELTA_TOKENS"])
delta = (delta_source * (delta_count // len(delta_source) + 1))[:delta_count]
prompt = base + delta
payload = {
    "model": os.environ["SERVED"],
    "prompt": prompt,
    "max_tokens": 8,
    "temperature": 0,
    "ignore_eos": True,
    "return_token_ids": True,
    "stream": False,
}
started = time.perf_counter()
response = httpx.post(
    f"{os.environ['PROXY_URL']}/v1/completions", json=payload, timeout=600
)
elapsed = time.perf_counter() - started
response.raise_for_status()
result = response.json()
record = {
    "phase": os.environ["PHASE"],
    "base_tokens": len(base),
    "delta_tokens": len(delta),
    "prompt_tokens": len(prompt),
    "elapsed_seconds": elapsed,
    "usage": result.get("usage"),
    "output_token_ids": result["choices"][0].get("token_ids"),
}
path = Path(os.environ["RUN_ROOT"]) / f"{os.environ['PHASE']}.json"
path.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record))
PY
}

(
  cd "$feature"
  CUDA_VISIBLE_DEVICES=$prefill_gpu "$python_bin" \
    -W ignore::DeprecationWarning -m pytest -q \
    tests/v1/kv_connector/unit/test_multi_connector.py::test_range_aware_load_assigns_contiguous_ranges \
    tests/v1/kv_connector/unit/test_multi_connector.py::test_range_aware_load_extends_a_local_hit \
    tests/v1/kv_connector/unit/test_multi_connector.py::test_range_aware_load_falls_back_to_longest_unaligned_hit \
    tests/v1/kv_connector/unit/test_multi_connector.py::test_range_aware_load_combines_three_sources \
    tests/v1/kv_connector/unit/test_multi_connector.py::test_range_aware_load_uses_one_source_for_equal_hits \
    tests/v1/kv_connector/unit/test_multi_connector.py::test_range_aware_load_waits_for_all_async_sources \
    tests/v1/kv_connector/unit/test_multi_connector.py::test_default_load_policy_still_selects_first_hit \
    tests/v1/kv_connector/unit/test_mooncake_connector.py::test_piecewise_load_uses_only_suffix_destination_blocks \
    tests/v1/kv_connector/unit/test_mooncake_connector.py::test_piecewise_load_rejects_incomplete_transfer_params
) >"$run_root/unit-tests.log" 2>&1

start_store "$run_root/master.log" "$mooncake_master_bin" \
  --rpc_port="$master_port" --enable_http_metadata_server=true \
  --http_metadata_server_port="$metadata_port" \
  --enable_metric_reporting=false --metrics_port="$metrics_port"
sleep 2
start_store "$run_root/store.log" env CUDA_VISIBLE_DEVICES="$store_gpu" \
  "$mooncake_client_bin" \
  --host="$host_ip" --port="$store_port" \
  --master_server_address="$host_ip:$master_port" \
  --metadata_server="http://$host_ip:$metadata_port/metadata" \
  --protocol=rdma --device_names="$rdma_device" \
  --global_segment_size="64 GB" --local_buffer_size="64 MB"
sleep 3

launch_pd seed
run_request seed 0
sleep 15
stop_group service_pids

launch_pd replay
run_request replay 32
sleep 15
stop_group service_pids

launch_pd reference
run_request reference 32
sleep 15

RUN_ROOT=$run_root "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
replay = json.loads((root / "replay.json").read_text())
reference = json.loads((root / "reference.json").read_text())
assert replay["output_token_ids"] == reference["output_token_ids"], (
    replay["output_token_ids"],
    reference["output_token_ids"],
)
PY

grep -q 'kvpool hit tokens: 1024, need to load: 1024' \
  "$run_root/replay/decode.log"
grep -q 'range load:.*range=\[1024, 1056)' \
  "$run_root/replay/decode.log"
grep -q 'Sending kv_caches.*(2 blocks)' "$run_root/replay/prefill.log"
grep -q 'Sending kv_caches.*(66 blocks)' "$run_root/reference/prefill.log"

grep -E -e 'kvpool hit tokens' -e 'range load' \
  -e 'Sending kv_caches' "$run_root"/replay/*.log \
  >"$run_root/evidence.log"

cat "$run_root/seed.json"
cat "$run_root/replay.json"
cat "$run_root/reference.json"
cat "$run_root/unit-tests.log"
cat "$run_root/evidence.log"
printf '%s\n' "$run_root" >"$run_root/RESULT_PATH"
touch "$run_root/PASS"
echo "PASS: $run_root"
