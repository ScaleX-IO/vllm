#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
vllm_root=${VLLM_ROOT:-$(cd -- "$script_dir/../../../../.." && pwd -P)}
job_id=${JOB_ID:-${SLURM_JOB_ID:-}}
case_name=${CASE:-piecewise}
mode=${MODE:-pull}
python_bin=${PYTHON_BIN:-/home/felixlinker/.venv/bin/python}
vllm_bin=${VLLM_BIN:-/home/felixlinker/.venv/bin/vllm}
model=${MODEL_PATH:-/mnt/nvme/shared/felixlinker/models/Qwen3-32B-FP8}
served=${SERVED_MODEL_NAME:-piecewise-nixl-e2e}
base_tokens=${BASE_TOKENS:-1024}
suffix_tokens=${SUFFIX_TOKENS:-256}
output_tokens=${OUTPUT_TOKENS:-8}
block_size=${BLOCK_SIZE:-16}
max_model_len=${MAX_MODEL_LEN:-2048}
kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES:-1073741824}
rdma_device=${RDMA_DEVICE:-mlx5_bond_0}
openssl3=${NIXL_OPENSSL3_DIR:-/home/felixlinker/opt/nixl-openssl3/root/usr/lib64}
master_bin=${MOONCAKE_MASTER_BIN:-/home/felixlinker/.venv/lib/python3.12/site-packages/mooncake/mooncake_master}
client_bin=${MOONCAKE_CLIENT_BIN:-/home/felixlinker/Mooncake/build/mooncake-store/src/mooncake_client}

if [[ -z $job_id ]]; then
  echo "Set JOB_ID to a single-node allocation with at least 3 GPUs" >&2
  exit 2
fi
if [[ $case_name != piecewise && $case_name != miss && $case_name != full ]]; then
  echo "CASE must be piecewise, miss, or full" >&2
  exit 2
fi
if [[ $mode != pull && $mode != push ]]; then
  echo "MODE must be pull or push" >&2
  exit 2
fi

node=$(squeue -h -j "$job_id" -o %N)
if [[ -z $node || $node == *"("* ]]; then
  echo "Job $job_id is not running on a node" >&2
  exit 2
fi
port_base=${PORT_BASE:-$((32000 + (job_id % 200) * 160))}
run_id=${RUN_ID:-${case_name}-${mode}-$(date +%Y%m%d-%H%M%S)}
run_root=${RESULT_ROOT:-/home/felixlinker/piecewise-nixl-e2e-$job_id/$run_id}
mkdir -p "$run_root/services"

run_node() {
  srun --jobid="$job_id" --overlap --nodes=1 --nodelist="$node" --ntasks=1 \
    --gres=none --cpus-per-task=2 \
    bash -lc 'ulimit -l unlimited; cd "$1"; shift; exec "$@"' \
    _ "$vllm_root" "$@"
}

service_ip=$(run_node bash -c "hostname -I | tr ' ' '\\n' | grep -m1 '^172\\.16\\.'")
rdma_ip=$(run_node bash -c "hostname -I | tr ' ' '\\n' | grep -m1 '^200\\.'")
if [[ -z $service_ip || -z $rdma_ip ]]; then
  echo "Could not resolve service and RDMA addresses on $node" >&2
  exit 2
fi

declare -a step_pids=()
last_step_pid=

start_step() {
  local name=$1
  local gpus=$2
  local cpus=$3
  shift 3
  local log=$run_root/services/$name.log
  local -a gres=(--gres=none)
  if ((gpus > 0)); then
    gres=(--gres="gpu:h20:$gpus")
  fi
  setsid srun --jobid="$job_id" --nodes=1 --nodelist="$node" --ntasks=1 \
    --exclusive --exact --unbuffered "${gres[@]}" --cpus-per-task="$cpus" \
    bash -lc 'ulimit -l unlimited; source /home/felixlinker/.venv/bin/activate; cd "$1"; shift; exec "$@"' \
    _ "$vllm_root" "$@" >"$log" 2>&1 &
  last_step_pid=$!
  step_pids+=("$last_step_pid")
}

stop_step() {
  local pid=$1
  kill -TERM -- "-$pid" 2>/dev/null || true
  sleep 5
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local pid
  for pid in "${step_pids[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  sleep 3
  for pid in "${step_pids[@]:-}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_url() {
  local pid=$1
  local url=$2
  local log=$3
  for _ in $(seq 1 "${HEALTH_WAIT_ATTEMPTS:-600}"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -160 "$log" >&2 || true
      return 1
    fi
    sleep 2
  done
  tail -160 "$log" >&2 || true
  return 1
}

common_env=(
  env
  PYTHONPATH="$vllm_root"
  PYTHONHASHSEED=0
  VLLM_KV_CACHE_LAYOUT=HND
  VLLM_SERVER_DEV_MODE=1
  VLLM_LOGGING_LEVEL=DEBUG
  VLLM_NIXL_SIDE_CHANNEL_HOST="$rdma_ip"
  VLLM_HOST_IP="$rdma_ip"
  VLLM_MOONCAKE_LOAD_RECV_THREADS=1
  VLLM_USE_DEEP_GEMM=0
  VLLM_MOE_USE_DEEP_GEMM=0
  UCX_TLS=tcp,cuda_copy,cuda_ipc
  UCX_NET_DEVICES=all
  MOONCAKE_CONFIG_PATH="$run_root/mooncake.json"
  LD_LIBRARY_PATH="$openssl3"
  PIECEWISE_VLLM_BIN="$vllm_bin"
)

master_port=$port_base
metadata_port=$((port_base + 1))
store_port=$((port_base + 2))
reference_port=$((port_base + 3))
seeder_port=$((port_base + 4))
prefill_port=$((port_base + 5))
decode_port=$((port_base + 6))
proxy_port=$((port_base + 7))
prefill_side_port=$((port_base + 8))
decode_side_port=$((port_base + 9))
seeder_lookup_port=$((port_base + 10))
prefill_lookup_port=$((port_base + 11))
decode_lookup_port=$((port_base + 12))
cache_prefix="$served-$job_id-$run_id"

"$python_bin" "$script_dir/workload.py" store-config \
  --output "$run_root/mooncake.json" --host "$rdma_ip" \
  --master-port "$master_port" --metadata-port "$metadata_port" \
  --device "$rdma_device"

run_node env LD_LIBRARY_PATH="$openssl3" PYTHONPATH="$vllm_root" \
  "$python_bin" -c \
  'import nixl, vllm, vllm._custom_ops; print(vllm.__file__, nixl.__file__)'
run_node rm -f \
  "/tmp/lookup_rpc_port_${seeder_lookup_port}_host_${node}_dp_rank0" \
  "/tmp/lookup_rpc_port_${prefill_lookup_port}_host_${node}_dp_rank0" \
  "/tmp/lookup_rpc_port_${decode_lookup_port}_host_${node}_dp_rank0"

start_step master 0 2 "$master_bin" --rpc_port="$master_port" \
  --enable_http_metadata_server=true \
  --http_metadata_server_port="$metadata_port" --enable_metric_reporting=false
master_pid=$last_step_pid
sleep 2
start_step store 0 4 env LD_LIBRARY_PATH="$openssl3" \
  "$client_bin" --host="$rdma_ip" --port="$store_port" \
  --master_server_address="$rdma_ip:$master_port" \
  --metadata_server="http://$rdma_ip:$metadata_port/metadata" --protocol=rdma \
  --device_names="$rdma_device" --global_segment_size="16 GB" \
  --local_buffer_size="64 MB"
store_pid=$last_step_pid
sleep 3
kill -0 "$master_pid"
kill -0 "$store_pid"

start_step reference 1 12 "${common_env[@]}" VLLM_PORT=$((port_base + 20)) \
  "$python_bin" "$script_dir/launch_server.py" --role reference --model "$model" \
  --served-model-name "$served" --port "$reference_port" --block-size "$block_size" \
  --max-model-len "$max_model_len" \
  --kv-cache-memory-bytes "$kv_cache_memory_bytes"
reference_pid=$last_step_pid
wait_url "$reference_pid" "http://$service_ip:$reference_port/health" \
  "$run_root/services/reference.log"
request_id="$run_id-request"
run_node "$python_bin" "$script_dir/workload.py" request \
  --url "http://$service_ip:$reference_port" --model-path "$model" \
  --served-model "$served" --base-tokens "$base_tokens" \
  --suffix-tokens "$suffix_tokens" --output-tokens "$output_tokens" \
  --request-id "$request_id-reference" --output "$run_root/reference.json"
stop_step "$reference_pid"
sleep 3

store_tokens=0
if [[ $case_name == piecewise ]]; then
  store_tokens=$base_tokens
elif [[ $case_name == full ]]; then
  store_tokens=$((base_tokens + suffix_tokens))
fi
if ((store_tokens > 0)); then
  seed_base_tokens=$((base_tokens + 1))
  seed_suffix_tokens=0
  if [[ $case_name == full ]]; then
    seed_base_tokens=$base_tokens
    seed_suffix_tokens=$((suffix_tokens + block_size))
  fi
  start_step seeder 1 12 "${common_env[@]}" VLLM_PORT=$((port_base + 21)) \
    "$python_bin" "$script_dir/launch_server.py" --role seeder --model "$model" \
    --served-model-name "$served" --port "$seeder_port" \
    --lookup-rpc-port "$seeder_lookup_port" --cache-prefix "$cache_prefix" \
    --store-tp-size 1 --block-size "$block_size" --max-model-len "$max_model_len" \
    --kv-cache-memory-bytes "$kv_cache_memory_bytes"
  seeder_pid=$last_step_pid
  wait_url "$seeder_pid" "http://$service_ip:$seeder_port/health" \
    "$run_root/services/seeder.log"
  run_node "$python_bin" "$script_dir/workload.py" request \
    --url "http://$service_ip:$seeder_port" --model-path "$model" \
    --served-model "$served" --base-tokens "$seed_base_tokens" \
    --suffix-tokens "$seed_suffix_tokens" \
    --output-tokens 1 --request-id "$run_id-seed" --output "$run_root/seed.json"
fi

start_step prefill 1 12 "${common_env[@]}" \
  VLLM_PORT=$((port_base + 22)) VLLM_NIXL_SIDE_CHANNEL_PORT="$prefill_side_port" \
  "$python_bin" "$script_dir/launch_server.py" --role prefill --mode "$mode" \
  --model "$model" --served-model-name "$served" --port "$prefill_port" \
  --lookup-rpc-port "$prefill_lookup_port" --cache-prefix "$cache_prefix" \
  --store-tp-size 1 --block-size "$block_size" --max-model-len "$max_model_len" \
  --kv-cache-memory-bytes "$kv_cache_memory_bytes"
prefill_pid=$last_step_pid
start_step decode 1 12 "${common_env[@]}" \
  VLLM_PORT=$((port_base + 23)) VLLM_NIXL_SIDE_CHANNEL_PORT="$decode_side_port" \
  "$python_bin" "$script_dir/launch_server.py" --role decode --mode "$mode" \
  --model "$model" --served-model-name "$served" --port "$decode_port" \
  --lookup-rpc-port "$decode_lookup_port" --cache-prefix "$cache_prefix" \
  --store-tp-size 1 --block-size "$block_size" --max-model-len "$max_model_len" \
  --kv-cache-memory-bytes "$kv_cache_memory_bytes"
decode_pid=$last_step_pid
wait_url "$prefill_pid" "http://$service_ip:$prefill_port/health" \
  "$run_root/services/prefill.log"
wait_url "$decode_pid" "http://$service_ip:$decode_port/health" \
  "$run_root/services/decode.log"

if ((store_tokens > 0)); then
  run_node "$python_bin" "$script_dir/workload.py" probe \
    --url "http://$service_ip:$prefill_port" --model-path "$model" \
    --served-model "$served" --base-tokens "$seed_base_tokens" \
    --suffix-tokens "$seed_suffix_tokens" --request-id "$run_id-probe" \
    --expected-cached-tokens "$store_tokens"
fi
run_node "$python_bin" "$script_dir/workload.py" reset \
  --url "http://$service_ip:$prefill_port"
run_node "$python_bin" "$script_dir/workload.py" reset \
  --url "http://$service_ip:$decode_port"

start_step proxy 0 2 env PYTHONPATH="$vllm_root" OPENAI_API_KEY=EMPTY \
  "$python_bin" "$vllm_root/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
  --host 0.0.0.0 --port "$proxy_port" --prefiller-host "$service_ip" \
  --prefiller-port "$prefill_port" --decoder-host "$service_ip" \
  --decoder-port "$decode_port"
proxy_pid=$last_step_pid
wait_url "$proxy_pid" "http://$service_ip:$proxy_port/healthcheck" \
  "$run_root/services/proxy.log"

for role in prefill decode; do
  port_var=${role}_port
  run_node "$python_bin" "$script_dir/metrics.py" \
    --url "http://$service_ip:${!port_var}" \
    --output "$run_root/$role-metrics-before.prom"
done
run_node "$python_bin" "$script_dir/workload.py" request \
  --url "http://$service_ip:$proxy_port" --model-path "$model" \
  --served-model "$served" --base-tokens "$base_tokens" \
  --suffix-tokens "$suffix_tokens" --output-tokens "$output_tokens" \
  --request-id "$request_id" --output "$run_root/response.json"
for role in prefill decode; do
  port_var=${role}_port
  run_node "$python_bin" "$script_dir/metrics.py" \
    --url "http://$service_ip:${!port_var}" \
    --output "$run_root/$role-metrics-after.prom" --settle-attempts 10
done
stop_step "$decode_pid"

run_node "$python_bin" -m \
  tests.v1.kv_connector.nixl_integration.piecewise_range.evidence check \
  --case "$case_name" --mode "$mode" --request-id "$request_id" \
  --base-tokens "$base_tokens" --block-size "$block_size" \
  --response "$run_root/response.json" --reference "$run_root/reference.json" \
  --decode-log "$run_root/services/decode.log" --plan-wait-seconds 10 \
  --prefill-metrics-before "$run_root/prefill-metrics-before.prom" \
  --prefill-metrics-after "$run_root/prefill-metrics-after.prom" \
  --decode-metrics-before "$run_root/decode-metrics-before.prom" \
  --decode-metrics-after "$run_root/decode-metrics-after.prom" \
  --output "$run_root/evidence.json"

"$python_bin" - "$run_root/manifest.json" <<PY
import json, pathlib, subprocess, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "commit": subprocess.check_output(["git", "-C", "$vllm_root", "rev-parse", "HEAD"], text=True).strip(),
    "job_id": "$job_id", "node": "$node", "service_ip": "$service_ip",
    "rdma_ip": "$rdma_ip",
    "case": "$case_name", "mode": "$mode", "model": "$model",
    "base_tokens": $base_tokens, "suffix_tokens": $suffix_tokens,
    "port_base": $port_base
}, indent=2))
PY
touch "$run_root/PASS"
echo "PASS: $run_root"
