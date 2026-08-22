#!/usr/bin/env bash
set -euo pipefail
ulimit -l unlimited

: "${FEATURE_ROOT:?Set FEATURE_ROOT to the feature worktree}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${VENV_ROOT:?Set VENV_ROOT}"
: "${MOONCAKE_CLIENT_BIN:?Set MOONCAKE_CLIENT_BIN}"
: "${RDMA_DEVICE:?Set RDMA_DEVICE}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
served=${SERVED_MODEL_NAME:-hybrid-heterogeneous-tp-test}
run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}
result_root=${RESULT_ROOT:-$PWD/hybrid-heterogeneous-tp-$run_id}
mkdir -p "$result_root/logs"

host_ip=${HOST_IP:-$(hostname -I | awk '{print $1}')}
port_base=${PORT_BASE:-$((52000 + ($$ % 100) * 10))}
master_port=$port_base
metadata_port=$((port_base + 1))
store_port=$((port_base + 2))
metrics_port=$((port_base + 3))
tp4_gpus=${TP4_GPUS:-0,1,2,3}
tp2_gpus=${TP2_GPUS:-4,5}
gpu_memory=${GPU_MEMORY_UTILIZATION:-0.8}

export PYTHONPATH=$FEATURE_ROOT
export PATH=$VENV_ROOT/bin:$PATH
export VLLM_BIN=$VENV_ROOT/bin/vllm
export PYTHONHASHSEED=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_MOONCAKE_LOAD_RECV_THREADS=1
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export MOONCAKE_PREFERRED_SEGMENT=$host_ip:$store_port

export MOONCAKE_CONFIG_PATH=$result_root/mooncake.json
export VLLM_MOONCAKE_STORE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH
printf '%s\n' "{\"metadata_server\":\"http://$host_ip:$metadata_port/metadata\",\"master_server_address\":\"$host_ip:$master_port\",\"protocol\":\"rdma\",\"device_name\":\"$RDMA_DEVICE\",\"mode\":\"standalone-store\",\"global_segment_size\":\"0 B\",\"local_buffer_size\":\"64 MB\",\"enable_offload\":false}" >"$MOONCAKE_CONFIG_PATH"

pids=()
last_pid=

start_service() {
  local log=$1
  shift
  setsid "$@" >"$log" 2>&1 &
  last_pid=$!
  pids+=("$last_pid")
}

stop_service() {
  local pid=$1
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${pids[@]:-}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

wait_url() {
  local pid=$1 url=$2 log=$3
  for _ in $(seq 1 360); do
    curl -fsS "$url/health" >/dev/null && return 0
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -200 "$log" || true
      return 1
    fi
    sleep 2
  done
  tail -200 "$log" || true
  return 1
}

start_service "$result_root/logs/master.log" mooncake_master \
  --rpc_port="$master_port" \
  --enable_http_metadata_server=true \
  --http_metadata_server_port="$metadata_port" \
  --enable_metric_reporting=false --metrics_port="$metrics_port"
sleep 2
start_service "$result_root/logs/store.log" "$MOONCAKE_CLIENT_BIN" \
  --host="$host_ip" --port="$store_port" \
  --master_server_address="$host_ip:$master_port" \
  --metadata_server="http://$host_ip:$metadata_port/metadata" \
  --protocol=rdma --device_names="$RDMA_DEVICE" \
  --global_segment_size="${STORE_SIZE:-128 GB}" --local_buffer_size="64 MB"
sleep 3

run_direction() {
  local producer_tp=$1 consumer_tp=$2 producer_gpus=$3 consumer_gpus=$4 label=$5
  local base=$6
  local reference_port=$base producer_port=$((base + 1)) consumer_port=$((base + 2))
  local producer_lookup=$((base + 3)) consumer_lookup=$((base + 4))
  local reference_json=$result_root/$label-reference.json
  local cache_prefix=hybrid-$label-$run_id

  start_service "$result_root/logs/$label-reference.log" \
    env CUDA_VISIBLE_DEVICES="$consumer_gpus" VLLM_HOST_IP="$host_ip" \
    "$VENV_ROOT/bin/python" "$script_dir/launch_server.py" --role reference \
    --model "$MODEL_PATH" --served-model-name "$served" --port "$reference_port" \
    --tensor-parallel-size "$consumer_tp" --gpu-memory-utilization "$gpu_memory"
  local reference_pid=$last_pid
  wait_url "$reference_pid" "http://$host_ip:$reference_port" \
    "$result_root/logs/$label-reference.log"
  "$VENV_ROOT/bin/python" "$script_dir/workload.py" reference \
    --reference-url "http://$host_ip:$reference_port" --model "$served" \
    --tokenizer "$MODEL_PATH" --reference-json "$reference_json" \
    --output-tokens "${REFERENCE_OUTPUT_TOKENS:-64}" \
    --prompt-repetitions "${PROMPT_REPETITIONS:-128}" \
    --prompt-tokens "${PROMPT_TOKENS:-785}"
  stop_service "$reference_pid"

  start_service "$result_root/logs/$label-producer.log" \
    env CUDA_VISIBLE_DEVICES="$producer_gpus" VLLM_HOST_IP="$host_ip" \
    "$VENV_ROOT/bin/python" "$script_dir/launch_server.py" --role producer \
    --model "$MODEL_PATH" --served-model-name "$served" --port "$producer_port" \
    --lookup-rpc-port "$producer_lookup" --tensor-parallel-size "$producer_tp" \
    --store-tp-size 4 --cache-prefix "$cache_prefix" \
    --gpu-memory-utilization "$gpu_memory"
  local producer_pid=$last_pid
  wait_url "$producer_pid" "http://$host_ip:$producer_port" \
    "$result_root/logs/$label-producer.log"

  start_service "$result_root/logs/$label-consumer.log" \
    env CUDA_VISIBLE_DEVICES="$consumer_gpus" VLLM_HOST_IP="$host_ip" \
    "$VENV_ROOT/bin/python" "$script_dir/launch_server.py" --role consumer \
    --model "$MODEL_PATH" --served-model-name "$served" --port "$consumer_port" \
    --lookup-rpc-port "$consumer_lookup" --tensor-parallel-size "$consumer_tp" \
    --store-tp-size 4 --cache-prefix "$cache_prefix" --save-decode-cache \
    --gpu-memory-utilization "$gpu_memory"
  local consumer_pid=$last_pid
  wait_url "$consumer_pid" "http://$host_ip:$consumer_port" \
    "$result_root/logs/$label-consumer.log"

  "$VENV_ROOT/bin/python" "$script_dir/workload.py" loop \
    --producer-url "http://$host_ip:$producer_port" \
    --consumer-url "http://$host_ip:$consumer_port" --model "$served" \
    --tokenizer "$MODEL_PATH" --reference-json "$reference_json" \
    --output-tokens "${OUTPUT_TOKENS:-800}" \
    --prompt-repetitions "${PROMPT_REPETITIONS:-128}" \
    --prompt-tokens "${PROMPT_TOKENS:-785}" \
    --visibility-timeout "${VISIBILITY_TIMEOUT:-300}" \
    | tee "$result_root/$label-result.json"

  stop_service "$producer_pid"
  stop_service "$consumer_pid"
}

run_direction 4 2 "$tp4_gpus" "$tp2_gpus" tp4-to-tp2 $((port_base + 10))
run_direction 2 4 "$tp2_gpus" "$tp4_gpus" tp2-to-tp4 $((port_base + 20))

grep -Ehi 'rdma|failed_keys=' "$result_root"/logs/*.log \
  >"$result_root/rdma-evidence.txt" || true
if grep -Ehi 'failed_keys=[1-9]|failed key|failed.*(get|put|pack|unpack)' \
  "$result_root"/logs/*-producer.log "$result_root"/logs/*-consumer.log \
  >"$result_root/failures.txt"; then
  cat "$result_root/failures.txt"
  exit 1
fi
touch "$result_root/PASS"
printf '%s\n' "$result_root" >"$result_root/RESULT_PATH"
echo "PASS: $result_root"
