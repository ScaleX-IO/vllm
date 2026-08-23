#!/usr/bin/env bash

set -euo pipefail
ulimit -l unlimited

feature=${FEATURE_ROOT:-/home/felixlinker/vllm-tp-share-v2}
tests=${TEST_ROOT:-/home/felixlinker/vllm-decode-offloading-tests}
scripts=$tests/examples/disaggregated/mooncake_store_connector/heterogeneous_tp_cluster
venv=${VENV_ROOT:-/home/felixlinker/.venv}
model=${MODEL_PATH:-/mnt/nvme/shared/felixlinker/models/Qwen3-32B-FP8}
served=heterogeneous-tp-rdma-matrix
run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}
result_root=${RESULT_ROOT:-/home/felixlinker/tp-layout-rdma-matrix-$run_id}
only_case=${ONLY_CASE:-}
layouts=${LAYOUTS:-"LBHNC LBNHC"}
mkdir -p "$result_root"

host_ip=$(hostname -I | awk '{print $1}')
port_base=${PORT_BASE:-$((48000 + ($$ % 50) * 200))}

export PYTHONPATH=$feature
export PATH=$venv/bin:$PATH
export VLLM_BIN=$venv/bin/vllm
export PYTHONHASHSEED=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_MOONCAKE_LOAD_RECV_THREADS=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

pids=()

start() {
  local log=$1
  shift
  setsid "$@" >"$log" 2>&1 &
  pids+=("$!")
}

stop_services() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  sleep 3
  for pid in "${pids[@]:-}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  pids=()
}

trap stop_services EXIT

wait_url() {
  local pid=$1
  local url=$2
  local log=$3
  local attempt
  for attempt in $(seq 1 360); do
    if curl -fsS "$url/health" >/dev/null; then
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

run_case() {
  local producer_tp=$1
  local decode_tp=$2
  local producer_layout=$3
  local decode_layout=$4
  local case_index=$5
  local label=p${producer_tp}-d${decode_tp}_${producer_layout,,}-to-${decode_layout,,}
  local run_dir=$result_root/$label
  local master_port=$((port_base + case_index * 20))
  local metadata_port=$((master_port + 1))
  local store_port=$((master_port + 2))
  local producer_port=$((master_port + 3))
  local consumer_port=$((master_port + 4))
  local reference_port=$((master_port + 5))
  local metrics_port=$((master_port + 6))
  local cache_prefix=tp-layout-rdma-$label-$run_id
  local producer_gpus consumer_gpus reference_gpus

  if [[ -n $only_case && $label != "$only_case" ]]; then
    return
  fi
  if ((producer_tp == 4)); then
    producer_gpus=0,1,2,3
    consumer_gpus=4,5
    reference_gpus=6,7
  else
    producer_gpus=0,1
    consumer_gpus=2
    reference_gpus=3
  fi

  mkdir -p "$run_dir/logs"
  printf '%s\n' \
    "case=$label producer_tp=$producer_tp decode_tp=$decode_tp producer_layout=$producer_layout decode_layout=$decode_layout" \
    | tee "$run_dir/case.txt"

  export MOONCAKE_CONFIG_PATH=$run_dir/mooncake.json
  export VLLM_MOONCAKE_STORE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH
  printf '%s\n' "{\"metadata_server\":\"http://$host_ip:$metadata_port/metadata\",\"master_server_address\":\"$host_ip:$master_port\",\"protocol\":\"rdma\",\"device_name\":\"mlx5_bond_0\",\"mode\":\"standalone-store\",\"global_segment_size\":\"0 B\",\"local_buffer_size\":\"64 MB\",\"enable_offload\":false}" >"$MOONCAKE_CONFIG_PATH"

  start "$run_dir/logs/master.log" mooncake_master \
    --rpc_port="$master_port" --enable_http_metadata_server=true \
    --http_metadata_server_port="$metadata_port" \
    --enable_metric_reporting=false --metrics_port="$metrics_port"
  sleep 2
  start "$run_dir/logs/store.log" \
    /home/felixlinker/Mooncake/build/mooncake-store/src/mooncake_client \
    --host="$host_ip" --port="$store_port" \
    --master_server_address="$host_ip:$master_port" \
    --metadata_server="http://$host_ip:$metadata_port/metadata" \
    --protocol=rdma --device_names=mlx5_bond_0 \
    --global_segment_size="64 GB" --local_buffer_size="64 MB"
  sleep 3

  cd "$feature"
  start "$run_dir/logs/producer.log" \
    env CUDA_VISIBLE_DEVICES="$producer_gpus" VLLM_HOST_IP="$host_ip" \
    "$venv/bin/python" "$scripts/launch_server.py" --role producer \
    --model "$model" --served-model-name "$served" --port "$producer_port" \
    --tensor-parallel-size "$producer_tp" --store-tp-size "$producer_tp" \
    --kv-cache-layout "$producer_layout" --cache-prefix "$cache_prefix" \
    --lookup-rpc-port "$producer_port" --gpu-memory-utilization 0.8 \
    --enforce-eager
  local producer_pid=${pids[-1]}

  start "$run_dir/logs/consumer.log" \
    env CUDA_VISIBLE_DEVICES="$consumer_gpus" VLLM_HOST_IP="$host_ip" \
    "$venv/bin/python" "$scripts/launch_server.py" --role consumer \
    --model "$model" --served-model-name "$served" --port "$consumer_port" \
    --tensor-parallel-size "$decode_tp" --store-tp-size "$producer_tp" \
    --kv-cache-layout "$decode_layout" --cache-prefix "$cache_prefix" \
    --lookup-rpc-port "$consumer_port" --gpu-memory-utilization 0.8 \
    --enforce-eager
  local consumer_pid=${pids[-1]}

  start "$run_dir/logs/reference.log" \
    env CUDA_VISIBLE_DEVICES="$reference_gpus" VLLM_HOST_IP="$host_ip" \
    "$venv/bin/python" "$scripts/launch_server.py" --role reference \
    --model "$model" --served-model-name "$served" --port "$reference_port" \
    --tensor-parallel-size "$decode_tp" --kv-cache-layout "$decode_layout" \
    --gpu-memory-utilization 0.8 --enforce-eager
  local reference_pid=${pids[-1]}

  wait_url "$producer_pid" "http://$host_ip:$producer_port" \
    "$run_dir/logs/producer.log"
  wait_url "$consumer_pid" "http://$host_ip:$consumer_port" \
    "$run_dir/logs/consumer.log"
  wait_url "$reference_pid" "http://$host_ip:$reference_port" \
    "$run_dir/logs/reference.log"

  PRODUCER_URL="http://$host_ip:$producer_port" \
  CONSUMER_URL="http://$host_ip:$consumer_port" \
  REFERENCE_URL="http://$host_ip:$reference_port" \
  MODEL="$served" TOKENIZER="$model" PYTHON_BIN="$venv/bin/python" \
    "$scripts/run_functional.sh" | tee "$run_dir/functional.json"

  PRODUCER_URL="http://$host_ip:$producer_port" \
  CONSUMER_URL="http://$host_ip:$consumer_port" \
  MODEL="$served" TOKENIZER="$model" PYTHON_BIN="$venv/bin/python" \
  REQUESTS=64 CONCURRENCY=16 SEED_CONCURRENCY=8 \
  PERFORMANCE_ORDER=cold-cached METRICS_SETTLE_SECONDS=5 \
    "$scripts/run_performance.sh" | tee "$run_dir/performance.json"

  grep -Ehi 'rdma|mlx5_bond_0' "$run_dir"/logs/*.log \
    >"$run_dir/rdma-evidence.txt" || true
  grep -Ehi 'failed_keys=[1-9]|failed key|failed.*(get|put|pack|unpack)' \
    "$run_dir/logs/producer.log" "$run_dir/logs/consumer.log" \
    >"$run_dir/failures.txt" || true
  touch "$run_dir/PASS"
  stop_services
}

nvidia-smi >"$result_root/nvidia-smi.txt"
ibv_devices >"$result_root/ibv-devices.txt" 2>&1 || true

case_index=0
for producer_tp_decode_tp in 4:2 2:1; do
  producer_tp=${producer_tp_decode_tp%:*}
  decode_tp=${producer_tp_decode_tp#*:}
  for layout in $layouts; do
    run_case "$producer_tp" "$decode_tp" "$layout" "$layout" "$case_index"
    case_index=$((case_index + 1))
  done
done

printf '%s\n' "$result_root" >"$result_root/RESULT_PATH"
echo "PASS: $result_root"
