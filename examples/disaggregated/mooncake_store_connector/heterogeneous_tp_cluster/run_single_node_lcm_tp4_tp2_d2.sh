#!/usr/bin/env bash

set -euo pipefail
ulimit -l unlimited

feature=${FEATURE_ROOT:-/home/felixlinker/vllm-tp-share-v2}
tests=${TEST_ROOT:-/home/felixlinker/vllm-decode-offloading-tests}
scripts=$tests/examples/disaggregated/mooncake_store_connector/heterogeneous_tp_cluster
venv=${VENV_ROOT:-/home/felixlinker/.venv}
model=${MODEL_PATH:-/mnt/nvme/shared/felixlinker/models/Qwen3-32B-FP8}
served=tp-lcm-p4-p2-d2
run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}
result_root=${RESULT_ROOT:-/home/felixlinker/tp-lcm-p4-p2-d2-$run_id}
mkdir -p "$result_root/logs"

host_ip=$(hostname -I | awk '{print $1}')
port_base=${PORT_BASE:-$((52000 + ($$ % 100) * 10))}
master_port=$port_base
metadata_port=$((port_base + 1))
store_port=$((port_base + 2))
prefill4_port=$((port_base + 3))
prefill2_port=$((port_base + 4))
decode_port=$((port_base + 5))
metrics_port=$((port_base + 6))
cache_prefix=tp-lcm-p4-p2-d2-$run_id
prefill4_gpus=${PREFILL4_GPUS:-0,1,2,3}
prefill2_gpus=${PREFILL2_GPUS:-4,5}
decode_gpus=${DECODE_GPUS:-6,7}
decode_gpus=${decode_gpus//:/,}
prefill4_gpu_memory=${PREFILL4_GPU_MEMORY_UTILIZATION:-0.8}
prefill2_gpu_memory=${PREFILL2_GPU_MEMORY_UTILIZATION:-0.8}
decode_gpu_memory=${DECODE_GPU_MEMORY_UTILIZATION:-0.8}

export PYTHONPATH=$feature
export PATH=$venv/bin:$PATH
export VLLM_BIN=$venv/bin/vllm
export PYTHONHASHSEED=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_MOONCAKE_LOAD_RECV_THREADS=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

export MOONCAKE_CONFIG_PATH=$result_root/mooncake.json
export VLLM_MOONCAKE_STORE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH
printf '%s\n' "{\"metadata_server\":\"http://$host_ip:$metadata_port/metadata\",\"master_server_address\":\"$host_ip:$master_port\",\"protocol\":\"rdma\",\"device_name\":\"mlx5_bond_0\",\"mode\":\"standalone-store\",\"global_segment_size\":\"0 B\",\"local_buffer_size\":\"64 MB\",\"enable_offload\":false}" >"$MOONCAKE_CONFIG_PATH"

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
      tail -200 "$log" || true
      return 1
    fi
    sleep 2
  done
  tail -200 "$log" || true
  return 1
}

start "$result_root/logs/master.log" mooncake_master \
  --rpc_port="$master_port" \
  --enable_http_metadata_server=true \
  --http_metadata_server_port="$metadata_port" \
  --enable_metric_reporting=false --metrics_port="$metrics_port"
sleep 2
start "$result_root/logs/store.log" \
  /home/felixlinker/Mooncake/build/mooncake-store/src/mooncake_client \
  --host="$host_ip" --port="$store_port" \
  --master_server_address="$host_ip:$master_port" \
  --metadata_server="http://$host_ip:$metadata_port/metadata" \
  --protocol=rdma --device_names=mlx5_bond_0 \
  --global_segment_size="128 GB" --local_buffer_size="64 MB"
sleep 3

common_args=(
  --model "$model"
  --served-model-name "$served"
  --enable-store-tp-lcm
  --prefill-tp-sizes 4 2
  --cache-prefix "$cache_prefix"
  --enforce-eager
)

cd "$feature"
start "$result_root/logs/prefill4.log" \
  env CUDA_VISIBLE_DEVICES="$prefill4_gpus" VLLM_HOST_IP="$host_ip" \
  "$venv/bin/python" "$scripts/launch_server.py" --role producer \
  --port "$prefill4_port" --lookup-rpc-port "$prefill4_port" \
  --tensor-parallel-size 4 --kv-cache-layout LBHNC \
  --gpu-memory-utilization "$prefill4_gpu_memory" "${common_args[@]}"
prefill4_pid=${pids[-1]}
wait_url "$prefill4_pid" "http://$host_ip:$prefill4_port" \
  "$result_root/logs/prefill4.log"

start "$result_root/logs/prefill2.log" \
  env CUDA_VISIBLE_DEVICES="$prefill2_gpus" VLLM_HOST_IP="$host_ip" \
  "$venv/bin/python" "$scripts/launch_server.py" --role producer \
  --port "$prefill2_port" --lookup-rpc-port "$prefill2_port" \
  --tensor-parallel-size 2 --kv-cache-layout LBHNC \
  --gpu-memory-utilization "$prefill2_gpu_memory" "${common_args[@]}"
prefill2_pid=${pids[-1]}
wait_url "$prefill2_pid" "http://$host_ip:$prefill2_port" \
  "$result_root/logs/prefill2.log"

start "$result_root/logs/decode2.log" \
  env CUDA_VISIBLE_DEVICES="$decode_gpus" VLLM_HOST_IP="$host_ip" \
  "$venv/bin/python" "$scripts/launch_server.py" --role consumer \
  --port "$decode_port" --lookup-rpc-port "$decode_port" \
  --tensor-parallel-size 2 --kv-cache-layout LBHNC --save-decode-cache \
  --gpu-memory-utilization "$decode_gpu_memory" "${common_args[@]}"
decode_pid=${pids[-1]}
wait_url "$decode_pid" "http://$host_ip:$decode_port" \
  "$result_root/logs/decode2.log"

PREFILL_FIRST_URL="http://$host_ip:$prefill4_port" \
PREFILL_SECOND_URL="http://$host_ip:$prefill2_port" \
DECODE_URL="http://$host_ip:$decode_port" \
MODEL="$served" TOKENIZER="$model" PYTHON_BIN="$venv/bin/python" \
  "$scripts/run_lcm_multi_prefill.sh" | tee "$result_root/result.json"

grep -Ehi 'rdma|mlx5_bond_0' "$result_root"/logs/*.log \
  >"$result_root/rdma-evidence.txt" || true
grep -Ehi 'failed_keys=[1-9]|failed key|failed.*(get|put|pack|unpack)' \
  "$result_root/logs/prefill4.log" "$result_root/logs/prefill2.log" \
  "$result_root/logs/decode2.log" >"$result_root/failures.txt" || true
touch "$result_root/PASS"
printf '%s\n' "$result_root" >"$result_root/RESULT_PATH"
echo "PASS: $result_root"
