#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
vllm_root=${VLLM_ROOT:-$(cd -- "$script_dir/../../../../.." && pwd -P)}
job_id=${JOB_ID:-${SLURM_JOB_ID:-}}
python_bin=${PYTHON_BIN:-/home/felixlinker/.venv/bin/python}
vllm_bin=${VLLM_BIN:-/home/felixlinker/.venv/bin/vllm}
model=${MODEL_PATH:-/mnt/nvme/shared/felixlinker/models/Qwen3-32B-FP8}
served=${SERVED_MODEL_NAME:-piecewise-nixl-perf}
base_tokens=${BASE_TOKENS:-1024}
suffix_tokens=${SUFFIX_TOKENS:-256}
iterations=${ITERATIONS:-10}
warmups=${WARMUPS:-2}
block_size=${BLOCK_SIZE:-16}
max_model_len=${MAX_MODEL_LEN:-2048}
kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES:-1073741824}
openssl3=${NIXL_OPENSSL3_DIR:-/home/felixlinker/opt/nixl-openssl3/root/usr/lib64}
master_bin=${MOONCAKE_MASTER_BIN:-/home/felixlinker/.venv/lib/python3.12/site-packages/mooncake/mooncake_master}
client_bin=${MOONCAKE_CLIENT_BIN:-/home/felixlinker/Mooncake/build/mooncake-store/src/mooncake_client}

if [[ -z $job_id ]]; then
  echo "Set JOB_ID to a two-node allocation with one GPU per node" >&2
  exit 2
fi
mapfile -t nodes < <(scontrol show hostnames "$(squeue -h -j "$job_id" -o %N)")
if ((${#nodes[@]} != 2)); then
  echo "Job $job_id must be running on exactly two nodes" >&2
  exit 2
fi
node1=${nodes[0]}
node2=${nodes[1]}
port_base=${PORT_BASE:-$((40000 + (job_id % 100) * 100))}
run_root=${RESULT_ROOT:-/home/felixlinker/piecewise-nixl-perf-$job_id}
mkdir -p "$run_root/services"

run_node() {
  local node=$1
  shift
  srun --jobid="$job_id" --overlap --nodes=1 --nodelist="$node" --ntasks=1 \
    --gres=none --cpus-per-task=2 \
    bash -lc 'ulimit -l unlimited; cd "$1"; shift; exec "$@"' \
    _ "$vllm_root" "$@"
}

service_ip() {
  run_node "$1" bash -c "hostname -I | tr ' ' '\n' | grep -m1 '^172\\.16\\.'"
}

rail_ip() {
  run_node "$1" bash -c \
    "ip -4 -o addr show dev '$2' | awk '{print \$4}' | cut -d/ -f1"
}

node1_service=$(service_ip "$node1")
node2_service=$(service_ip "$node2")
node1_store=$(rail_ip "$node1" bond0)
node2_store=$(rail_ip "$node2" bond0)
node1_nixl=$(rail_ip "$node1" bond1)
node2_nixl=$(rail_ip "$node2" bond1)

cat >"$run_root/manifest.json" <<EOF
{
  "job_id": "$job_id",
  "node1": "$node1",
  "node2": "$node2",
  "node1_service_ip": "$node1_service",
  "node2_service_ip": "$node2_service",
  "node1_store_ip": "$node1_store",
  "node2_store_ip": "$node2_store",
  "node1_nixl_ip": "$node1_nixl",
  "node2_nixl_ip": "$node2_nixl",
  "store_device": "mlx5_bond_0",
  "nixl_device": "mlx5_bond_1",
  "base_tokens": $base_tokens,
  "suffix_tokens": $suffix_tokens,
  "iterations": $iterations,
  "warmups": $warmups
}
EOF

declare -a step_pids=()
last_step_pid=

start_step() {
  local name=$1
  local node=$2
  local gpus=$3
  local cpus=$4
  shift 4
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
  sleep 4
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

base_env=(
  env
  PYTHONPATH="$vllm_root"
  PYTHONHASHSEED=0
  VLLM_KV_CACHE_LAYOUT=HND
  VLLM_SERVER_DEV_MODE=1
  VLLM_LOGGING_LEVEL=INFO
  VLLM_MOONCAKE_LOAD_RECV_THREADS=1
  VLLM_USE_DEEP_GEMM=0
  VLLM_MOE_USE_DEEP_GEMM=0
  UCX_TLS=rc,cuda_copy
  UCX_NET_DEVICES=mlx5_bond_1:1
  MC_NUM_QP_PER_EP=8
  MC_WORKERS_PER_CTX=8
  MC_SLICE_SIZE=262144
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
cache_prefix="$served-$job_id"

run_node "$node1" "$python_bin" "$script_dir/workload.py" store-config \
  --output "$run_root/mooncake.json" --host "$node1_store" \
  --master-port "$master_port" --metadata-port "$metadata_port" \
  --device mlx5_bond_0

for node in "$node1" "$node2"; do
  run_node "$node" env LD_LIBRARY_PATH="$openssl3" PYTHONPATH="$vllm_root" \
    UCX_TLS=rc,cuda_copy UCX_NET_DEVICES=mlx5_bond_1:1 \
    "$python_bin" -c 'import nixl, vllm, vllm._custom_ops; print(vllm.__file__, nixl.__file__)'
  run_node "$node" test -r \
    /sys/class/infiniband/mlx5_bond_0/ports/1/counters/port_xmit_data
  run_node "$node" test -r \
    /sys/class/infiniband/mlx5_bond_1/ports/1/counters/port_xmit_data
done
run_node "$node1" rm -f \
  "/tmp/lookup_rpc_port_${seeder_lookup_port}_host_${node1}_dp_rank0" \
  "/tmp/lookup_rpc_port_${prefill_lookup_port}_host_${node1}_dp_rank0"
run_node "$node2" rm -f \
  "/tmp/lookup_rpc_port_${decode_lookup_port}_host_${node2}_dp_rank0"

start_step master "$node1" 0 2 "${base_env[@]}" \
  "$master_bin" --rpc_port="$master_port" --enable_http_metadata_server=true \
  --http_metadata_server_port="$metadata_port" --enable_metric_reporting=false
master_pid=$last_step_pid
sleep 2
start_step store "$node1" 0 6 "${base_env[@]}" \
  "$client_bin" --host="$node1_store" --port="$store_port" \
  --master_server_address="$node1_store:$master_port" \
  --metadata_server="http://$node1_store:$metadata_port/metadata" --protocol=rdma \
  --device_names=mlx5_bond_0 --global_segment_size="16 GB" \
  --local_buffer_size="64 MB"
store_pid=$last_step_pid
sleep 3
kill -0 "$master_pid"
kill -0 "$store_pid"

start_step reference "$node1" 1 18 "${base_env[@]}" \
  "$python_bin" "$script_dir/launch_server.py" --role reference --model "$model" \
  --served-model-name "$served" --port "$reference_port" \
  --block-size "$block_size" --max-model-len "$max_model_len" \
  --kv-cache-memory-bytes "$kv_cache_memory_bytes"
reference_pid=$last_step_pid
wait_url "$reference_pid" "http://$node1_service:$reference_port/health" \
  "$run_root/services/reference.log"
run_node "$node2" "$python_bin" "$script_dir/workload.py" request \
  --url "http://$node1_service:$reference_port" --model-path "$model" \
  --served-model "$served" --base-tokens "$base_tokens" \
  --suffix-tokens "$suffix_tokens" --output-tokens 1 \
  --request-id perf-reference --output "$run_root/reference.json"
stop_step "$reference_pid"

start_step seeder "$node1" 1 18 "${base_env[@]}" \
  VLLM_HOST_IP="$node1_store" \
  "$python_bin" "$script_dir/launch_server.py" --role seeder --model "$model" \
  --served-model-name "$served" --port "$seeder_port" \
  --lookup-rpc-port "$seeder_lookup_port" --cache-prefix "$cache_prefix" \
  --store-tp-size 1 --block-size "$block_size" --max-model-len "$max_model_len" \
  --kv-cache-memory-bytes "$kv_cache_memory_bytes"
seeder_pid=$last_step_pid
wait_url "$seeder_pid" "http://$node1_service:$seeder_port/health" \
  "$run_root/services/seeder.log"
run_node "$node2" "$python_bin" "$script_dir/workload.py" request \
  --url "http://$node1_service:$seeder_port" --model-path "$model" \
  --served-model "$served" --base-tokens "$((base_tokens + 1))" \
  --suffix-tokens 0 --output-tokens 1 --request-id perf-seed \
  --output "$run_root/seed.json"
run_node "$node2" "$python_bin" "$script_dir/metrics.py" \
  --url "http://$node1_service:$seeder_port" --output "$run_root/seed-metrics.prom" \
  --wait-name vllm:mooncake_store_operation_bytes_total \
  --wait-label operation=save_put --wait-label status=ok --wait-minimum 1
stop_step "$seeder_pid"

snapshot_metrics() {
  local scenario=$1
  local when=$2
  run_node "$node2" "$python_bin" "$script_dir/metrics.py" \
    --url "http://$node1_service:$prefill_port" \
    --output "$run_root/$scenario/prefill-metrics-$when.prom" \
    --settle-attempts 2
  run_node "$node2" "$python_bin" "$script_dir/metrics.py" \
    --url "http://$node2_service:$decode_port" \
    --output "$run_root/$scenario/decode-metrics-$when.prom" \
    --settle-attempts 2
}

run_benchmark() {
  local scenario=$1
  local count=$2
  local output=$3
  run_node "$node2" "$python_bin" -m \
    tests.v1.kv_connector.nixl_integration.piecewise_range.perf_workload \
    --url "http://$node2_service:$proxy_port" \
    --prefill-url "http://$node1_service:$prefill_port" \
    --decode-url "http://$node2_service:$decode_port" \
    --model-path "$model" --served-model "$served" \
    --base-tokens "$base_tokens" --suffix-tokens "$suffix_tokens" \
    --output-tokens 4 --iterations "$count" \
    --request-prefix "$scenario-$output" --output "$run_root/$scenario/$output.json"
}

run_scenario() {
  local scenario=$1
  mkdir -p "$run_root/$scenario"
  start_step "$scenario-prefill" "$node1" 1 18 "${base_env[@]}" \
    VLLM_HOST_IP="$node1_store" VLLM_NIXL_SIDE_CHANNEL_HOST="$node1_nixl" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$prefill_side_port" \
    "$python_bin" "$script_dir/launch_server.py" --role prefill --mode pull \
    --scenario "$scenario" --model "$model" --served-model-name "$served" \
    --port "$prefill_port" --lookup-rpc-port "$prefill_lookup_port" \
    --cache-prefix "$cache_prefix" --store-tp-size 1 --block-size "$block_size" \
    --max-model-len "$max_model_len" \
    --kv-cache-memory-bytes "$kv_cache_memory_bytes"
  prefill_pid=$last_step_pid
  start_step "$scenario-decode" "$node2" 1 18 "${base_env[@]}" \
    VLLM_HOST_IP="$node2_store" VLLM_NIXL_SIDE_CHANNEL_HOST="$node2_nixl" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$decode_side_port" \
    "$python_bin" "$script_dir/launch_server.py" --role decode --mode pull \
    --scenario "$scenario" --model "$model" --served-model-name "$served" \
    --port "$decode_port" --lookup-rpc-port "$decode_lookup_port" \
    --cache-prefix "$cache_prefix" --store-tp-size 1 --block-size "$block_size" \
    --max-model-len "$max_model_len" \
    --kv-cache-memory-bytes "$kv_cache_memory_bytes"
  decode_pid=$last_step_pid
  wait_url "$prefill_pid" "http://$node1_service:$prefill_port/health" \
    "$run_root/services/$scenario-prefill.log"
  wait_url "$decode_pid" "http://$node2_service:$decode_port/health" \
    "$run_root/services/$scenario-decode.log"

  start_step "$scenario-proxy" "$node2" 0 2 env PYTHONPATH="$vllm_root" \
    OPENAI_API_KEY=EMPTY "$python_bin" \
    "$vllm_root/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
    --host 0.0.0.0 --port "$proxy_port" --prefiller-host "$node1_service" \
    --prefiller-port "$prefill_port" --decoder-host "$node2_service" \
    --decoder-port "$decode_port"
  proxy_pid=$last_step_pid
  wait_url "$proxy_pid" "http://$node2_service:$proxy_port/healthcheck" \
    "$run_root/services/$scenario-proxy.log"

  run_node "$node2" "$python_bin" "$script_dir/workload.py" probe \
    --url "http://$node1_service:$prefill_port" --model-path "$model" \
    --served-model "$served" --base-tokens "$((base_tokens + 1))" \
    --suffix-tokens 0 --request-id "$scenario-probe" \
    --expected-cached-tokens "$base_tokens"
  run_node "$node2" "$python_bin" "$script_dir/workload.py" reset \
    --url "http://$node1_service:$prefill_port"
  run_node "$node2" "$python_bin" "$script_dir/workload.py" reset \
    --url "http://$node2_service:$decode_port"
  run_node "$node2" "$python_bin" "$script_dir/workload.py" request \
    --url "http://$node2_service:$proxy_port" --model-path "$model" \
    --served-model "$served" --base-tokens "$base_tokens" \
    --suffix-tokens "$suffix_tokens" --output-tokens 1 \
    --request-id "$scenario-correctness" \
    --output "$run_root/$scenario/correctness.json"
  run_benchmark "$scenario" "$warmups" warmup
  snapshot_metrics "$scenario" before

  stop_file=$run_root/$scenario/stop-sampling
  ready1=$run_root/$scenario/net-node1.ready
  ready2=$run_root/$scenario/net-node2.ready
  rm -f "$stop_file" "$ready1" "$ready2"
  start_step "$scenario-net-node1" "$node1" 0 1 "$python_bin" -m \
    tests.v1.kv_connector.nixl_integration.piecewise_range.net_sampler \
    --devices mlx5_bond_0 mlx5_bond_1 --interval 0.02 \
    --stop-file "$stop_file" --ready-file "$ready1" \
    --output "$run_root/$scenario/net-node1.json"
  sampler1_pid=$last_step_pid
  start_step "$scenario-net-node2" "$node2" 0 1 "$python_bin" -m \
    tests.v1.kv_connector.nixl_integration.piecewise_range.net_sampler \
    --devices mlx5_bond_0 mlx5_bond_1 --interval 0.02 \
    --stop-file "$stop_file" --ready-file "$ready2" \
    --output "$run_root/$scenario/net-node2.json"
  sampler2_pid=$last_step_pid
  for _ in $(seq 1 600); do
    if [[ -f $ready1 && -f $ready2 ]]; then
      break
    fi
    if ! kill -0 "$sampler1_pid" 2>/dev/null; then
      tail -80 "$run_root/services/$scenario-net-node1.log" >&2 || true
      return 1
    fi
    if ! kill -0 "$sampler2_pid" 2>/dev/null; then
      tail -80 "$run_root/services/$scenario-net-node2.log" >&2 || true
      return 1
    fi
    sleep 0.1
  done
  if [[ ! -f $ready1 || ! -f $ready2 ]]; then
    echo "Network samplers did not become ready within 60 seconds" >&2
    return 1
  fi
  run_benchmark "$scenario" "$iterations" benchmark
  run_node "$node2" touch "$stop_file"
  wait "$sampler1_pid"
  wait "$sampler2_pid"
  snapshot_metrics "$scenario" after

  stop_step "$proxy_pid"
  stop_step "$decode_pid"
  stop_step "$prefill_pid"
}

run_scenario legacy
run_scenario piecewise

run_node "$node2" "$python_bin" -m \
  tests.v1.kv_connector.nixl_integration.piecewise_range.perf_evidence \
  --root "$run_root" --output "$run_root/performance.json"
run_node "$node2" touch "$run_root/PASS"
echo "PASS: $run_root"
