#!/usr/bin/env bash
set -euo pipefail

: "${GPUKV_MODEL:?Set GPUKV_MODEL to a local Hugging Face model path}"

run_user=${SUDO_USER:-${USER:-}}
run_home=$(getent passwd "$run_user" 2>/dev/null | cut -d: -f6)
run_home=${run_home:-$HOME}
bam_root=${BAM_ROOT:-$run_home/bam-master}
export LD_LIBRARY_PATH="$bam_root/build/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
vllm_bin=${GPUKV_VLLM_BIN:-$run_home/miniconda3/envs/vllm-gpukv/bin/vllm}
if [[ ! -x "$vllm_bin" ]]; then
  echo "vLLM executable is missing; set GPUKV_VLLM_BIN" >&2
  exit 2
fi

mode=${GPUKV_BENCH_MODE:-gpukv}
host=${GPUKV_HOST:-127.0.0.1}
port=${GPUKV_PORT:-8000}
served_model=${GPUKV_SERVED_MODEL:-gpukv-benchmark}
max_num_seqs=${GPUKV_MAX_NUM_SEQS:-4}
kv_cache_bytes=${GPUKV_KV_CACHE_BYTES:-536870912}

common=(
  "$GPUKV_MODEL"
  --host "$host"
  --port "$port"
  --served-model-name "$served_model"
  --enable-prefix-caching
  --max-model-len "${GPUKV_MAX_MODEL_LEN:-4096}"
  --kv-cache-memory-bytes "$kv_cache_bytes"
  --max-num-seqs "$max_num_seqs"
  --enforce-eager
)

if [[ "$mode" == recompute ]]; then
  exec "$vllm_bin" serve "${common[@]}"
fi
if [[ "$mode" != gpukv ]]; then
  echo "GPUKV_BENCH_MODE must be gpukv or recompute" >&2
  exit 2
fi

kv_config=$(printf '%s' "{
  \"kv_connector\":\"GPUKVConnector\",
  \"kv_role\":\"kv_both\",
  \"kv_connector_module_path\":\"gpu_kv_connector.connector\",
  \"kv_connector_extra_config\":{
    \"device_path\":\"${GPUKV_DEVICE:-/dev/libnvm0}\",
    \"catalog_path\":\"${GPUKV_CATALOG_PATH:-/tmp/vllm-gpu-kv/benchmark.sqlite3}\",
    \"disk_start_page\":${GPUKV_DISK_START_PAGE:-1048576},
    \"capacity_pages\":${GPUKV_CAPACITY_PAGES:-1000000},
    \"max_objects\":${GPUKV_MAX_OBJECTS:-8192},
    \"max_batch\":${GPUKV_MAX_BATCH:-8192},
    \"queue_depth\":${GPUKV_QUEUE_DEPTH:-64},
    \"num_queues\":${GPUKV_NUM_QUEUES:-16},
    \"max_request_pages\":${GPUKV_MAX_REQUEST_PAGES:-0},
    \"max_superrequest_objects\":${GPUKV_MAX_SUPERREQUEST_OBJECTS:-512},
    \"superrequest_target_bytes\":${GPUKV_SUPERREQUEST_TARGET_BYTES:-524288},
    \"min_superrequest_bytes\":${GPUKV_MIN_SUPERREQUEST_BYTES:-65536},
    \"read_executor_blocks\":${GPUKV_READ_EXECUTOR_BLOCKS:-16},
    \"write_executor_blocks\":${GPUKV_WRITE_EXECUTOR_BLOCKS:-8},
    \"prefetch_layers\":${GPUKV_PREFETCH_LAYERS:-2},
    \"ready_cache_entries\":${GPUKV_READY_CACHE_ENTRIES:-65536},
    \"fuse_kv_planes\":${GPUKV_FUSE_KV_PLANES:-true},
    \"reset_catalog\":${GPUKV_RESET_CATALOG:-true}
  }
}")

exec "$vllm_bin" serve "${common[@]}" --kv-transfer-config "$kv_config"
