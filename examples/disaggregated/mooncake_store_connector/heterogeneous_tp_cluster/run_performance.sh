#!/usr/bin/env bash
set -euo pipefail

: "${PRODUCER_URL:?Set PRODUCER_URL}"
: "${CONSUMER_URL:?Set CONSUMER_URL}"
: "${MODEL:?Set MODEL to the served model name}"
: "${TOKENIZER:?Set TOKENIZER to the tokenizer name or path}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON_BIN:-python3}" "$script_dir/workload.py" performance \
  --producer-url "$PRODUCER_URL" \
  --consumer-url "$CONSUMER_URL" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --block-size "${BLOCK_SIZE:-16}" \
  --prompt-repetitions "${PROMPT_REPETITIONS:-80}" \
  --output-tokens "${OUTPUT_TOKENS:-64}" \
  --requests "${REQUESTS:-64}" \
  --concurrency "${CONCURRENCY:-16}" \
  --seed-concurrency "${SEED_CONCURRENCY:-8}" \
  --warmup-requests "${WARMUP_REQUESTS:-2}" \
  --metrics-settle-seconds "${METRICS_SETTLE_SECONDS:-2}" \
  --order "${PERFORMANCE_ORDER:-cold-cached}" \
  --visibility-timeout "${VISIBILITY_TIMEOUT:-300}"
