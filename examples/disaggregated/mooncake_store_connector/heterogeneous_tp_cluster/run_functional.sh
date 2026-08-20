#!/usr/bin/env bash
set -euo pipefail

: "${PRODUCER_URL:?Set PRODUCER_URL}"
: "${CONSUMER_URL:?Set CONSUMER_URL}"
: "${REFERENCE_URL:?Set REFERENCE_URL}"
: "${MODEL:?Set MODEL to the served model name}"
: "${TOKENIZER:?Set TOKENIZER to the tokenizer name or path}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
args=(
  functional
  --producer-url "$PRODUCER_URL"
  --consumer-url "$CONSUMER_URL"
  --reference-url "$REFERENCE_URL"
  --model "$MODEL"
  --tokenizer "$TOKENIZER"
  --block-size "${BLOCK_SIZE:-16}"
  --prompt-repetitions "${PROMPT_REPETITIONS:-80}"
  --output-tokens "${OUTPUT_TOKENS:-64}"
  --visibility-timeout "${VISIBILITY_TIMEOUT:-300}"
)
if [[ ${ALLOW_TOKEN_MISMATCH:-0} == 1 ]]; then
  args+=(--allow-token-mismatch)
fi
"${PYTHON_BIN:-python3}" "$script_dir/workload.py" "${args[@]}"
