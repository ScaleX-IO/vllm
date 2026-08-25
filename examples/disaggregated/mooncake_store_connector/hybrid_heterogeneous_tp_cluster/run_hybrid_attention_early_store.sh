#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

: "${VENV_ROOT:?Set VENV_ROOT}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${REFERENCE_URL:?Set REFERENCE_URL}"
: "${PRODUCER_URL:?Set PRODUCER_URL}"
: "${CONSUMER_URL:?Set CONSUMER_URL}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
served=${SERVED_MODEL_NAME:-hybrid-heterogeneous-tp-test}
result_root=${RESULT_ROOT:-$PWD/hybrid-attention-early-store}
reference_json=$result_root/reference.json
mkdir -p "$result_root"

"$VENV_ROOT/bin/python" "$script_dir/checkpoint_workload.py" reference \
  --reference-url "$REFERENCE_URL" --model "$served" \
  --tokenizer "$MODEL_PATH" --reference-json "$reference_json" \
  --prompt-tokens "${PROMPT_TOKENS:-1601}" \
  --prompt-repetitions "${PROMPT_REPETITIONS:-256}" \
  --output-tokens "${OUTPUT_TOKENS:-401}"

"$VENV_ROOT/bin/python" "$script_dir/checkpoint_workload.py" checkpoint \
  --producer-url "$PRODUCER_URL" --consumer-url "$CONSUMER_URL" \
  --model "$served" --tokenizer "$MODEL_PATH" \
  --reference-json "$reference_json" \
  --prompt-tokens "${PROMPT_TOKENS:-1601}" \
  --prompt-repetitions "${PROMPT_REPETITIONS:-256}" \
  --output-tokens "${OUTPUT_TOKENS:-401}" \
  --expected-decode-cached-tokens "${EXPECTED_DECODE_CACHED_TOKENS:-1600}" \
  --expected-extended-cached-tokens \
    "${EXPECTED_EXTENDED_CACHED_TOKENS:-1600}" \
  --visibility-timeout "${VISIBILITY_TIMEOUT:-300}" \
  | tee "$result_root/result.json"
