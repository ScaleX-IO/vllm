#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${PREFILL_FIRST_URL:?set PREFILL_FIRST_URL}"
: "${PREFILL_SECOND_URL:?set PREFILL_SECOND_URL}"
: "${DECODE_URL:?set DECODE_URL}"
: "${MODEL:?set MODEL}"
: "${TOKENIZER:?set TOKENIZER}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python3}

exec "$python_bin" "$script_dir/workload.py" lcm-multi-prefill \
  --prefill-first-url "$PREFILL_FIRST_URL" \
  --prefill-second-url "$PREFILL_SECOND_URL" \
  --decode-url "$DECODE_URL" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --block-size "${BLOCK_SIZE:-16}" \
  --prompt-repetitions "${PROMPT_REPETITIONS:-80}" \
  --decode-tokens "${DECODE_TOKENS:-64}" \
  --visibility-timeout "${VISIBILITY_TIMEOUT:-300}" \
  --request-timeout "${REQUEST_TIMEOUT:-300}"
