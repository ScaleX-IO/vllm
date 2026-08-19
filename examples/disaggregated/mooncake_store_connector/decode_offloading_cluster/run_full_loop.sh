#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the served model name}"
: "${TOKENIZER:?Set TOKENIZER to the tokenizer name or path}"
: "${PD_PROXY_URL:?Set PD_PROXY_URL to the P-D proxy endpoint}"
: "${NEXT_PREFILL_URL:?Set NEXT_PREFILL_URL to a fresh P-side Store reader}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python3}
output_dir=${OUTPUT_DIR:-decode-offloading-loop-result}
mkdir -p "$output_dir"

"$python_bin" "$script_dir/workload.py" probe \
  --url "$PD_PROXY_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --output "$output_dir/probe.json" \
  --decode-tokens "${PROBE_DECODE_TOKENS:-65}" \
  --prompt-repetitions "${PROMPT_REPETITIONS:-80}"

[[ $NEXT_PREFILL_URL != "$PD_PROXY_URL" ]] || {
  echo "NEXT_PREFILL_URL must be a fresh endpoint to exclude local KV hits" >&2
  exit 2
}
"$python_bin" "$script_dir/workload.py" verify-recorded \
  --cached-url "$NEXT_PREFILL_URL" \
  --model "$MODEL" --probe "$output_dir/probe.json" \
  --block-size "${BLOCK_SIZE:-16}" | tee "$output_dir/verification.json"

echo "PASS: full P-D-Store-P loop; artifacts: $output_dir"
