#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the served model name}"
: "${TOKENIZER:?Set TOKENIZER to the tokenizer name or path}"
: "${PROBE_URL:?Set PROBE_URL to the endpoint that generates decode KV}"
: "${CACHED_URL:?Set CACHED_URL to a separate endpoint that reads the shared Store}"
: "${REFERENCE_URL:?Set REFERENCE_URL to an endpoint without a KV connector}"
: "${TARGET_URLS:?Set TARGET_URLS to one or more space-separated serving endpoints}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python3}
output_dir=${OUTPUT_DIR:-decode-offloading-result}
mkdir -p "$output_dir"

read -r -a target_urls <<<"$TARGET_URLS"
(( ${#target_urls[@]} > 0 )) || { echo "TARGET_URLS is empty" >&2; exit 2; }

"$python_bin" "$script_dir/workload.py" probe \
  --url "$PROBE_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --output "$output_dir/probe.json" \
  --decode-tokens "${PROBE_DECODE_TOKENS:-65}" \
  --prompt-repetitions "${PROMPT_REPETITIONS:-80}"

# Verify before the pressure phase so unrelated Store eviction cannot turn a
# correctness test into a capacity test. CACHED_URL must not be PROBE_URL.
[[ $CACHED_URL != "$PROBE_URL" ]] || {
  echo "CACHED_URL must be separate from PROBE_URL to exclude local KV hits" >&2
  exit 2
}
"$python_bin" "$script_dir/workload.py" verify \
  --cached-url "$CACHED_URL" --reference-url "$REFERENCE_URL" \
  --model "$MODEL" --probe "$output_dir/probe.json" \
  --block-size "${BLOCK_SIZE:-16}" | tee "$output_dir/verification.json"

stress_args=()
for url in "${target_urls[@]}"; do stress_args+=(--urls "$url"); done
"$python_bin" "$script_dir/workload.py" stress "${stress_args[@]}" \
  --model "$MODEL" --requests "${REQUESTS:-64}" \
  --concurrency "${CONCURRENCY:-16}" \
  --output-tokens "${OUTPUT_TOKENS:-128}" \
  --prompt-repetitions "${PROMPT_REPETITIONS:-80}" \
  | tee "$output_dir/stress.json"

echo "PASS: artifacts: $output_dir"
