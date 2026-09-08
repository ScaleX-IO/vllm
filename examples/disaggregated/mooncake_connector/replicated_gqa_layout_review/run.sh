#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${VLLM_REPO:?Set VLLM_REPO to a checkout of PR 52516 at d61e44b85b}"
: "${VENV:?Set VENV to the Python environment containing torch and vLLM deps}"
export VLLM_REPO
export PYTHONPATH="$VLLM_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
source "$VENV/bin/activate"

expected_head=d61e44b85b04cb9b3c90938e57a1ba72a98cab5d
actual_head=$(git -C "$VLLM_REPO" rev-parse HEAD)
if [[ "$actual_head" != "$expected_head" ]]; then
    echo "Expected PR head $expected_head; got $actual_head" >&2
    exit 1
fi
git -C "$VLLM_REPO" cat-file -e 5db652225f00b55783823ae6606d36925e3e3efe^{commit}

case "${1:-}" in
    padded) script=reproduce_padded_page.py ;;
    nvfp4) script=reproduce_nvfp4_slots.py ;;
    *) echo "Usage: $0 {padded|nvfp4}" >&2; exit 2 ;;
esac
cd "$VLLM_REPO"
exec "$VENV/bin/python" -u "$TEST_DIR/$script"
