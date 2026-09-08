#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${VLLM_REPO:?Set VLLM_REPO to a checkout of PR 52516 at 3d52c8329d}"
: "${VENV:?Set VENV to the Python environment containing torch and mooncake}"
export VLLM_REPO
export PYTHONPATH="$VLLM_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
source "$VENV/bin/activate"
ulimit -l unlimited

expected_head=3d52c8329ded27d62e043c960cc3e2250a5c02de
actual_head=$(git -C "$VLLM_REPO" rev-parse HEAD)
if [[ "$actual_head" != "$expected_head" ]]; then
    echo "Expected PR head $expected_head; got $actual_head" >&2
    exit 1
fi
git -C "$VLLM_REPO" cat-file -e e862c2f45bc81c72bb84f3807d88d60cfb93d694^{commit}

case "${1:-}" in
    hybrid) script=reproduce_hybrid_gdn.py ;;
    region) script=reproduce_region_length.py ;;
    *) echo "Usage: $0 {hybrid|region}" >&2; exit 2 ;;
esac
cd "$VLLM_REPO"
exec "$VENV/bin/python" -u "$TEST_DIR/$script"
