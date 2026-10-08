#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJ_DIR"
mkdir -p logs

EXP_NAME="M-VICE-MGPT-EIC-1000"
LOG_FILE="logs/${EXP_NAME}.log"

SCOPE_CKPT="${SCOPE_CKPT:-/path/to/vice_scope_EIC.pt}"

if [ ! -f "$SCOPE_CKPT" ]; then
    echo "Set SCOPE_CKPT to an existing VICE scope-head checkpoint file (.pt). See README.md." >&2
    echo "Checkpoint not found: $SCOPE_CKPT" >&2
    exit 1
fi

nohup python -u test_vllm_edit.py \
    -dvc "4" \
    -en "vice_vl" \
    -mn "minigpt-4-vicuna-7b" \
    -dn "EIC" \
    -dnp "" \
    -dsn 999999 \
    -sen 1000 \
    -thr "0.65" \
    -kmeans true \
    -usesum true \
    -online false \
    --scope_classifier_ckpt "$SCOPE_CKPT" \
    -exp "$EXP_NAME" \
    > "$LOG_FILE" 2>&1 &
