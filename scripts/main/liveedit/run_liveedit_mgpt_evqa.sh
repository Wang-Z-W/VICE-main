#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJ_DIR"
mkdir -p logs

CKPT="${CKPT:-/path/to/liveedit_mgpt_evqa.pt}"
EXP_NAME="M-LE-MGPT-EVQA-1000"
LOG_FILE="logs/${EXP_NAME}.log"

if [ ! -f "$CKPT" ]; then
    echo "Set CKPT to an existing LiveEdit checkpoint file (.pt). See README.md." >&2
    echo "Checkpoint not found: $CKPT" >&2
    exit 1
fi

nohup python -u test_vllm_edit.py \
    -dvc "6" \
    -en "liveedit" \
    -mn "minigpt-4-vicuna-7b" \
    -dn "EVQA" \
    -dnp "" \
    -dsn 999999 \
    -sen 1000 \
    -online false \
    -ckpt "$CKPT" \
    -exp "$EXP_NAME" \
    > "$LOG_FILE" 2>&1 &
