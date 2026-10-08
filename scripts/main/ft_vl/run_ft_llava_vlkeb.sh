#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJ_DIR"
mkdir -p logs

EXP_NAME="M-FT-LLaVA-VLKEB-1000"
LOG_FILE="logs/${EXP_NAME}.log"

nohup python -u test_vllm_edit.py \
    -dvc "2" \
    -en "ft_vl" \
    -mn "llava-v1.5-7b" \
    -dn "VLKEB" \
    -dnp "" \
    -dsn 999999 \
    -sen 1000 \
    -online false \
    -exp "$EXP_NAME" \
    > "$LOG_FILE" 2>&1 &
