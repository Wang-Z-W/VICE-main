#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJ_DIR"
mkdir -p logs

nohup python -u train_vllm_editor.py \
    -dvc "4,5" \
    -edvc "6,7" \
    -en "liveedit" \
    -mn "minigpt-4-vicuna-7b" \
    -dna "VLKEB" \
    -bs 2 \
    -eps 100 \
    -sci 500 \
    -tnp "liveedit-mgpt-VLKEB-train" \
    > logs/train-liveedit-mgpt-VLKEB.log 2>&1 &
