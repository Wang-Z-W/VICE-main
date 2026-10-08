#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJ_DIR"
mkdir -p logs

nohup python -u train_vllm_editor.py \
    -dvc "0,1" \
    -edvc "2,3" \
    -en "liveedit" \
    -mn "minigpt-4-vicuna-7b" \
    -dna "EIC" \
    -bs 2 \
    -eps 100 \
    -sci 500 \
    -tnp "liveedit-mgpt-EIC-train" \
    > logs/train-liveedit-mgpt-EIC.log 2>&1 &
