#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJ_DIR"

DATA_NAMES=(evqa eic vlkeb waterv2)

for DATA_NAME in "${DATA_NAMES[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training VICE's scope head for $DATA_NAME..."
    python -u train_vice_cls.py \
        --data_name "$DATA_NAME" \
        --device "0" \
        --train_epochs 80 \
        --train_batch_size 32 \
        --train_lr 1e-3 \
        --projection_dim 256 \
        --loc_margin 0.5 \
        --loc_lambda 1.0
done
