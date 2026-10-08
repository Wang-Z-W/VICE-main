#!/bin/bash
# Generate image summaries and diverse embeddings using LLaVA-v1.5.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJ_DIR"
DATA_NAME="${DATA_NAME:-waterv2}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG="logs/generate_summaries_${DATA_NAME}_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG")"

python -u preprocess_image_summaries.py \
    --data_name "$DATA_NAME" "$@" \
    2>&1 | tee -a "$LOG"
