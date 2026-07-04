#!/bin/bash
# VTX-300M staged pretraining: 2k → 8k → 32k (RoPE 128k)
# Total: ~8B tokens across 3 curriculum stages
#
# Stage 1: 5B tokens @ 2k context  (76,300 steps)
# Stage 2: 2B tokens @ 8k context  (7,630 steps)
# Stage 3: 1B tokens @ 32k context (1,900 steps, RoPE θ=50M → 128k effective)

set -e

TOKEN="${HF_TOKEN:-$1}"
if [ -z "$TOKEN" ]; then
    echo "Usage: bash train_staged.sh <HF_TOKEN>"
    exit 1
fi

CKPT_DIR="/tmp/vtx_300m_ckpts"
mkdir -p "$CKPT_DIR"

echo "============================================"
echo "Stage 1: 5B tokens @ 2k context"
echo "============================================"
python src/pretrain.py \
    --config src/stage1_2k.json \
    --token "$TOKEN" \
    --save-dir "$CKPT_DIR/stage1_2k"

echo ""
echo "============================================"
echo "Stage 2: 2B tokens @ 8k context"
echo "============================================"
# Resume from stage 1's latest checkpoint
python src/pretrain.py \
    --config src/stage2_8k.json \
    --token "$TOKEN" \
    --save-dir "$CKPT_DIR/stage2_8k"

echo ""
echo "============================================"
echo "Stage 3: 1B tokens @ 32k context (RoPE 128k)"
echo "============================================"
python src/pretrain.py \
    --config src/stage3_32k.json \
    --token "$TOKEN" \
    --save-dir "$CKPT_DIR/stage3_32k"

echo ""
echo "============================================"
echo "Training complete!"
echo "Final checkpoint: $CKPT_DIR/stage3_32k/"
echo "============================================"
