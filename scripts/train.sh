#!/bin/bash
# Training script for Skin and Abdominal Wall Segmentation

# Default values
MODEL="swin_unetr"
EPOCHS=500
BATCH_SIZE=2
LR=1e-4
GPU="2,3,4,5"
EXP_NAME=""
OUTPUT_DIR="./outputs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set experiment name if not provided
if [ -z "$EXP_NAME" ]; then
    EXP_NAME="${MODEL}_$(date +%Y%m%d_%H%M%S)"
fi

# Run training
echo "Starting training with model: $MODEL"
echo "Experiment: $EXP_NAME"
echo "GPU: $GPU"

python -m src.train \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --gpu "$GPU" \
    --exp-name "$EXP_NAME" \
    --output-dir "$OUTPUT_DIR"
