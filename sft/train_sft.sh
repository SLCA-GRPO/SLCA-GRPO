#!/bin/bash
# SFT training launcher for SLCA-GRPO.
# Usage:
#   bash sft/train_sft.sh <yaml>             # 8 GPUs by default
#   bash sft/train_sft.sh <yaml> <num_gpus>
#
# Examples:
#   bash sft/train_sft.sh sft/qwen2_5_7b_split_sft.yaml
#   bash sft/train_sft.sh sft/qwen2_5_3b_split_sft.yaml 4
#
# Prerequisites:
#   - LLaMA-Factory is available at $LLAMAFACTORY_DIR (default: ./LLaMA-Factory)
#   - The dataset referenced in the YAML is registered in
#     $LLAMAFACTORY_DIR/data/dataset_info.json (see sft/README.md)
#
set -e

CONFIG_FILE=${1:-"$(dirname "$0")/qwen2_5_7b_split_sft.yaml"}
NUM_GPUS=${2:-8}
LLAMAFACTORY_DIR=${LLAMAFACTORY_DIR:-"./LLaMA-Factory"}

if [ ! -f "$CONFIG_FILE" ]; then
  echo "[ERROR] config file not found: $CONFIG_FILE" >&2
  exit 1
fi
if [ ! -d "$LLAMAFACTORY_DIR" ]; then
  echo "[ERROR] LLaMA-Factory not found at $LLAMAFACTORY_DIR" >&2
  echo "        clone https://github.com/hiyouga/LLaMA-Factory and set LLAMAFACTORY_DIR" >&2
  exit 1
fi

CONFIG_ABS="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"

cd "$LLAMAFACTORY_DIR"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "SLCA-GRPO SFT training"
echo "  config   : $CONFIG_ABS"
echo "  num_gpus : $NUM_GPUS"
echo "  LLaMA-F. : $(pwd)"
echo "=========================================="

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --master_port=29500 \
  src/train.py \
  "$CONFIG_ABS"
