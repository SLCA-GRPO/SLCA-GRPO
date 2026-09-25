#!/usr/bin/env bash
# Summary-reward LLM judge vLLM launcher (GPT-OSS-120B).
#
# Provides the endpoint consumed by `reward_fn.py` through the
# `LLM_JUDGE_BASE_URL` env var.
#
# Usage:
#   bash rl/judge/serve_llm_judge_gpt_oss.sh
#
# Env overrides:
#   MODEL_PATH=...   PORT=8016   TENSOR_PARALLEL_SIZE=8
#   GPU_MEMORY_UTILIZATION=0.85  MAX_MODEL_LEN=16384  MAX_NUM_SEQS=6000
#
# Hardware target: 8 x H20 96GB+. On smaller clusters use TENSOR_PARALLEL_SIZE=4
# and lower MAX_NUM_SEQS.

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"./pretrained_models/gpt-oss-120b"}

PORT=${PORT:-8016}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-6000}
HOST=${HOST:-"0.0.0.0"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"gpt-oss-120b"}

LOG_FILE=${LOG_FILE:-"llm_judge_gpt_oss_$(date +%Y%m%d_%H%M%S).log"}

while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --tp) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --max-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-util) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--port N] [--tp N] [--model PATH] [--max-len N] [--gpu-util R]"
            exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model path does not exist: $MODEL_PATH" >&2
    exit 1
fi

echo "=============================================="
echo "  LLM Judge (GPT-OSS-120B) vLLM server"
echo "=============================================="
echo "  MODEL_PATH            = $MODEL_PATH"
echo "  SERVED_MODEL_NAME     = $SERVED_MODEL_NAME"
echo "  HOST:PORT             = $HOST:$PORT"
echo "  TENSOR_PARALLEL_SIZE  = $TENSOR_PARALLEL_SIZE"
echo "  GPU_MEMORY_UTILIZATION= $GPU_MEMORY_UTILIZATION"
echo "  MAX_MODEL_LEN         = $MAX_MODEL_LEN"
echo "  MAX_NUM_SEQS          = $MAX_NUM_SEQS"
echo "  LOG_FILE              = $LOG_FILE"
echo "=============================================="

export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-75}

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --trust-remote-code \
    --disable-log-requests \
    --async-scheduling \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    2>&1 | tee "$LOG_FILE"
