#!/usr/bin/env bash
# Schema-Guided LLM Simulator (SGLS) vLLM launcher.
#
# Spawns an OpenAI-compatible endpoint backed by Qwen3-235B-A22B-Instruct-2507.
# During RL rollout, ToucanVLLMTool (see rl/slca_grpo/tools/toucan_vllm_tool.py)
# sends each `<tool_call>` here and parses the `<tool_response>` that comes
# back.
#
# Usage:
#   bash rl/sgls/serve_sgls_qwen3_235b.sh
#
# Env overrides:
#   MODEL_PATH=...  TP_SIZE=8  PORT=8003  GPU_MEM_UTIL=0.70  MAX_NUM_SEQS=300
#   bash rl/sgls/serve_sgls_qwen3_235b.sh
#
# Hardware target: 8 x H20 / H100 80GB+. Adjust `TP_SIZE` / `GPU_MEM_UTIL`
# for smaller nodes.

set -euo pipefail

# Point at your local copy / mount of Qwen3-235B-A22B-Instruct-2507.
MODEL_PATH="${MODEL_PATH:-./pretrained_models/Qwen3-235B-A22B-Instruct-2507}"

TP_SIZE="${TP_SIZE:-8}"
PORT="${PORT:-8003}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-300}"
MODEL_NAME="${MODEL_NAME:-qwen3-235b-a22b}"
DTYPE="${DTYPE:-bfloat16}"

# Reduce large-allocation fragmentation on long runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Starting SGLS vLLM server (Qwen3-235B-A22B)..."
echo "  MODEL_PATH   = ${MODEL_PATH}"
echo "  MODEL_NAME   = ${MODEL_NAME}"
echo "  TP_SIZE      = ${TP_SIZE}"
echo "  HOST:PORT    = ${HOST}:${PORT}"
echo "  MAX_MODEL_LEN= ${MAX_MODEL_LEN}"
echo "  GPU_MEM_UTIL = ${GPU_MEM_UTIL}"
echo "  MAX_NUM_SEQS = ${MAX_NUM_SEQS}"
echo

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --dtype "${DTYPE}" \
  --disable-custom-all-reduce \
  --enable-prefix-caching \
  --enable-expert-parallel \
  --trust-remote-code
