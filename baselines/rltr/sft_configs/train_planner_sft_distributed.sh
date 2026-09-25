#!/usr/bin/env bash
# Multi-node planner SFT for the RLTR baseline (LLaMA-Factory).
#
# The 8B-Base planner was trained on 4 nodes x 8 GPUs. Run this script once per
# node with a different NODE_RANK; rank 0 must be reachable at MASTER_ADDR.
#
# Usage (on every node):
#   MASTER_ADDR=<rank-0 hostname or IP> \
#   NODE_RANK=<0..NNODES-1> \
#   NNODES=4 \
#   LLAMAFACTORY_DIR=/path/to/LLaMA-Factory \
#   bash baselines/rltr/sft_configs/train_planner_sft_distributed.sh
#
# For the 3B / 7B planners a single node is enough; just use
#   bash sft/train_sft.sh baselines/rltr/sft_configs/planner_sft_qwen2.5-7B.yaml
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONFIG_FILE=${CONFIG_FILE:-${SCRIPT_DIR}/planner_sft_qwen3-8B-Base.yaml}
LLAMAFACTORY_DIR=${LLAMAFACTORY_DIR:-${REPO_ROOT}/LLaMA-Factory}

# Rendezvous. MASTER_ADDR has no default on purpose: it is cluster-specific.
if [ -z "${MASTER_ADDR:-}" ]; then
    echo "[ERROR] MASTER_ADDR is not set. Point it at the rank-0 node." >&2
    exit 1
fi
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=${NNODES:-4}
export NODE_RANK=${NODE_RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# Interconnect. Override NCCL_SOCKET_IFNAME to match your fabric; set
# NCCL_IB_DISABLE=0 if you do have InfiniBand.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export PYTHONUNBUFFERED=1

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "[ERROR] config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi
if [ ! -d "${LLAMAFACTORY_DIR}" ]; then
    echo "[ERROR] LLaMA-Factory not found at ${LLAMAFACTORY_DIR}" >&2
    echo "        clone https://github.com/hiyouga/LLaMA-Factory and set LLAMAFACTORY_DIR" >&2
    exit 1
fi

echo "=========================================="
echo " RLTR planner SFT (multi-node)"
echo "   config     : ${CONFIG_FILE}"
echo "   node rank  : ${NODE_RANK} / ${NNODES}"
echo "   master     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "=========================================="

cd "${LLAMAFACTORY_DIR}"
llamafactory-cli train "${CONFIG_FILE}"
