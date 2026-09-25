#!/usr/bin/env bash
# SLCA-GRPO main training launcher (Qwen2.5-7B backbone, SLCA + token-mean
# advantage normalisation + HierR reward).
#
# Usage:
#   bash rl/slca_grpo/train_slca_grpo.sh
#
# Override via env vars, e.g.:
#   MODEL_PATH=./outputs/sft_split/qwen2_5_3b_toucan_toolcall \
#   TRAIN_DATA=./data/toucan_toolcall_rl.parquet \
#   VAL_DATA=./data/toucan_eval_4k_unified.parquet \
#   LLM_JUDGE_BASE_URL=http://127.0.0.1:8016/v1 \
#   LLM_JUDGE_MODEL=gpt-oss-120b \
#   bash rl/slca_grpo/train_slca_grpo.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------- paths (user-configurable) ----------------------
PYTHON=${PYTHON:-python}

DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data}
TRAIN_DATA=${TRAIN_DATA:-${DATA_DIR}/toucan_toolcall_rl.parquet}
VAL_DATA=${VAL_DATA:-${DATA_DIR}/toucan_eval_4k_unified.parquet}

MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/outputs/sft_split/qwen2_5_7b_toucan_toolcall}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/rl/qwen2_5_7b_slca_grpo}
REWARD_FN_PATH=${REWARD_FN_PATH:-${SCRIPT_DIR}/reward_fn.py}
TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-${SCRIPT_DIR}/config/tool_config/toucan_tool_config.yaml}

# ---------------------- HierR reward weights ----------------------
# Process: format + tool-name F1 + param-key F1 + param-value F1 + parallel match (sum = 1.0).
export SLCA_WEIGHT_FORMAT=${SLCA_WEIGHT_FORMAT:-0.10}
export SLCA_WEIGHT_NAME=${SLCA_WEIGHT_NAME:-0.25}
export SLCA_WEIGHT_KEY=${SLCA_WEIGHT_KEY:-0.15}
export SLCA_WEIGHT_VALUE=${SLCA_WEIGHT_VALUE:-0.20}
export SLCA_WEIGHT_PARALLEL=${SLCA_WEIGHT_PARALLEL:-0.30}
# Weights mixing process vs summary rewards across segments.
export SLCA_WEIGHT_PROCESS=${SLCA_WEIGHT_PROCESS:-1.0}
export SLCA_WEIGHT_RESPQ=${SLCA_WEIGHT_RESPQ:-1.0}

# ---------------------- LLM Judge (summary reward) ----------------------
export LLM_JUDGE_ENABLED=${LLM_JUDGE_ENABLED:-true}
export LLM_JUDGE_BASE_URL=${LLM_JUDGE_BASE_URL:-http://127.0.0.1:8016/v1}
export LLM_JUDGE_MODEL=${LLM_JUDGE_MODEL:-gpt-oss-120b}
export LLM_JUDGE_TIMEOUT=${LLM_JUDGE_TIMEOUT:-540}
export LLM_JUDGE_MAX_RETRIES=${LLM_JUDGE_MAX_RETRIES:-3}
export JUDGE_FAILURE_SCORE=${JUDGE_FAILURE_SCORE:-0.0}
export NO_CALL_PENALTY=${NO_CALL_PENALTY:--0.5}
export ENABLE_WANDB_LOGGING=${ENABLE_WANDB_LOGGING:-true}
export REWARD_LOG_DIR=${REWARD_LOG_DIR:-${REPO_ROOT}/logs/reward_function}

# ---------------------- validation cadence ----------------------
TEST_FREQ=${TEST_FREQ:-10}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-0}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-false}

# ---------------------- training knobs ----------------------
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-24576}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
ROLLOUT_N=${ROLLOUT_N:-16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.82}

ACTOR_LR=${ACTOR_LR:-1e-6}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}

PROJECT_NAME=${PROJECT_NAME:-slca_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_7b_slca_grpo}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
SAVE_FREQ=${SAVE_FREQ:-10}
RESUME_MODE=${RESUME_MODE:-auto}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

# Ensure Python can resolve the tools package referenced by the tool config.
export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}/verl:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

echo "=========================================="
echo " SLCA-GRPO training"
echo "   model      : ${MODEL_PATH}"
echo "   train data : ${TRAIN_DATA}"
echo "   val   data : ${VAL_DATA}"
echo "   output dir : ${OUTPUT_DIR}"
echo "   reward fn  : ${REWARD_FN_PATH}"
echo "   tool cfg   : ${TOOL_CONFIG_PATH}"
echo "   SGLS / judge endpoints"
echo "     sgls     : (see ${TOOL_CONFIG_PATH})"
echo "     judge    : ${LLM_JUDGE_BASE_URL}  model=${LLM_JUDGE_MODEL}"
echo "=========================================="

${PYTHON} -m verl.trainer.main_ppo \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=toucan_grpo \
    algorithm.adv_estimator=slca_grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.norm_adv_by_std_in_grpo=true \
    \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    data.return_raw_chat=true \
    data.train_max_samples=-1 \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.use_shm=false \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS} \
    actor_rollout_ref.rollout.enable_prefix_caching=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
    \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=6 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=20 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=32 \
    \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
    \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    \
    reward_model.enable=false \
    reward_model.reward_manager=naive \
    reward_model.enable_resource_pool=false \
    reward_model.launch_reward_fn_async=true \
    \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.max_actor_ckpt_to_keep=8 \
    trainer.max_critic_ckpt_to_keep=8 \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.resume_mode="${RESUME_MODE}" \
    ${RESUME_FROM_PATH:+trainer.resume_from_path="${RESUME_FROM_PATH}"} \
    trainer.logger='["console","wandb"]' \
    "$@"

echo ""
echo "=========================================="
echo " Training finished. Outputs under: ${OUTPUT_DIR}"
echo "=========================================="
