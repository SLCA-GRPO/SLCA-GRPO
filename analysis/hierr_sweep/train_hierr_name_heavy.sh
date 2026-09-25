#!/usr/bin/env bash
#
# HierR weight-sensitivity sweep -- Name-heavy configuration
# Weights: format=0.05, name=0.40, key=0.15, value=0.15, parallel=0.25
# Emphasises tool-name matching. This configuration is an extra sweep point and
# has no row in the paper's HierR weight-sensitivity table.
# data.seed=43 is pinned below purely for reproducibility; it does not identify
# any particular run of the paper's three-run set.
#

set -euo pipefail

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Python interpreter
PYTHON="${PYTHON:-python}"

# Data paths
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/toucan_toolcall_rl.parquet}"

# Validation data path
VAL_DATA_DIR="${VAL_DATA_DIR:-${REPO_ROOT}/data}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/toucan_eval_4k_unified.parquet}"

# Model path
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/outputs/sft_split/qwen2_5_7b_toucan_toolcall}"

# Output directory: HierR sweep, Default configuration
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/analysis/hierr_sweep/name_heavy}"

# Reward function shipped in this directory
REWARD_FN_PATH="${REWARD_FN_PATH:-${SCRIPT_DIR}/reward_fn_hierr_v82.py}"

# ============================================================================
# HierR weights -- Name-heavy
# ============================================================================
export SLCA_WEIGHT_PROCESS="${SLCA_WEIGHT_PROCESS:-1.0}"
export SLCA_WEIGHT_FORMAT="${SLCA_WEIGHT_FORMAT:-0.05}"
export SLCA_WEIGHT_NAME="${SLCA_WEIGHT_NAME:-0.40}"
export SLCA_WEIGHT_KEY="${SLCA_WEIGHT_KEY:-0.15}"
export SLCA_WEIGHT_VALUE="${SLCA_WEIGHT_VALUE:-0.15}"
export SLCA_WEIGHT_PARALLEL="${SLCA_WEIGHT_PARALLEL:-0.25}"
export SLCA_WEIGHT_RESPQ="${SLCA_WEIGHT_RESPQ:-1.0}"

# ============================================================================
# Online validation settings
# ============================================================================
TEST_FREQ="${TEST_FREQ:-10}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-0}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-false}"

# ============================================================================
# Reward / judge toggles
# ============================================================================
export LLM_JUDGE_ENABLED="${LLM_JUDGE_ENABLED:-true}"
export JUDGE_FAILURE_SCORE="${JUDGE_FAILURE_SCORE:-0.0}"
export NO_CALL_PENALTY="${NO_CALL_PENALTY:--0.5}"
export ENABLE_WANDB_LOGGING=true

echo "=========================================="
echo "HierR weight sensitivity -- Name-heavy configuration"
echo "data.seed=43 (reproducibility parameter)"
echo "=========================================="
echo ""
echo "Weights:"
echo "  Process-segment weight: ${SLCA_WEIGHT_PROCESS}"
echo "  Process-segment sub-weights:"
echo "    - format:   ${SLCA_WEIGHT_FORMAT}"
echo "    - name:     ${SLCA_WEIGHT_NAME}"
echo "    - key:      ${SLCA_WEIGHT_KEY}"
echo "    - value:    ${SLCA_WEIGHT_VALUE}"
echo "    - parallel: ${SLCA_WEIGHT_PARALLEL}"
echo "  Summary-segment weight: ${SLCA_WEIGHT_RESPQ}"
echo ""
echo "Validation settings:"
echo "  - test_freq:       ${TEST_FREQ}"
echo "  - val_data:        ${VAL_DATA}"
echo ""
echo "LLM Judge: ${LLM_JUDGE_ENABLED}"
echo "=========================================="

# ============================================================================
# Environment
# ============================================================================
export PYTHONPATH="${REPO_ROOT}/rl/slca_grpo:${REPO_ROOT}/verl:${PYTHONPATH:-}"

if [ "${LLM_JUDGE_ENABLED}" = "true" ]; then
    export LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-http://127.0.0.1:8016/v1}"
    export LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-gpt-oss-120b}"
    export LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-540}"
    export LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-3}"
fi

# ============================================================================
# Training hyper-parameters
# ============================================================================
MAX_PROMPT_LENGTH=24576
MAX_RESPONSE_LENGTH=8192
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"

ROLLOUT_N="${ROLLOUT_N:-16}"
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=32
GPU_MEMORY_UTILIZATION=0.82

ACTOR_LR="${ACTOR_LR:-1e-6}"
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE_PER_GPU=1
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"

PROJECT_NAME="Toucan_tool_call_slca_grpo_qwen2_5_7b_v1"
EXPERIMENT_NAME="hierr_name_heavy"
N_GPUS_PER_NODE=8
NNODES="${NNODES:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
SAVE_FREQ="${SAVE_FREQ:-10}"

RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

# ============================================================================
# Detect an existing checkpoint
# ============================================================================
LATEST_CKPT_FILE="${OUTPUT_DIR}/latest_checkpointed_iteration.txt"
if [ -f "${LATEST_CKPT_FILE}" ]; then
    LATEST_STEP=$(cat "${LATEST_CKPT_FILE}")
    echo ""
    echo "Found checkpoint: global_step_${LATEST_STEP}"
    if [ "${RESUME_MODE}" = "auto" ]; then
        echo "Resuming automatically from the latest checkpoint."
    elif [ "${RESUME_MODE}" = "disable" ]; then
        echo "Warning: resume_mode=disable -- training will restart from scratch."
    fi
else
    echo ""
    echo "No existing checkpoint found; training from scratch."
fi
echo ""

# ============================================================================
# Validate the validation set
# ============================================================================
if [ "${TEST_FREQ}" != "-1" ]; then
    if [ ! -f "${VAL_DATA}" ]; then
        echo "Warning: validation file not found: ${VAL_DATA}"
        echo "   Disabling online validation (test_freq=-1)"
        TEST_FREQ=-1
    fi
fi

# ============================================================================
# Launch training
# ============================================================================
echo ""
echo "train data : ${TRAIN_DATA}"
echo "val   data : ${VAL_DATA}"
echo "model      : ${MODEL_PATH}"
echo "output dir : ${OUTPUT_DIR}"
echo "reward fn  : ${REWARD_FN_PATH}"
echo "=========================================="
echo ""

${PYTHON} -m verl.trainer.main_ppo \
    --config-path="${REPO_ROOT}/rl/slca_grpo/config" \
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
    data.seed=43 \
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
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${REPO_ROOT}/rl/slca_grpo/config/tool_config/toucan_tool_config.yaml" \
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
echo "Training finished. Outputs under: ${OUTPUT_DIR}"
echo "=========================================="
