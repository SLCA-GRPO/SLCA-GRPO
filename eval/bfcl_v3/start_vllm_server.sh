#!/usr/bin/env bash
set -euo pipefail

# vLLM OpenAI-compatible server for the BFCL-v3 agent.
#
# Features:
# - Prepends a Toucan-style system instruction that matches SFT training.
# - Injects tools into <tools>...</tools> on the system turn.
# - Expects the model to output <think>...</think> + <tool_call>[...]</tool_call>,
#   parsed back into OpenAI tool_calls by the toucan_xlam plugin.
#
# Usage:
#   bash eval/bfcl_v3/start_vllm_server.sh
#
# Common overrides:
#   MODEL_PATH=...   HOST=...   PORT=...   SERVED_MODEL_NAME=...
#   CUDA_VISIBLE_DEVICES=...   TP_SIZE=...   PP_SIZE=...
#
# Hardware target: 4-8 GPUs (A100 / H20 80GB+). TP/PP auto-tuning below picks
# a layout that (a) uses all visible GPUs and (b) divides num_attention_heads
# and num_hidden_layers cleanly.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${ROOT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/outputs/rl/qwen2_5_7b_slca_grpo}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# Must match BFCL_VLLM_MODEL_NAME consumed by bfcl_v3_eval.py.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-toucan_toolcall_v4}"

# Tool parser plugin that normalises Toucan-style <tool_call> into OpenAI tool_calls.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-toucan_xlam}"
TOOL_PARSER_PLUGIN="${TOOL_PARSER_PLUGIN:-${ROOT_DIR}/toucan_xlam_tool_parser_plugin.py}"

# Interpreter with vLLM installed. Override PYTHON_BIN to point at your env.
PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

TP_SIZE="${TP_SIZE:-}"
PP_SIZE="${PP_SIZE:-}"

IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#_GPU_IDS[@]}"

NUM_HEADS=""
NUM_LAYERS=""
if [[ -e "${MODEL_PATH}" ]]; then
  # Best-effort: infer heads/layers from HF config to pick a valid TP/PP default.
  set +e
  _MODEL_META="$(MODEL_PATH="${MODEL_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
from transformers import AutoConfig

model_path = os.environ["MODEL_PATH"]
cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
heads = getattr(cfg, "num_attention_heads", "")
layers = getattr(cfg, "num_hidden_layers", "")
print(f"{heads} {layers}")
PY
  2>/dev/null)"
  _RC=$?
  set -e
  if [[ ${_RC} -eq 0 ]]; then
    read -r NUM_HEADS NUM_LAYERS <<< "${_MODEL_META}"
  fi
fi

_user_set_tp=0
_user_set_pp=0
if [[ -n "${TP_SIZE}" ]]; then _user_set_tp=1; fi
if [[ -n "${PP_SIZE}" ]]; then _user_set_pp=1; fi

if [[ ${_user_set_tp} -eq 0 || ${_user_set_pp} -eq 0 ]]; then
  if [[ -n "${NUM_HEADS}" && "${NUM_HEADS}" =~ ^[0-9]+$ ]]; then
    _best_tp=""
    _best_pp=""
    for (( _tp = GPU_COUNT; _tp >= 1; _tp-- )); do
      if (( _tp > GPU_COUNT )); then
        continue
      fi
      if (( NUM_HEADS % _tp != 0 )); then
        continue
      fi
      if (( GPU_COUNT % _tp != 0 )); then
        continue
      fi
      _pp=$(( GPU_COUNT / _tp ))
      if [[ -n "${NUM_LAYERS}" && "${NUM_LAYERS}" =~ ^[0-9]+$ ]]; then
        if (( NUM_LAYERS % _pp != 0 )); then
          continue
        fi
      fi
      _best_tp="${_tp}"
      _best_pp="${_pp}"
      break
    done

    if [[ -n "${_best_tp}" && -n "${_best_pp}" ]]; then
      if [[ ${_user_set_tp} -eq 0 ]]; then TP_SIZE="${_best_tp}"; fi
      if [[ ${_user_set_pp} -eq 0 ]]; then PP_SIZE="${_best_pp}"; fi
    fi
  fi
fi

TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-1}"

if [[ -n "${NUM_HEADS}" && "${NUM_HEADS}" =~ ^[0-9]+$ ]]; then
  if (( NUM_HEADS % TP_SIZE != 0 )); then
    echo "ERROR: num_attention_heads (${NUM_HEADS}) must be divisible by TP_SIZE (${TP_SIZE})." >&2
    exit 2
  fi
fi

if (( TP_SIZE > GPU_COUNT )); then
  echo "ERROR: TP_SIZE (${TP_SIZE}) > visible GPU count (${GPU_COUNT})." >&2
  exit 2
fi

if [[ -n "${NUM_LAYERS}" && "${NUM_LAYERS}" =~ ^[0-9]+$ ]]; then
  if (( NUM_LAYERS % PP_SIZE != 0 )); then
    echo "ERROR: num_hidden_layers (${NUM_LAYERS}) must be divisible by PP_SIZE (${PP_SIZE})." >&2
    exit 2
  fi
fi

if (( TP_SIZE * PP_SIZE != GPU_COUNT )); then
  echo "WARN: TP_SIZE*PP_SIZE (${TP_SIZE}*${PP_SIZE}=$((TP_SIZE*PP_SIZE))) != visible GPU count (${GPU_COUNT})." >&2
fi

echo "Starting vLLM OpenAI server (BFCL-v3 Toucan template)"
echo "  MODEL_PATH           = ${MODEL_PATH}"
echo "  HOST:PORT            = ${HOST}:${PORT}"
echo "  SERVED_MODEL_NAME    = ${SERVED_MODEL_NAME}"
echo "  PYTHON_BIN           = ${PYTHON_BIN}"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  TP_SIZE              = ${TP_SIZE}"
echo "  PP_SIZE              = ${PP_SIZE}"
echo "  TOOL_CALL_PARSER     = ${TOOL_CALL_PARSER}"

TEMPLATE_FILE="${TEMPLATE_FILE:-/tmp/toucan_toolcall_v4_template.jinja}"
cat >"${TEMPLATE_FILE}" <<'JINJA'
{# Toucan-style chat template (ChatML) with tools injection. #}
{%- set sys_prefix -%}
You are a helpful assistant.

# Instructions
Answer the user's question and output your thinking process within the <think> and </think> tags.

If tools are needed, output a **JSON list** wrapped in <tool_call></tool_call> XML tags like this:
<tool_call>
[{"name": <function-name>, "arguments": <args-json-object>}]
</tool_call>
You can place one or multiple tool calls in the list for parallel execution.

If the task requires multiple steps, output tool calls, wait for results, and then continue reasoning.

# Tool Definitions
<tools>
{%- endset -%}

{%- set sys_suffix -%}
</tools>
{%- endset -%}

{{- '<|im_start|>system\n' -}}
{{- sys_prefix -}}
{%- if tools -%}
{%- for tool in tools -%}
{{- "\n" -}}
{{- tool | tojson -}}
{%- endfor -%}
{%- endif -%}
{{- "\n" -}}
{{- sys_suffix -}}
{{- '<|im_end|>\n' -}}

{%- for message in messages -%}
  {# Ignore upstream system messages; we always use the Toucan system prompt above. #}
  {%- if message.role == "system" -%}
    {%- continue -%}
  {%- endif -%}

  {%- if message.role == "user" -%}
    {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' -}}

  {%- elif message.role == "assistant" and (not message.tool_calls) -%}
    {{- '<|im_start|>assistant' -}}
    {%- if message.content -%}
      {{- '\n' + message.content -}}
    {%- endif -%}
    {{- '<|im_end|>\n' -}}

  {%- elif message.role == "assistant" and message.tool_calls -%}
    {{- '<|im_start|>assistant' -}}
    {%- if message.content -%}
      {{- '\n' + message.content -}}
    {%- endif -%}
    {{- '\n<tool_call>\n[' -}}
    {%- for tool_call in message.tool_calls -%}
      {%- if tool_call.function is defined -%}
        {%- set tool_call = tool_call.function -%}
      {%- endif -%}
      {{- '\n  {"name": "' -}}
      {{- tool_call.name -}}
      {{- '", "arguments": ' -}}
      {{- tool_call.arguments | tojson -}}
      {{- '}' -}}
      {%- if not loop.last -%}
        {{- ',' -}}
      {%- endif -%}
    {%- endfor -%}
    {{- '\n]\n</tool_call>' -}}
    {{- '<|im_end|>\n' -}}

  {%- elif message.role == "tool" -%}
    {# Render tool results as <tool_response> blocks inside a user message. #}
    {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") -%}
      {{- '<|im_start|>user' -}}
    {%- endif -%}
    {{- '\n<tool_response>\n' -}}
    {{- message.content -}}
    {{- '\n</tool_response>' -}}
    {%- if loop.last or (messages[loop.index0 + 1].role != "tool") -%}
      {{- '<|im_end|>\n' -}}
    {%- endif -%}
  {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
  {{- '<|im_start|>assistant\n' -}}
{%- endif -%}
JINJA

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --pipeline-parallel-size "${PP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enforce-eager \
  --trust-remote-code \
  --tool-parser-plugin "${TOOL_PARSER_PLUGIN}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  --chat-template "${TEMPLATE_FILE}"
