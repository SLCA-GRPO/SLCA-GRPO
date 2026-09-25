#!/usr/bin/env bash
set -euo pipefail

# Optional: vLLM OpenAI-compatible server for the tau2-Bench USER simulator.
#
# Only needed if you want to self-host the user-simulator model locally
# instead of calling an external OpenAI-compatible API (the paper uses DeepSeek-V3.2).
#
# Key constraint:
#   Do NOT strip/ignore upstream system messages here - tau2 uses them to
#   condition the model to "act as user", not assistant.
#
# Usage:
#   bash eval/tau2_bench/start_tau2_user_vllm_server.sh
#
# Common overrides:
#   MODEL_PATH=...   HOST=...   PORT=...   SERVED_MODEL_NAME=...
#   CUDA_VISIBLE_DEVICES=...   TP_SIZE=...   PP_SIZE=...
#
# If you use this script, point the evaluator at it via:
#   TAU2_USER_API_BASE=http://127.0.0.1:8001/v1
#   TAU2_USER_MODEL_NAME=tau2_user
#   TAU2_USER_API_KEY=EMPTY    # vLLM ignores the value but evalscope wants it set

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${ROOT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/outputs/sft_split/qwen2_5_7b_toucan_toolcall}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"

# Must match TAU2_USER_MODEL_NAME consumed by tau2_bench_eval.py.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-tau2_user}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

echo "Starting vLLM OpenAI server (tau2-Bench user simulator)"
echo "  MODEL_PATH           = ${MODEL_PATH}"
echo "  HOST:PORT            = ${HOST}:${PORT}"
echo "  SERVED_MODEL_NAME    = ${SERVED_MODEL_NAME}"
echo "  PYTHON_BIN           = ${PYTHON_BIN}"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  TP_SIZE              = ${TP_SIZE}"
echo "  PP_SIZE              = ${PP_SIZE}"

TEMPLATE_FILE="${TEMPLATE_FILE:-/tmp/tau2_user_chat_template.jinja}"
cat >"${TEMPLATE_FILE}" <<'JINJA'
{# tau2 user-simulator chat template (ChatML). Preserves upstream system messages. #}
{%- set sys_msgs = messages | selectattr("role", "equalto", "system") | list -%}
{%- set non_sys_msgs = messages | rejectattr("role", "equalto", "system") | list -%}

{{- '<|im_start|>system\n' -}}
{%- if sys_msgs -%}
  {%- for m in sys_msgs -%}
{{- (m.content or '') -}}
{{- '\n' -}}
  {%- endfor -%}
{%- endif -%}
{%- if tools -%}
{{- '\n<tools>\n' -}}
  {%- for tool in tools -%}
{{- tool | tojson -}}
{{- '\n' -}}
  {%- endfor -%}
{{- '</tools>\n' -}}
{%- endif -%}
{{- '<|im_end|>\n' -}}

{%- for message in non_sys_msgs -%}
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
    {%- if (loop.index0 == 0) or (non_sys_msgs[loop.index0 - 1].role != "tool") -%}
      {{- '<|im_start|>user' -}}
    {%- endif -%}
    {{- '\n<tool_response>\n' -}}
    {{- message.content -}}
    {{- '\n</tool_response>' -}}
    {%- if loop.last or (non_sys_msgs[loop.index0 + 1].role != "tool") -%}
      {{- '<|im_end|>\n' -}}
    {%- endif -%}
  {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
  {{- '<|im_start|>assistant\n' -}}
{%- endif -%}
JINJA

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
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --chat-template "${TEMPLATE_FILE}"
