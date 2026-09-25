#!/usr/bin/env bash
set -euo pipefail

# vLLM OpenAI-compatible server for the tau2-Bench agent (system under test).
#
# The critical difference between this template and eval/bfcl_v3/start_vllm_server.sh:
# 1) We PRESERVE the upstream system messages that tau2 injects
#    (domain policy / resolution steps / role constraints).
# 2) We APPEND a Toucan-style tool-call suffix so the model still outputs
#    <think>...</think> + <tool_call>[...]</tool_call> that the toucan_xlam
#    parser can extract back into OpenAI tool_calls.
#
# Usage:
#   bash eval/tau2_bench/start_tau2_agent_vllm_server.sh
#
# Common overrides:
#   MODEL_PATH=...   HOST=...   PORT=...   SERVED_MODEL_NAME=...
#   CUDA_VISIBLE_DEVICES=...   TP_SIZE=...   PP_SIZE=...

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${ROOT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/outputs/rl/qwen2_5_7b_slca_grpo}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# Must match TAU2_AGENT_MODEL_NAME consumed by tau2_bench_eval.py.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-toucan_toolcall_v4}"

TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-toucan_xlam}"
TOOL_PARSER_PLUGIN="${TOOL_PARSER_PLUGIN:-${REPO_ROOT}/eval/bfcl_v3/toucan_xlam_tool_parser_plugin.py}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

echo "Starting vLLM OpenAI server (tau2-Bench agent, Toucan toolcall)"
echo "  MODEL_PATH           = ${MODEL_PATH}"
echo "  HOST:PORT            = ${HOST}:${PORT}"
echo "  SERVED_MODEL_NAME    = ${SERVED_MODEL_NAME}"
echo "  PYTHON_BIN           = ${PYTHON_BIN}"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  TP_SIZE              = ${TP_SIZE}"
echo "  PP_SIZE              = ${PP_SIZE}"
echo "  TOOL_CALL_PARSER     = ${TOOL_CALL_PARSER}"
echo "  MAX_MODEL_LEN        = ${MAX_MODEL_LEN}"

TEMPLATE_FILE="${TEMPLATE_FILE:-/tmp/tau2_agent_toucan_toolcall_template.jinja}"
cat >"${TEMPLATE_FILE}" <<'JINJA'
{# tau2 agent chat template: preserve upstream system + append Toucan toolcall instructions. #}
{%- set sys_msgs = messages | selectattr("role", "equalto", "system") | list -%}

{%- set toucan_suffix -%}

# Toucan Tool-Calling Output Format
Answer normally, but put your internal reasoning inside <think>...</think>.

If tools are needed, output a JSON list wrapped in <tool_call></tool_call>, e.g.
<tool_call>
[{"name": "<function-name>", "arguments": {"arg": "value"}}]
</tool_call>
You may include multiple tool calls in the list.
{%- endset -%}

{{- '<|im_start|>system\n' -}}
{%- if sys_msgs -%}
  {%- for m in sys_msgs -%}
{{- (m.content or '') -}}
{{- '\n' -}}
  {%- endfor -%}
{%- endif -%}
{{- toucan_suffix -}}
{%- if tools -%}
{{- '\n\n<tools>\n' -}}
  {%- for tool in tools -%}
{{- tool | tojson -}}
{{- '\n' -}}
  {%- endfor -%}
{{- '</tools>\n' -}}
{%- endif -%}
{{- '<|im_end|>\n' -}}

{%- for message in messages -%}
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
