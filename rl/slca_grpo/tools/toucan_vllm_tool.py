"""Toucan vLLM tool: forwards model-generated tool calls to the Schema-Guided
LLM Simulator (SGLS) running on an OpenAI-compatible vLLM endpoint.

At rollout time, the ToolAgentLoop hands each `<tool_call>` back to this tool,
which constructs an "API server" prompt, calls the SGLS server, and returns
the content between `<tool_response>...</tool_response>` tags as the tool
observation.

Environment:
  - SGLS endpoint is configured in config/tool_config/toucan_tool_config.yaml
  - Per-instance tool schemas come from the training data's `extra_info`
    via the ToolAgentLoop's `create()` call.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional

import aiohttp

from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)

# Extract content inside <tool_response>...</tool_response>; fall back to
# <tool_call>...</tool_call> because some instruct models leak the client tag.
TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# Shared TCP connector -> reuse connections across many concurrent tool calls.
_global_connector: Optional[aiohttp.TCPConnector] = None


def get_global_connector() -> aiohttp.TCPConnector:
    global _global_connector
    if _global_connector is None or _global_connector.closed:
        _global_connector = aiohttp.TCPConnector(
            limit=300,
            limit_per_host=300,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=60,
        )
    return _global_connector


TOOL_SIMULATOR_PROMPT_TEMPLATE = """# CRITICAL ROLE: You are an API SERVER, NOT an AI assistant or agent.

You are role-playing as a **backend API server** that executes function calls and returns results.
You are NOT making tool calls - you are RESPONDING to them as if you were the actual API endpoint.

## YOUR ONLY OUTPUT FORMAT:
<tool_response>{"key": "value", ...}</tool_response>

## ABSOLUTE PROHIBITIONS:
- NEVER output <tool_call> tags (you are the SERVER, not the client)
- NEVER output markdown code blocks
- NEVER output explanations or conversational text
- ONLY output the <tool_response> tag with JSON inside

## Examples of CORRECT server responses:

Input: calculator(expression="124 * 5")
<tool_response>{"result": 620}</tool_response>

Input: get_weather(city="Seattle")
<tool_response>{"temp": "12°C", "condition": "Rainy"}</tool_response>

Input: google_search(query="capital of Australia")
<tool_response>{"snippets": ["Canberra is the capital city of Australia."]}</tool_response>

---
## NOW EXECUTE THIS API CALL:

Tool Definition:
%s

Incoming Request:
%s

Return the execution result in <tool_response>...</tool_response> format:
"""


class ToucanVLLMTool(BaseTool):
    """Forwards one tool call per `execute()` to the SGLS vLLM endpoint."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.base_url: str = self.config.get("base_url", "http://127.0.0.1:8000/v1")
        self.model_name: str = self.config.get("model", "qwen3-235b-a22b")
        self.temperature: float = float(self.config.get("temperature", 0.6))
        self.max_new_tokens: int = int(self.config.get("max_new_tokens", 4096))
        self.timeout: int = int(self.config.get("timeout", 540))
        self.max_retries: int = int(self.config.get("max_retries", 2))
        self.retry_delay: float = float(self.config.get("retry_delay", 1.0))

        # Per-trajectory tool schemas, populated by create().
        self._instance_tool_definitions: Dict[str, Any] = {}
        self.default_tool_definition_str = ""

        # Optional debug logging of the last prompt / response pair.
        self.log_every: int = int(self.config.get("log_every", 8))
        self.log_dir: str = self.config.get("log_dir", os.environ.get("TOOL_DEBUG_DIR", "./logs/tool_debug"))
        os.makedirs(self.log_dir, exist_ok=True)
        self._log_counter: int = 0
        self._log_prompt_path = os.path.join(self.log_dir, "debug_prompt.txt")
        self._log_response_path = os.path.join(self.log_dir, "debug_response.txt")

    async def _call_vllm(self, tool_definition_str: str, tool_call_payload: Dict[str, Any]) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        tool_call_block = "<tool_call>" + json.dumps([tool_call_payload], ensure_ascii=False) + "</tool_call>"
        prompt = TOOL_SIMULATOR_PROMPT_TEMPLATE % (tool_definition_str, tool_call_block)

        self._log_counter += 1
        need_log = self.log_every > 0 and self._log_counter % self.log_every == 0

        messages = [{"role": "user", "content": prompt}]
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            # vLLM does not validate the key; any placeholder is fine.
            "Authorization": "Bearer EMPTY",
        }
        timeout = aiohttp.ClientTimeout(
            total=self.timeout,
            connect=60,
            sock_read=self.timeout,
        )

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                async with aiohttp.ClientSession(
                    connector=get_global_connector(),
                    connector_owner=False,
                    timeout=timeout,
                ) as session:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()

                try:
                    content = data["choices"][0]["message"]["content"]
                except Exception as e:
                    content = f"[ToucanVLLMTool] Failed to parse response: {e}\nRaw: {json.dumps(data, ensure_ascii=False)[:1024]}"

                if need_log:
                    try:
                        with open(self._log_prompt_path, "w", encoding="utf-8") as f:
                            f.write(prompt)
                        with open(self._log_response_path, "w", encoding="utf-8") as f:
                            f.write(content)
                    except Exception as log_err:
                        logger.warning(f"Failed to write log files: {log_err}")
                return content

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(f"Timeout calling SGLS (attempt {attempt + 1}/{self.max_retries + 1}): {url}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(f"Client error calling SGLS (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        raise last_error or TimeoutError(f"Failed to call SGLS after {self.max_retries + 1} attempts")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        original_tool_name = kwargs.get("original_tool_name") or self.name
        tool_def = self._instance_tool_definitions.get(instance_id)
        if tool_def is None:
            error_text = f"Tool '{original_tool_name}' not found in the current available tools set."
            return ToolResponse(text=error_text), 0.0, {"tool_name": original_tool_name, "tool_not_found": True}

        # Check required parameters before bothering the simulator.
        try:
            required_params = tool_def.get("function", {}).get("parameters", {}).get("required", [])
            if required_params and isinstance(parameters, dict):
                missing_params = [p for p in required_params if p not in parameters]
                if missing_params:
                    error_text = f"Tool '{original_tool_name}' missing required parameters: {missing_params}"
                    return ToolResponse(text=error_text), 0.0, {"tool_name": original_tool_name, "missing_params": missing_params}
        except Exception:
            pass

        try:
            tool_definition_str = json.dumps(tool_def, ensure_ascii=False, indent=2)
        except Exception:
            tool_definition_str = self.default_tool_definition_str

        tool_call_payload: Dict[str, Any] = {
            "name": original_tool_name,
            "parameters": parameters,
            "depend_on": [],
        }
        text = await self._call_vllm(tool_definition_str, tool_call_payload)

        # Strip nested / truncated tool tags so the observation contains just
        # the simulated API body.
        text = text.strip()
        while True:
            matches = TOOL_RESPONSE_PATTERN.findall(text)
            if matches:
                text = matches[-1].strip()
                continue
            matches = TOOL_CALL_PATTERN.findall(text)
            if matches:
                text = matches[-1].strip()
                continue
            tool_response_start = "<tool_response>"
            if tool_response_start in text:
                idx = text.rfind(tool_response_start)
                text = text[idx + len(tool_response_start):].strip()
                continue
            tool_call_start = "<tool_call>"
            if tool_call_start in text:
                idx = text.rfind(tool_call_start)
                text = text[idx + len(tool_call_start):].strip()
                continue
            break

        return ToolResponse(text=text), 0.0, {"tool_name": original_tool_name}

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        tool_def = create_kwargs.get("tool_definition")

        if isinstance(tool_def, str):
            try:
                loaded = json.loads(tool_def)
                tool_def = loaded if isinstance(loaded, dict) else None
            except Exception:
                tool_def = None

        if instance_id is None:
            from uuid import uuid4
            instance_id = str(uuid4())

        if tool_def is not None:
            self._instance_tool_definitions[instance_id] = tool_def

        return instance_id, ToolResponse()

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_tool_definitions.pop(instance_id, None)
