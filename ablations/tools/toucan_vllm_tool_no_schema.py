"""
Ablation (ii): w/o SGLS.

Drop-in replacement for `ToucanVLLMTool` that removes Schema-Guided LLM
Simulation: the frozen simulator LLM is shown only the tool name and the
arguments, never the tool definition (JSON schema), so it has to invent the
return payload.

Expected effect: malformed tool returns, unexpected JSON structures and
stronger drift between simulated and real API responses.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp

from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)

# Regex to extract content from <tool_response>...</tool_response> tags
TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# Global connection pool for better performance
_global_connector: Optional[aiohttp.TCPConnector] = None


def get_global_connector() -> aiohttp.TCPConnector:
    """Get or create a global TCP connector for connection pooling."""
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


# ================================================================================
# [Key difference] prompt template without schema guidance
# ================================================================================
# Original version:
#   Tool Definition:
#   %s  <-- the tool JSON schema is filled in here
#
# Ablation version:
#   no Tool Definition is provided, the LLM generates freely
# ================================================================================

TOOL_SIMULATOR_PROMPT_NO_SCHEMA = """# CRITICAL ROLE: You are an API SERVER, NOT an AI assistant or agent.

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

Incoming Request:
%s

Return the execution result in <tool_response>...</tool_response> format:
"""


class ToucanVLLMToolNoSchema(BaseTool):
    """Tool that calls an external vLLM server WITHOUT schema guidance.

    Ablation version: the Tool Definition (schema) is removed, only the tool call request is passed in.

    Key differences from ToucanVLLMTool:
    1. The prompt template has no "Tool Definition:" section
    2. _instance_tool_definitions is neither cached nor used
    3. The LLM has to infer the returned JSON structure itself
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.base_url: str = self.config.get("base_url", "http://127.0.0.1:8003/v1")
        self.model_name: str = self.config.get("model", "qwen3-235b-a22b")
        self.temperature: float = float(self.config.get("temperature", 0.6))
        self.max_new_tokens: int = int(self.config.get("max_new_tokens", 4096))
        self.timeout: int = int(self.config.get("timeout", 540))
        self.max_retries: int = int(self.config.get("max_retries", 2))
        self.retry_delay: float = float(self.config.get("retry_delay", 1.0))

        # logging settings
        self.log_every: int = int(self.config.get("log_every", 8))
        self.log_dir: str = self.config.get(
            "log_dir", os.environ.get("TOOL_DEBUG_DIR", "./logs/tool_debug")
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self._log_counter: int = 0
        self._log_prompt_path = os.path.join(self.log_dir, "debug_prompt_no_schema.txt")
        self._log_response_path = os.path.join(self.log_dir, "debug_response_no_schema.txt")

    async def _call_vllm(self, tool_call_payload: Dict[str, Any]) -> str:
        """Call external vLLM WITHOUT schema guidance."""
        url = self.base_url.rstrip("/") + "/chat/completions"

        # Build the agent tool call text
        tool_call_block = "<tool_call>" + json.dumps([tool_call_payload], ensure_ascii=False) + "</tool_call>"

        # [Key difference] use the schema-free prompt template
        prompt = TOOL_SIMULATOR_PROMPT_NO_SCHEMA % tool_call_block

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
                    content = f"[ToucanVLLMToolNoSchema] Failed to parse response: {e}\nRaw: {json.dumps(data, ensure_ascii=False)[:1024]}"

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
                logger.warning(f"Timeout calling vLLM (attempt {attempt + 1}/{self.max_retries + 1}): {url}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(f"Client error calling vLLM (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        raise last_error or TimeoutError(f"Failed to call vLLM after {self.max_retries + 1} attempts")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the Toucan tool WITHOUT schema guidance."""
        original_tool_name = kwargs.get("original_tool_name") or self.name

        tool_call_payload: Dict[str, Any] = {
            "name": original_tool_name,
            "parameters": parameters,
            "depend_on": [],
        }

        # [Key difference] tool_definition_str is not passed in
        text = await self._call_vllm(tool_call_payload)

        # Extract the tag content (same post-processing as the original)
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

        tool_response = ToolResponse(text=text)
        reward = 0.0
        metrics: Dict[str, Any] = {"tool_name": original_tool_name, "no_schema": True}
        return tool_response, reward, metrics

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance (no schema caching needed)."""
        if instance_id is None:
            from uuid import uuid4
            instance_id = str(uuid4())
        # [Key difference] tool_definition is not cached, return directly
        return instance_id, ToolResponse()

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance (nothing to clean up)."""
        pass
