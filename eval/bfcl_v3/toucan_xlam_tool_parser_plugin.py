# Tool parser plugin for vLLM's OpenAI-compatible server.
#
# Why this plugin exists:
# - Toucan-style `<tool_call>` payloads sometimes use `parameters` instead of
#   `arguments` for the JSON key that carries the function arguments.
# - vLLM's built-in `xlam` parser only accepts {"name": ..., "arguments": ...}.
# - This plugin registers `toucan_xlam` which accepts BOTH `arguments` and
#   `parameters` and normalises to the OpenAI `arguments` field.
#
# Plug it in via vLLM:
#   --tool-parser-plugin /abs/path/to/toucan_xlam_tool_parser_plugin.py
#   --tool-call-parser toucan_xlam

from __future__ import annotations

import json
from typing import Any, Optional

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParserManager)
from vllm.entrypoints.openai.tool_parsers.xlam_tool_parser import xLAMToolParser
from vllm.utils import random_uuid


@ToolParserManager.register_module("toucan_xlam")
class ToucanXLAMToolParser(xLAMToolParser):
    """xLAM-compatible parser that also accepts the ``parameters`` key.

    Supported payload shapes (inside ``<tool_call>...</tool_call>``):
      - Single call: ``{"name": "...", "arguments": {...}}`` or
                     ``{"name": "...", "parameters": {...}}``
      - Parallel calls: ``[{"name": "...", "arguments": {...}}, ...]``
    """

    @staticmethod
    def _get_args_payload(call: dict[str, Any]) -> Optional[Any]:
        if "arguments" in call:
            return call["arguments"]
        if "parameters" in call:
            return call["parameters"]
        return None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Reuse xLAM preprocessing (handles <tool_call> tags, JSON blocks, etc.)
        content, potential_tool_calls = self.preprocess_model_output(model_output)

        if not potential_tool_calls:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=content)

        try:
            tool_calls_data = json.loads(potential_tool_calls)
        except Exception:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=content or model_output,
            )

        # Normalise to a list; Toucan usually outputs an array but some
        # templates instruct a single object.
        if isinstance(tool_calls_data, dict):
            tool_calls_data = [tool_calls_data]
        elif not isinstance(tool_calls_data, list):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=content or model_output,
            )

        tool_calls: list[ToolCall] = []
        for idx, call in enumerate(tool_calls_data):
            if not isinstance(call, dict) or "name" not in call:
                continue

            args_payload = self._get_args_payload(call)
            if args_payload is None:
                continue

            tool_calls.append(
                ToolCall(
                    id=f"call_{idx}_{random_uuid()}",
                    type="function",
                    function=FunctionCall(
                        name=call["name"],
                        arguments=(
                            json.dumps(args_payload)
                            if isinstance(args_payload, dict)
                            else args_payload
                        ),
                    ),
                )
            )

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=content,
        )
