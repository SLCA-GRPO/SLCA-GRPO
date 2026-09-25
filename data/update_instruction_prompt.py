#!/usr/bin/env python3
"""
Update the instruction prompt of the training data.
Replace the old prompt before "Available Tools" with a new ReAct-format prompt.

Input: train_rl_from_00001_single_turn.pretty.json
Output: train_rl_from_00001_single_turn_updated.pretty.json
"""

import json
import os
import re
from pathlib import Path

INPUT_PATH = Path(os.environ.get("INPUT_JSON", "./data/train_rl_single_turn.pretty.json"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_JSON", "./data/train_rl_single_turn_updated.pretty.json"))

NEW_PROMPT_PREFIX = """# Role
You are an autonomous AI agent capable of using tools to solve problems. You must follow the "ReAct" (Reasoning + Acting) loop strictly.

# Tool Use Protocol
1. **Analyze**: Review the user input or the latest tool execution result.
2. **Think**: Always begin with a <thinking> block to analyze the current state, what information is missing, and what step to take next.
3. **Act**:
   - If you need external information, output a single <tool_call>.
   - If you have sufficient information to answer the user request, output <answer>.

# Strict Constraints
- **SINGLE TOOL EXECUTION**: In each response, you must output exactly ONE <tool_call> block containing a SINGLE tool object. Do not use arrays or lists.
- **NO PARALLEL CALLS**: Do not output multiple <tool_call> blocks in a single response. If multiple tools are needed, use them sequentially (wait for the result of the first tool before calling the second).
- **VALID JSON**: The content inside <tool_call> must be a strictly valid JSON object.
- **NO HALLUCINATION**: You MUST NOT generate <tool_response> tags yourself. After outputting <tool_call>, stop generating immediately.

# Response Formats

## Format 1: Calling a Tool
<thinking>
Reasoning about why a tool is needed and which arguments to use.
</thinking>
<tool_call>
{"name": "tool_name", "arguments": {"arg_name": "value"}}
</tool_call>

## Format 2: Final Answer
<thinking>
Reasoning that sufficient info is gathered and the answer can be synthesized.
</thinking>
<answer>
The final response to the user.
</answer>

# Examples

User: What is the current weather in the capital of France?

A:
<thinking>
The user is asking for the weather in the capital of France.
First, I need to identify the capital city of France using the `get_capital` tool.
Then, once I have the city name, I will use the `get_weather` tool.
</thinking>
<tool_call>{"name": "get_capital", "arguments": {"country": "France"}}</tool_call>

User (System):
<tool_response>{"capital": "Paris"}</tool_response>

A:
<thinking>
The capital of France is Paris. Now I have the specific city name needed to check the weather.
I will proceed to call the `get_weather` tool for Paris.
</thinking>
<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>

User (System):
<tool_response>{"temperature": "18°C", "condition": "Partly Cloudy"}</tool_response>

A:
<thinking>
I have successfully retrieved the weather for Paris (the capital of France).
I now have all the necessary information to answer the user's request explicitly.
</thinking>
<answer>
The capital of France is Paris, and the current weather there is 18°C and partly cloudy.
</answer>

"""


def update_instruction(instruction: str) -> str:
    """
    Replace everything before "Available Tools:" in the instruction with the new prompt.
    Keep "Available Tools:" and the tool list that follows it.
    """
    # locate the position of "Available Tools:"
    # supports variants such as "Available Tools:" and "Available Tools:\n"
    match = re.search(r'(Available Tools:\s*\n)', instruction)

    if match:
        # found "Available Tools:", so keep it and everything after it
        tools_section = instruction[match.start():]
        new_instruction = NEW_PROMPT_PREFIX + tools_section
    else:
        # not found; the format may differ, so try another way
        # locate the start of the JSON array (the tool list)
        json_match = re.search(r'\n\[{', instruction)
        if json_match:
            # insert "Available Tools:\n" before the JSON
            tools_json = instruction[json_match.start():]
            new_instruction = NEW_PROMPT_PREFIX + "Available Tools:" + tools_json
        else:
            # nothing matched; keep the text as is and print a warning
            print(f"Warning: Could not find 'Available Tools:' pattern in instruction")
            new_instruction = instruction

    return new_instruction


def main():
    print(f"[INFO] Loading data from {INPUT_PATH}")
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Total samples: {len(data)}")

    updated_count = 0
    for i, sample in enumerate(data):
        old_instruction = sample.get("instruction", "")
        new_instruction = update_instruction(old_instruction)

        if new_instruction != old_instruction:
            sample["instruction"] = new_instruction
            updated_count += 1

        if i == 0:
            # print a before/after comparison for the first record
            print("\n" + "=" * 70)
            print("Preview of the updated instruction for the first record:")
            print("=" * 70)
            print("\n--- first 500 characters of the new instruction ---")
            print(new_instruction[:500])
            print("\n...")

    print(f"\n[INFO] Updated {updated_count}/{len(data)} samples")

    print(f"[INFO] Writing to {OUTPUT_PATH}")
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("[INFO] Done!")

    # verify
    print("\n" + "=" * 70)
    print("Verifying the updated data:")
    print("=" * 70)
    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        verify_data = json.load(f)

    first_instruction = verify_data[0]["instruction"]

    # check the key content
    checks = [
        ("# Role", "# Role" in first_instruction),
        ("SINGLE TOOL EXECUTION", "SINGLE TOOL EXECUTION" in first_instruction),
        ("NO PARALLEL CALLS", "NO PARALLEL CALLS" in first_instruction),
        ("Available Tools:", "Available Tools:" in first_instruction),
    ]

    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} Contains '{name}'")


if __name__ == "__main__":
    main()
