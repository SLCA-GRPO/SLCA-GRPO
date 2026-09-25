"""
Convert the Toucan tool-calling dataset into verl RL training format.

Two conversion modes:
1. first_turn_only: keep only the first user turn as the prompt.
2. split_turns:     split a multi-turn dialogue into independent samples
                    (recommended).

split_turns, by example. Raw record:

    human[0] -> function_call -> observation -> gpt[0] -> human[1] -> gpt[1]

becomes two training samples:

    sample 1: prompt = [human[0]],                     gold_tool_calls = turn-1 tools
    sample 2: prompt = [human[0], gpt[0], human[1]],   gold_tool_calls = turn-2 tools

so every sample is a "given this context, predict the next reply" task.

Format notes:
  - verl reads the gold data out of reward_model.ground_truth.
  - tool_schemas is serialised to a JSON string to sidestep PyArrow typing.
"""

import json
import argparse
import os
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import re


def parse_function_call(value: str) -> Optional[Dict[str, Any]]:
    """Parse the value field of a function_call message."""
    if not value:
        return None

    value = value.strip()

    # try to parse the JSON directly
    if value.startswith('{'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    # retry parsing after stripping the <think> tags
    think_pattern = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
    cleaned = think_pattern.sub('', value).strip()
    if cleaned.startswith('{'):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    return None


def extract_allowed_tools(tools_str: str) -> Tuple[List[str], List[Dict]]:
    """Extract the allowed tool names and schemas from the tools JSON string."""
    if not tools_str:
        return [], []

    try:
        tools = json.loads(tools_str)
    except json.JSONDecodeError:
        return [], []

    allowed_tools = []
    tool_schemas = []

    for tool in tools:
        if isinstance(tool, dict):
            if 'function' in tool:
                func = tool['function']
                name = func.get('name', '')
                if name:
                    allowed_tools.append(name)
                    tool_schemas.append(tool)
            elif 'name' in tool:
                name = tool.get('name', '')
                if name:
                    allowed_tools.append(name)
                    tool_schemas.append({
                        "type": "function",
                        "function": tool
                    })

    return allowed_tools, tool_schemas


def format_tools_for_prompt(tool_schemas: List[Dict], style: str = "json") -> str:
    """Format the tool schemas into a readable string for injection into the prompt.

    Args:
        tool_schemas: list of tool schemas
        style: formatting style
            - "json": raw JSON format
            - "compact": compact name + description + parameters format
            - "markdown": Markdown format

    Returns:
        The formatted tool definition string
    """
    if not tool_schemas:
        return ""

    if style == "json":
        return json.dumps(tool_schemas, ensure_ascii=False, indent=2)

    elif style == "compact":
        lines = []
        for tool in tool_schemas:
            func = tool.get('function', tool)
            name = func.get('name', 'unknown')
            desc = func.get('description', '')
            params = func.get('parameters', {})
            props = params.get('properties', {})
            required = params.get('required', [])

            param_strs = []
            for pname, pinfo in props.items():
                ptype = pinfo.get('type', 'any')
                pdesc = pinfo.get('description', '')
                req_mark = '*' if pname in required else ''
                param_strs.append(f"{pname}{req_mark}: {ptype}" + (f" ({pdesc})" if pdesc else ""))

            params_str = ", ".join(param_strs) if param_strs else "none"
            lines.append(f"- {name}: {desc}\n  Parameters: {params_str}")

        return "\n".join(lines)

    elif style == "markdown":
        lines = ["## Available Tools\n"]
        for tool in tool_schemas:
            func = tool.get('function', tool)
            name = func.get('name', 'unknown')
            desc = func.get('description', '')
            params = func.get('parameters', {})
            props = params.get('properties', {})
            required = params.get('required', [])

            lines.append(f"### `{name}`")
            lines.append(f"{desc}\n")

            if props:
                lines.append("**Parameters:**")
                for pname, pinfo in props.items():
                    ptype = pinfo.get('type', 'any')
                    pdesc = pinfo.get('description', '')
                    req_mark = " (required)" if pname in required else ""
                    lines.append(f"- `{pname}` ({ptype}){req_mark}: {pdesc}")
                lines.append("")

        return "\n".join(lines)

    elif style == "qwen":
        # Qwen style: identical to QwenToolUtils.tool_formatter in LLaMA-Factory
        # this ensures SFT and RL training use the same tool prompt format
        tool_text = ""
        for tool in tool_schemas:
            # ensure the tool has the form {"type": "function", "function": {...}}
            if tool.get('type') == 'function':
                wrapped_tool = tool
            else:
                wrapped_tool = {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        qwen_prompt = (
            "\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>{tool_text}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )
        return qwen_prompt

    else:
        return json.dumps(tool_schemas, ensure_ascii=False)


def find_turn_boundaries(conversations: List[Dict]) -> List[int]:
    """Find the start index of every user turn.

    Returns:
        List of indices of the user messages
    """
    boundaries = []
    for i, conv in enumerate(conversations):
        if conv.get('from') == 'human':
            boundaries.append(i)
    return boundaries


def extract_turn_tool_calls(conversations: List[Dict], start_idx: int, end_idx: int) -> List[str]:
    """Extract the tool call names within the given range."""
    tool_calls = []
    for conv in conversations[start_idx:end_idx]:
        if conv.get('from') == 'function_call':
            parsed = parse_function_call(conv.get('value', ''))
            if parsed and 'name' in parsed:
                tool_calls.append(parsed['name'])
    return tool_calls


def convert_messages_to_prompt(
    conversations: List[Dict],
    end_idx: int,
    tool_schemas: Optional[List[Dict]] = None,
    inject_tools: bool = False,
    tool_format_style: str = "json",
    inject_position: str = "system"
) -> List[Dict]:
    """Convert the dialogue into an OpenAI-format prompt.

    Converts every message type, including function_call and observation, to keep the full dialogue context.
    - human -> user
    - gpt -> assistant
    - function_call -> assistant (with <tool_call> tags, Qwen format)
    - observation -> user (with a tool response marker)

    Args:
        conversations: full dialogue list
        end_idx: cut-off index (exclusive)
        tool_schemas: list of tool schemas (optional)
        inject_tools: whether to inject the tool definitions into the prompt
        tool_format_style: tool formatting style ("json", "compact", "markdown", "qwen")
        inject_position: injection position
            - "system": prepend as a system message
            - "first_user": prepend to the first user message

    Returns:
        List of messages in OpenAI format
    """
    prompt = []

    # inject the tool definitions into the system message if requested
    tools_text = ""
    if inject_tools and tool_schemas:
        tools_text = format_tools_for_prompt(tool_schemas, style=tool_format_style)
        if inject_position == "system" and tools_text:
            system_content = f"You have access to the following tools:\n\n{tools_text}\n\nUse them when appropriate to help answer the user's questions."
            prompt.append({"role": "system", "content": system_content})

    first_user_processed = False
    for conv in conversations[:end_idx]:
        role = conv.get('from', '')
        value = conv.get('value', '')

        if role == 'human':
            # inject the tools into the first user message if requested
            if inject_tools and inject_position == "first_user" and not first_user_processed and tools_text:
                value = f"Available tools:\n{tools_text}\n\n{value}"
            first_user_processed = True
            prompt.append({"role": "user", "content": value})
        elif role == 'gpt':
            prompt.append({"role": "assistant", "content": value})
        elif role == 'function_call':
            # convert function_call into an assistant message using the Qwen <tool_call> format
            # this keeps tool calls in the history in the same format the model generates
            parsed = parse_function_call(value)
            if parsed:
                # wrap the tool call in Qwen format
                tool_call_content = f"<tool_call>\n{json.dumps(parsed, ensure_ascii=False)}\n</tool_call>"
                prompt.append({"role": "assistant", "content": tool_call_content})
            else:
                # keep the raw content if parsing fails
                prompt.append({"role": "assistant", "content": value})
        elif role == 'observation':
            # convert observation into a user message (the tool execution result)
            # use an explicit marker so the model knows this is a tool result
            observation_content = f"<tool_response>\n{value}\n</tool_response>"
            prompt.append({"role": "user", "content": observation_content})

    return prompt


def get_question_content(conversations: List[Dict], turn_start_idx: int) -> str:
    """Get the user question content of the current turn."""
    if turn_start_idx < len(conversations):
        conv = conversations[turn_start_idx]
        if conv.get('from') == 'human':
            return conv.get('value', '')
    return ''


def convert_sample_split_turns(
    sample: Dict,
    serialize_for_parquet: bool = True,
    inject_tools: bool = False,
    tool_format_style: str = "json",
    inject_position: str = "system"
) -> List[Dict]:
    """Split one multi-turn record into several training samples.

    Args:
        sample: the source record
        serialize_for_parquet: whether to serialise complex structures
        inject_tools: whether to inject the tool definitions into the prompt
        tool_format_style: tool formatting style ("json", "compact", "markdown")
        inject_position: injection position ("system" or "first_user")

    Returns:
        List of samples produced by the split
    """
    conversations = sample.get('conversations', [])
    tools_str = sample.get('tools', '')

    allowed_tools, tool_schemas = extract_allowed_tools(tools_str)

    # find the boundaries of all user turns
    user_indices = find_turn_boundaries(conversations)

    if not user_indices:
        return []

    results = []

    for i, turn_start in enumerate(user_indices):
        # determine where the current turn ends
        if i + 1 < len(user_indices):
            turn_end = user_indices[i + 1]
        else:
            turn_end = len(conversations)

        # extract the tool calls of the current turn
        gold_tool_calls = extract_turn_tool_calls(conversations, turn_start, turn_end)

        # determine subset_name
        # a turn with no tool call is treated as irrelevant
        subset_name = "tool_call" if gold_tool_calls else "irrelevant"

        # build the prompt: all history from the start up to the current user turn
        # include up to and including the current user message
        # every split sample needs the tool definitions, because each one is an independent training sample
        # the model needs the available tools during rollout to decide correctly
        prompt = convert_messages_to_prompt(
            conversations, turn_start + 1,
            tool_schemas=tool_schemas if inject_tools else None,
            inject_tools=inject_tools,
            tool_format_style=tool_format_style,
            inject_position=inject_position
        )

        if not prompt:
            continue

        # get the question content of the current turn
        question_content = get_question_content(conversations, turn_start)

        # build ground_truth
        ground_truth = {
            "allowed_tools": allowed_tools,
            "gold_tool_calls": gold_tool_calls,
            "subset_name": subset_name,
            "question_content": question_content,
            "tool_schemas": json.dumps(tool_schemas, ensure_ascii=False) if serialize_for_parquet else tool_schemas,
        }

        output = {
            "prompt": prompt,
            "data_source": "toucan_toolcall",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "need_tools_kwargs": False,
                "original_turn_index": i,  # which turn of the source record this came from
            }
        }

        if tool_schemas:
            output["tools"] = json.dumps(tool_schemas, ensure_ascii=False) if serialize_for_parquet else tool_schemas

        results.append(output)

    return results


def convert_sample_first_turn_only(
    sample: Dict,
    serialize_for_parquet: bool = True,
    inject_tools: bool = False,
    tool_format_style: str = "json",
    inject_position: str = "system"
) -> Optional[Dict]:
    """Convert the first turn only (original mode).

    Args:
        sample: the source record
        serialize_for_parquet: whether to serialise complex structures
        inject_tools: whether to inject the tool definitions into the prompt
        tool_format_style: tool formatting style ("json", "compact", "markdown")
        inject_position: injection position ("system" or "first_user")

    Returns:
        The converted record, or None if the conversion fails
    """
    conversations = sample.get('conversations', [])
    tools_str = sample.get('tools', '')

    allowed_tools, tool_schemas = extract_allowed_tools(tools_str)

    # extract all tool calls (used for gold_tool_calls)
    all_tool_calls = []
    for conv in conversations:
        if conv.get('from') == 'function_call':
            parsed = parse_function_call(conv.get('value', ''))
            if parsed and 'name' in parsed:
                all_tool_calls.append(parsed['name'])

    # use only the first user message as the prompt
    # find the index of the first human message
    first_human_idx = -1
    for i, conv in enumerate(conversations):
        if conv.get('from') == 'human':
            first_human_idx = i
            break

    if first_human_idx < 0:
        return None

    # build the prompt with convert_messages_to_prompt (supports tool injection)
    prompt = convert_messages_to_prompt(
        conversations, first_human_idx + 1,
        tool_schemas=tool_schemas if inject_tools else None,
        inject_tools=inject_tools,
        tool_format_style=tool_format_style,
        inject_position=inject_position
    )

    if not prompt:
        return None

    # get the question content
    question_content = conversations[first_human_idx].get('value', '')

    # determine subset_name
    subset_name = "tool_call" if all_tool_calls else "irrelevant"

    ground_truth = {
        "allowed_tools": allowed_tools,
        "gold_tool_calls": all_tool_calls,
        "subset_name": subset_name,
        "question_content": question_content,
        "tool_schemas": json.dumps(tool_schemas, ensure_ascii=False) if serialize_for_parquet else tool_schemas,
    }

    output = {
        "prompt": prompt,
        "data_source": "toucan_toolcall",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "need_tools_kwargs": False,
        }
    }

    if tool_schemas:
        output["tools"] = json.dumps(tool_schemas, ensure_ascii=False) if serialize_for_parquet else tool_schemas

    return output


def convert_dataset(
    input_path: str,
    output_path: str,
    mode: str = "split_turns",
    max_samples: int = -1,
    inject_tools: bool = False,
    tool_format_style: str = "json",
    inject_position: str = "system",
) -> Dict[str, Any]:
    """Convert the whole dataset.

    Args:
        input_path: path to the input JSON file
        output_path: path to the output parquet file
        mode: conversion mode, "split_turns" or "first_turn_only"
        max_samples: maximum number of samples; -1 means all
        inject_tools: whether to inject the tool definitions into the prompt
        tool_format_style: tool formatting style ("json", "compact", "markdown")
        inject_position: injection position ("system" or "first_user")

    Returns:
        Conversion statistics
    """
    print(f"Reading input file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    if max_samples > 0:
        data = data[:max_samples]

    print(f"Total records: {total}, processed: {len(data)}")
    print(f"Conversion mode: {mode}")
    print(f"Inject tool definitions: {inject_tools}")
    if inject_tools:
        print(f"  - format style: {tool_format_style}")
        print(f"  - injection position: {inject_position}")

    converted = []
    stats = {
        "total_input": len(data),
        "total_output": 0,
        "tool_call": 0,
        "irrelevant": 0,
        "errors": 0,
        "tool_call_counts": Counter(),
        "multi_turn_samples": 0,
    }

    for i, sample in enumerate(data):
        try:
            if mode == "split_turns":
                samples = convert_sample_split_turns(
                    sample,
                    inject_tools=inject_tools,
                    tool_format_style=tool_format_style,
                    inject_position=inject_position
                )
                if not samples:
                    stats["errors"] += 1
                    continue

                # count the multi-turn records
                if len(samples) > 1:
                    stats["multi_turn_samples"] += 1

                for s in samples:
                    converted.append(s)
                    gt = s["reward_model"]["ground_truth"]
                    if gt["subset_name"] == "tool_call":
                        stats["tool_call"] += 1
                    else:
                        stats["irrelevant"] += 1
                    stats["tool_call_counts"][len(gt["gold_tool_calls"])] += 1
            else:
                # first_turn_only mode
                converted_sample = convert_sample_first_turn_only(
                    sample,
                    inject_tools=inject_tools,
                    tool_format_style=tool_format_style,
                    inject_position=inject_position
                )
                if converted_sample is None:
                    stats["errors"] += 1
                    continue

                converted.append(converted_sample)
                gt = converted_sample["reward_model"]["ground_truth"]
                if gt["subset_name"] == "tool_call":
                    stats["tool_call"] += 1
                else:
                    stats["irrelevant"] += 1
                stats["tool_call_counts"][len(gt["gold_tool_calls"])] += 1

        except Exception as e:
            print(f"Record {i} failed to convert: {e}")
            stats["errors"] += 1

    stats["total_output"] = len(converted)

    print(f"\nConversion complete:")
    print(f"  - input records: {stats['total_input']}")
    print(f"  - output records: {stats['total_output']}")
    print(f"  - tool_call records: {stats['tool_call']}")
    print(f"  - irrelevant records: {stats['irrelevant']}")
    print(f"  - errors: {stats['errors']}")
    if mode == "split_turns":
        print(f"  - multi-turn records (split): {stats['multi_turn_samples']}")
    print(f"  - tool call count distribution: {dict(sorted(stats['tool_call_counts'].items())[:10])}")

    # save as parquet
    try:
        import datasets

        dataset = datasets.Dataset.from_list(converted)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        dataset.to_parquet(output_path)
        print(f"\nSaved to: {output_path}")
        print(f"Dataset features: {dataset.features}")

    except ImportError as e:
        print(f"Warning: datasets library unavailable: {e}")
        output_json_path = output_path.replace('.parquet', '.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {output_json_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert the Toucan dataset into verl format")
    parser.add_argument(
        "--input", "-i",
        default="./data/toucan_toolcall_rl.json",
        help="Path to the input JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/toucan_toolcall_rl.parquet",
        help="Path to the output parquet file"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["split_turns", "first_turn_only"],
        default="split_turns",
        help="Conversion mode: split_turns (split multi-turn dialogues) or first_turn_only (keep only the first turn)"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=-1,
        help="Maximum number of samples; -1 means all"
    )
    parser.add_argument(
        "--inject-tools",
        action="store_true",
        help="Whether to inject the tool definitions into the prompt (recommended for offline training)"
    )
    parser.add_argument(
        "--tool-format",
        choices=["json", "compact", "markdown", "qwen"],
        default="qwen",
        help="Formatting style for the tool definitions: qwen (recommended, matches SFT), json, compact, markdown"
    )
    parser.add_argument(
        "--inject-position",
        choices=["system", "first_user"],
        default="system",
        help="Where to inject the tool definitions: system (system message) or first_user (start of the first user message)"
    )

    args = parser.parse_args()

    convert_dataset(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        max_samples=args.max_samples,
        inject_tools=args.inject_tools,
        tool_format_style=args.tool_format,
        inject_position=args.inject_position,
    )


if __name__ == "__main__":
    main()
