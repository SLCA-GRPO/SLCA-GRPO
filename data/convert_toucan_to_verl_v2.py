"""
Convert the Toucan tool-calling dataset into verl RL training format (v2).

This is the converter the released RL parquet was built with, and the one the
parallel sub-term of the HierR reward requires.

Changes relative to v1
----------------------
1. KEY CHANGE: `gold_tool_calls` becomes a NESTED list, so parallelism survives
   the conversion.

       v1: [{"name": "A", "arguments": {...}}, {"name": "B", ...}]
       v2: [[{"name": "A", ...}, {"name": "B", ...}], [{"name": "C", ...}]]

   Each inner list is one step; several tools inside one step are a parallel
   call. `SLCA_WEIGHT_PARALLEL` / `R_parallel` in the reward functions is
   computed from exactly this structure, so a v1 parquet silently disables the
   parallel term.
2. Reports the distribution of parallel-call shapes (single step / single tool,
   single step with parallel calls, multi-step with parallel calls, ...).
3. `gold_tool_calls` keeps the full argument payload; records without
   `arguments` are back-filled with an empty dict.
4. Parquet is written through pandas to avoid `datasets` metadata issues.

Format notes
------------
  - verl reads the gold data out of reward_model.ground_truth.
  - ground_truth.gold_tool_calls is List[List[Dict]].
  - tool_schemas is serialised to a JSON string to sidestep PyArrow typing.

Conversion modes
----------------
1. first_turn_only: keep only the first user turn as the prompt.
2. split_turns:     split a multi-turn dialogue into independent samples
                    (recommended).

Usage
-----
python data/convert_toucan_to_verl_v2.py \
    --input  /path/to/toucan_toolcall_rl.json \
    --output /path/to/toucan_toolcall_rl.parquet \
    --mode split_turns \
    --inject-tools \
    --tool-format qwen
"""

import json
import argparse
import os
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import re


def parse_function_call(value: str) -> Optional[List[Dict[str, Any]]]:
    """Parse the value field of a function_call message and return the full list of tool calls.

    Two formats are supported:
    1. JSON object: {"name": "xxx", "arguments": {...}}
    2. JSON array: [{"name": "xxx", "arguments": {...}}, ...]

    Returns:
        List of tool calls, e.g. [{"name": "xxx", "arguments": {...}}, ...]
        Returns None if parsing fails
    """
    if not value:
        return None

    value = value.strip()

    def normalize_tool_call(obj):
        """Normalise a single tool call object"""
        if isinstance(obj, dict) and "name" in obj:
            return {
                "name": obj.get("name"),
                "arguments": obj.get("arguments") or obj.get("parameters") or {}
            }
        return None

    def try_parse(text):
        """Try to parse the JSON text"""
        text = text.strip()
        if text.startswith('['):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    results = []
                    for item in parsed:
                        normalized = normalize_tool_call(item)
                        if normalized:
                            results.append(normalized)
                    return results if results else None
            except json.JSONDecodeError:
                pass
        elif text.startswith('{'):
            try:
                parsed = json.loads(text)
                normalized = normalize_tool_call(parsed)
                return [normalized] if normalized else None
            except json.JSONDecodeError:
                pass
        return None

    # try a direct parse
    result = try_parse(value)
    if result:
        return result

    # retry parsing after stripping the <think> tags
    think_pattern = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
    cleaned = think_pattern.sub('', value).strip()
    result = try_parse(cleaned)
    if result:
        return result

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
    """Format the tool schemas into a readable string."""
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
        # Qwen style: kept consistent with the system prompt of the SFT data
        tool_text = ""
        for tool in tool_schemas:
            if tool.get('type') == 'function':
                wrapped_tool = tool
            else:
                wrapped_tool = {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        qwen_prompt = (
            "You are a helpful assistant.\n\n"
            "# Instructions\n"
            "Answer the user's question and output your thinking process within the <think> and </think> tags.\n\n"
            "If tools are needed, output a **JSON list** wrapped in <tool_call></tool_call> XML tags like this:\n"
            "<tool_call>\n"
            '[{"name": <function-name>, "arguments": <args-json-object>}]\n'
            "</tool_call>\n"
            "You can place one or multiple tool calls in the list for parallel execution.\n\n"
            "If the task requires multiple steps, output tool calls, wait for results, and then continue reasoning.\n\n"
            "# Tool Definitions\n"
            f"<tools>{tool_text}\n</tools>"
        )
        return qwen_prompt

    else:
        return json.dumps(tool_schemas, ensure_ascii=False)


def find_turn_boundaries(conversations: List[Dict]) -> List[int]:
    """Find the start index of every user turn."""
    boundaries = []
    for i, conv in enumerate(conversations):
        if conv.get('from') == 'human':
            boundaries.append(i)
    return boundaries


def extract_turn_tool_calls(conversations: List[Dict], start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
    """Extract the tool calls within the given range (flat format, backward compatible).

    Returns:
        List[Dict]: [{"name": "...", "arguments": {...}}, ...]
    """
    tool_calls = []
    for conv in conversations[start_idx:end_idx]:
        if conv.get('from') == 'function_call':
            parsed_list = parse_function_call(conv.get('value', ''))
            if parsed_list:
                tool_calls.extend(parsed_list)
    return tool_calls


def extract_parallel_tool_calls(conversations: List[Dict], start_idx: int = 0, end_idx: int = None) -> List[List[Dict[str, Any]]]:
    """Extract the tool calls while preserving the parallel structure.

    Each function_call message is one step, and a step may contain several parallel tool calls.

    Args:
        conversations: dialogue list
        start_idx: start index (inclusive)
        end_idx: end index (exclusive); None means up to the end

    Returns:
        Nested list: [[step1_calls], [step2_calls], ...]
        Each step is a List[Dict] holding all tool calls of that step (with name and arguments)

    Example:
        The input dialogue contains:
          function_call: [{"name": "A", "arguments": {...}}, {"name": "B", "arguments": {...}}]
          observation: ...
          function_call: {"name": "C", "arguments": {...}}

        Output: [
            [{"name": "A", "arguments": {...}}, {"name": "B", "arguments": {...}}],  # parallel
            [{"name": "C", "arguments": {...}}]  # serial
        ]
    """
    if end_idx is None:
        end_idx = len(conversations)

    result = []

    for conv in conversations[start_idx:end_idx]:
        if conv.get('from') == 'function_call':
            parsed_list = parse_function_call(conv.get('value', ''))
            if parsed_list:
                # parsed_list is a List[Dict]; each Dict holds name and arguments
                # one function_call message = one step
                result.append(parsed_list)

    return result


def classify_parallel_structure(gold_tool_calls: List[List[Dict]]) -> str:
    """Classify the parallel structure type of a record.

    Args:
        gold_tool_calls: tool calls in nested list format

    Returns:
        Classification labels:
        - "irrelevant": no tool call
        - "single_step_single_tool": single step, single tool [[A]]
        - "single_step_parallel": single step, parallel [[A, B]]
        - "multi_step_all_serial": multi-step, all serial [[A], [B], [C]]
        - "multi_step_has_parallel": multi-step with parallel [[A, B], [C]]
        - "multi_step_all_parallel": multi-step, all parallel [[A, B], [C, D]]
    """
    if not gold_tool_calls:
        return "irrelevant"

    step_count = len(gold_tool_calls)
    parallel_steps = [s for s in gold_tool_calls if len(s) > 1]
    parallel_count = len(parallel_steps)

    if step_count == 1:
        if len(gold_tool_calls[0]) == 1:
            return "single_step_single_tool"
        else:
            return "single_step_parallel"
    else:
        if parallel_count == 0:
            return "multi_step_all_serial"
        elif parallel_count == step_count:
            return "multi_step_all_parallel"
        else:
            return "multi_step_has_parallel"


def convert_messages_to_prompt(
    conversations: List[Dict],
    end_idx: int,
    tool_schemas: Optional[List[Dict]] = None,
    inject_tools: bool = False,
    tool_format_style: str = "json",
    inject_position: str = "system"
) -> List[Dict]:
    """Convert the dialogue into an OpenAI-format prompt."""
    prompt = []

    tools_text = ""
    if inject_tools and tool_schemas:
        tools_text = format_tools_for_prompt(tool_schemas, style=tool_format_style)
        if inject_position == "system" and tools_text:
            # use the qwen-format tool definitions directly as the system content, with no extra wrapper
            prompt.append({"role": "system", "content": tools_text})

    first_user_processed = False
    for conv in conversations[:end_idx]:
        role = conv.get('from', '')
        value = conv.get('value', '')

        if role == 'human':
            if inject_tools and inject_position == "first_user" and not first_user_processed and tools_text:
                value = f"Available tools:\n{tools_text}\n\n{value}"
            first_user_processed = True
            prompt.append({"role": "user", "content": value})
        elif role == 'gpt':
            prompt.append({"role": "assistant", "content": value})
        elif role == 'function_call':
            parsed_list = parse_function_call(value)
            if parsed_list:
                # wrap each tool call separately
                tool_call_parts = []
                for tc in parsed_list:
                    tool_call_parts.append(f"<tool_call>\n{json.dumps(tc, ensure_ascii=False)}\n</tool_call>")
                tool_call_content = "\n".join(tool_call_parts)
                prompt.append({"role": "assistant", "content": tool_call_content})
            else:
                prompt.append({"role": "assistant", "content": value})
        elif role == 'observation':
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

    gold_tool_calls uses a nested list format so that parallel call information is preserved
    """
    conversations = sample.get('conversations', [])
    tools_str = sample.get('tools', '')

    allowed_tools, tool_schemas = extract_allowed_tools(tools_str)
    user_indices = find_turn_boundaries(conversations)

    if not user_indices:
        return []

    results = []

    for i, turn_start in enumerate(user_indices):
        if i + 1 < len(user_indices):
            turn_end = user_indices[i + 1]
        else:
            turn_end = len(conversations)

        # extract the tool calls as a nested list to preserve the parallel structure
        gold_tool_calls = extract_parallel_tool_calls(conversations, turn_start, turn_end)

        subset_name = "tool_call" if gold_tool_calls else "irrelevant"

        prompt = convert_messages_to_prompt(
            conversations, turn_start + 1,
            tool_schemas=tool_schemas if inject_tools else None,
            inject_tools=inject_tools,
            tool_format_style=tool_format_style,
            inject_position=inject_position
        )

        if not prompt:
            continue

        question_content = get_question_content(conversations, turn_start)

        # ground_truth.gold_tool_calls is now List[List[Dict]] (a nested list)
        # serialise to a JSON string to avoid PyArrow typing issues
        ground_truth = {
            "allowed_tools": allowed_tools,
            "gold_tool_calls": json.dumps(gold_tool_calls, ensure_ascii=False) if serialize_for_parquet else gold_tool_calls,
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
                "original_turn_index": i,
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
    """Convert the first turn only.

    gold_tool_calls uses a nested list format so that parallel call information is preserved
    """
    conversations = sample.get('conversations', [])
    tools_str = sample.get('tools', '')

    allowed_tools, tool_schemas = extract_allowed_tools(tools_str)

    # extract the tool calls as a nested list to preserve the parallel structure
    gold_tool_calls = extract_parallel_tool_calls(conversations)

    first_human_idx = -1
    for i, conv in enumerate(conversations):
        if conv.get('from') == 'human':
            first_human_idx = i
            break

    if first_human_idx < 0:
        return None

    prompt = convert_messages_to_prompt(
        conversations, first_human_idx + 1,
        tool_schemas=tool_schemas if inject_tools else None,
        inject_tools=inject_tools,
        tool_format_style=tool_format_style,
        inject_position=inject_position
    )

    if not prompt:
        return None

    question_content = conversations[first_human_idx].get('value', '')
    subset_name = "tool_call" if gold_tool_calls else "irrelevant"

    # ground_truth.gold_tool_calls is now List[List[Dict]] (a nested list)
    ground_truth = {
        "allowed_tools": allowed_tools,
        "gold_tool_calls": json.dumps(gold_tool_calls, ensure_ascii=False) if serialize_for_parquet else gold_tool_calls,
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

    Adds parallel structure statistics and reports the edge case distribution
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
        "tool_call_counts": Counter(),  # counted by number of steps (nested list length)
        "multi_turn_samples": 0,
        "with_arguments": 0,  # count of tool calls that carry arguments
        "without_arguments": 0,  # count of tool calls without arguments
        # parallel structure statistics
        "parallel_structure": Counter(),  # counted by parallel structure type
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

                if len(samples) > 1:
                    stats["multi_turn_samples"] += 1

                for s in samples:
                    converted.append(s)
                    gt = s["reward_model"]["ground_truth"]
                    if gt["subset_name"] == "tool_call":
                        stats["tool_call"] += 1
                    else:
                        stats["irrelevant"] += 1

                    # gold_tool_calls is now a nested list, List[List[Dict]]
                    gold_tc_str = gt["gold_tool_calls"]
                    gold_tc_nested = json.loads(gold_tc_str) if isinstance(gold_tc_str, str) else gold_tc_str

                    # count by number of steps
                    stats["tool_call_counts"][len(gold_tc_nested)] += 1

                    # classify the parallel structure
                    structure_type = classify_parallel_structure(gold_tc_nested)
                    stats["parallel_structure"][structure_type] += 1

                    # argument statistics (walk the nested list)
                    for step_calls in gold_tc_nested:
                        for tc in step_calls:
                            if isinstance(tc, dict) and tc.get("arguments"):
                                stats["with_arguments"] += 1
                            else:
                                stats["without_arguments"] += 1
            else:
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

                # gold_tool_calls is now a nested list, List[List[Dict]]
                gold_tc_str = gt["gold_tool_calls"]
                gold_tc_nested = json.loads(gold_tc_str) if isinstance(gold_tc_str, str) else gold_tc_str

                # count by number of steps
                stats["tool_call_counts"][len(gold_tc_nested)] += 1

                # classify the parallel structure
                structure_type = classify_parallel_structure(gold_tc_nested)
                stats["parallel_structure"][structure_type] += 1

                # argument statistics (walk the nested list)
                for step_calls in gold_tc_nested:
                    for tc in step_calls:
                        if isinstance(tc, dict) and tc.get("arguments"):
                            stats["with_arguments"] += 1
                        else:
                            stats["without_arguments"] += 1

        except Exception as e:
            print(f"Record {i} failed to convert: {e}")
            stats["errors"] += 1

    stats["total_output"] = len(converted)

    # ================================================================================
    # print the statistics
    # ================================================================================
    print(f"\n{'='*60}")
    print("Conversion complete - statistics")
    print(f"{'='*60}")
    print(f"Basic statistics:")
    print(f"  - input records: {stats['total_input']}")
    print(f"  - output records: {stats['total_output']}")
    print(f"  - tool_call records: {stats['tool_call']}")
    print(f"  - irrelevant records: {stats['irrelevant']}")
    print(f"  - errors: {stats['errors']}")
    if mode == "split_turns":
        print(f"  - multi-turn records (split): {stats['multi_turn_samples']}")

    print(f"\nStep count distribution (nested list length):")
    for step_count, count in sorted(stats['tool_call_counts'].items())[:10]:
        pct = 100.0 * count / stats['total_output'] if stats['total_output'] > 0 else 0
        print(f"  - {step_count} steps: {count} ({pct:.1f}%)")

    print(f"\nArgument statistics:")
    print(f"  - tool calls with arguments: {stats['with_arguments']}")
    print(f"  - tool calls without arguments: {stats['without_arguments']}")

    # parallel structure statistics (edge case distribution)
    print(f"\n{'='*60}")
    print("Parallel structure statistics (edge case distribution)")
    print(f"{'='*60}")
    total_samples = stats['total_output']

    # display order and labels
    structure_labels = {
        "irrelevant": "no tool call (irrelevant)",
        "single_step_single_tool": "single step, single tool [[A]]",
        "single_step_parallel": "single step, parallel [[A, B]]",
        "multi_step_all_serial": "multi-step, all serial [[A], [B], [C]]",
        "multi_step_has_parallel": "multi-step with parallel [[A, B], [C]]",
        "multi_step_all_parallel": "multi-step, all parallel [[A, B], [C, D]]",
    }

    for struct_type, label in structure_labels.items():
        count = stats["parallel_structure"].get(struct_type, 0)
        pct = 100.0 * count / total_samples if total_samples > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    # aggregate the records that contain parallel calls
    parallel_count = (
        stats["parallel_structure"].get("single_step_parallel", 0) +
        stats["parallel_structure"].get("multi_step_has_parallel", 0) +
        stats["parallel_structure"].get("multi_step_all_parallel", 0)
    )
    parallel_pct = 100.0 * parallel_count / total_samples if total_samples > 0 else 0
    print(f"\n  Total records with parallel calls: {parallel_count} ({parallel_pct:.1f}%)")
    print(f"{'='*60}")

    # save as parquet
    # save through pandas to avoid compatibility issues caused by the metadata the datasets library writes
    # parquet written by the datasets library uses the "List" type, while loading expects "Sequence"
    try:
        import pandas as pd

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # convert to a DataFrame
        df = pd.DataFrame(converted)
        print(f"\nColumns: {list(df.columns)}")
        print(f"Rows: {len(df)}")

        # save the parquet through pandas (produces compatible metadata)
        df.to_parquet(output_path, index=False, engine='pyarrow')
        print(f"Saved to: {output_path}")

        # verify the file is readable
        df_check = pd.read_parquet(output_path)
        print(f"Verified: file readable, rows={len(df_check)}")

    except ImportError as e:
        print(f"Warning: pandas library unavailable: {e}")
        # fall back to JSON format
        output_json_path = output_path.replace('.parquet', '.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {output_json_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert the Toucan dataset into verl format (v2, keeps arguments)")
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
        help="Conversion mode"
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
        help="Whether to inject the tool definitions into the prompt"
    )
    parser.add_argument(
        "--tool-format",
        choices=["json", "compact", "markdown", "qwen"],
        default="qwen",
        help="Formatting style for the tool definitions"
    )
    parser.add_argument(
        "--inject-position",
        choices=["system", "first_user"],
        default="system",
        help="Where to inject the tool definitions"
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
