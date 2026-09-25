"""
Build the planner-only SFT dataset for the RLTR baseline.

Converts the Toucan SFT data into planner-only form:
1. Drop the trailing summary turn (the last assistant message with no
   <tool_call> in it).
2. Append an <answer> terminator as the final assistant message.

Input : ./data/toucan_toolcall_sft.json  (ShareGPT-shaped SFT data, summaries included;
        produced by data/convert_sft_parquet_to_json.py)
Output: ./baselines/rltr/data/toucan_planner_sft.json  (summaries stripped, <answer> appended)

Register the output in LLaMA-Factory's data/dataset_info.json as
`toucan_planner_sft` before running the planner SFT configs in sft_configs/.
"""

import json
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = str(_REPO_ROOT / "data" / "toucan_toolcall_sft.json")
DEFAULT_OUTPUT = str(Path(__file__).parent / "data" / "toucan_planner_sft.json")

ANSWER_TOKEN = "<answer>"


def process_conversation(conv: list[dict]) -> list[dict] | None:
    """
    Process a single conversation:
    - Find the last gpt message; if it has no <tool_call>, treat it as a summary and drop it
    - Append a gpt: <answer> terminator at the end

    Returns:
        The processed conversation, or None if the record is malformed
    """
    if len(conv) < 3:
        return None

    # Find the index of the last gpt message
    last_gpt_idx = None
    for i in range(len(conv) - 1, -1, -1):
        if conv[i].get("from") == "gpt":
            last_gpt_idx = i
            break

    if last_gpt_idx is None:
        return None

    last_gpt_value = conv[last_gpt_idx].get("value", "")

    # Last gpt message has no <tool_call> -> it is a summary, drop it
    if "<tool_call>" not in last_gpt_value:
        conv = conv[:last_gpt_idx]
    # Otherwise keep it (the record itself has no summary)

    if len(conv) < 2:
        return None

    # Append the <answer> terminator
    conv.append({"from": "gpt", "value": ANSWER_TOKEN})

    return conv


def validate_result(data: list[dict]) -> dict:
    """Validate the processed data"""
    stats = {
        "total": len(data),
        "last_msg_is_answer": 0,
        "has_summary_leak": 0,
        "min_turns": float("inf"),
        "max_turns": float("inf") * -1,
    }

    for item in data:
        conv = item["conversations"]
        stats["min_turns"] = min(stats["min_turns"], len(conv))
        stats["max_turns"] = max(stats["max_turns"], len(conv))

        last_msg = conv[-1]
        if last_msg.get("from") == "gpt" and last_msg.get("value") == ANSWER_TOKEN:
            stats["last_msg_is_answer"] += 1

        # Check for summary leakage: a second-to-last gpt message with no tool_call that is not system/human
        for msg in conv[:-1]:
            if msg.get("from") == "gpt" and "<tool_call>" not in msg.get("value", ""):
                # This may be an intermediate gpt message (no tool call), not necessarily a summary
                # It is fine as long as the last message is <answer>
                pass

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare Planner SFT data")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSON file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON file")
    args = parser.parse_args()

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    with open(args.input, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Raw record count: {len(raw_data)}")

    processed = []
    skipped = 0

    for item in raw_data:
        conv = item.get("conversations", [])
        result = process_conversation(list(conv))  # pass in a copy
        if result is None:
            skipped += 1
            continue
        processed.append({"conversations": result})

    print(f"Processed record count: {len(processed)}")
    print(f"Skipped record count: {skipped}")

    # Validate
    stats = validate_result(processed)
    print(f"\nValidation result:")
    print(f"  total records: {stats['total']}")
    print(f"  last message is <answer>: {stats['last_msg_is_answer']}/{stats['total']}")
    print(f"  conversation turn range: {stats['min_turns']} ~ {stats['max_turns']}")

    assert stats["last_msg_is_answer"] == stats["total"], \
        f"Validation failed: the last message is not <answer> in {stats['total'] - stats['last_msg_is_answer']} records"

    # Output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"\nData saved to: {args.output}")

    # Sample preview
    print(f"\nSample preview (first 3 records):")
    for i, item in enumerate(processed[:3]):
        conv = item["conversations"]
        print(f"\n  === Sample {i} ({len(conv)} messages) ===")
        for j, msg in enumerate(conv):
            role = msg["from"]
            val = msg["value"]
            has_tc = "<tool_call>" in val
            has_tr = "<tool_response>" in val
            preview = val[:80].replace("\n", "\\n")
            print(f"    [{j}] {role}: tc={has_tc}, tr={has_tr} | {preview}...")


if __name__ == "__main__":
    main()
