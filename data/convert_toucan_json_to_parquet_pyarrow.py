#!/usr/bin/env python
"""
Convert the Toucan RL pretty.json into a parquet file for the VERL RLHFDataset (using pyarrow instead of datasets.to_parquet).

Input:
  ./data/train_rl_single_turn.pretty.json  (override with INPUT_JSON)

Output:
  ./data/train_rl_single_turn.parquet  (override with OUTPUT_PARQUET)
"""

import json
import os
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# use the updated data (with the new instruction prompt)
JSON_PATH = Path(
    os.environ.get("INPUT_JSON", "./data/train_rl_single_turn_updated.pretty.json")
)
OUT_PATH = Path(
    os.environ.get("OUTPUT_PARQUET", "./data/train_rl_single_turn_updated.parquet")
)

TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def extract_tool_names(text: str):
    names = []
    for match in TOOL_CALL_PATTERN.findall(text or ""):
        content = match.strip()
        if not content:
            continue
        try:
            parsed = json.loads(content)
        except Exception:
            continue
        if isinstance(parsed, dict):
            objs = [parsed]
        elif isinstance(parsed, list):
            objs = parsed
        else:
            continue
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            name = obj.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def main():
    print(f"[INFO] Loading JSON from {JSON_PATH}")
    with JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for i, ex in enumerate(data):
        instruction = ex.get("instruction", "")
        inp = ex.get("input", "")
        out = ex.get("output", "")
        meta = ex.get("meta", {}) or {}
        tools = meta.get("tools", []) or []

        allowed_tools = []
        tools_kwargs = {}
        for t in tools:
            if not isinstance(t, dict):
                continue
            name = t.get("name")
            if isinstance(name, str):
                allowed_tools.append(name)
                tools_kwargs[name] = {
                    "create_kwargs": {
                        # store as a JSON string to avoid parquet limits on complex structs
                        "tool_definition": json.dumps(t, ensure_ascii=False),
                    }
                }

        gold_tool_calls = extract_tool_names(out)

        row = {
            "prompt": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": inp},
            ],
            "data_source": "toucan_tool_rl",
            "reward_model": {
                "ground_truth": {
                    "allowed_tools": allowed_tools,
                    "gold_tool_calls": gold_tool_calls,
                }
            },
            "extra_info": {
                "index": i,
                "allowed_tools": allowed_tools,
                "gold_tool_calls": gold_tool_calls,
                "need_tools_kwargs": True,
                "tools_kwargs": tools_kwargs,
            },
            "raw_instruction": instruction,
            "raw_input": inp,
            "raw_output": out,
        }
        rows.append(row)

    print(f"[INFO] Total examples: {len(rows)}")
    table = pa.Table.from_pylist(rows)

    print(f"[INFO] Writing parquet to {OUT_PATH}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    pq.write_table(table, str(OUT_PATH))
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

