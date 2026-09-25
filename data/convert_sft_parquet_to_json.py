#!/usr/bin/env python3
"""
Convert a Toucan-Toolcall SFT parquet into the LLaMA-Factory-ready JSON file
referenced by `data/dataset_info.json`.

Input parquet (as published on HuggingFace dataset
`YanZhanPKU/toucan-toolcall-slca`): a single `conversations` column
in ShareGPT `from/value` form, with hermes-style `<tools>` / `<tool_call>` /
`<tool_response>` markup embedded in the system / gpt / human turns.

Output JSON: a list of records, each with the following fields (same schema
as `data/samples/toucan_toolcall_sft.preview.jsonl`):

    messages : JSON-serialized list[{role, content}], where
               role in {"user", "assistant", "tool_call", "tool_response"}.
               Parallel tool calls in a single gpt turn are expanded into
               separate `tool_call` messages, one per call.
    tools    : JSON-serialized list[OpenAI function-schema dict], extracted
               from the `<tools>...</tools>` block in the system turn.

Usage:
    python data/convert_sft_parquet_to_json.py \
        --input  data/toucan_toolcall_sft_split_42k.parquet \
        --output data/toucan_toolcall_sft.json

    python data/convert_sft_parquet_to_json.py \
        --input  data/toucan_toolcall_sft_full_74k.parquet \
        --output data/toucan_toolcall_full.json

LLaMA-Factory then reads these JSON files via the `toucan_toolcall_sft` and
`toucan_toolcall_full` dataset ids registered in `data/dataset_info.json`.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

TOOLS_BLOCK_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
TOOL_CALL_RE   = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESP_RE   = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def _parse_json_or_pyliteral(blob: str):
    """Parquet rows were produced by a pipeline that occasionally emits
    Python-literal syntax (single quotes) rather than strict JSON. Try
    json.loads first, fall back to ast.literal_eval for the Python-literal
    form. Fail loudly if neither works so we notice pipeline drift."""
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return ast.literal_eval(blob)


def convert_row(conversations):
    """Turn one parquet row's `conversations` list into (messages, tools)."""
    if not conversations or conversations[0]["from"] != "system":
        raise ValueError("expected first turn to be `system`")

    system_value = conversations[0]["value"]
    m = TOOLS_BLOCK_RE.search(system_value)
    if not m:
        raise ValueError("no <tools>...</tools> block in system turn")
    # The <tools> block is NDJSON (newline-delimited JSON), one tool schema
    # per line — matches the hermes tool-calling system-prompt convention.
    # A handful of splits fall back to a single JSON object (1-tool) or a
    # JSON array; normalize all three to a list.
    tools_blob = m.group(1).strip()
    tools = []
    try:
        parsed = json.loads(tools_blob)
        tools = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        for line in tools_blob.splitlines():
            line = line.strip()
            if line:
                tools.append(_parse_json_or_pyliteral(line))

    messages = []
    for turn in conversations[1:]:
        frm, val = turn["from"], turn["value"]
        if frm == "human":
            resp = TOOL_RESP_RE.search(val)
            if resp:
                messages.append({"role": "tool_response",
                                 "content": resp.group(1)})
            else:
                messages.append({"role": "user", "content": val})
        elif frm == "gpt":
            call = TOOL_CALL_RE.search(val)
            if call:
                calls = _parse_json_or_pyliteral(call.group(1))
                if not isinstance(calls, list):
                    calls = [calls]
                for c in calls:
                    # Match the preview format: outer dict stringified via
                    # Python's `str(dict)`; inner `arguments` re-serialized
                    # as a JSON string (the author's original pipeline stores
                    # tool_call content this way).
                    call_dict = {
                        "name": c["name"],
                        "arguments": json.dumps(c["arguments"],
                                                ensure_ascii=False),
                    }
                    messages.append({"role": "tool_call",
                                     "content": str(call_dict)})
            else:
                messages.append({"role": "assistant", "content": val})
        else:
            raise ValueError(f"unexpected `from` value: {frm!r}")

    return messages, tools


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  required=True,
                    help="Path to input parquet (single `conversations` column)")
    ap.add_argument("--output", required=True,
                    help="Path to output JSON (list of LF-ready records)")
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        print(f"ERROR: input parquet not found: {inp}", file=sys.stderr)
        sys.exit(1)

    pf = pq.ParquetFile(str(inp))
    total_rows = pf.metadata.num_rows

    records = []
    converted = 0
    for batch in pf.iter_batches(batch_size=args.batch_size,
                                 columns=["conversations"]):
        for row in batch.to_pylist():
            messages, tools = convert_row(row["conversations"])
            records.append({
                "messages": json.dumps(messages, ensure_ascii=False),
                "tools":    json.dumps(tools,    ensure_ascii=False),
            })
            converted += 1
            if converted % 5000 == 0:
                print(f"  converted {converted}/{total_rows} rows ...",
                      file=sys.stderr, flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"wrote {converted} records to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
