# Data

The SLCA-GRPO pipeline consumes four splits:

| split              | rows   | used by                                   | format                                     |
| :----------------- | :----- | :---------------------------------------- | :----------------------------------------- |
| SFT (split)        | 42,423 | `sft/qwen2_5_7b_split_sft.yaml` (SFT stage of the main-table split protocol) | LLaMA-Factory `sharegpt` (parquet → JSON)  |
| SFT (full)         | 74,241 | `sft/qwen2_5_7b_full_sft.yaml` (single-stage "SFT full" ablation) | LLaMA-Factory `sharegpt` (parquet → JSON)  |
| RL (tool-call)     | 31,818 | `rl/slca_grpo/train_slca_grpo.sh`         | verl-native parquet                        |
| Eval (in-domain)   | 4,000  | optional `VAL_DATA` in RL; in-domain Name-F1 / ArgMatch / Process / Success | verl-native parquet                        |

The **out-of-domain** evals (BFCL-v3 and τ²-Bench) are **not** shipped here;
`evalscope` fetches them from ModelScope on first run.

The training parquets are **not** committed to this repo (each file is
~150-250 MB, which exceeds GitHub's 100 MB/file cap). Download them from the
Hugging Face release that accompanies the paper (link in the top-level
`README.md`), then place them here:

```
data/
├── toucan_toolcall_sft_split_42k.parquet      # raw download for SFT (split)
├── toucan_toolcall_sft_full_74k.parquet       # raw download for SFT (full)
├── toucan_toolcall_rl.parquet                 # used directly by rl/slca_grpo/train_slca_grpo.sh
├── toucan_eval_4k_unified.parquet             # used directly as VAL_DATA / in-domain eval
├── dataset_info.json                          # LLaMA-Factory registration (points at .json, below)
├── convert_sft_parquet_to_json.py             # materialiser: SFT parquet -> LF-ready JSON
├── convert_toucan_to_verl_v2.py               # RL: Toucan JSON -> verl parquet (nested gold_tool_calls)
├── convert_toucan_to_verl.py                  # RL: v1 converter, flat gold_tool_calls (superseded)
├── update_instruction_prompt.py               # legacy: rewrite the ReAct system prompt in place
├── convert_toucan_json_to_parquet_pyarrow.py  # legacy: single-turn JSON -> parquet via pyarrow
├── toucan_toolcall_sft.json                   # OUTPUT: materialised SFT (split), consumed by LF
├── toucan_toolcall_full.json                  # OUTPUT: materialised SFT (full), consumed by LF
└── samples/
    ├── toucan_toolcall_sft.preview.jsonl      # 5-row preview (same schema as the materialised JSON)
    └── toucan_toolcall_rl.preview.jsonl       # 5-row preview (same schema as the RL parquet)
```

A small preview of each split is included under `data/samples/` so you can
sanity-check the expected schema without downloading the full files.

## Provenance

All four splits are derived from
[Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M), the
open tool-calling corpus released by Agent-Ark, following the paper's
four-phase pipeline:

1. **Noise filtering and schema validation.** From 119,279 raw Toucan
   trajectories, drop ~40,000 that are irrelevant or badly formatted, then
   strict-filter the remaining 79,279 to remove 1,038 with broken JSON,
   non-standard tool tokens or hallucinations → a valid pool of **78,241**.
2. **Evaluation isolation.** Sample **4,000** held-out rows *before* any
   training split is made (3,000 native single-turn + 1,000 decomposed
   multi-turn) → `toucan_eval_4k_unified.parquet`. The remaining **74,241**
   are the total training pool, released as `sft_full`.
3. **Trajectory decomposition.** Multi-turn dialogues are decomposed into
   atomic context-action pairs for RL; the first tool-call turn is excluded,
   so decomposed samples start at turn ≥ 2 and carry prior conversation
   history.
4. **SFT / RL partitioning.** The training pool splits into **42,423** SFT
   rows and **31,818** RL rows (42,423 + 31,818 = 74,241). Each RL row is
   tagged with `ground_truth = {allowed_tools, gold_tool_calls,
   question_content, ...}` and serialised into verl's prompt/reward-model
   parquet layout.

Phases 1–2 and the partitioning are internal; the released parquets are the
final materialised splits. Phase 4's serialisation into verl's layout ships as
`convert_toucan_to_verl_v2.py`, described next.

> The step from the full Toucan-1.5M corpus down to the 119,279 raw
> trajectories that phase 1 starts from is not documented in the paper.

## RL converters

`convert_toucan_to_verl_v2.py` is the script the released RL parquet was built
with, and the one you need if you rebuild the RL split yourself:

```bash
python data/convert_toucan_to_verl_v2.py \
    --input  ./data/toucan_toolcall_rl.json \
    --output ./data/toucan_toolcall_rl.parquet \
    --mode split_turns \
    --inject-tools \
    --tool-format qwen
```

> **Use v2, not v1.** v2 writes `gold_tool_calls` as a **nested** list
> (`List[List[Dict]]`), where each inner list is one step and several tools in
> one step means a parallel call. The `parallel` sub-term of the HierR reward
> (`SLCA_WEIGHT_PARALLEL`, `R_parallel`) is computed from exactly that nesting.
> `convert_toucan_to_verl.py` (v1) writes a flat list, which silently collapses
> parallelism and zeroes the parallel reward. v1 ships for provenance only.

`--mode split_turns` (recommended) turns each multi-turn dialogue into one
sample per turn — "given this context, predict the next reply" —
rather than keeping only the first user turn.

Two legacy helpers from an earlier single-turn pipeline are also included:
`update_instruction_prompt.py` (rewrites the ReAct system prompt inside a
`pretty.json`) and `convert_toucan_json_to_parquet_pyarrow.py` (writes that
JSON to parquet through pyarrow instead of `datasets`). Both read and write
paths from environment variables (`INPUT_JSON`, `OUTPUT_JSON`,
`OUTPUT_PARQUET`) and are not part of the current pipeline.

Backbones without instruct priors (Qwen3-8B-Base) train on `<think>`-tagged
variants of these parquets (`*_thinking.parquet`). Those are **not** part of the
Hugging Face release; regenerate them with the same converter, or override
`TRAIN_DATA` / `VAL_DATA` to the standard parquets.

## Materialising SFT data for LLaMA-Factory

LLaMA-Factory wants SFT data as a JSON file where each record has a
`messages` column (stringified chat) and a `tools` column (stringified tool
schema list), matching `data/samples/toucan_toolcall_sft.preview.jsonl`.
The raw SFT parquets do **not** ship in that shape — they carry a single
`conversations` column holding ShareGPT `from/value` turns with hermes
`<tools>` / `<tool_call>` / `<tool_response>` markup embedded in the text.

Run the materialiser once after you download the two parquets:

```bash
python data/convert_sft_parquet_to_json.py \
    --input  data/toucan_toolcall_sft_split_42k.parquet \
    --output data/toucan_toolcall_sft.json

python data/convert_sft_parquet_to_json.py \
    --input  data/toucan_toolcall_sft_full_74k.parquet \
    --output data/toucan_toolcall_full.json
```

What the converter does per row: extracts the NDJSON `<tools>` block from
the `system` turn, expands every `<tool_call>` (parallel calls become one
`tool_call` message each), and rewrites every `<tool_response>` wrapper as
a flat `tool_response` role. The output matches the `samples/*.jsonl`
preview one-for-one, so the preview files double as a golden contract for
conversion correctness.

## LLaMA-Factory registration

`data/dataset_info.json` ships two registrations:

- `toucan_toolcall_sft` → `toucan_toolcall_sft.json` (produced by the
  materialiser above, used by `sft/qwen2_5_7b_split_sft.yaml`).
- `toucan_toolcall_full` → `toucan_toolcall_full.json` (produced likewise,
  used by `sft/qwen2_5_7b_full_sft.yaml`).

Both use `formatting: sharegpt`, `columns: {messages, tools}`, and tag the
roles as `user / assistant / tool_call / tool_response / system` (matching
hermes tool-calling conventions).

Either set `dataset_dir: ./data` inside each SFT YAML, or pass
`--dataset_dir ./data` to LLaMA-Factory's CLI. The shipped YAMLs already
reference the correct dataset name.

## Schema

### SFT parquet (the file you download)

`toucan_toolcall_sft_split_42k.parquet`, `toucan_toolcall_sft_full_74k.parquet`.

Single column:

| column          | type           | notes                                                                |
| :-------------- | :------------- | :------------------------------------------------------------------- |
| `conversations` | list\<struct\> | ShareGPT turns `{from: "system"/"human"/"gpt", value: str}`. The `system` turn embeds the tool schema as **NDJSON** inside a hermes `<tools>…</tools>` block; `gpt` turns may embed `<tool_call>[...]</tool_call>`; `human` turns may embed `<tool_response>…</tool_response>`. |

### SFT JSON (what the materialiser writes)

`toucan_toolcall_sft.json`, `toucan_toolcall_full.json`. Flat records; same
schema as `samples/toucan_toolcall_sft.preview.jsonl`:

| column     | type          | notes                                                                  |
| :--------- | :------------ | :--------------------------------------------------------------------- |
| `messages` | string (JSON) | `[{"role": "user"/"assistant"/"tool_call"/"tool_response", "content": str}, ...]` — parallel tool calls are split into one `tool_call` message per call. |
| `tools`    | string (JSON) | `[{"type": "function", "function": {...}}, ...]` — OpenAI-style tool schema list, extracted from the system turn's `<tools>` block. |

Upstream metadata columns (`uuid`, `subset_name`, `question`, `target_tools`)
are intentionally **not** preserved during materialisation — LLaMA-Factory
doesn't read them.

### RL parquet (`toucan_toolcall_rl.parquet`)

| column         | type           | notes                                                                  |
| :------------- | :------------- | :--------------------------------------------------------------------- |
| `prompt`       | list\<struct\> | chat-formatted prompt consumed by verl's rollout loop                  |
| `data_source`  | string         | always `toucan_toolcall`                                               |
| `reward_model` | struct         | `{"style": "rule", "ground_truth": {"allowed_tools":..., "gold_tool_calls":..., "question_content":..., "tool_schemas":..., "subset_name":...}}` — consumed by `rl/slca_grpo/reward_fn.py` |
| `extra_info`   | struct         | `{"need_tools_kwargs": bool, "original_turn_index": int, ...}`         |
| `tools`        | string (JSON)  | OpenAI-style tool schemas available to the rollout                     |

### Eval parquet (`toucan_eval_4k_unified.parquet`)

Shares the RL parquet schema. Reports the in-domain Name-F1 / ArgMatch /
Process / Success numbers. Can also be passed as `VAL_DATA` to
`rl/slca_grpo/train_slca_grpo.sh` for periodic in-training validation (skip
with `VAL_MAX_SAMPLES=0`).

## Reproducing the splits from raw Toucan-1.5M

The internal filtering pipeline is out of scope for this release, but the
main steps map 1:1 to public tools:
`datasets.load_dataset("Agent-Ark/Toucan-1.5M")` to pull the upstream
parquets, then a small pandas / pyarrow script that filters by
`subset_name`, validates `tools`/`messages` JSON, and writes the four
splits. Reach out via the paper contact if you need the internal filter
script for an exact re-derivation.
