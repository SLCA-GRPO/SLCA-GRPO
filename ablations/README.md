# Ablations

Four component ablations of SLCA-GRPO, three of which the paper reports. Each
removes exactly one piece and leaves everything else — data, SFT initialisation,
SGLS environment, rollout budget, optimiser settings — identical to the full
method.

**Backs:** the 7B ablation table (`tab:ablation`, Qwen2.5-7B-Instruct, three-run
mean±std) and the cross-scale ablation table (`tab:full_ablation`, the same
ablations across Qwen2.5-3B / Qwen2.5-7B / Qwen3-8B-Base). Both paper tables
report **three** ablations, so only three of the four scripts here back a
published number — see "Which ablations appear in the paper" below.

> Table **numbers** shift between paper revisions, so this README refers to
> paper tables by their LaTeX `\label` and by descriptive name instead. Search
> the paper source for the label to find the table.

| # | Ablation | What is removed | How | paper row |
| :-: | :--- | :--- | :--- | :--- |
| (i) | w/o SLCA | segment-locked credit routing | `algorithm.adv_estimator=grpo` | `w/o SLCA` |
| (ii) | w/o SGLS | schema guidance in the tool simulator | `ToucanVLLMToolNoSchema` via `config/toucan_tool_config_no_schema.yaml` | `w/o SGLS` |
| (iii) | w/o Tool-Call Reward | the whole process reward | all five `SLCA_WEIGHT_*` process weights set to 0 | `w/o HierR` |
| (iv) | w/o Parallel Reward | only the parallel term | `SLCA_WEIGHT_PARALLEL=0`, remaining weights rescaled to sum to 1 | **none** |

```
ablations/
├── README.md                              # (this file)
├── reward_fn_hierr_v82.py                 # HierR reward shared by all four ablations
├── config/
│   └── toucan_tool_config_no_schema.yaml  # (ii) schema-free tool routing
├── tools/
│   └── toucan_vllm_tool_no_schema.py      # (ii) the schema-free simulator class
├── train_ablation_wo_slca.sh              # (i)
├── train_ablation_wo_sgls.sh              # (ii)
├── train_ablation_wo_tool_reward.sh       # (iii)
└── train_ablation_wo_parallel.sh          # (iv) extra ablation, no paper row
```

## Which ablations appear in the paper

Besides the full method, the paper's ablation tables have exactly three rows:
`w/o SLCA`, `w/o SGLS` and `w/o HierR`. Ablations (i)–(iii) above map onto them.
(iii) is the paper's `w/o HierR` condition under a different name: zeroing every
process sub-weight leaves only the summary reward in the objective, and the
paper likewise still reports Process as a post-hoc evaluation metric for that
row.

**Ablation (iv), `w/o Parallel Reward` (`train_ablation_wo_parallel.sh`), has no
row in any paper table.** It is an additional ablation not reported in the
paper. It ships because it is a real, runnable configuration, but no published
number is backed by it — do not present its output as reproducing a paper
figure.

## Which reward function these use

All four launchers point at **`ablations/reward_fn_hierr_v82.py`**, not at
`rl/slca_grpo/reward_fn.py`. The two are different:

- `rl/slca_grpo/reward_fn.py` is the main-method reward. Its summary judge sees
  only `<tool_response>` blocks, uses a response-quality-only prompt, and scores
  0 when there is no tool response at all.
- `ablations/reward_fn_hierr_v82.py` is its direct predecessor. Its judge sees
  the full tool interaction (`<tool_call>` + `<tool_response>`) under the older
  prompt.

The ablation runs in the paper were produced with the v8.2 file, so it ships
verbatim. **Do not swap one for the other** — the scores are not comparable.
`analysis/hierr_sweep/reward_fn_hierr_v82.py` is a byte-identical copy, kept so
that each experiment directory is self-contained.

Upstream, ablations (iii) and (iv) each carried their own copy of this file that
differed only in the default `SLCA_WEIGHT_*` constants. Because every launcher
`export`s those weights before starting the trainer, the file-level defaults are
never read, and the copies are consolidated into the single file above.

## Prerequisites

Everything the RL stage needs (`rl/README.md`): the 7B SFT checkpoint, the RL /
eval parquets under `data/`, the vendored `verl/` installed, the SGLS server on
port 8003 and the LLM judge on port 8016.

## Running them

Each launcher is backbone-agnostic — point `MODEL_PATH` at the SFT checkpoint
for the backbone you want and, for (i)–(iii), you get the corresponding row of
the cross-scale ablation table (`tab:full_ablation`).

```bash
# (i) w/o SLCA
MODEL_PATH=./outputs/sft_split/qwen2_5_7b_toucan_toolcall \
OUTPUT_DIR=./outputs/ablations/wo_slca \
NNODES=4 bash ablations/train_ablation_wo_slca.sh

# (ii) w/o SGLS
bash ablations/train_ablation_wo_sgls.sh

# (iii) w/o Tool-Call Reward
bash ablations/train_ablation_wo_tool_reward.sh

# (iv) w/o Parallel Reward  -- extra ablation, not reported in the paper
bash ablations/train_ablation_wo_parallel.sh

# Cross-scale ablation table, other backbones: same scripts, different MODEL_PATH
MODEL_PATH=./outputs/sft_split/qwen2_5_3b_toucan_toolcall \
OUTPUT_DIR=./outputs/ablations/wo_slca_3b \
bash ablations/train_ablation_wo_slca.sh
```

The full-method reference row for each table is
`rl/slca_grpo/train_slca_grpo.sh`.

### Weight rebalancing in (iv)

Dropping `parallel` frees 0.30, which is redistributed over the other four so
the process weights still sum to 1.0:

| term | full method | w/o Parallel |
| :--- | :---: | :---: |
| `SLCA_WEIGHT_FORMAT` | 0.10 | 0.15 |
| `SLCA_WEIGHT_NAME` | 0.25 | 0.30 |
| `SLCA_WEIGHT_KEY` | 0.15 | 0.25 |
| `SLCA_WEIGHT_VALUE` | 0.20 | 0.30 |
| `SLCA_WEIGHT_PARALLEL` | 0.30 | 0.00 |

Without this rescaling the ablation would confound "no parallel term" with
"weaker process reward overall".

## What to watch

`process_score` is still computed and logged for every setting, including
(iii) where it does not enter the objective — it is a post-hoc evaluation
metric, which is how the paper reports it.

| metric | most informative for |
| :--- | :--- |
| `total_score`, `success` | all |
| `format_score` | (ii), (iii) |
| `name_match_score`, `key_match_score`, `value_match_score` | (iii) |
| `parallel_score`, `pred_tool_count` | (iv) |
| `response_quality_score` | all |
| `process_score`, `summary_score` | (i), (iii) |

## Knobs

| variable | default | what it does |
| :--- | :--- | :--- |
| `MODEL_PATH` | `./outputs/sft_split/qwen2_5_7b_toucan_toolcall` | SFT init; change for the other cross-scale ablation rows |
| `TRAIN_DATA` / `VAL_DATA` | `./data/toucan_toolcall_rl.parquet`, `./data/toucan_eval_4k_unified.parquet` | verl-native parquets |
| `REWARD_FN_PATH` | `ablations/reward_fn_hierr_v82.py` | see the section above before changing |
| `TOOL_CONFIG_PATH` | (ii) only: `ablations/config/toucan_tool_config_no_schema.yaml` | which simulator class is used |
| `SLCA_WEIGHT_*` | per ablation | reward-weight overrides |
| `NNODES` / `TOTAL_EPOCHS` / `SAVE_FREQ` | `4` / `5` / `10` | cluster shape and cadence |
| `RESUME_MODE` | `auto` | set `disable` to force a cold start |

## Troubleshooting

- **`Unknown advantage estimator 'slca_grpo'`**: you are running stock verl.
  Install the vendored copy (`cd verl && pip install -e . --no-deps`).
- **(ii) fails with `No module named 'tools'`**: `train_ablation_wo_sgls.sh`
  puts `ablations/` on `PYTHONPATH` so that
  `tools.toucan_vllm_tool_no_schema` resolves. If you invoke the trainer by
  hand, do the same.
- **(iii) with the judge disabled trains on an all-zero reward.** Every process
  weight is 0, so `response_quality` is the only signal left; the launcher
  warns if `LLM_JUDGE_ENABLED=false`.
