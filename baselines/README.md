# Baselines (ToolPO, RLTR)

Re-implementations of the two external RL baselines the paper uses as reference
points, adapted to Toucan tool-calling data and run through the same SGLS
environment, the same data partitions and the same compute budget as SLCA-GRPO.
They stand for the other two credit-assignment paradigms — ToolPO for additive
augmentation, RLTR for pipeline separation. Unlike the controlled
GRPO ↔ SLCA-GRPO isolation comparison, both **retain their method-specific
initialisation and reward protocol**: RLTR starts from a planner-only SFT
checkpoint (Step 1 below), and ToolPO keeps its own additive estimator and
outcome-reward protocol.

**Backs:** the `RLTR` and `ToolPO` rows of the main Toucan-Test table
(`tab:toucan_main`) and of the full per-benchmark appendix tables
(`tab:toucan_full`, `tab:bfcl_full`, `tab:tau2_full`). Those rows use the
**LLM-judge outcome reward**; the paper also reports a reference-rule condition
as a protocol-sensitivity diagnostic, and the two are not interchangeable — see
"The paper's two ToolPO outcome rewards" below.

> Table **numbers** shift between paper revisions, so this README refers to
> paper tables by their LaTeX `\label` and by descriptive name instead. Search
> the paper source for the label to find the table.

```
baselines/
├── toolpo/
│   ├── reward_fn_toolpo.py             # R_succ + R_action (R_action normalised by |gold|)
│   ├── reward_fn_toolpo_sum_action.py  # variant: R_action = sum(C), unnormalised (see below)
│   ├── train_toolpo_qwen2.5-3B.sh
│   ├── train_toolpo_qwen2.5-7B.sh
│   └── train_toolpo_qwen3-8B-Base.sh
└── rltr/
    ├── reward_fn_rltr.py               # R_comp + R_repeat + R_error, planner only
    ├── prepare_planner_sft_data.py     # builds the planner-only SFT set
    ├── sft_configs/
    │   ├── planner_sft_qwen2.5-3B.yaml
    │   ├── planner_sft_qwen2.5-7B.yaml
    │   ├── planner_sft_qwen3-8B-Base.yaml
    │   └── train_planner_sft_distributed.sh   # multi-node launcher for the 8B planner
    ├── train_rltr_2stage_qwen2.5-3B.sh
    ├── train_rltr_2stage_qwen2.5-7B.sh
    └── train_rltr_2stage_qwen3-8B-Base.sh
```

Both baselines need advantage estimators that live in the vendored verl:
`toolpo_grpo` and `rltr_planner`. See `verl/SLCA_PATCHES.md` for what they do.

## Prerequisites

Same as the RL stage (`rl/README.md`): SFT checkpoint, RL / eval parquets under
`data/`, the vendored `verl/` installed, and the SGLS server running. ToolPO
additionally needs the LLM judge; RLTR instead needs a completeness-checker
endpoint (any OpenAI-compatible chat endpoint will do).

## ToolPO

### The paper's two ToolPO outcome rewards

ToolPO appears under two conditions that differ **only** in the outcome reward
`R_succ`. They are not interchangeable, and a ToolPO number is meaningless until
you say which one produced it.

| condition | outcome reward `R_succ` | where it appears | 7B values (Toucan Succ. / BFCL Acc. / τ²-Bench Pass¹) |
| :--- | :--- | :--- | :--- |
| **LLM judge** | LLM judge, Solved / Unsolved | the `ToolPO` rows of the main and full per-benchmark tables, at **all three scales** | `20.00±1.36` / `24.59±0.87` / `30.44±2.15` |
| **Reference rule** | a rule over the available gold tool calls (judge removed) | the outcome-reward protocol comparison, `tab:toolpo_reward_protocol` — **7B only** | `77.35±1.19` / `68.86±0.43` / `34.18±2.42` |

How to read these:

- **The `20.00` Toucan figure is a property of the LLM-judge condition at 7B, not
  a general statement about ToolPO.** The reference-rule condition holds the
  ToolPO estimator, initialisation, data, training budget and evaluation fixed
  and changes only the outcome reward; that alone takes the same comparison to
  `77.35±1.19` on Toucan and removes the observed 7B collapse. What the low
  number tracks is the outcome-reward protocol, not the method as such. Under the
  reference rule ToolPO lands close to unified GRPO (`76.60` / `68.41` / `31.87`)
  and still below SLCA-GRPO (`79.13` / `69.77` / `41.02`) on τ²-Bench.
- **No reference-rule runs exist at 3B or 8B.** The 3B and 8B ToolPO rows in the
  scale tables are all LLM-judge. Do not carry the 7B reference-rule number
  across scales, and do not subtract it from an LLM-judge row.
- **7B format behaviour.** Under the LLM-judge condition the paper reports low
  format validity for ToolPO on Qwen2.5-7B-Instruct (`format_passed = 38%`),
  while the 3B and 8B runs retain higher format validity. The accompanying
  training-trace analysis (unmatched opening tags after step 80 at 7B, silent
  process-score plateau at 3B) comes from representative **single-run** traces,
  and the paper adds that "these observations are specific to the LLM-judge
  diagnostic and do not imply that ToolPO always fails."

**What this directory implements.** All three launchers here run the **LLM-judge**
condition: `R_succ` comes from the Solved / Unsolved judge in
`reward_fn_toolpo.py`, and `S_pref = 0` because Toucan has no memory-fold action
(the variant file states it as `TOOLPO_MEMORY_FOLD_ENABLED = False`). The
reference-rule outcome reward is not shipped as a launcher, so nothing here
reproduces the `tab:toolpo_reward_protocol` reference-rule row.

### Reward decomposition (LLM-judge condition)

The reward implements the ToolPO paper's outcome + action decomposition:

```
A(y_i)   = A_succ + M(y_i) * A_action          (ToolPO Eq. 7)
A_succ   = R_succ   - mean(R_succ)             (Eq. 5)  <- the `score` field
A_action = R_action - mean(R_action)           (Eq. 6)  <- `toolpo_correct_ratio`
```

`R_succ` is binary. ToolPO scores it by rule-matching the final answer against a
ground-truth answer; Toucan carries no ground-truth answer, only
`gold_tool_calls`, so we substitute the Solved / Unsolved judge from the
DeepAgent repository (prompt copied verbatim). `C(a_t^call)` is binary tool
correctness under a strict match, assigned greedily one-to-one against the gold
calls, and `R_action = sum(C) / |gold_flat|`. `S_pref = 0`, because Toucan has
no memory-fold action.

```bash
MODEL_PATH=./outputs/sft_split/qwen2_5_7b_toucan_toolcall \
OUTPUT_DIR=./outputs/rl/qwen2_5_7b_toolpo \
NNODES=4 \
bash baselines/toolpo/train_toolpo_qwen2.5-7B.sh
```

### The two ToolPO reward files

They are **not** cosmetic variants of each other, so both ship:

| file | `R_action` | exported as | used by |
| :--- | :--- | :--- | :--- |
| `reward_fn_toolpo.py` | `sum(C) / \|gold_flat\|`, in [0, 1] | `toolpo_correct_ratio` | all three launchers here; the paper's LLM-judge ToolPO rows |
| `reward_fn_toolpo_sum_action.py` | `sum(C)`, unnormalised | `toolpo_action_reward` | nothing by default |

`reward_fn_toolpo_sum_action.py` is a later, more heavily instrumented rewrite:
it adds a per-process concurrency semaphore, an explicit judge prompt/context
token budget, frozen formula- and parser-identifier constants, and
structured-output constraints on the outcome judgement. It changes the reward
definition, which is why it is kept rather than merged. To use it:

```bash
REWARD_FN_PATH=baselines/toolpo/reward_fn_toolpo_sum_action.py \
bash baselines/toolpo/train_toolpo_qwen2.5-7B.sh
```

The upstream repository also carried per-model copies of the reward
(`_3B`, `_7B`). They differed only in a hardcoded judge URL, so they are
consolidated into the single `reward_fn_toolpo.py` above; point
`LLM_JUDGE_BASE_URL` wherever your judge lives.

## RLTR (2-stage)

RLTR splits tool use across two models: an RL-trained **planner** that emits the
tool trajectory, and a **frozen SFT summariser** that writes the final answer.
Only the planner is trained here, so there is no LLM judge and no summary
reward; the `rltr_planner` estimator forces the summary-token advantage to zero.

Reward: `R_comp + R_repeat + R_error`, with a hard `-1` for malformed output
(RLTR Eq. 3). `R_comp` follows the paper's Algorithm 1 —
`R_comp = (1/N) * sum_j gamma_j(tau)` over `N` samples of a verification LLM.

**Scope of the comparison.** RLTR retains its own planner-only initialisation and
completeness reward, so its rows are a reference point rather than a controlled
contrast. On τ²-Bench the paper reports RLTR trailing SLCA-GRPO at 7B
(`0.3372` vs `0.4102`) and 3B (`0.2850` vs `0.3563`), attributing the gap to the
frozen summariser in tasks that need a full conversational loop. The penalty
coefficients used here (`RLTR_LAMBDA_REPEAT=0.1`, `RLTR_MU_ERROR=0.2`) are set
to reasonable values because the original RLTR paper does not specify them, so
they are a choice of this re-implementation rather than a published setting.

### Step 1 — build the planner-only SFT set

The planner is initialised from an SFT checkpoint trained on trajectories with
the summary turn removed and an `<answer>` terminator appended:

```bash
python baselines/rltr/prepare_planner_sft_data.py \
    --input  ./data/toucan_toolcall_sft.json \
    --output ./data/toucan_planner_sft.json
```

The two large planner SFT JSONs (~534 MB each, plain and `<thinking>`-tagged)
are **not** committed; regenerate them with the command above. Register the
result in LLaMA-Factory's `data/dataset_info.json` as `toucan_planner_sft`
(and `toucan_planner_sft_thinking` for the Qwen3-Base config).

### Step 2 — planner SFT

```bash
bash sft/train_sft.sh baselines/rltr/sft_configs/planner_sft_qwen2.5-7B.yaml
```

The 8B-Base planner was trained on 4 nodes. Run this on each node with a
different `NODE_RANK`:

```bash
MASTER_ADDR=<rank-0 host> NODE_RANK=<0..3> NNODES=4 \
LLAMAFACTORY_DIR=/path/to/LLaMA-Factory \
bash baselines/rltr/sft_configs/train_planner_sft_distributed.sh
```

### Step 3 — planner RL

```bash
MODEL_PATH=./outputs/planner_sft/qwen2_5_7b_planner \
OUTPUT_DIR=./outputs/rl/qwen2_5_7b_rltr_2stage \
RLTR_COMP_CHECKER_BASE_URL=http://127.0.0.1:8016/v1 \
RLTR_COMP_CHECKER_MODEL=Qwen3-30B-A3B \
NNODES=4 \
bash baselines/rltr/train_rltr_2stage_qwen2.5-7B.sh
```

The upstream `_7B` / `_8B` reward copies differed only in a hardcoded checker
URL and are consolidated into `reward_fn_rltr.py`.

## Knobs

| variable | default | what it does |
| :--- | :--- | :--- |
| `MODEL_PATH` | `./outputs/...` per backbone | SFT (ToolPO) or planner-SFT (RLTR) init |
| `TRAIN_DATA` / `VAL_DATA` | `./data/toucan_toolcall_rl.parquet`, `./data/toucan_eval_4k_unified.parquet` | verl-native parquets |
| `REWARD_FN_PATH` | the reward next to the launcher | swap in the ToolPO variant |
| `LLM_JUDGE_BASE_URL` / `LLM_JUDGE_MODEL` | `http://127.0.0.1:8016/v1`, `gpt-oss-120b` | ToolPO outcome judge |
| `RLTR_COMP_CHECKER_BASE_URL` / `_MODEL` / `_N` | `http://127.0.0.1:8016/v1`, `Qwen3-30B-A3B`, `3` | RLTR completeness checker |
| `RLTR_LAMBDA_REPEAT` / `RLTR_MU_ERROR` | `0.1` / `0.2` | penalty coefficients (the paper does not fix these) |
| `NNODES` / `N_GPUS_PER_NODE` | `4` / `8` | cluster shape |

## Troubleshooting

- **`Unknown advantage estimator 'toolpo_grpo'`**: you are running stock verl.
  Install the vendored copy (`cd verl && pip install -e . --no-deps`).
- **ToolPO scores drop to 0**: check the outcome judge is reachable. A dead
  judge makes `R_succ = JUDGE_FAILURE_SCORE` for every sample, which zeroes
  `A_succ` group-wide.
- **`No module named 'tools'`**: the launchers put `rl/slca_grpo` on
  `PYTHONPATH` for you; if you invoke the trainer directly, do the same.
- **Qwen3-8B-Base needs `<think>`-tagged parquets** (`*_thinking.parquet`),
  which are not part of the released dataset. Regenerate them from the source
  JSON with `data/convert_toucan_to_verl_v2.py`, or override
  `TRAIN_DATA` / `VAL_DATA` to point at the non-thinking parquets.
