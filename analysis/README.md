# Analysis experiments

Three diagnostic studies that are not part of the main pipeline. Each is a
matched pair or sweep run on top of the standard SLCA-GRPO setup, changing one
thing at a time.

```
analysis/
├── README.md                          # (this file)
├── hierr_sweep/                       # HierR weight sensitivity      -> tab:hierr_sensitivity
│   ├── reward_fn_hierr_v82.py
│   ├── train_hierr_default.sh
│   ├── train_hierr_uniform.sh
│   ├── train_hierr_name_heavy.sh      #   (extra sweep point, no paper row)
│   └── train_hierr_value_heavy.sh
├── grad_cosine/                       # cos(grad_tool, grad_summary)  -> App. B, "Empirical
│   ├── train_grad_cosine_slca.sh      #   Analysis of Optimization Stability"
│   └── train_grad_cosine_grpo.sh
└── execution_reward/                  # execution-based reward        -> tab:exec_reward_main
    ├── reward_fn_exec_based.py
    ├── train_exec_slca.sh
    └── train_exec_grpo.sh
```

All of these need the same prerequisites as the RL stage (`rl/README.md`): an
SFT checkpoint, the RL / eval parquets, the vendored `verl/`, the SGLS server on
port 8003 and the LLM judge on port 8016.

> Table **numbers** shift between paper revisions, so this README refers to
> paper tables by their LaTeX `\label` (e.g. `tab:hierr_sensitivity`) and by
> descriptive name instead. Search the source for the label to find the table.

---

## 1. `hierr_sweep/` — HierR weight sensitivity (`tab:hierr_sensitivity`)

Four HierR weightings on Qwen2.5-7B-Instruct, everything else fixed. The paper
reports three of them as a three-run mean±std sensitivity check and reads it
narrowly: the Default configuration has the highest values among the tested
settings, and the Uniform configuration stays above the matched GRPO baseline
(+0.77 pp on BFCL, +6.75 pp on τ²-Bench). That is the whole claim — a
sensitivity check over a small set of tested weightings, not evidence about
whether the gain is structural or a product of weight tuning.

| configuration | format | name | key | value | parallel | row in the paper? |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Default (paper) | 0.10 | 0.25 | 0.15 | 0.20 | 0.30 | yes |
| Uniform | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | yes |
| Name-heavy | 0.05 | 0.40 | 0.15 | 0.15 | 0.25 | **no** |
| Value-heavy | 0.05 | 0.20 | 0.20 | 0.35 | 0.20 | yes |

`train_hierr_name_heavy.sh` is an **additional sweep point that the paper does
not report**: the HierR weight-sensitivity table lists only Default, Uniform and
Value-heavy (plus w/o HierR and GRPO reference rows). It ships because it is a
real, runnable configuration, but no number in the paper is backed by it.

```bash
MODEL_PATH=./outputs/sft_split/qwen2_5_7b_toucan_toolcall NNODES=4 \
  bash analysis/hierr_sweep/train_hierr_default.sh
# ... likewise for _uniform, _name_heavy, _value_heavy
```

Each launcher `export`s its own weights before starting the trainer, so all four
share one reward implementation (`reward_fn_hierr_v82.py`). Upstream these were
four near-identical copies of the same 1,086-line file differing only in default
constants that the launchers already overrode; they are consolidated here. You
can also sweep without touching any file:

```bash
SLCA_WEIGHT_FORMAT=0.05 SLCA_WEIGHT_NAME=0.40 SLCA_WEIGHT_KEY=0.15 \
SLCA_WEIGHT_VALUE=0.15 SLCA_WEIGHT_PARALLEL=0.25 \
  bash analysis/hierr_sweep/train_hierr_default.sh
```

These runs pin `data.seed=43`. It is a reproducibility parameter and nothing
more: the paper reports mean±std over three runs and deliberately does not
identify individual runs, so this value must not be read as naming a particular
run of that set.

---

## 2. `grad_cosine/` — gradient-conflict diagnostic

Backs the measurement in **Appendix B, "Empirical Analysis of Optimization
Stability"**: `cos(grad_tool, grad_summary)` stays near zero — in
`[-0.03, 0.08]` — throughout training for *both* SLCA and standard GRPO on the
3B backbone. The paper's reading of this is narrow: in this diagnostic,
advantage magnitudes, rather than gradient direction, are the more visible
source of cross-segment mismatch. It is one diagnostic on one backbone, not a
mechanism claim about how SLCA works.

```bash
MODEL_PATH=./outputs/sft_split/qwen2_5_3b_toucan_toolcall NNODES=4 \
  bash analysis/grad_cosine/train_grad_cosine_slca.sh   # adv_estimator=slca_grpo
  bash analysis/grad_cosine/train_grad_cosine_grpo.sh   # adv_estimator=grpo
```

The pair is identical except for `algorithm.adv_estimator`. Both set
`+actor_rollout_ref.actor.compute_grad_cosine=true`, which is the opt-in switch
patched into `verl/workers/actor/dp_actor.py`; it runs two extra masked
forward+backward passes on the first micro-batch of each step and logs
`actor/grad_cosine_sim`, `actor/grad_norm_tool` and `actor/grad_norm_sum` to
Weights & Biases.

> **This roughly triples actor step cost.** It is a diagnostic, off by default
> everywhere else. Do not leave it on for a production run.

The gradient-norm standard deviations in the appendix's gradient-stability table
(`tab:grad_std`, same appendix section) come from `actor/grad_norm` on the
ordinary main-method and GRPO runs, not from these two. The reductions reported
there are 25.0% (3B), 13.1% (7B) and 2.5% (8B), from representative single-run
traces.

These runs pin `data.seed=44` — again a reproducibility parameter only, not an
index into the paper's three-run set. Both launchers read the reward from
`analysis/hierr_sweep/reward_fn_hierr_v82.py`
(upstream shipped a byte-identical copy in this directory); override
`REWARD_FN_PATH` if you want a different one.

---

## 3. `execution_reward/` — execution-based reward (`tab:exec_reward_main`)

Control for the objection "SLCA only helps because the reward is gold-matching".
It swaps the process term for an execution-outcome term and re-runs the SLCA vs
GRPO contrast:

```
R_tool = R_succ      (DeepAgent Solved / Unsolved LLM judge, binary 0/1)
R_sum  = S_summary   (Toucan LLM judge, 5-point scale, unchanged)
score  = w_process * R_succ + w_respq * S_summary
```

Under SLCA the two terms are routed separately — `process_score = R_succ` drives
`A^tool`, `summary_score = S_summary` drives `A^sum`. Under plain GRPO only
`score` is read and a single advantage is broadcast to every token.

```bash
MODEL_PATH=./outputs/sft_split/qwen2_5_7b_toucan_toolcall NNODES=4 \
  bash analysis/execution_reward/train_exec_slca.sh   # adv_estimator=slca_grpo
  bash analysis/execution_reward/train_exec_grpo.sh   # adv_estimator=grpo
```

Both arms use the same `reward_fn_exec_based.py` (upstream shipped two copies
differing only in a hardcoded judge URL); the only difference between the two
launchers is the advantage estimator. Gold-matching metrics are still computed
and logged for comparison but never enter the RL objective.

> **What the paper's execution-reward table reports.** Entries are mean ±
> standard deviation over three runs, with data, initialization, SGLS and
> compute matched across the two arms. The SLCA advantage persists under the
> execution-based reward: on Qwen2.5-7B-Instruct, `76.43 ± 1.21` vs
> `73.51 ± 1.45` Toucan Success, `67.61 ± 0.51` vs `66.28 ± 0.32` BFCL Acc, and
> `30.18 ± 1.76` vs `25.54 ± 2.13` τ²-Bench Pass¹ (SLCA-GRPO vs SFT+GRPO).
> Absolute scores are lower than under the dense HierR reward because the
> execution signal is sparser. The only difference between the two launchers
> here is `algorithm.adv_estimator`, which is exactly the contrast that table
> isolates.

---

## Common knobs

| variable | default | what it does |
| :--- | :--- | :--- |
| `MODEL_PATH` | 7B (3B for `grad_cosine`) SFT checkpoint | training init |
| `TRAIN_DATA` / `VAL_DATA` | `./data/toucan_toolcall_rl.parquet`, `./data/toucan_eval_4k_unified.parquet` | verl-native parquets |
| `REWARD_FN_PATH` | per experiment, see above | reward implementation |
| `SLCA_WEIGHT_*` | Default weighting | HierR weights (the sweep axis) |
| `LLM_JUDGE_BASE_URL` / `LLM_JUDGE_MODEL` | `http://127.0.0.1:8016/v1`, `gpt-oss-120b` | summary judge |
| `NNODES` / `TOTAL_EPOCHS` / `SAVE_FREQ` | `4` / `5` / `10` | cluster shape and cadence |

## Troubleshooting

- **`Unknown advantage estimator 'slca_grpo'`**: you are running stock verl.
  Install the vendored copy (`cd verl && pip install -e . --no-deps`).
- **`actor/grad_cosine_sim` never appears in W&B**: `compute_grad_cosine` is a
  dataclass field on the actor config (`verl/workers/config/actor.py`). If that
  patch is missing, the `getattr(..., False)` guards are permanently false and
  the diagnostic silently no-ops.
- **Sweep runs all report the same weights**: the launcher `export`s take effect
  only if they precede the trainer process. Check the banner each script prints
  before training starts — it echoes the weights actually in the environment.
