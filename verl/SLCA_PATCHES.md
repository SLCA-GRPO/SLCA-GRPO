# verl Patches for SLCA-GRPO

This directory is a **vendored copy of [verl](https://github.com/volcengine/verl)**
with the SLCA-GRPO modifications applied in place. It is not a patch series or a
submodule: the edits are baked into the source files, so `git log` on this
directory will not show them.

**Upstream base:** commit
[`7ca9d8a069c6d966d3d87d2a5ee4b02654fafd0a`](https://github.com/volcengine/verl/commit/7ca9d8a069c6d966d3d87d2a5ee4b02654fafd0a)
(2025-12-08), which reports itself as version `0.7.0.dev` in `verl/version/version`.
Note this is a development snapshot between the `v0.6.0` and `v0.7.0` tags — diffing
against either release tag will show hundreds of unrelated files. Pin the commit above.

Relative to that commit, this tree contains:

| Category | Count | Detail |
| :--- | ---: | :--- |
| Modified Python sources | **15** | Listed exhaustively in the two tables below. |
| Modified other files | 1 | `.gitignore` — appends `wandb` and `verl.egg-info`. Not functional. |
| Added files | 2 | This document, and `tests/trainer/ppo/test_slca_grpo_on_cpu.py` (the SLCA estimator unit tests). |
| Files absent vs upstream | 6 | `.gemini/config.yaml`, `.vscode/settings.json`, `recipe/vla/workers/env/{env_loop_wg_test,env_manager,env_worker}.py`, `tests/utils/ckpt/test_esi_save_ckpt_on_cpu.py`. None are used by SLCA-GRPO. |
| Byte-identical to upstream | 1140 | Everything not named above. |

> **Correction.** Earlier revisions of this document listed only **5** patched
> files and asserted *"everything else in this directory is unmodified upstream
> verl."* **That assertion was false** — 10 further source files carry
> SLCA-specific edits. If you re-applied this patch set using the old list,
> re-check the "Infrastructure patches" table below, especially the two
> rollout/tool-schema entries, which fail *silently* rather than loudly.

## Regenerating the exact diff

```bash
git clone https://github.com/volcengine/verl /tmp/verl_up
git -C /tmp/verl_up checkout 7ca9d8a069c6d966d3d87d2a5ee4b02654fafd0a
diff -r --brief /tmp/verl_up /path/to/this/verl | grep -v '^Only in /tmp/verl_up/\.git'
```

An ownership-based shortcut also works on the shipped archive, because the tree
was unpacked as a non-root user and then edited as root (run from the release
repository root, i.e. the parent of this directory):

```bash
find verl -type f -user root | sort
```

Be aware this heuristic is *over*-inclusive: it also reports
`verl/experimental/agent_loop/agent_loop.py`, which is **byte-identical to
upstream** (it was touched but not changed), plus this document itself. The
tables below are the authoritative record.

---

## Core algorithm patches

These five files **are** the method. Anything re-implementing SLCA-GRPO on a
different verl release must port all of them; without them
`algorithm.adv_estimator=slca_grpo` does not exist and the paper's numbers
cannot be reproduced.

| File | What SLCA-GRPO adds | Paper component |
| :--- | :--- | :--- |
| `verl/trainer/ppo/core_algos.py` | Adds `SLCA_GRPO`, `TOOLPO_GRPO` and `RLTR_PLANNER` to the `AdvantageEstimator` enum. Defines the segment machinery — `_extract_segments_from_mask_1d`, `_build_two_segment_masks` (process = all learnable spans but the last, summary = the last span), `_groupwise_normalize_with_present` (group z-score restricted to samples where the segment exists, zeroing any group with `count <= 1`) — and the estimator `compute_slca_grpo_advantage`, which folds per-segment KL, applies the negative-score / `should_no_call` penalty guard, normalizes each segment independently, then applies `w_process` / `w_respq` **after** normalization and returns `(advantages, returns, segment_metrics)`. | The SLCA estimator itself (Method §3); `compute_toolpo_grpo_advantage` and `compute_rltr_planner_advantage` are the ToolPO and RLTR baselines in the comparison table. |
| `verl/trainer/ppo/ray_trainer.py` | Adds `elif` branches in `compute_advantage` for the three new estimators. For `slca_grpo` it assembles `reward_components` from `non_tensor_batch` (`reward/tool_call/process_score`, `reward/tool_call/summary_score`, `score`, `reward/tool_call/should_call_but_no_call`), reads the segment weights from the `SLCA_WEIGHT_PROCESS` / `SLCA_WEIGHT_RESPQ` environment variables, unpacks the 3-tuple return, and writes `adv_process` / `adv_summary` back into `data.batch`. | Wires the estimator to the HierR reward in `rl/slca_grpo/reward_fn.py`; the two env vars are the λ_tool / λ_sum sweep. |
| `verl/trainer/ppo/metric_utils.py` | Emits `critic/advantages/{process,summary}_{mean,std,max,min}` from the `adv_process` / `adv_summary` tensors, plus a whitelist of ~24 `reward/tool_call/*` component keys (format / name / key / value / parallel match, process & summary scores, `success@{0.7..1.0}`, `tool_success@{0.7..1.0}`) logged as mean/max/min. | Produces the per-segment advantage and reward-component curves in the training-dynamics figures. |
| `verl/workers/actor/dp_actor.py` | Adds an opt-in `compute_grad_cosine` block (guarded by `getattr(self.config, 'compute_grad_cosine', False)`, first micro-batch only). It runs two extra independent forward+backward passes — one masked to tool tokens, one to summary tokens — then all-reduces the reduce-scattered gradient shards to log `actor/grad_cosine_sim`, `actor/grad_norm_tool`, `actor/grad_norm_sum`. | The gradient-conflict diagnostic: the cos(∇_tool, ∇_summary) measurement motivating segment-locked credit. **Diagnostic only — off by default; it roughly triples actor step cost when on.** |
| `verl/workers/utils/losses.py` | Under the same `compute_grad_cosine` guard, computes per-segment policy losses over `process_mask` / `summary_mask` while retaining the autograd graph, so `dp_actor.py` can differentiate each segment separately. Wrapped in a bare `except: pass` so a mask failure degrades to normal training rather than crashing. | Supplies the segment-level loss tensors the gradient-cosine diagnostic differentiates. |

### Config surface these depend on

`verl/workers/config/actor.py` (see infrastructure table) must carry
`compute_grad_cosine: bool = False`, otherwise the `getattr(...)` guards in
`dp_actor.py` and `losses.py` are permanently false and the diagnostic silently
never runs.

---

## Infrastructure patches

These ten files are **not** part of the credit-assignment method. They exist to
make the method runnable in our environment: heterogeneous GPU clusters, an
external LLM tool simulator standing in for thousands of real APIs, and
CPU/NPU backend compatibility. Someone reproducing SLCA-GRPO on a homogeneous
cluster with real tools can skip most of them — **except the two rollout
entries flagged REQUIRED**, which the shipped tool config depends on.

| File | What changed | Why it exists / who depends on it |
| :--- | :--- | :--- |
| `verl/workers/config/rollout.py` | **REQUIRED.** Adds two fields to `MultiTurnConfig`: `disable_tool_schema_injection: bool = False` and `wildcard_tool_name: Optional[str] = None`. | Declares the config surface consumed by `tool_agent_loop.py`. See the callout below — omitting this is the single most dangerous thing to miss. |
| `verl/experimental/agent_loop/tool_agent_loop.py` | **REQUIRED.** Five regions. (1) Reads the two fields above off `config.actor_rollout_ref.rollout.multi_turn`, raising `ValueError` if `wildcard_tool_name` names an unregistered tool. (2) When `disable_tool_schema_injection` is set, passes `tools=None` to `apply_chat_template`, so tool definitions come from the dataset prompt instead of the chat template. (3) Appends the parser's `</tool_call>` token to the sampling `stop` list, preventing the model from hallucinating its own tool results. (4) For the `hermes` parser, prepends `<\|im_end\|>\n` to tool-response token ids to match the SFT tokenization. (5) Rewrites `_call_tool` to route *every* emitted tool name to `wildcard_tool_name`, forwarding `original_tool_name=` and keying `tools_kwargs` by the original name. | The wildcard router is what lets one SGLS simulator endpoint stand in for arbitrary Toucan APIs. Item (4) has real training semantics — it aligns RL rollout tokenization with the split-SFT checkpoint. Consumer: `rl/slca_grpo/config/tool_config/toucan_tool_config.yaml`. |
| `verl/experimental/agent_loop/tool_parser.py` | Adds `HermesToolParser._parse_single_tool_call`, which accepts either `"arguments"` or `"parameters"` as the argument key and coerces dict/str payloads to a JSON string, returning `None` on malformed input. `extract_tool_calls` now handles a JSON **list** of calls as well as a single dict. | Robustness against the xLAM/Toucan tool-call formats in our RL data; upstream indexed `["name"]`/`["arguments"]` directly and raised on anything else. |
| `verl/utils/dataset/rl_dataset.py` | When `tools_kwargs` is empty, synthesizes it from `reward_model.ground_truth.tool_schemas` (JSON string tolerated, OpenAI-style `{"function": {...}}` envelope unwrapped), producing `{"create_kwargs": {"tool_definition": tool}}` per schema. Also fixes an upstream logging bug where brace placeholders were passed to stdlib `logging` (`{}` → `%s`). | Feeds each sample's own tool schemas to the wildcard simulator at rollout time, since the schemas are per-example rather than global. |
| `verl/trainer/main_ppo.py` | Reads the `GPUS_PER_NODE_LIST` environment variable in `TaskRunner.init_resource_pool_mgr`; when set, splits it on commas into the Ray `resource_pool_spec` instead of `[n_gpus_per_node] * nnodes`. Falls back to upstream behaviour when unset. | Heterogeneous-cluster support, e.g. `GPUS_PER_NODE_LIST="4,4,6,6,6,6"` for two 4-GPU nodes plus four 6-GPU nodes. Purely environment-specific; irrelevant on a uniform cluster. |
| `verl/utils/device.py` | Adds `get_distributed_backend()`, returning `"gloo"` on CPU and `f"cpu:gloo,{device}:{nccl_backend}"` otherwise. | Upstream unconditionally built the composite string, which raises a duplicate-device-type error in CPU-only mode. |
| `verl/utils/distributed.py` | Two lines: imports `get_distributed_backend` and calls it instead of inlining the composite backend string. | Call-site adoption of the fix above. |
| `verl/workers/fsdp_workers.py` | Two unrelated changes. (a) Same `get_distributed_backend` adoption. (b) Adds an `lr_override` hook in `update_actor`: if `data.meta_info["lr_override"]` is present, temporarily overwrites every optimizer param-group LR, runs the update, restores the originals, and logs `actor/lr` plus `actor/lr_base`. | (a) is the CPU/NPU compat fix. (b) is **vestigial** — nothing in this repository ever writes `lr_override`, so the branch is never taken. It is documented here for completeness, not because it is used. |
| `verl/workers/config/actor.py` | One line: `compute_grad_cosine: bool = False`. | Master switch for the gradient-conflict diagnostic. Strictly speaking this is core-algorithm config, but it is a single dataclass field with an infrastructure-shaped footprint. **If you omit it, the `getattr(..., False)` guards in `dp_actor.py` and `losses.py` are permanently false and the diagnostic silently never runs.** |
| `verl/workers/engine/fsdp/transformer_impl.py` | A second, independent implementation of the gradient-cosine diagnostic for the model-engine path: pops the non-detached `_gc_tool_loss` / `_gc_sum_loss` tensors out of `meta_info["metrics"]`, runs `torch.autograd.grad(..., retain_graph=True, allow_unused=True)` on each, all-reduces over the DP group, and logs `grad_cosine_sim` / `grad_norm_tool` / `grad_norm_sum`. | Same diagnostic as `dp_actor.py`, different execution path. **Caveat:** this variant relies on `retain_graph=True` across two `autograd.grad` calls under FSDP, whereas `dp_actor.py` explicitly documents that FSDP v1 does *not* support that (PyTorch #106637) and instead performs two independent forward+backward passes. The two paths disagree, and this one swallows the resulting exception via a bare `except Exception: pass`. Prefer the `dp_actor.py` path; treat this one as unvalidated. |

---

## Critical: the two rollout fields are load-bearing

`rl/slca_grpo/config/tool_config/toucan_tool_config.yaml` registers exactly one
tool, `toucan_tool`, whose schema is a generic `{"query": string}` envelope. It
is **not** a description of any real API. The config only works because the
trainer config `rl/slca_grpo/config/toucan_grpo.yaml` sets:

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      disable_tool_schema_injection: true
      wildcard_tool_name: toucan_tool
```

Together these two flags mean: *do not inject the registered tool schema into
the chat template* (the real per-example schemas arrive in the prompt text via
the `rl_dataset.py` patch), and *route every tool name the model emits to
`toucan_tool`* regardless of what it was called.

**Re-applying this patch set to a newer verl without these two fields fails
silently, not loudly.** The `getattr(config, ..., default)` reads in
`tool_agent_loop.py` return the defaults, so:

- `disable_tool_schema_injection` defaults to `False` → the generic
  `{"query": string}` envelope gets injected into every prompt, overriding the
  real per-example schemas. The model is trained against the wrong tool surface.
- `wildcard_tool_name` defaults to `None` → every tool call the model emits
  under its real name (`get_weather`, `search_flights`, ...) misses the registry
  and is scored as a failed call.

Neither raises. Training proceeds, loss decreases, and the tool-call reward
simply sits near zero — which is easy to misdiagnose as a reward-shaping or
learning-rate problem. If you port this patch set, verify these two flags first.

## Note: `verl/.github/workflows/` is inert

This directory carries 36 upstream CI workflow files (42 files total under
`verl/.github/`). **None of them run.** GitHub Actions only reads workflows from
`.github/workflows/` at the *repository root*, and this vendored verl sits one
level down. They are retained solely to keep the vendored tree byte-comparable
with upstream. Do not interpret their presence as CI coverage for this release,
and do not spend time fixing them.

---

## Installation

```bash
# from the repo root
cd verl
pip install -e . --no-deps
```

`--no-deps` is preferred so pip does not rewrite `torch` / `vllm`. Install the
heavy dependencies separately with `pip install -r ../requirements.txt`.

## Using the estimator

`rl/slca_grpo/train_slca_grpo.sh` already sets these:

```bash
algorithm.adv_estimator=slca_grpo \
algorithm.norm_adv_by_std_in_grpo=true \
actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
```

The two segment weights are read from the environment, not from Hydra:

```bash
export SLCA_WEIGHT_PROCESS=1.0   # lambda_tool
export SLCA_WEIGHT_RESPQ=1.0     # lambda_sum
```

Both default to `1.0`. They are applied **after** per-segment z-score
normalization — applying them before would cancel them out exactly, since
`(w*s - w*mu) / (w*sigma) == (s - mu) / sigma`. See `rl/README.md` for the full
knob list and `rl/slca_grpo/reward_fn.py` for the matching HierR reward shape.

Two additional estimators ship for the baseline comparisons in the paper:
`toolpo_grpo` (additive tool advantage: tool tokens get `A_global + lambda * A_tool`)
and `rltr_planner` (tool tokens only; summary advantage forced to zero).

## Tests

CPU-only unit tests for the estimator:

```bash
python -m pytest verl/tests/trainer/ppo/test_slca_grpo_on_cpu.py -q
```

They cover segment extraction and process/summary routing, per-segment
group-normalization, the `count <= 1` guard, KL folding, the negative-score and
`should_no_call` penalty guards, advantage leakage, and post-normalization
weighting. No GPU required.

## Behaviour when the patches are dormant

With `algorithm.adv_estimator` left at `grpo`, `compute_grad_cosine` at `False`,
`GPUS_PER_NODE_LIST` unset, and the two `multi_turn` flags at their defaults,
this tree behaves as upstream verl `7ca9d8a`. The `rl_dataset.py`,
`tool_parser.py` and `device.py` patches are always active, but each is a
strict widening of the accepted input — they add fallbacks rather than change
existing behaviour.
