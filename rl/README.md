# RL Stage (SLCA-GRPO)

This folder hosts the RL-stage code: the main launcher, the HierR reward
function, the Schema-Guided LLM Simulator (SGLS) tool definition, and the
vLLM launchers for the two auxiliary servers used during training.

```
rl/
├── slca_grpo/
│   ├── train_slca_grpo.sh          # main launcher (verl.trainer.main_ppo)
│   ├── reward_fn.py                # HierR reward (process + summary)
│   ├── config/
│   │   ├── toucan_grpo.yaml        # hydra base (data/model/rollout/trainer)
│   │   └── tool_config/
│   │       └── toucan_tool_config.yaml  # wildcard routing to SGLS
│   └── tools/
│       └── toucan_vllm_tool.py     # forwards <tool_call> to SGLS
├── sgls/
│   └── serve_sgls_qwen3_235b.sh    # vLLM server for the environment simulator
└── judge/
    └── serve_llm_judge_gpt_oss.sh  # vLLM server for the response-quality judge
```

## Prerequisites

1. SFT checkpoint available at `MODEL_PATH` (see `sft/README.md`).
2. RL / eval parquets available under `data/` (see `data/README.md`).
3. Local `verl/` installed in a Python env that has `vllm>=0.8`, `torch>=2.3`,
   `flash-attn`, `ray`, `hydra-core`, `aiohttp`, `transformers`.
   ```bash
   conda create -n slca_grpo python=3.10 -y
   conda activate slca_grpo
   cd verl && pip install -e . && cd -
   pip install -r requirements.txt
   ```

## Three processes, three GPUs groups

SLCA-GRPO trains with three concurrent vLLM groups:

| Role    | Script                                    | Default port | Why                                    |
| :------ | :---------------------------------------- | :----------- | :------------------------------------- |
| Actor   | `rl/slca_grpo/train_slca_grpo.sh`         | (none)       | The policy we update.                  |
| SGLS    | `rl/sgls/serve_sgls_qwen3_235b.sh`        | 8003         | Schema-guided tool-call simulator.     |
| Judge   | `rl/judge/serve_llm_judge_gpt_oss.sh`     | 8016         | Response-quality scorer (HierR summary). |

You can co-locate SGLS and Judge on the same 8-GPU node, or put the policy
actor on a separate node if you have enough GPUs.

## Step 1. Launch SGLS

```bash
bash rl/sgls/serve_sgls_qwen3_235b.sh
# -> Ready at http://0.0.0.0:8003/v1/chat/completions
```

Confirm the URL and `served-model-name` in
`rl/slca_grpo/config/tool_config/toucan_tool_config.yaml` match.

## Step 2. Launch the LLM judge

```bash
bash rl/judge/serve_llm_judge_gpt_oss.sh
# -> Ready at http://0.0.0.0:8016/v1/chat/completions
```

The reward function picks up the endpoint from `LLM_JUDGE_BASE_URL` (default
`http://127.0.0.1:8016/v1`).

## Step 3. Launch training

```bash
MODEL_PATH=./outputs/sft_split/qwen2_5_7b_toucan_toolcall \
TRAIN_DATA=./data/toucan_toolcall_rl.parquet \
VAL_DATA=./data/toucan_eval_4k_unified.parquet \
OUTPUT_DIR=./outputs/rl/qwen2_5_7b_slca_grpo \
NNODES=4 \
bash rl/slca_grpo/train_slca_grpo.sh
```

Key algorithmic knobs (exposed as CLI overrides in the launcher):

- `algorithm.adv_estimator=slca_grpo` — segment-locked credit assignment.
- `algorithm.norm_adv_by_std_in_grpo=true` — per-group std normalisation.
- `actor_rollout_ref.actor.policy_loss.loss_mode=vanilla` — standard PPO loss.
- `actor_rollout_ref.rollout.multi_turn.enable=true` — multi-turn rollout.
- `actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent` — use verl's
  ToolAgentLoop to interleave `<tool_call>` / `<tool_response>`.

HierR weights (process = format / name / key / value / parallel, summary =
response-quality from the judge) are set through `SLCA_WEIGHT_*` env vars;
defaults mirror the paper:

```
SLCA_WEIGHT_FORMAT=0.10
SLCA_WEIGHT_NAME=0.25
SLCA_WEIGHT_KEY=0.15
SLCA_WEIGHT_VALUE=0.20
SLCA_WEIGHT_PARALLEL=0.30
SLCA_WEIGHT_RESPQ=1.00
```

## Troubleshooting

- **`No module named 'tools'`**: make sure `PYTHONPATH` includes
  `rl/slca_grpo/` (the launcher sets this automatically).
- **Judge timeouts**: raise `LLM_JUDGE_TIMEOUT` (default 540s). If the judge is
  swamped, lower the RL rollout batch size or scale the judge's `MAX_NUM_SEQS`.
- **SGLS returns empty `<tool_response>`**: the simulator was truncated by
  `max_new_tokens=4096`; bump it in `toucan_tool_config.yaml`.
