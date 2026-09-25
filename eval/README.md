# Evaluation (BFCL-v3 + τ²-Bench)

Reproduces the two benchmarks reported in the main table of the paper:

- **BFCL-v3** (in-distribution): single-turn / multi-turn / live function-calling.
- **τ²-Bench** (out-of-distribution): multi-turn agent-user-environment dialogue
  across `airline`, `retail`, `telecom`.

Both benchmarks go through [`evalscope`](https://github.com/modelscope/evalscope),
but each one needs a local vLLM server to host the agent. τ²-Bench additionally
needs a *user simulator* endpoint.

```
eval/
├── bfcl_v3/
│   ├── start_vllm_server.sh                 # local vLLM for the agent
│   ├── toucan_xlam_tool_parser_plugin.py    # parses <tool_call> back to OpenAI tool_calls
│   └── bfcl_v3_eval.py                      # evalscope entry point
└── tau2_bench/
    ├── start_tau2_agent_vllm_server.sh      # local vLLM for the agent
    ├── start_tau2_user_vllm_server.sh       # (optional) local vLLM for the user simulator
    └── tau2_bench_eval.py                   # evalscope entry point
```

## Prerequisites

```bash
pip install "evalscope[tau2-bench]>=0.17"  pip install "vllm>=0.8"
# plus transformers/torch already installed from the top-level requirements.txt
```

The agent backbone must be available locally at the path you pass as
`MODEL_PATH`. Default value in the scripts is
`./outputs/rl/qwen2_5_7b_slca_grpo` (the checkpoint produced by
`rl/slca_grpo/train_slca_grpo.sh`). You can also evaluate a pre-SFT checkpoint
by pointing `MODEL_PATH` at `./outputs/sft_split/qwen2_5_7b_toucan_toolcall`.

## BFCL-v3

Two processes:

1. Agent vLLM server (in terminal A):

   ```bash
   MODEL_PATH=./outputs/rl/qwen2_5_7b_slca_grpo \
   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash eval/bfcl_v3/start_vllm_server.sh
   # -> http://127.0.0.1:8000/v1  (served name: toucan_toolcall_v4)
   ```

   The script auto-tunes `TP_SIZE / PP_SIZE` from the HF config
   (`num_attention_heads % TP == 0`, `num_hidden_layers % PP == 0`).

2. Evaluator (in terminal B):

   ```bash
   BFCL_VLLM_API_URL=http://127.0.0.1:8000/v1 \
   BFCL_VLLM_MODEL_NAME=toucan_toolcall_v4 \
   python eval/bfcl_v3/bfcl_v3_eval.py
   ```

   Results and per-sample traces are written to `./eval_results/bfcl_v3/toucan_v4_<ts>/`.

### What's included in the subset_list

`simple, multiple, parallel, parallel_multiple, java, javascript, live_simple,
live_multiple, live_parallel, live_parallel_multiple, multi_turn_base,
multi_turn_miss_func, multi_turn_miss_param`. Irrelevance / live-relevance /
live-irrelevance / `multi_turn_long_context` are disabled by default
(follow the commented block in `bfcl_v3_eval.py` to re-enable).

## τ²-Bench

Three processes if you self-host the user simulator, two if you use an external
OpenAI-compatible API (the paper uses DeepSeek-V3.2).

1. Agent vLLM server (terminal A):

   ```bash
   MODEL_PATH=./outputs/rl/qwen2_5_7b_slca_grpo \
   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash eval/tau2_bench/start_tau2_agent_vllm_server.sh
   # -> http://127.0.0.1:8000/v1  (served name: toucan_toolcall_v4)
   ```

2. User simulator — pick one:

   **Option A: external OpenAI-compatible API** (recommended, matches the paper setup):

   ```bash
   export TAU2_USER_API_BASE=https://your-openai-compat-host/v1
   export TAU2_USER_MODEL_NAME=your-user-sim-model        # e.g. deepseek-v3.2
   export TAU2_USER_API_KEY=<your-bearer-token>          # REQUIRED
   ```

   **Option B: local vLLM user simulator** (terminal C):

   ```bash
   CUDA_VISIBLE_DEVICES=4 bash eval/tau2_bench/start_tau2_user_vllm_server.sh
   # -> http://127.0.0.1:8001/v1 (served name: tau2_user)
   export TAU2_USER_API_BASE=http://127.0.0.1:8001/v1
   export TAU2_USER_MODEL_NAME=tau2_user
   export TAU2_USER_API_KEY=EMPTY
   ```

3. Evaluator (terminal B):

   ```bash
   TAU2_AGENT_API_URL=http://127.0.0.1:8000/v1 \
   TAU2_AGENT_MODEL_NAME=toucan_toolcall_v4 \
   python eval/tau2_bench/tau2_bench_eval.py
   ```

   Results are written to `./eval_results/tau2_bench/tau2_local_agent_external_user_<ts>/`.

### Common τ²-Bench knobs

| env                        | default               | purpose                                             |
| :------------------------- | :-------------------- | :-------------------------------------------------- |
| `TAU2_SUBSET_LIST`         | `airline,retail,telecom` | domains to evaluate                              |
| `TAU2_EVAL_BATCH_SIZE`     | 5                     | concurrent rollouts                                |
| `TAU2_AGENT_MAX_TOKENS`    | 1024                  | per-turn agent generation cap                      |
| `TAU2_USER_MAX_TOKENS`     | 1024                  | per-turn user-sim generation cap                   |
| `TAU2_RETRY_PASSES`        | 1                     | re-run with cache to catch 429 / timeout drops     |
| `TAU2_MAX_STEPS`           | *(unset)*             | debug-only: raises tau2 simulation step cap        |
| `TAU2_DATA_DIR`            | auto-discovered       | path to `tau2/domains/*/tasks.json` assets         |
| `TAU2_AUTO_DOWNLOAD`       | `0`                   | set `1` to auto-download `evalscope/tau2-bench-data` |

### First-time setup for the tau2 dataset

`tau2_bench_eval.py` looks for the benchmark assets under
`~/.cache/modelscope/hub/datasets/**/tau2/domains/airline/tasks.json`. If you
haven't downloaded them yet:

```bash
TAU2_AUTO_DOWNLOAD=1 python -c "from eval.tau2_bench import tau2_bench_eval"
```

or point `TAU2_DATA_DIR` at a pre-existing copy.

## Troubleshooting

- **`toucan_xlam` parser not registered**: make sure
  `--tool-parser-plugin` in the vLLM launcher points at
  `eval/bfcl_v3/toucan_xlam_tool_parser_plugin.py` (tau2 reuses the same file).
- **BFCL `multi_turn_*` subsets OOM**: drop `eval_batch_size` via
  `BFCL_EVAL_BATCH_SIZE=64` or lower.
- **τ²-Bench says `AssistantMessage must have either content or tool calls`**:
  handled by the built-in `_patch_tau2_empty_assistant_fallback`; if you still
  see it, raise `TAU2_EMPTY_ASSISTANT_RETRY` (default 3).
- **τ²-Bench user simulator hits 429 repeatedly**: raise
  `TAU2_USER_OPENAI_MAX_RETRIES` (default 8) and `TAU2_RETRY_PASSES` (default 1).
