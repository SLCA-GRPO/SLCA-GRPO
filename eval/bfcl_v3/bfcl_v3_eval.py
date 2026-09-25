#!/usr/bin/env python3
"""
BFCL-v3 evaluation entry point (in-distribution benchmark).

Pair this with `eval/bfcl_v3/start_vllm_server.sh`, which launches a local
vLLM OpenAI-compatible endpoint that uses the Toucan chat template + the
`toucan_xlam` tool-call parser.

The evaluator itself is just a thin wrapper over `evalscope.run_task` that
fixes the BFCL-v3 subset list used in the main table of the paper.

Environment overrides:
  BFCL_VLLM_API_URL   (default: http://127.0.0.1:8000/v1)
  BFCL_VLLM_API_KEY   (default: EMPTY)
  BFCL_VLLM_MODEL_NAME(default: toucan_toolcall_v4 - must match the vLLM
                       `--served-model-name`)
  BFCL_V4_OUTPUTS_DIR (default: ./eval_results/bfcl_v3)
  BFCL_V4_WORK_DIR    (default: <OUTPUTS_DIR>/toucan_v4_<ts>)
"""

from __future__ import annotations

import os
from datetime import datetime

from evalscope import TaskConfig, run_task


API_URL = os.getenv("BFCL_VLLM_API_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("BFCL_VLLM_API_KEY", "EMPTY")
# Must match SERVED_MODEL_NAME in start_vllm_server.sh.
MODEL_NAME = os.getenv("BFCL_VLLM_MODEL_NAME", "toucan_toolcall_v4")

_DEFAULT_OUTPUTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "eval_results", "bfcl_v3")
)
_outputs_base = os.getenv("BFCL_V4_OUTPUTS_DIR", _DEFAULT_OUTPUTS_DIR)
_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
WORK_DIR = os.getenv(
    "BFCL_V4_WORK_DIR",
    os.path.join(_outputs_base, f"toucan_v4_{_run_id}"),
)
os.makedirs(WORK_DIR, exist_ok=True)


task_cfg = TaskConfig(
    model=MODEL_NAME,
    api_url=API_URL,
    api_key=API_KEY,
    eval_type="openai_api",
    datasets=["bfcl_v3"],
    dataset_args={
        "bfcl_v3": {
            # Main-table subsets (multi_turn_long_context is excluded for stability).
            "subset_list": [
                "simple",
                "multiple",
                "parallel",
                "parallel_multiple",
                "java",
                "javascript",
                "live_simple",
                "live_multiple",
                "live_parallel",
                "live_parallel_multiple",
                # "irrelevance",
                # "live_relevance",
                # "live_irrelevance",
                "multi_turn_base",
                "multi_turn_miss_func",
                "multi_turn_miss_param",
            ],
            "extra_params": {
                "underscore_to_dot": True,
                "is_fc_model": True,
            },
        }
    },
    generation_config={
        "temperature": 0,
        "max_tokens": 8192,
        "parallel_tool_calls": True,
    },
    eval_batch_size=int(os.getenv("BFCL_EVAL_BATCH_SIZE", "256")),
    ignore_errors=True,
    work_dir=WORK_DIR,
)


if __name__ == "__main__":
    run_task(task_cfg=task_cfg)
