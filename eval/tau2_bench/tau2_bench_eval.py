#!/usr/bin/env python3
"""
tau2-Bench evaluation entry point (out-of-distribution benchmark).

Architecture:
- Agent (system under test) = local vLLM OpenAI-compat server
  (launched by `eval/tau2_bench/start_tau2_agent_vllm_server.sh`).
- User simulator = any external OpenAI-compatible endpoint (the paper uses DeepSeek-V3.2)
  OR a second local vLLM instance (see `start_tau2_user_vllm_server.sh`).

Required env vars for an external user simulator:
  TAU2_USER_API_KEY  - Bearer token for the user backend.

Key env knobs:
  TAU2_AGENT_API_URL        (default: http://127.0.0.1:8000/v1)
  TAU2_AGENT_API_KEY        (default: EMPTY)
  TAU2_AGENT_MODEL_NAME     (default: toucan_toolcall_v4)

  TAU2_USER_API_BASE        - user simulator endpoint base URL.
  TAU2_USER_MODEL_NAME      - user simulator model name.
  TAU2_USER_API_KEY         - Bearer token (REQUIRED).

  TAU2_SUBSET_LIST          (default: airline,retail,telecom)
  TAU2_EVAL_BATCH_SIZE      (default: 5)
  TAU2_AGENT_MAX_TOKENS     (default: 1024)
  TAU2_USER_MAX_TOKENS      (default: 1024)
  TAU2_AGENT_TIMEOUT_SEC    (default: 200)
  TAU2_USER_TIMEOUT_SEC     (default: 200)
  TAU2_RETRY_PASSES         (default: 1)
  TAU2_MAX_STEPS            (optional; debug-only; raises simulation cap)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from evalscope import TaskConfig, run_task


def _configure_logging() -> None:
    log_level = os.getenv("TAU2_LOG_LEVEL", os.getenv("LOGURU_LEVEL", "INFO")).upper()
    os.environ.setdefault("LOGURU_LEVEL", log_level)
    os.environ.setdefault("EVALSCOPE_LOG_LEVEL", log_level)

    numeric_level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    for name in ("tau2", "evalscope", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(numeric_level)

    try:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, level=log_level)
    except Exception:
        pass


def _get_int_env(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return int(val)


def _get_list_env(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return [x.strip() for x in val.split(",") if x.strip()]


def _get_required_env(*names: str, hint: str) -> str:
    for name in names:
        val = os.getenv(name)
        if val and val.strip():
            return val.strip()
    joined = " / ".join(names)
    raise SystemExit(f"Missing required env var ({joined}). {hint}")


def _ensure_tau2_data_dir() -> None:
    """Make sure tau2 can find its dataset files (tau2/domains/*/tasks.json).

    tau2 reads TAU2_DATA_DIR only at import time and caches it. If unset, it
    falls back to a non-existent packaged path and fails.
    """
    if os.getenv("TAU2_DATA_DIR"):
        return

    candidate_roots = [
        Path("~/.cache/modelscope/hub/datasets").expanduser(),
        Path("~/.cache/evalscope/datasets").expanduser(),
    ]
    for root in candidate_roots:
        if not root.exists():
            continue
        try:
            hit = next(root.rglob("tau2/domains/airline/tasks.json"), None)
        except Exception:
            hit = None
        if hit:
            os.environ["TAU2_DATA_DIR"] = str(hit.parents[3])
            return

    if os.getenv("TAU2_AUTO_DOWNLOAD", "0") == "1":
        from modelscope import dataset_snapshot_download

        dataset_path = dataset_snapshot_download("evalscope/tau2-bench-data")
        os.environ["TAU2_DATA_DIR"] = str(dataset_path)


_ensure_tau2_data_dir()
_configure_logging()

RETRY_PASSES = _get_int_env("TAU2_RETRY_PASSES", 1)
RETRY_SLEEP_SEC = _get_int_env("TAU2_RETRY_SLEEP_SEC", 60)

# Agent (system under test): local vLLM OpenAI-compat endpoint.
AGENT_API_URL = os.getenv("TAU2_AGENT_API_URL", "http://127.0.0.1:8000/v1")
AGENT_API_KEY = os.getenv("TAU2_AGENT_API_KEY", "EMPTY")
AGENT_MODEL_NAME = os.getenv("TAU2_AGENT_MODEL_NAME", "toucan_toolcall_v4")

# User simulator: external OpenAI-compat API (or a second local vLLM).
USER_API_BASE = os.getenv("TAU2_USER_API_BASE", "")
USER_MODEL_NAME = os.getenv("TAU2_USER_MODEL_NAME", "")
if not USER_API_BASE or not USER_MODEL_NAME:
    raise SystemExit(
        "tau2 user simulator is not configured. Set TAU2_USER_API_BASE "
        "and TAU2_USER_MODEL_NAME to point at the user-simulator backend."
    )

USER_API_KEY = _get_required_env(
    "TAU2_USER_API_KEY",
    hint="Set TAU2_USER_API_KEY to the Bearer token for the user-simulator API.",
)

SUBSET_LIST = _get_list_env("TAU2_SUBSET_LIST", ["airline", "retail", "telecom"])
EVAL_BATCH_SIZE = _get_int_env("TAU2_EVAL_BATCH_SIZE", 5)
AGENT_MAX_TOKENS = _get_int_env("TAU2_AGENT_MAX_TOKENS", 1024)
USER_MAX_TOKENS = _get_int_env("TAU2_USER_MAX_TOKENS", 1024)
AGENT_TIMEOUT_SEC = float(os.getenv("TAU2_AGENT_TIMEOUT_SEC", "200"))
USER_TIMEOUT_SEC = float(os.getenv("TAU2_USER_TIMEOUT_SEC", "200"))

# OpenAI python client retry for the external user model (handles 429/5xx).
USER_OPENAI_MAX_RETRIES = _get_int_env("TAU2_USER_OPENAI_MAX_RETRIES", 8)

_DEFAULT_OUTPUTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "eval_results", "tau2_bench")
)
_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
WORK_DIR = os.getenv(
    "TAU2_WORK_DIR",
    os.path.join(_DEFAULT_OUTPUTS_DIR, f"tau2_local_agent_external_user_{_run_id}"),
)
os.makedirs(WORK_DIR, exist_ok=True)


def _patch_evalscope_get_model_for_user() -> None:
    """Inject OpenAI client retries/timeouts for the external user backend.

    tau2-bench builds the user-simulator model via `evalscope.get_model()`
    (EvalType.SERVICE). We monkey-patch `evalscope.get_model` so only calls
    targeting our user backend receive the extra `max_retries` / `timeout`
    kwargs - without importing tau2 early (which would freeze TAU2_DATA_DIR).
    """
    try:
        from evalscope.api.model import model as model_mod
    except Exception:
        return

    original_get_model = model_mod.get_model

    def wrapped_get_model(*args, **kwargs):
        base_url = kwargs.get("base_url", args[2] if len(args) > 2 else None)
        model_name = kwargs.get("model", args[0] if len(args) > 0 else None)

        if base_url and str(base_url).startswith(str(USER_API_BASE)) and (model_name == USER_MODEL_NAME):
            model_args = dict(kwargs.get("model_args") or {})
            model_args.setdefault("max_retries", USER_OPENAI_MAX_RETRIES)
            model_args.setdefault("timeout", USER_TIMEOUT_SEC)
            kwargs["model_args"] = model_args

        return original_get_model(*args, **kwargs)

    model_mod.get_model = wrapped_get_model


_patch_evalscope_get_model_for_user()


def _patch_tau2_empty_assistant_fallback() -> None:
    """Retry + fall back when the assistant turn is entirely empty.

    Some providers return assistant messages with only hidden reasoning and an
    empty visible content field. tau2 validates that `AssistantMessage` has
    either non-empty content or tool_calls; otherwise it raises and the whole
    sample is skipped. We patch `patched_generate` to retry a few times and
    then fall back to a minimal "." content, so the run continues.
    """
    try:
        from evalscope.benchmarks.tau_bench.tau2_bench import generation as tau2_gen
        from tau2.data_model.message import AssistantMessage
    except Exception:
        return

    original = tau2_gen.patched_generate
    retries = _get_int_env("TAU2_EMPTY_ASSISTANT_RETRY", 3)
    retry_sleep = float(os.getenv("TAU2_EMPTY_ASSISTANT_RETRY_SLEEP_SEC", "0.5"))

    def wrapped_generate(*args, **kwargs):
        last_exc = None
        for _ in range(max(0, retries) + 1):
            try:
                return original(*args, **kwargs)
            except ValueError as e:
                if "AssistantMessage must have either content or tool calls" not in str(e):
                    raise
                last_exc = e
                time.sleep(max(0.0, retry_sleep))
        return AssistantMessage(role="assistant", content=".", tool_calls=None, cost=None, usage={}, raw_data={})

    tau2_gen.patched_generate = wrapped_generate


_patch_tau2_empty_assistant_fallback()


def _patch_tau2_max_steps() -> None:
    """Optional debug-only patch: raise tau2 simulation max_steps / max_errors.

    The default behaviour is unchanged unless `TAU2_MAX_STEPS` or
    `TAU2_MAX_ERRORS` is set. Enabling this makes the run not directly
    comparable to the standard tau2-bench setting.
    """
    max_steps = os.getenv("TAU2_MAX_STEPS")
    max_errors = os.getenv("TAU2_MAX_ERRORS")
    if not max_steps and not max_errors:
        return

    try:
        from evalscope.benchmarks.tau_bench.tau2_bench import generation as tau2_gen
    except Exception:
        return

    original_run_task = getattr(tau2_gen, "run_task", None)
    if original_run_task is None:
        return

    steps = int(max_steps) if max_steps else None
    errors = int(max_errors) if max_errors else None

    def wrapped_run_task(*args, **kwargs):
        if steps is not None:
            kwargs["max_steps"] = steps
        if errors is not None:
            kwargs["max_errors"] = errors
        return original_run_task(*args, **kwargs)

    tau2_gen.run_task = wrapped_run_task


_patch_tau2_max_steps()


task_cfg = TaskConfig(
    model=AGENT_MODEL_NAME,
    api_url=AGENT_API_URL,
    api_key=AGENT_API_KEY,
    eval_type="openai_api",
    datasets=["tau2_bench"],
    dataset_args={
        "tau2_bench": {
            "subset_list": SUBSET_LIST,
            "extra_params": {
                "user_model": USER_MODEL_NAME,
                "api_key": USER_API_KEY,
                "api_base": USER_API_BASE,
                "generation_config": {
                    "temperature": float(os.getenv("TAU2_USER_TEMPERATURE", "0.7")),
                    "max_tokens": USER_MAX_TOKENS,
                    "timeout": USER_TIMEOUT_SEC,
                },
            },
        }
    },
    eval_batch_size=EVAL_BATCH_SIZE,
    generation_config={
        "temperature": float(os.getenv("TAU2_AGENT_TEMPERATURE", "0.6")),
        "max_tokens": AGENT_MAX_TOKENS,
        "timeout": AGENT_TIMEOUT_SEC,
        "parallel_tool_calls": True,
    },
    ignore_errors=True,
    work_dir=WORK_DIR,
)


if __name__ == "__main__":
    # Multi-pass retry: re-run with cache enabled so transient failures
    # (429s / timeouts) don't permanently skip a sample.
    if RETRY_PASSES > 1:
        task_cfg.use_cache = WORK_DIR
        task_cfg.work_dir = WORK_DIR

    for i in range(max(1, RETRY_PASSES)):
        if i > 0:
            time.sleep(max(1, RETRY_SLEEP_SEC))
        run_task(task_cfg)
