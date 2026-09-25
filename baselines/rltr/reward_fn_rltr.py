"""
RLTR-2Stage baseline reward function (planner only).

RLTR splits tool use across two models: an RL-trained planner that emits the
tool trajectory, and a frozen SFT summariser that writes the final answer. This
file rewards the planner, so there is no LLM judge and no summary score.

Reward terms, following the RLTR paper:
- R_comp:   completeness reward from an LLM-based checker. Following the
            paper's Algorithm 1, R_comp = (1/N) * sum_j gamma_j(tau), where
            gamma: trajectory -> {0, 1} is the verification LLM's verdict over
            N samples.
- R_repeat: penalty for consecutively repeated tool calls.
- R_error:  penalty for malformed / invalid tool calls.
- A malformed trajectory scores -1 outright (RLTR Eq. 3).

The SLCA process metrics (format / name / key / value / parallel) are still
computed and logged so this baseline can be compared under one protocol; they
do not enter the RL objective.

Environment variables:

| variable                        | default                    | meaning                                  |
| ------------------------------- | -------------------------- | ---------------------------------------- |
| RLTR_COMP_CHECKER_BASE_URL      | http://127.0.0.1:8016/v1   | completeness-checker endpoint            |
| RLTR_COMP_CHECKER_MODEL         | Qwen3-30B-A3B              | completeness-checker model name          |
| RLTR_COMP_CHECKER_N             | 3                          | number of samples N                      |
| RLTR_COMP_CHECKER_TEMPERATURE   | 0.7                        | sampling temperature                     |
| RLTR_COMP_CHECKER_TIMEOUT       | 120                        | per-call timeout, seconds                |
| RLTR_COMP_CHECKER_MAX_RETRIES   | 3                          | per-call retry budget                    |
| RLTR_LAMBDA_REPEAT              | 0.1                        | repeat-penalty coefficient (not fixed by the paper) |
| RLTR_MU_ERROR                   | 0.2                        | error-penalty coefficient (not fixed by the paper)  |
| SLCA_WEIGHT_FORMAT              | 0.10                       | format weight (process metric only)      |
| SLCA_WEIGHT_NAME                | 0.25                       | tool-name match weight                   |
| SLCA_WEIGHT_KEY                 | 0.15                       | argument-key match weight                |
| SLCA_WEIGHT_VALUE               | 0.20                       | argument-value match weight              |
| SLCA_WEIGHT_PARALLEL            | 0.30                       | parallel-call weight                     |
| NO_CALL_PENALTY                 | -0.5                       | penalty when no call is emitted          |
"""

from __future__ import annotations

import json
import os
import re
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

# ======================== Logging setup ========================
import os as _os
from datetime import datetime as _datetime

_LOGGER_NAME = "reward_fn_rltr"
logger = logging.getLogger(_LOGGER_NAME)

logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.propagate = False

_log_formatter = logging.Formatter(
    '[%(asctime)s][RewardFnRLTR][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

_log_dir = os.environ.get("REWARD_LOG_DIR", "./logs/reward_function")
_os.makedirs(_log_dir, exist_ok=True)
_log_timestamp = _datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file_path = _os.path.join(_log_dir, f'reward_fn_rltr_{_log_timestamp}_pid{_os.getpid()}.log')

_file_handler = logging.FileHandler(filename=_log_file_path, mode='a', encoding='utf-8')
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_log_formatter)

_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.WARNING)
_stream_handler.setFormatter(_log_formatter)

logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)


# ======================== Configuration constants ========================
# RLTR Comp. Checker configuration (paper Algorithm 1, Eq. 2: LLM-based completeness)
# Paper Section 4.3: Qwen3-30B-A3B is used as the scoring model
# Serve any OpenAI-compatible endpoint for it; rl/judge/serve_llm_judge_gpt_oss.sh
# is a working template (swap the model and the port).
RLTR_COMP_CHECKER_BASE_URL = os.environ.get("RLTR_COMP_CHECKER_BASE_URL") or "http://127.0.0.1:8016/v1"
RLTR_COMP_CHECKER_MODEL = os.environ.get("RLTR_COMP_CHECKER_MODEL") or "Qwen3-30B-A3B"
RLTR_COMP_CHECKER_N = int(os.environ.get("RLTR_COMP_CHECKER_N", "3"))
RLTR_COMP_CHECKER_TEMPERATURE = float(os.environ.get("RLTR_COMP_CHECKER_TEMPERATURE", "0.7"))
RLTR_COMP_CHECKER_TIMEOUT = int(os.environ.get("RLTR_COMP_CHECKER_TIMEOUT", "600"))
RLTR_COMP_CHECKER_MAX_RETRIES = int(os.environ.get("RLTR_COMP_CHECKER_MAX_RETRIES", "3"))

# RLTR-specific configuration
RLTR_LAMBDA_REPEAT = float(os.environ.get("RLTR_LAMBDA_REPEAT", "0.1"))
RLTR_MU_ERROR = float(os.environ.get("RLTR_MU_ERROR", "0.2"))

# Penalty configuration
NO_CALL_PENALTY = float(os.environ.get("NO_CALL_PENALTY") or "-0.5")

# Regular expressions
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)

# ======================== RLTR Comp. Checker prompt (paper Figure 6A) ========================
RLTR_COMP_CHECKER_PROMPT = """You are an agent expert, committed to fully meeting query needs through precise tool combinations. When the tool returns unsatisfactory results, you must adjust the parameters and try a new call again. Ultimately, the output should be 0, indicating a missing invocation, and 1, indicating completeness.
Tool List: {tools}
# input
query: {query}
agent_action: {trajectory}"""


# ======================== Comp. Checker connection pool and helpers ========================
_comp_checker_connector = None


def _get_comp_checker_connector():
    """Get or create the aiohttp connection pool used by the Comp. Checker."""
    import aiohttp
    global _comp_checker_connector
    if _comp_checker_connector is None or _comp_checker_connector.closed:
        _comp_checker_connector = aiohttp.TCPConnector(
            limit=100, limit_per_host=50,
            ttl_dns_cache=300, enable_cleanup_closed=True, keepalive_timeout=60,
        )
    return _comp_checker_connector


def _extract_trajectory_text(solution_str: str) -> str:
    """
    Extract the tool_call + tool_response sequence from solution_str as agent_action.
    Strip <think>...</think> tags and keep only the tool interaction sequence (paper Figure 4 template).
    """
    # Strip think tags
    text = THINK_PATTERN.sub("", solution_str)

    # Extract every <tool_call>...</tool_call> and <tool_response>...</tool_response> block
    parts = []
    # Use a regex to locate every tool_call / tool_response block and its position
    combined_pattern = re.compile(
        r"(<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>)",
        re.DOTALL,
    )
    for m in combined_pattern.finditer(text):
        parts.append(m.group(0))

    if parts:
        return "\n".join(parts)
    # fallback: if no tags matched, return the whole text with think removed
    return text.strip()


async def _call_comp_checker_async(prompt: str) -> Optional[str]:
    """
    Call the Comp. Checker LLM endpoint and return the raw response text.
    Reuses the aiohttp connection-pool pattern of the main experiment.
    """
    import aiohttp
    import asyncio

    url = RLTR_COMP_CHECKER_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": RLTR_COMP_CHECKER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": RLTR_COMP_CHECKER_TEMPERATURE,
        "max_tokens": 512,
        # Disable Qwen3 thinking mode: 12 tokens vs 97 tokens, ~8x throughput
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    timeout = aiohttp.ClientTimeout(
        total=RLTR_COMP_CHECKER_TIMEOUT, connect=60,
        sock_read=RLTR_COMP_CHECKER_TIMEOUT,
    )

    for attempt in range(RLTR_COMP_CHECKER_MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(
                connector=_get_comp_checker_connector(),
                connector_owner=False, timeout=timeout,
            ) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        if resp.status >= 500 and attempt < RLTR_COMP_CHECKER_MAX_RETRIES:
                            await asyncio.sleep(2 * (attempt + 1))
                            continue
                        logger.warning(f"[CompChecker] HTTP {resp.status} on attempt {attempt}")
                        return None
                    data = await resp.json()
                    if "choices" not in data or len(data["choices"]) == 0:
                        return None
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"[CompChecker] attempt {attempt} failed: {e}")
            if attempt < RLTR_COMP_CHECKER_MAX_RETRIES:
                await asyncio.sleep(2 * (attempt + 1))
    return None


def _parse_comp_checker_response(text: str) -> Optional[int]:
    """
    Parse the Comp. Checker LLM output and extract 0 or 1.
    The LLM may emit `<think>...</think> 1` or a bare `0`/`1`,
    Parse robustly: strip the think tags, then take the last 0 or 1 that appears.
    """
    if not text:
        return None
    # Strip think tags
    cleaned = THINK_PATTERN.sub("", text).strip()
    if not cleaned:
        return None
    # Find every standalone 0 or 1 (word-boundary match, so digits inside other numbers are ignored)
    matches = re.findall(r'\b([01])\b', cleaned)
    if matches:
        return int(matches[-1])
    # fallback: check whether the whole cleaned string is "0" or "1"
    if cleaned in ("0", "1"):
        return int(cleaned)
    return None


async def _compute_rcomp_llm(
    solution_str: str,
    question: str,
    tools_str: str,
) -> Tuple[float, float]:
    """
    Core routine: build the prompt, call the Comp. Checker N times concurrently, average.

    Paper Algorithm 1:
      R_comp = (1/N) * Σ_{j=1}^{N} γ_j(τ)
      γ_j ∈ {0, 1}

    Returns:
        r_comp: continuous value in [0, 1] (mean over the successful samples)
        r_comp_mean: same as r_comp, scalar form for numpy packing
    """
    import asyncio

    trajectory = _extract_trajectory_text(solution_str)
    prompt = RLTR_COMP_CHECKER_PROMPT.format(
        tools=tools_str,
        query=question,
        trajectory=trajectory,
    )

    # Send N requests concurrently
    tasks = [_call_comp_checker_async(prompt) for _ in range(RLTR_COMP_CHECKER_N)]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    raw_scores: List[int] = []
    for i, resp in enumerate(responses):
        if isinstance(resp, Exception):
            logger.warning(f"[CompChecker] sample {i} exception: {resp}")
            continue
        if resp is None:
            logger.warning(f"[CompChecker] sample {i} returned None")
            continue
        score = _parse_comp_checker_response(resp)
        if score is not None:
            raw_scores.append(score)
        else:
            logger.warning(f"[CompChecker] sample {i} parse failed, response: {resp[:200]}")

    if not raw_scores:
        logger.warning("[CompChecker] all N samples failed, fallback R_comp=0")
        return 0.0, 0.0

    r_comp = sum(raw_scores) / len(raw_scores)
    logger.info(f"[CompChecker] raw_scores={raw_scores}, R_comp={r_comp:.3f} ({len(raw_scores)}/{RLTR_COMP_CHECKER_N} succeeded)")
    return r_comp, r_comp


# ======================== Data structures ========================
class ToolCall(NamedTuple):
    name: str
    arguments: Dict[str, Any]
    raw_json: str


class FormatCheckResult(NamedTuple):
    is_valid: bool
    error_type: Optional[str]
    error_message: Optional[str]
    tool_calls: List[ToolCall]
    tool_calls_by_block: List[List[ToolCall]]
    open_count: int
    close_count: int
    denom: int
    valid_pair_n: int


# ======================== Loose value matching ========================
def _loose_value_equal(pred_val: Any, gold_val: Any) -> bool:
    """Loose value-equality rule (same as the SLCA version)."""
    if pred_val is None and gold_val is None:
        return True
    if pred_val is None or gold_val is None:
        return False

    pred_str = str(pred_val).strip().lower()
    gold_str = str(gold_val).strip().lower()
    if pred_str == gold_str:
        return True

    try:
        pred_float = float(pred_val)
        gold_float = float(gold_val)
        if abs(pred_float - gold_float) < 1e-9:
            return True
    except (ValueError, TypeError):
        pass

    bool_map = {"true": True, "false": False, "1": True, "0": False}
    pred_bool = bool_map.get(pred_str)
    gold_bool = bool_map.get(gold_str)
    if pred_bool is not None and gold_bool is not None:
        return pred_bool == gold_bool

    if isinstance(pred_val, (list, dict)) and isinstance(gold_val, (list, dict)):
        try:
            return json.dumps(pred_val, sort_keys=True) == json.dumps(gold_val, sort_keys=True)
        except:
            pass

    return False


# ======================== Format checking ========================
def _check_tool_call_format(
    text: str,
    allowed_tools: List[str],
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
) -> FormatCheckResult:
    """Check tool-call format; return the tool calls grouped by block."""
    tool_calls: List[ToolCall] = []
    tool_calls_by_block: List[List[ToolCall]] = []
    allowed_set = {str(t) for t in allowed_tools}

    open_count = text.count("<tool_call>")
    close_count = text.count("</tool_call>")
    denom = max(open_count, close_count)

    if denom == 0:
        return FormatCheckResult(
            is_valid=True, error_type="no_tool_call", error_message=None,
            tool_calls=[], tool_calls_by_block=[],
            open_count=0, close_count=0, denom=0, valid_pair_n=0,
        )

    matches = TOOL_CALL_PATTERN.findall(text)
    valid_pair_n = 0
    first_error_type = None
    first_error_message = None

    for match in matches:
        content = match.strip()
        if not content:
            if first_error_type is None:
                first_error_type = "json_error"
                first_error_message = "empty tool_call content"
            continue

        block_tool_calls: List[ToolCall] = []
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            if first_error_type is None:
                first_error_type = "json_error"
                first_error_message = f"JSON parse failed: {str(e)[:100]}"
            continue

        objs = [parsed] if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else None)
        if objs is None:
            if first_error_type is None:
                first_error_type = "json_error"
                first_error_message = f"JSON must be an object or an array"
            continue

        block_ok = True
        for obj in objs:
            if not isinstance(obj, dict):
                block_ok = False
                break
            name = obj.get("name")
            if not name or not isinstance(name, str):
                block_ok = False
                break
            if name not in allowed_set:
                block_ok = False
                if first_error_type is None:
                    first_error_type = "tool_not_found"
                    first_error_message = f"tool '{name}' is not in the allowed list"
                break
            arguments = obj.get("arguments") or obj.get("parameters") or {}
            if not isinstance(arguments, dict):
                block_ok = False
                break
            block_tool_calls.append(ToolCall(name=name, arguments=arguments, raw_json=json.dumps(obj, ensure_ascii=False)))

        if not block_ok:
            continue

        valid_pair_n += 1
        tool_calls.extend(block_tool_calls)
        tool_calls_by_block.append(block_tool_calls)

    fully_passed = (open_count == close_count) and (valid_pair_n == denom)
    error_type = "none" if fully_passed else (first_error_type or "partial")

    return FormatCheckResult(
        is_valid=True, error_type=error_type, error_message=first_error_message,
        tool_calls=tool_calls, tool_calls_by_block=tool_calls_by_block,
        open_count=open_count, close_count=close_count, denom=denom, valid_pair_n=valid_pair_n,
    )


def _compute_format_score(text: str, allowed_tools: List[str], tool_schemas=None) -> Tuple[float, FormatCheckResult]:
    """Compute the format score, normalised to 0~1."""
    text_clean = THINK_PATTERN.sub("", text)
    result = _check_tool_call_format(text_clean, allowed_tools, tool_schemas)
    if result.denom <= 0:
        return 0.0, result
    ratio = float(result.valid_pair_n) / float(result.denom)
    return ratio, result


# ======================== Tool-name F1 ========================
def _compute_name_f1(pred_names: List[str], gold_names: List[str]) -> float:
    """Compute the tool-name F1 score."""
    if not pred_names and not gold_names:
        return 1.0
    if not pred_names or not gold_names:
        return 0.0
    pred_counter = Counter(pred_names)
    gold_counter = Counter(gold_names)
    intersection = sum(min(pred_counter.get(t, 0), gold_counter.get(t, 0)) for t in set(pred_counter) | set(gold_counter))
    return (2.0 * intersection) / (len(pred_names) + len(gold_names))


# ======================== Detailed match scoring ========================
def _safe_parse_args(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            obj = json.loads(args)
            return obj if isinstance(obj, dict) else {}
        except:
            return {}
    return {}


def _compute_detailed_match_scores(
    pred_calls: List[ToolCall],
    gold_tool_calls_flat: List[Dict[str, Any]]
) -> Tuple[float, float, float]:
    """
    Compute the detailed match scores.

    Returns:
        name_match_score: tool-name F1 (0~1)
        key_match_score: argument-key match score (0~1)
        value_match_score: argument-value match score (0~1)
    """
    pred_names = [c.name for c in pred_calls]
    gold_names = [g.get("name", "") if isinstance(g, dict) else g for g in gold_tool_calls_flat]

    name_match_score = _compute_name_f1(pred_names, gold_names)

    gold_with_args = [g for g in gold_tool_calls_flat if isinstance(g, dict) and _safe_parse_args(g.get("arguments"))]

    if not gold_with_args:
        return name_match_score, 1.0, 1.0

    if not pred_calls:
        return name_match_score, 0.0, 0.0

    total_key_score = 0.0
    total_value_score = 0.0
    matched_pred = set()

    for gold in gold_with_args:
        gold_name = gold.get("name", "")
        gold_args = _safe_parse_args(gold.get("arguments"))
        gold_keys = set(gold_args.keys())

        best_key_score = 0.0
        best_value_score = 0.0
        best_pred_idx = -1

        for pred_idx, pred in enumerate(pred_calls):
            if pred_idx in matched_pred:
                continue
            if pred.name != gold_name:
                continue

            pred_keys = set(pred.arguments.keys())

            if not gold_keys:
                key_score = 1.0
            elif not pred_keys:
                key_score = 0.0
            else:
                intersection = len(pred_keys & gold_keys)
                union = len(pred_keys | gold_keys)
                key_score = intersection / union if union > 0 else 0.0

            if not gold_keys:
                value_score = 1.0
            else:
                matched_keys = pred_keys & gold_keys
                if not matched_keys:
                    value_score = 0.0
                else:
                    value_matches = sum(
                        1 for k in matched_keys
                        if _loose_value_equal(pred.arguments.get(k), gold_args.get(k))
                    )
                    value_score = value_matches / len(gold_keys)

            combined_score = key_score + value_score
            if combined_score > best_key_score + best_value_score:
                best_key_score = key_score
                best_value_score = value_score
                best_pred_idx = pred_idx

        if best_pred_idx >= 0:
            matched_pred.add(best_pred_idx)

        total_key_score += best_key_score
        total_value_score += best_value_score

    key_match_score = total_key_score / len(gold_with_args)
    value_match_score = total_value_score / len(gold_with_args)

    return name_match_score, key_match_score, value_match_score


# ======================== Parallel scoring ========================
def _compute_parallel_score(
    pred_tool_calls_by_block: List[List[ToolCall]],
    gold_tool_calls_nested: List[List[Dict[str, Any]]],
) -> Tuple[float, int, int]:
    """
    Compute the parallel-call score: only the first-step tool-call count is compared

    Returns:
        parallel_score: 0 or 1
        first_gold_count: number of gold calls in the first step
        first_pred_count: number of predicted calls in the first step
    """
    gold_first_count = len(gold_tool_calls_nested[0]) if gold_tool_calls_nested else 0
    pred_first_count = len(pred_tool_calls_by_block[0]) if pred_tool_calls_by_block else 0

    parallel_score = 1.0 if gold_first_count == pred_first_count else 0.0

    logger.info(f"[Parallel] gold_first={gold_first_count}, pred_first={pred_first_count}, score={parallel_score:.1f}")

    return parallel_score, gold_first_count, pred_first_count


# ======================== Gold tool call normalisation ========================
def _normalize_gold_tool_calls_nested(gold_tool_calls: Any) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Convert to a nested list plus a flat list."""
    if gold_tool_calls is None:
        return [], []
    try:
        import numpy as np
        if isinstance(gold_tool_calls, np.ndarray):
            gold_tool_calls = gold_tool_calls.tolist()
    except ImportError:
        pass
    if not gold_tool_calls:
        return [], []
    if isinstance(gold_tool_calls, str):
        try:
            gold_tool_calls = json.loads(gold_tool_calls)
        except:
            return [], []
    if not isinstance(gold_tool_calls, list):
        return [], []

    is_nested = gold_tool_calls and isinstance(gold_tool_calls[0], list)
    if is_nested:
        nested_list, flat_list = [], []
        for step in gold_tool_calls:
            if not isinstance(step, list):
                continue
            step_calls = []
            for item in step:
                normalized = {"name": item} if isinstance(item, str) else (item if isinstance(item, dict) else None)
                if normalized:
                    step_calls.append(normalized)
                    flat_list.append(normalized)
            if step_calls:
                nested_list.append(step_calls)
        return nested_list, flat_list
    else:
        flat_list = [{"name": item} if isinstance(item, str) else item for item in gold_tool_calls if isinstance(item, (str, dict))]
        return [flat_list] if flat_list else [], flat_list


# ======================== RLTR-specific: repeated tool call detection ========================
def _compute_repeat_penalty(tool_calls: List[ToolCall]) -> Tuple[float, int]:
    """
    Detect consecutively repeated tool calls (identical name + arguments).

    Follows RLTR Algorithm 1: R_repeat = -lambda_repeat * repeat_count

    Returns:
        penalty: -λ_repeat × repeat_count
        repeat_count: number of consecutive repeats
    """
    if len(tool_calls) <= 1:
        return 0.0, 0

    repeat_count = 0
    for t in range(1, len(tool_calls)):
        if (tool_calls[t].name == tool_calls[t - 1].name and
                tool_calls[t].arguments == tool_calls[t - 1].arguments):
            repeat_count += 1

    penalty = -RLTR_LAMBDA_REPEAT * repeat_count
    return penalty, repeat_count


# ======================== RLTR-specific: invalid tool call penalty ========================
def _compute_error_penalty(format_result: FormatCheckResult) -> Tuple[float, int]:
    """
    Detect invalid tool calls (malformed, not in allowed_tools, etc.).

    Follows RLTR Algorithm 1: R_error = -mu_error * error_count

    error_count = denom - valid_pair_n (already computed by the format check)

    Returns:
        penalty: -μ_error × error_count
        error_count: number of invalid tool calls
    """
    error_count = format_result.denom - format_result.valid_pair_n
    penalty = -RLTR_MU_ERROR * error_count
    return penalty, error_count


# ======================== Error result construction ========================
def _build_error_result(score: float, error_type: str) -> Dict[str, Any]:
    """Build the result dict for the error case."""
    return {
        "score": score,
        "reward/error_type": error_type,
        "reward/tool_call/judge_failed": 0,
        "reward/tool_call/judge_parse_failed": 0,
        "reward/tool_call/should_call_but_no_call": 0,
        "reward/tool_call/no_tool_response": 0,
        "reward/tool_call/format_passed": 0,
        "reward/tool_call/format_error_type": "error",
        "reward/tool_call/format_score": 0.0,
        "reward/tool_call/tag_open_n": 0,
        "reward/tool_call/tag_close_n": 0,
        "reward/tool_call/tag_denom": 0,
        "reward/tool_call/tag_valid_pair_n": 0,
        "reward/tool_call/name_match_score": 0.0,
        "reward/tool_call/key_match_score": 0.0,
        "reward/tool_call/value_match_score": 0.0,
        "reward/tool_call/parallel_score": 0.0,
        "reward/tool_call/parallel_match": 0,
        "reward/tool_call/process_score": 0.0,
        "reward/tool_call/summary_score": 0.0,
        "reward/tool_call/weighted_score_norm": 0.0,
        "reward/tool_call/success@0.7": 0,
        "reward/tool_call/success@0.8": 0,
        "reward/tool_call/success@0.9": 0,
        "reward/tool_call/success@1.0": 0,
        "reward/tool_call/tool_success@0.7": 0,
        "reward/tool_call/tool_success@0.8": 0,
        "reward/tool_call/tool_success@0.9": 0,
        "reward/tool_call/tool_success@1.0": 0,
        "reward/tool_call/tool_match_score": 0.0,
        "reward/tool_call/arg_match_score": 0.0,
        "reward/tool_call/first_step_gold_count": 0,
        "reward/tool_call/first_step_pred_count": 0,
        "reward/tool_call/response_quality_score": 0.0,
        "reward/tool_call/pred_tool_count": 0,
        "reward/tool_call/gold_tool_count": 0,
        "reward/tool_call/total_score": 0.0,
        # RLTR-specific fields
        "reward/tool_call/rltr_comp": 0.0,
        "reward/tool_call/rltr_comp_raw_scores": 0.0,
        "reward/tool_call/rltr_comp_n": 0,
        "reward/tool_call/rltr_repeat_penalty": 0.0,
        "reward/tool_call/rltr_error_penalty": 0.0,
        "reward/tool_call/rltr_format_valid": 0.0,
        "reward/tool_call/rltr_repeat_count": 0,
        "reward/tool_call/rltr_error_count": 0,
    }


# ======================== Main scoring function ========================
async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    RLTR planner reward function.

    Differences from the SLCA version:
    1. No LLM judge (summary_score = 0)
    2. Three RLTR reward terms: R_comp, R_repeat, R_error
    3. score = R_comp + R_repeat + R_error (-1 when the format is invalid)

    Args:
        data_source: data-source identifier
        solution_str: full model-generated response
        ground_truth: ground-truth dict
        extra_info: optional dict of extra information

    Returns:
        A dict holding score and all reported metrics.
    """
    logger.info(f"[ComputeScore] RLTR reward start, data_source={data_source}")

    if not isinstance(ground_truth, dict):
        return _build_error_result(0.0, "invalid_ground_truth")

    # Note: allowed_tools in the RL data may be a numpy.ndarray, so `or []` cannot be used
    allowed_tools = ground_truth.get("allowed_tools")
    if allowed_tools is None:
        allowed_tools = []
    elif not isinstance(allowed_tools, list):
        allowed_tools = list(allowed_tools)  # numpy array → list
    tool_schemas = ground_truth.get("tool_schemas")
    if isinstance(tool_schemas, str):
        try:
            tool_schemas = json.loads(tool_schemas)
        except:
            tool_schemas = None

    if not allowed_tools:
        return _build_error_result(0.0, "missing_allowed_tools")

    raw_gold = ground_truth.get("gold_tool_calls")
    if raw_gold is None:
        raw_gold = []
    gold_nested, gold_flat = _normalize_gold_tool_calls_nested(raw_gold)

    format_score, format_result = _compute_format_score(solution_str, allowed_tools, tool_schemas)

    # process_score sub-item weights (same as the SLCA version, to keep the comparison fair)
    w_format = float(os.environ.get("SLCA_WEIGHT_FORMAT", "0.10"))
    w_name = float(os.environ.get("SLCA_WEIGHT_NAME", "0.25"))
    w_key = float(os.environ.get("SLCA_WEIGHT_KEY", "0.15"))
    w_value = float(os.environ.get("SLCA_WEIGHT_VALUE", "0.20"))
    w_parallel = float(os.environ.get("SLCA_WEIGHT_PARALLEL", "0.30"))

    # Should have called a tool but did not
    if gold_flat and format_result.open_count == 0:
        first_gold = len(gold_nested[0]) if gold_nested else 0
        return {
            "score": float(NO_CALL_PENALTY),
            "reward/error_type": "none",
            "reward/tool_call/judge_failed": 0,
            "reward/tool_call/judge_parse_failed": 0,
            "reward/tool_call/should_call_but_no_call": 1,
            "reward/tool_call/no_tool_response": 0,
            "reward/tool_call/format_passed": 1,
            "reward/tool_call/format_error_type": "no_tool_call",
            "reward/tool_call/format_score": 0.0,
            "reward/tool_call/tag_open_n": 0,
            "reward/tool_call/tag_close_n": 0,
            "reward/tool_call/tag_denom": 0,
            "reward/tool_call/tag_valid_pair_n": 0,
            "reward/tool_call/name_match_score": 0.0,
            "reward/tool_call/key_match_score": 0.0,
            "reward/tool_call/value_match_score": 0.0,
            "reward/tool_call/parallel_score": 0.0,
            "reward/tool_call/parallel_match": 0,
            "reward/tool_call/process_score": 0.0,
            "reward/tool_call/summary_score": 0.0,
            "reward/tool_call/weighted_score_norm": 0.0,
            "reward/tool_call/success@0.7": 0,
            "reward/tool_call/success@0.8": 0,
            "reward/tool_call/success@0.9": 0,
            "reward/tool_call/success@1.0": 0,
            "reward/tool_call/tool_success@0.7": 0,
            "reward/tool_call/tool_success@0.8": 0,
            "reward/tool_call/tool_success@0.9": 0,
            "reward/tool_call/tool_success@1.0": 0,
            "reward/tool_call/tool_match_score": 0.0,
            "reward/tool_call/arg_match_score": 0.0,
            "reward/tool_call/first_step_gold_count": first_gold,
            "reward/tool_call/first_step_pred_count": 0,
            "reward/tool_call/response_quality_score": 0.0,
            "reward/tool_call/pred_tool_count": 0,
            "reward/tool_call/gold_tool_count": len(gold_flat),
            "reward/tool_call/total_score": 0.0,
            # RLTR-specific fields
            "reward/tool_call/rltr_comp": 0.0,
            "reward/tool_call/rltr_comp_raw_scores": 0.0,
            "reward/tool_call/rltr_comp_n": 0,
            "reward/tool_call/rltr_repeat_penalty": 0.0,
            "reward/tool_call/rltr_error_penalty": 0.0,
            "reward/tool_call/rltr_format_valid": 0.0,
            "reward/tool_call/rltr_repeat_count": 0,
            "reward/tool_call/rltr_error_count": 0,
        }

    # Detailed match scoring
    name_match_score, key_match_score, value_match_score = _compute_detailed_match_scores(
        format_result.tool_calls, gold_flat
    )

    # Parallel scoring
    parallel_score, first_gold, first_pred = _compute_parallel_score(
        format_result.tool_calls_by_block, gold_nested
    )
    parallel_match = 1 if parallel_score == 1.0 else 0

    # ========== process_score (same sub-item weighting as SLCA) ==========
    process_score = (
        w_format * format_score +
        w_name * name_match_score +
        w_key * key_match_score +
        w_value * value_match_score +
        w_parallel * parallel_score
    )

    # ========== RLTR-specific reward computation ==========

    # Format validity check
    denom = format_result.denom
    format_fully_passed = (denom == 0) or (
        (format_result.open_count == format_result.close_count) and
        (format_result.valid_pair_n == denom)
    )
    format_error_type = format_result.error_type or ("none" if format_fully_passed else "partial")

    # 1) R_comp: completeness reward (RLTR Algorithm 1, Eq. 2)
    #    Sample the Comp. Checker LLM N times and average
    #    Collect question_content and tools_str for the Comp. Checker prompt
    question_content = ground_truth.get("question_content", "") or ground_truth.get("question", "") or ground_truth.get("query", "") or ""
    tools_str = ""
    if tool_schemas:
        tools_str = json.dumps(tool_schemas, ensure_ascii=False, indent=2) if isinstance(tool_schemas, (list, dict)) else str(tool_schemas)
    elif allowed_tools:
        tools_str = ", ".join(str(t) for t in allowed_tools)
    R_comp, rcomp_raw_scores = await _compute_rcomp_llm(solution_str, question_content, tools_str)

    # 2) R_repeat: repeat penalty (RLTR Algorithm 1)
    R_repeat, repeat_count = _compute_repeat_penalty(format_result.tool_calls)

    # 3) R_error: error penalty (RLTR Algorithm 1)
    R_error, error_count = _compute_error_penalty(format_result)

    # 4) Whether the format is fully valid
    format_valid = 1.0 if format_fully_passed else 0.0

    # 5) Total score (RLTR Eq. 3)
    if not format_fully_passed:
        score = -1.0
    else:
        score = R_comp + R_repeat + R_error

    # ========== weighted_score_norm (aligned with the main experiment) ==========
    w_process = float(os.environ.get("SLCA_WEIGHT_PROCESS", "1.0"))
    w_respq = float(os.environ.get("SLCA_WEIGHT_RESPQ", "1.0"))
    summary_score = 0.0  # RLTR does not score the summary
    weighted_score_raw = w_process * process_score + w_respq * summary_score
    weighted_score_norm = weighted_score_raw / (w_process + w_respq)

    # success@ thresholds (based on weighted_score_norm, same as the main experiment)
    success_at_07 = 1 if weighted_score_norm >= 0.7 else 0
    success_at_08 = 1 if weighted_score_norm >= 0.8 else 0
    success_at_09 = 1 if weighted_score_norm >= 0.9 else 0
    success_at_10 = 1 if weighted_score_norm >= 1.0 - 1e-9 else 0

    # tool_success@ thresholds (based on name_match_score, for WandB logging)
    tool_success_at_07 = 1 if name_match_score >= 0.7 else 0
    tool_success_at_08 = 1 if name_match_score >= 0.8 else 0
    tool_success_at_09 = 1 if name_match_score >= 0.9 else 0
    tool_success_at_10 = 1 if name_match_score >= 1.0 - 1e-9 else 0

    # Compatibility fields
    tool_match_score = name_match_score
    arg_match_score = (key_match_score + value_match_score) / 2.0

    logger.info(f"[ComputeScore] RLTR: process={process_score:.4f}, R_comp={R_comp:.3f}(raw={rcomp_raw_scores}), "
                f"R_repeat={R_repeat:.3f}(cnt={repeat_count}), R_error={R_error:.3f}(cnt={error_count}), "
                f"format_valid={format_valid:.0f}, score={score:.4f}")
    logger.info(f"[ComputeScore] weighted_score_norm={weighted_score_norm:.4f}, "
                f"success@: 0.7={success_at_07}, 0.8={success_at_08}, 0.9={success_at_09}, 1.0={success_at_10}")

    return {
        # score = RLTR total reward (used for RL training)
        "score": float(score),
        "reward/error_type": "none",
        "reward/tool_call/judge_failed": 0,
        "reward/tool_call/judge_parse_failed": 0,
        "reward/tool_call/should_call_but_no_call": 0,
        "reward/tool_call/no_tool_response": 0,
        "reward/tool_call/format_passed": int(format_fully_passed),
        "reward/tool_call/format_error_type": "none" if format_fully_passed else format_error_type,
        "reward/tool_call/format_score": format_score,
        "reward/tool_call/tag_open_n": format_result.open_count,
        "reward/tool_call/tag_close_n": format_result.close_count,
        "reward/tool_call/tag_denom": denom,
        "reward/tool_call/tag_valid_pair_n": format_result.valid_pair_n,
        # Sub-item scores (for WandB logging)
        "reward/tool_call/name_match_score": name_match_score,
        "reward/tool_call/key_match_score": key_match_score,
        "reward/tool_call/value_match_score": value_match_score,
        "reward/tool_call/parallel_score": parallel_score,
        "reward/tool_call/parallel_match": parallel_match,
        # Segment-level scores
        "reward/tool_call/process_score": process_score,
        "reward/tool_call/summary_score": 0.0,  # RLTR does not score the summary
        # Evaluation metrics aligned with the main experiment
        "reward/tool_call/weighted_score_norm": weighted_score_norm,
        "reward/tool_call/success@0.7": success_at_07,
        "reward/tool_call/success@0.8": success_at_08,
        "reward/tool_call/success@0.9": success_at_09,
        "reward/tool_call/success@1.0": success_at_10,
        "reward/tool_call/tool_success@0.7": tool_success_at_07,
        "reward/tool_call/tool_success@0.8": tool_success_at_08,
        "reward/tool_call/tool_success@0.9": tool_success_at_09,
        "reward/tool_call/tool_success@1.0": tool_success_at_10,
        # Compatibility fields
        "reward/tool_call/tool_match_score": tool_match_score,
        "reward/tool_call/arg_match_score": arg_match_score,
        "reward/tool_call/first_step_gold_count": first_gold,
        "reward/tool_call/first_step_pred_count": first_pred,
        "reward/tool_call/response_quality_score": 0.0,  # RLTR has no LLM judge
        "reward/tool_call/pred_tool_count": len(format_result.tool_calls),
        "reward/tool_call/gold_tool_count": len(gold_flat),
        "reward/tool_call/total_score": weighted_score_norm,
        # RLTR-specific reward fields (read by the advantage estimator)
        "reward/tool_call/rltr_comp": R_comp,
        "reward/tool_call/rltr_comp_raw_scores": rcomp_raw_scores,
        "reward/tool_call/rltr_comp_n": RLTR_COMP_CHECKER_N,
        "reward/tool_call/rltr_repeat_penalty": R_repeat,
        "reward/tool_call/rltr_error_penalty": R_error,
        "reward/tool_call/rltr_format_valid": format_valid,
        "reward/tool_call/rltr_repeat_count": repeat_count,
        "reward/tool_call/rltr_error_count": error_count,
    }


def compute_score_sync(data_source: str, solution_str: str, ground_truth: Any, extra_info=None, **kwargs) -> Dict[str, Any]:
    """Synchronous wrapper around compute_score."""
    import asyncio
    return asyncio.run(compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs))


# ======================== Test entry point ========================
if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("RLTR reward function test (LLM-based Comp. Checker)")
    print("=" * 60)

    print(f"\nRLTR Comp. Checker config:")
    print(f"  RLTR_COMP_CHECKER_BASE_URL: {RLTR_COMP_CHECKER_BASE_URL}")
    print(f"  RLTR_COMP_CHECKER_MODEL: {RLTR_COMP_CHECKER_MODEL}")
    print(f"  RLTR_COMP_CHECKER_N: {RLTR_COMP_CHECKER_N}")
    print(f"  RLTR_COMP_CHECKER_TEMPERATURE: {RLTR_COMP_CHECKER_TEMPERATURE}")
    print(f"\nRLTR other config:")
    print(f"  RLTR_LAMBDA_REPEAT: {RLTR_LAMBDA_REPEAT}")
    print(f"  RLTR_MU_ERROR: {RLTR_MU_ERROR}")

    # Test 1: perfect tool call
    ground_truth_1 = {
        "allowed_tools": ["get_weather"],
        "gold_tool_calls": '[[{"name": "get_weather", "arguments": {"city": "Beijing"}}]]',
        "question": "What is the weather in Beijing?",
    }
    solution_1 = '<think>Let me check</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>\n<tool_response>\n{"temperature": 25}\n</tool_response>\nThe weather in Beijing is 25 degrees.'

    def print_result(name, result):
        print(f"\n{name}: score={result['score']:.3f}")
        print(f"  process_score={result['reward/tool_call/process_score']:.3f}")
        print(f"  weighted_score_norm={result['reward/tool_call/weighted_score_norm']:.3f}")
        print(f"  total_score={result['reward/tool_call/total_score']:.3f}")
        print(f"  R_comp={result['reward/tool_call/rltr_comp']:.3f}, raw_scores={result['reward/tool_call/rltr_comp_raw_scores']}, n={result['reward/tool_call/rltr_comp_n']}")
        print(f"  R_repeat={result['reward/tool_call/rltr_repeat_penalty']:.3f}, R_error={result['reward/tool_call/rltr_error_penalty']:.3f}")
        print(f"  format_valid={result['reward/tool_call/rltr_format_valid']:.0f}")
        print(f"  success@: 0.7={result['reward/tool_call/success@0.7']}, 0.8={result['reward/tool_call/success@0.8']}, 0.9={result['reward/tool_call/success@0.9']}, 1.0={result['reward/tool_call/success@1.0']}")
        print(f"  tool_success@: 0.7={result['reward/tool_call/tool_success@0.7']}, 0.8={result['reward/tool_call/tool_success@0.8']}, 0.9={result['reward/tool_call/tool_success@0.9']}, 1.0={result['reward/tool_call/tool_success@1.0']}")
        # Check that the fields exist
        for k in ["judge_failed", "judge_parse_failed", "no_tool_response", "response_quality_score",
                   "first_step_gold_count", "first_step_pred_count",
                   "rltr_comp", "rltr_comp_raw_scores", "rltr_comp_n"]:
            full_key = f"reward/tool_call/{k}"
            assert full_key in result, f"missing field: {full_key}"
        # Check that R_comp lies within [0, 1]
        assert 0.0 <= result["reward/tool_call/rltr_comp"] <= 1.0, \
            f"R_comp out of range: {result['reward/tool_call/rltr_comp']}"

    result_1 = asyncio.run(compute_score("test", solution_1, ground_truth_1))
    print_result("Test 1 (perfect)", result_1)

    # Test 2: repeated tool call
    solution_2 = '<think>Let me check</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>\n<tool_response>\n{"temperature": 25}\n</tool_response>\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>\n<tool_response>\n{"temperature": 25}\n</tool_response>\nDone.'

    result_2 = asyncio.run(compute_score("test", solution_2, ground_truth_1))
    print_result("Test 2 (repeat)", result_2)

    # Test 3: malformed format
    solution_3 = '<think>hmm</think>\n\n<tool_call>\n{invalid json\n</tool_call>'

    result_3 = asyncio.run(compute_score("test", solution_3, ground_truth_1))
    print_result("Test 3 (malformed)", result_3)

    # Test 4: no tool call
    solution_4 = '<think>I know the answer</think>\n\nThe weather is nice today.'

    result_4 = asyncio.run(compute_score("test", solution_4, ground_truth_1))
    print_result("Test 4 (no call)", result_4)

    # Test 5: _parse_comp_checker_response unit test
    print("\n--- _parse_comp_checker_response unit test ---")
    test_cases = [
        ("1", 1),
        ("0", 0),
        ("<think>analysis...</think>\n1", 1),
        ("<think>analysis...</think>\n0", 0),
        ("<thinking>let me check</thinking> 1", 1),
        ("The trajectory is complete. 1", 1),
        ("Missing invocation. 0", 0),
        ("", None),
        ("no number here", None),
    ]
    for text, expected in test_cases:
        got = _parse_comp_checker_response(text)
        status = "OK" if got == expected else "FAIL"
        print(f"  [{status}] parse({text!r:.50}) -> {got} (expected {expected})")

    # Test 6: _extract_trajectory_text unit test
    print("\n--- _extract_trajectory_text unit test ---")
    test_sol = '<think>reasoning</think>\n\n<tool_call>\n{"name":"f","arguments":{}}\n</tool_call>\n<tool_response>\n{"r":1}\n</tool_response>\nfinal answer'
    traj = _extract_trajectory_text(test_sol)
    assert "<tool_call>" in traj, "trajectory must contain <tool_call>"
    assert "<tool_response>" in traj, "trajectory must contain <tool_response>"
    assert "<think>" not in traj, "trajectory must not contain <think>"
    print(f"  [OK] trajectory extracted correctly, length={len(traj)}")

    print(f"\nAll tests passed!")
