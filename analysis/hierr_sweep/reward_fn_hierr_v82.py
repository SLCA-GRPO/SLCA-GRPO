"""
HierR reward function, v8.2 -- used by the ablations and by the HierR / gradient
analyses in `analysis/`.

This is the direct predecessor of `rl/slca_grpo/reward_fn.py` (the main-method
reward). It is kept verbatim because the ablation and analysis runs reported in
the paper were produced with it; do NOT substitute the main-method file, whose
summary judge sees only `<tool_response>` blocks and uses a different prompt.

Score composition:
- process_score = format * w_format + name * w_name + key * w_key
                  + value * w_value + parallel * w_parallel        (in [0, 1])
- summary_score = LLM-judge response quality                       (in [0, 1])
- score         = SLCA_WEIGHT_PROCESS * process_score
                  + SLCA_WEIGHT_RESPQ * summary_score              (weighted_score)
- weighted_score_norm = score / (SLCA_WEIGHT_PROCESS + SLCA_WEIGHT_RESPQ), the
  quantity behind the reported success@0.7 / 0.8 / 0.9 / 1.0 metrics.

`compute_score` is the async entry point verl calls through
`custom_reward_function.path`.

Environment variables:
- LLM_JUDGE_BASE_URL / LLM_JUDGE_MODEL / LLM_JUDGE_ENABLED: judge endpoint and toggle
- SLCA_WEIGHT_FORMAT / NAME / KEY / VALUE / PARALLEL: process sub-weights
- SLCA_WEIGHT_PROCESS / SLCA_WEIGHT_RESPQ: process vs summary mixing
- REWARD_LOG_DIR: directory for the per-process reward logs
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

_LOGGER_NAME = "reward_fn_llm_v8_slca_v8_2"
logger = logging.getLogger(_LOGGER_NAME)

logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.propagate = False

_log_formatter = logging.Formatter(
    '[%(asctime)s][RewardFnV8.2][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

_log_dir = os.environ.get("REWARD_LOG_DIR", "./logs/reward_function")
_os.makedirs(_log_dir, exist_ok=True)
_log_timestamp = _datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file_path = _os.path.join(_log_dir, f'reward_fn_llm_v8_slca_v8_2_{_log_timestamp}_pid{_os.getpid()}.log')

_file_handler = logging.FileHandler(filename=_log_file_path, mode='a', encoding='utf-8')
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_log_formatter)

_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.WARNING)
_stream_handler.setFormatter(_log_formatter)

logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)


# ======================== Configuration constants ========================
# LLM judge configuration
LLM_JUDGE_BASE_URL = os.environ.get("LLM_JUDGE_BASE_URL") or "http://127.0.0.1:8016/v1"
LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL") or "gpt-oss-120b"
LLM_JUDGE_ENABLED = os.environ.get("LLM_JUDGE_ENABLED", "true").lower() in ("true", "1", "yes")
LLM_JUDGE_TIMEOUT = int(os.environ.get("LLM_JUDGE_TIMEOUT") or "540")
LLM_JUDGE_MAX_RETRIES = int(os.environ.get("LLM_JUDGE_MAX_RETRIES") or "3")
LLM_JUDGE_RETRY_DELAY = float(os.environ.get("LLM_JUDGE_RETRY_DELAY") or "2.0")

# Penalty configuration
JUDGE_FAILURE_SCORE = float(os.environ.get("JUDGE_FAILURE_SCORE") or "0.0")
NO_CALL_PENALTY = float(os.environ.get("NO_CALL_PENALTY") or "-0.5")

# WandB logging toggle
ENABLE_WANDB_LOGGING = os.environ.get("ENABLE_WANDB_LOGGING", "true").lower() in ("true", "1", "yes")

# Rating -> score map
RESPONSE_QUALITY_SCORE_MAP = {
    "very poor": 1, "poor": 2, "acceptable": 3,
    "good": 4, "excellent": 5,
}

# Regular expressions
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
# Support both the <think> and <thinking> tag forms
THINK_PATTERN = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)
RESPONSE_QUALITY_RATING_PATTERN = re.compile(
    r"<response_quality>.*?<rating>(.*?)</rating>.*?</response_quality>", re.DOTALL | re.IGNORECASE
)


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
    """Loose value-equality rule."""
    # 1. Both None or empty
    if pred_val is None and gold_val is None:
        return True
    if pred_val is None or gold_val is None:
        return False

    # 2. String comparison (strip whitespace, lowercase)
    pred_str = str(pred_val).strip().lower()
    gold_str = str(gold_val).strip().lower()
    if pred_str == gold_str:
        return True

    # 3. Numeric comparison: 10 == "10" == 10.0
    try:
        pred_float = float(pred_val)
        gold_float = float(gold_val)
        if abs(pred_float - gold_float) < 1e-9:
            return True
    except (ValueError, TypeError):
        pass

    # 4. Boolean comparison: true == "true" == True
    bool_map = {"true": True, "false": False, "1": True, "0": False}
    pred_bool = bool_map.get(pred_str)
    gold_bool = bool_map.get(gold_str)
    if pred_bool is not None and gold_bool is not None:
        return pred_bool == gold_bool

    # 5. List / dict: compare sorted JSON
    if isinstance(pred_val, (list, dict)) and isinstance(gold_val, (list, dict)):
        try:
            return json.dumps(pred_val, sort_keys=True) == json.dumps(gold_val, sort_keys=True)
        except:
            pass

    return False


# ======================== Final response extraction ========================
def _extract_final_response(solution_str: str) -> str:
    """Extract the final response from solution_str."""
    if not solution_str:
        return ""
    text = THINK_PATTERN.sub("", solution_str)
    last_response_end = text.rfind("</tool_response>")
    if last_response_end != -1:
        final_text = text[last_response_end + len("</tool_response>"):].strip()
        if final_text:
            return final_text
    last_call_end = text.rfind("</tool_call>")
    if last_call_end != -1:
        final_text = text[last_call_end + len("</tool_call>"):].strip()
        if "<tool_response>" in final_text:
            return ""
        if final_text and not final_text.startswith("<tool_call>"):
            return final_text
        return ""
    text_without_tools = TOOL_CALL_PATTERN.sub("", text).strip()
    if text_without_tools:
        return text_without_tools
    return ""


def _extract_tool_interactions(solution_str: str) -> str:
    """Extract the tool-call / tool-response interaction text from solution_str."""
    if not solution_str:
        return ""
    interactions = []
    tool_calls = TOOL_CALL_PATTERN.findall(solution_str)
    tool_responses = TOOL_RESPONSE_PATTERN.findall(solution_str)
    for i, call_content in enumerate(tool_calls):
        interactions.append(f"[Tool Call {i+1}]")
        interactions.append(call_content.strip())
        interactions.append("")
    for i, response_content in enumerate(tool_responses):
        interactions.append(f"[Tool Response {i+1}]")
        interactions.append(response_content.strip())
        interactions.append("")
    return "\n".join(interactions).strip()


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
    """Compute the format score, normalised to 0~1.

    Note: <think> content is stripped first, so that a tool_call produced inside
    the model's reasoning is not parsed by mistake.
    """
    # Strip <think> content so a tool_call inside the reasoning is not parsed
    text_clean = THINK_PATTERN.sub("", text)
    result = _check_tool_call_format(text_clean, allowed_tools, tool_schemas)
    if result.denom <= 0:
        return 0.0, result
    ratio = float(result.valid_pair_n) / float(result.denom)
    return ratio, result  # normalised to 0~1


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

    # 1. Tool-name F1
    name_match_score = _compute_name_f1(pred_names, gold_names)

    # 2. Argument match (tools have to be matched first)
    gold_with_args = [g for g in gold_tool_calls_flat if isinstance(g, dict) and _safe_parse_args(g.get("arguments"))]

    if not gold_with_args:
        # No tool takes arguments, so key and value both count as a perfect match
        return name_match_score, 1.0, 1.0

    if not pred_calls:
        return name_match_score, 0.0, 0.0

    # Greedy match: for each gold call, find the best pred call
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

            # Key Jaccard
            if not gold_keys:
                key_score = 1.0
            elif not pred_keys:
                key_score = 0.0
            else:
                intersection = len(pred_keys & gold_keys)
                union = len(pred_keys | gold_keys)
                key_score = intersection / union if union > 0 else 0.0

            # Value match (over the matched keys)
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

            # Keep the pred with the highest combined score
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
    Compute the parallel-call score (simple count match).

    Only compares whether the first step has the same number of tool calls:
    - returns 1.0 if they are equal
    - returns 0.0 otherwise

    Returns:
        parallel_score: 0 or 1
        first_gold_count: number of gold calls in the first step
        first_pred_count: number of predicted calls in the first step
    """
    gold_first_count = len(gold_tool_calls_nested[0]) if gold_tool_calls_nested else 0
    pred_first_count = len(pred_tool_calls_by_block[0]) if pred_tool_calls_by_block else 0

    parallel_score = 1.0 if gold_first_count == pred_first_count else 0.0

    logger.info(f"[Parallel] v8.2: gold_first={gold_first_count}, pred_first={pred_first_count}, score={parallel_score:.1f}")

    return parallel_score, gold_first_count, pred_first_count


# ======================== LLM judge helpers ========================
_global_connector = None

def _get_global_connector():
    import aiohttp
    global _global_connector
    if _global_connector is None or _global_connector.closed:
        _global_connector = aiohttp.TCPConnector(limit=100, limit_per_host=50, ttl_dns_cache=300, enable_cleanup_closed=True, keepalive_timeout=60)
    return _global_connector


LLM_JUDGE_PROMPT_TEMPLATE = """## Task
Evaluate the quality of an AI assistant's final response after using tools to help the user.

## Assessment Criteria
### Response Quality (1-5 scale)
- 1 (very poor): Response is completely wrong, irrelevant, or ignores tool results
- 2 (poor): Seriously misinterprets tool results or misses critical information
- 3 (acceptable): Basically correct but has minor errors or incomplete coverage
- 4 (good): Correctly uses tool results, response is clear and complete
- 5 (excellent): Perfect interpretation of results, professional and helpful response

## User Question
```
{question_content}
```

## Tool Interactions
```
{tool_interactions}
```

## Model's Final Response
```
{final_response}
```

## Evaluation Points
1. **Result Interpretation**: Does the model correctly understand and explain the data returned by tools?
2. **Information Completeness**: Does the response address all parts of the user's question?
3. **Expression Quality**: Is the response clear, well-organized, and helpful to the user?

## Output Format
<response>
  <response_quality>
    <reasoning>Brief analysis of the response quality based on the three evaluation points above.</reasoning>
    <rating><!-- one of: very poor, poor, acceptable, good, excellent --></rating>
  </response_quality>
</response>
"""


async def _call_llm_judge_async(prompt: str) -> Optional[str]:
    import aiohttp
    import asyncio
    url = LLM_JUDGE_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {"model": LLM_JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 8192}
    headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    timeout = aiohttp.ClientTimeout(total=LLM_JUDGE_TIMEOUT, connect=60, sock_read=LLM_JUDGE_TIMEOUT)

    for attempt in range(LLM_JUDGE_MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(connector=_get_global_connector(), connector_owner=False, timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        if resp.status >= 500 and attempt < LLM_JUDGE_MAX_RETRIES:
                            await asyncio.sleep(LLM_JUDGE_RETRY_DELAY * (attempt + 1))
                            continue
                        return None
                    data = await resp.json()
                    if "choices" not in data or len(data["choices"]) == 0:
                        return None
                    return data["choices"][0]["message"]["content"]
        except Exception:
            if attempt < LLM_JUDGE_MAX_RETRIES:
                import asyncio
                await asyncio.sleep(LLM_JUDGE_RETRY_DELAY * (attempt + 1))
    return None


def _parse_response_quality_rating(llm_response: str) -> Optional[str]:
    match = RESPONSE_QUALITY_RATING_PATTERN.search(llm_response)
    if match:
        rating = match.group(1).strip().lower()
        return rating if rating in RESPONSE_QUALITY_SCORE_MAP else None
    return None


def _normalize_score(raw_score: int) -> float:
    return (raw_score - 1) / 4.0


async def _compute_response_quality_score(
    question_content: str,
    tool_interactions: str,
    final_response: str
) -> Tuple[float, Optional[str], Optional[str], bool, bool]:
    """Compute the final-response quality score."""
    if not LLM_JUDGE_ENABLED:
        return 0.0, None, None, False, False

    if not final_response or len(final_response.strip()) < 10:
        logger.info("[ResponseQuality] final response empty or too short, scoring low")
        return 0.0, "very poor", None, False, False

    prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(
        question_content=question_content,
        tool_interactions=tool_interactions,
        final_response=final_response
    )
    llm_response = await _call_llm_judge_async(prompt)
    if llm_response is None:
        return JUDGE_FAILURE_SCORE, None, None, True, False
    rating = _parse_response_quality_rating(llm_response)
    if rating is None:
        return JUDGE_FAILURE_SCORE, None, llm_response, False, True
    return _normalize_score(RESPONSE_QUALITY_SCORE_MAP[rating]), rating, llm_response, False, False


# ======================== Nested list handling ========================
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


def _build_error_result(score: float, error_type: str) -> Dict[str, Any]:
    """Build the result dict for the error case."""
    # Weights used in the computation
    w_process = float(os.environ.get("SLCA_WEIGHT_PROCESS", "1.0"))
    w_respq = float(os.environ.get("SLCA_WEIGHT_RESPQ", "1.0"))

    return {
        "score": score,
        "reward/error_type": error_type,
        "reward/tool_call/judge_failed": 0,
        "reward/tool_call/judge_parse_failed": 0,
        "reward/tool_call/should_call_but_no_call": 0,
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
        # Segment scores
        "reward/tool_call/process_score": 0.0,
        "reward/tool_call/summary_score": 0.0,
        # Weighted score
        "reward/tool_call/weighted_score_norm": 0.0,
        # success@ thresholds
        "reward/tool_call/success@0.7": 0,
        "reward/tool_call/success@0.8": 0,
        "reward/tool_call/success@0.9": 0,
        "reward/tool_call/success@1.0": 0,
        # tool_success@ thresholds (based on tool_match_score)
        "reward/tool_call/tool_success@0.7": 0,
        "reward/tool_call/tool_success@0.8": 0,
        "reward/tool_call/tool_success@0.9": 0,
        "reward/tool_call/tool_success@1.0": 0,
        # Compatibility fields
        "reward/tool_call/tool_match_score": 0.0,
        "reward/tool_call/arg_match_score": 0.0,
        "reward/tool_call/first_step_gold_count": 0,
        "reward/tool_call/first_step_pred_count": 0,
        "reward/tool_call/response_quality_score": 0.0,
        "reward/tool_call/pred_tool_count": 0,
        "reward/tool_call/gold_tool_count": 0,
        "reward/tool_call/total_score": 0.0,
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
    Main scoring function (v8.2 SLCA weighted).

    Scoring:
    1. score = weighted_score = w_process × process_score + w_respq × summary_score
    2. weighted_score_norm = weighted_score / (w_process + w_respq)
    3. success@0.7/0.8/0.9/1.0 are based on weighted_score_norm

    Args:
        data_source: data-source identifier
        solution_str: full model-generated response
        ground_truth: ground-truth dict
        extra_info: optional dict of extra information

    Returns:
        A dict holding score and all reported metrics.
    """
    logger.info(f"[ComputeScore] v8.2_slca start, data_source={data_source}")

    if not isinstance(ground_truth, dict):
        return _build_error_result(0.0, "invalid_ground_truth")

    allowed_tools = ground_truth.get("allowed_tools") or []
    tool_schemas = ground_truth.get("tool_schemas")
    if isinstance(tool_schemas, str):
        try:
            tool_schemas = json.loads(tool_schemas)
        except:
            tool_schemas = None

    if not allowed_tools:
        return _build_error_result(0.0, "missing_allowed_tools")

    question_content = ground_truth.get("question_content", "")
    if not question_content and extra_info:
        question_content = extra_info.get("input", "") or extra_info.get("question", "")

    raw_gold = ground_truth.get("gold_tool_calls") or []
    gold_nested, gold_flat = _normalize_gold_tool_calls_nested(raw_gold)

    format_score, format_result = _compute_format_score(solution_str, allowed_tools, tool_schemas)

    # Weight configuration
    w_process = float(os.environ.get("SLCA_WEIGHT_PROCESS", "1.0"))
    w_format = float(os.environ.get("SLCA_WEIGHT_FORMAT", "0.10"))
    w_name = float(os.environ.get("SLCA_WEIGHT_NAME", "0.25"))
    w_key = float(os.environ.get("SLCA_WEIGHT_KEY", "0.15"))
    w_value = float(os.environ.get("SLCA_WEIGHT_VALUE", "0.20"))
    w_parallel = float(os.environ.get("SLCA_WEIGHT_PARALLEL", "0.30"))
    w_respq = float(os.environ.get("SLCA_WEIGHT_RESPQ", "1.0"))

    # Should have called a tool but did not
    if gold_flat and format_result.open_count == 0:
        first_gold = len(gold_nested[0]) if gold_nested else 0
        # Compute weighted_score for the penalty case
        weighted_score = float(NO_CALL_PENALTY)
        weighted_score_norm = max(0.0, weighted_score / (w_process + w_respq))

        return {
            "score": weighted_score,
            "reward/error_type": "none",
            "reward/tool_call/judge_failed": 0,
            "reward/tool_call/judge_parse_failed": 0,
            "reward/tool_call/should_call_but_no_call": 1,
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
            # Segment scores
            "reward/tool_call/process_score": 0.0,
            "reward/tool_call/summary_score": 0.0,
            "reward/tool_call/weighted_score_norm": weighted_score_norm,
            "reward/tool_call/success@0.7": 0,
            "reward/tool_call/success@0.8": 0,
            "reward/tool_call/success@0.9": 0,
            "reward/tool_call/success@1.0": 0,
            # tool_success@ thresholds (based on tool_match_score)
            "reward/tool_call/tool_success@0.7": 0,
            "reward/tool_call/tool_success@0.8": 0,
            "reward/tool_call/tool_success@0.9": 0,
            "reward/tool_call/tool_success@1.0": 0,
            # Compatibility fields
            "reward/tool_call/tool_match_score": 0.0,
            "reward/tool_call/arg_match_score": 0.0,
            "reward/tool_call/first_step_gold_count": first_gold,
            "reward/tool_call/first_step_pred_count": 0,
            "reward/tool_call/response_quality_score": 0.0,
            "reward/tool_call/pred_tool_count": 0,
            "reward/tool_call/gold_tool_count": len(gold_flat),
            "reward/tool_call/total_score": weighted_score_norm,
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

    # Extract the tool interactions and the final response
    tool_interactions = _extract_tool_interactions(solution_str)
    final_response = _extract_final_response(solution_str)

    logger.info(f"[ComputeScore] final response: {final_response[:200] if final_response else '(empty)'}...")

    # Compute the final-response quality score
    response_quality_score, rating, _, judge_failed, parse_failed = await _compute_response_quality_score(
        question_content, tool_interactions, final_response
    )

    logger.info(f"[ComputeScore] response_quality_score={response_quality_score:.3f}, rating={rating}")
    logger.info(f"[ComputeScore] name={name_match_score:.3f}, key={key_match_score:.3f}, value={value_match_score:.3f}")

    # ========== Core computation ==========
    # Step 1: process_score (weighted sub-items, range [0,1])
    process_score = (
        w_format * format_score +
        w_name * name_match_score +
        w_key * key_match_score +
        w_value * value_match_score +
        w_parallel * parallel_score
    )

    # Step 2: summary_score = response_quality (range [0,1])
    summary_score = response_quality_score

    # Step 3: weighted_score (un-normalised, used for RL)
    weighted_score = w_process * process_score + w_respq * summary_score

    # Step 4: weighted_score_norm (normalised, used for success@)
    weighted_score_norm = weighted_score / (w_process + w_respq)

    # Step 5: success@ thresholds
    success_at_07 = 1 if weighted_score_norm >= 0.7 else 0
    success_at_08 = 1 if weighted_score_norm >= 0.8 else 0
    success_at_09 = 1 if weighted_score_norm >= 0.9 else 0
    success_at_10 = 1 if weighted_score_norm >= 1.0 - 1e-9 else 0  # float tolerance

    # tool_success@ thresholds (based on tool_match_score = name_match_score)
    tool_success_at_07 = 1 if name_match_score >= 0.7 else 0
    tool_success_at_08 = 1 if name_match_score >= 0.8 else 0
    tool_success_at_09 = 1 if name_match_score >= 0.9 else 0
    tool_success_at_10 = 1 if name_match_score >= 1.0 - 1e-9 else 0  # float tolerance

    logger.info(f"[ComputeScore] ★ v8.2 weighted: process={process_score:.4f}, summary={summary_score:.4f}")
    logger.info(f"[ComputeScore] ★ weighted_score={weighted_score:.4f}, norm={weighted_score_norm:.4f}")
    logger.info(f"[ComputeScore] ★ success@: 0.7={success_at_07}, 0.8={success_at_08}, 0.9={success_at_09}, 1.0={success_at_10}")

    denom = format_result.denom
    format_fully_passed = (denom == 0) or ((format_result.open_count == format_result.close_count) and (format_result.valid_pair_n == denom))
    format_error_type = format_result.error_type or ("none" if format_fully_passed else "partial")

    # Compatibility fields
    tool_match_score = name_match_score  # alias
    arg_match_score = (key_match_score + value_match_score) / 2.0  # compatibility

    logger.info(f"[ComputeScore] ★ score={weighted_score:.4f}, format={format_score:.3f}, name={name_match_score:.3f}, key={key_match_score:.3f}, value={value_match_score:.3f}, parallel={parallel_score:.3f}, respq={response_quality_score:.3f}")

    return {
        # score = weighted_score (used for RL training)
        "score": float(weighted_score),
        "reward/error_type": "none",
        "reward/tool_call/judge_failed": int(judge_failed),
        "reward/tool_call/judge_parse_failed": int(parse_failed),
        "reward/tool_call/should_call_but_no_call": 0,
        "reward/tool_call/format_passed": int(format_fully_passed),
        "reward/tool_call/format_error_type": "none" if format_fully_passed else format_error_type,
        "reward/tool_call/format_score": format_score,
        "reward/tool_call/tag_open_n": format_result.open_count,
        "reward/tool_call/tag_close_n": format_result.close_count,
        "reward/tool_call/tag_denom": denom,
        "reward/tool_call/tag_valid_pair_n": format_result.valid_pair_n,
        # Raw sub-item scores, before weighting
        "reward/tool_call/name_match_score": name_match_score,
        "reward/tool_call/key_match_score": key_match_score,
        "reward/tool_call/value_match_score": value_match_score,
        "reward/tool_call/parallel_score": parallel_score,
        "reward/tool_call/parallel_match": parallel_match,
        # Segment scores (process is sub-item weighted, summary is not)
        "reward/tool_call/process_score": process_score,
        "reward/tool_call/summary_score": summary_score,
        # Normalised weighted score
        "reward/tool_call/weighted_score_norm": weighted_score_norm,
        # success@ thresholds (based on weighted_score_norm)
        "reward/tool_call/success@0.7": success_at_07,
        "reward/tool_call/success@0.8": success_at_08,
        "reward/tool_call/success@0.9": success_at_09,
        "reward/tool_call/success@1.0": success_at_10,
        # tool_success@ thresholds (based on tool_match_score)
        "reward/tool_call/tool_success@0.7": tool_success_at_07,
        "reward/tool_call/tool_success@0.8": tool_success_at_08,
        "reward/tool_call/tool_success@0.9": tool_success_at_09,
        "reward/tool_call/tool_success@1.0": tool_success_at_10,
        # Compatibility fields
        "reward/tool_call/tool_match_score": tool_match_score,
        "reward/tool_call/arg_match_score": arg_match_score,
        "reward/tool_call/first_step_gold_count": first_gold,
        "reward/tool_call/first_step_pred_count": first_pred,
        "reward/tool_call/response_quality_score": response_quality_score,
        "reward/tool_call/pred_tool_count": len(format_result.tool_calls),
        "reward/tool_call/gold_tool_count": len(gold_flat),
        # total_score kept for compatibility; equals weighted_score_norm
        "reward/tool_call/total_score": weighted_score_norm,
    }


def compute_score_sync(data_source: str, solution_str: str, ground_truth: Any, extra_info=None, **kwargs) -> Dict[str, Any]:
    """Synchronous wrapper around compute_score."""
    import asyncio
    return asyncio.run(compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs))


def test_reward_function():
    """Test the v8.2 reward function."""
    def print_result(name, result):
        print(f"\n{name}")
        print(f"  Score (weighted): {result.get('score'):.4f}")
        print(f"  weighted_score_norm: {result.get('reward/tool_call/weighted_score_norm'):.4f}")
        print(f"  process_score: {result.get('reward/tool_call/process_score'):.4f}")
        print(f"  summary_score: {result.get('reward/tool_call/summary_score'):.4f}")
        print(f"  success@: 0.7={result.get('reward/tool_call/success@0.7')}, 0.8={result.get('reward/tool_call/success@0.8')}, 0.9={result.get('reward/tool_call/success@0.9')}, 1.0={result.get('reward/tool_call/success@1.0')}")

    print("=" * 60)
    print("Testing v8.2 weighted scoring + success@ thresholds")
    print("=" * 60)
    print(f"Weight config:")
    print(f"  SLCA_WEIGHT_PROCESS: {os.environ.get('SLCA_WEIGHT_PROCESS', '1.0')}")
    print(f"  SLCA_WEIGHT_RESPQ: {os.environ.get('SLCA_WEIGHT_RESPQ', '1.0')}")

    # Test 1: exact match
    s1 = '''<tool_call>
{"name": "get_weather", "arguments": {"city": "Beijing"}}
</tool_call>
<tool_response>
{"temperature": 25, "weather": "sunny"}
</tool_response>
It is sunny in Beijing today, 25 degrees.'''
    g1 = {
        "allowed_tools": ["get_weather"],
        "gold_tool_calls": [[{"name": "get_weather", "arguments": {"city": "Beijing"}}]],
        "question_content": "What is the weather in Beijing?"
    }
    print_result("Test 1: exact match (assumes response_quality=1.0)", compute_score_sync("test", s1, g1))

    # Test 2: parallel calls - count matches
    s2 = '''<tool_call>
[{"name": "get_weather", "arguments": {"city": "Beijing"}}, {"name": "get_weather", "arguments": {"city": "Shanghai"}}]
</tool_call>
<tool_response>
{"temperature": 25}
</tool_response>
<tool_response>
{"temperature": 28}
</tool_response>
Beijing 25 degrees, Shanghai 28 degrees'''
    g2 = {
        "allowed_tools": ["get_weather"],
        "gold_tool_calls": [[{"name": "get_weather", "arguments": {"city": "Beijing"}}, {"name": "get_weather", "arguments": {"city": "Shanghai"}}]],
        "question_content": "weather"
    }
    print_result("Test 2: parallel calls - count matches (2 vs 2)", compute_score_sync("test", s2, g2))

    # Test 3: parallel calls - count mismatch
    s3 = '''<tool_call>
{"name": "get_weather", "arguments": {"city": "Beijing"}}
</tool_call>
<tool_response>
{"temperature": 25}
</tool_response>
Beijing 25 degrees'''
    g3 = {
        "allowed_tools": ["get_weather"],
        "gold_tool_calls": [[{"name": "get_weather", "arguments": {"city": "Beijing"}}, {"name": "get_weather", "arguments": {"city": "Shanghai"}}]],
        "question_content": "weather"
    }
    print_result("Test 3: parallel calls - count mismatch (1 vs 2)", compute_score_sync("test", s3, g3))

    print("\nTests done!")


if __name__ == "__main__":
    os.environ["LLM_JUDGE_ENABLED"] = "false"
    globals()["LLM_JUDGE_ENABLED"] = False
    # Test weights
    os.environ["SLCA_WEIGHT_PROCESS"] = "1.0"
    os.environ["SLCA_WEIGHT_RESPQ"] = "1.0"
    test_reward_function()
