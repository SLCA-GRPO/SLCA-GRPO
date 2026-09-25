"""
Execution-based reward function.

Control for the question "does segment-locked routing only help when the reward
is gold-matching?". Relative to the HierR reward it swaps the process term for
an execution-outcome term:

    R_tool = R_succ      (DeepAgent Solved / Unsolved LLM judge, binary 0/1)
    R_sum  = S_summary   (Toucan LLM judge, 5-point scale, unchanged)
    score  = w_process * R_succ + w_respq * S_summary

Under SLCA the two terms are routed separately -- `process_score = R_succ`
drives A^tool and `summary_score = S_summary` drives A^sum. Under plain GRPO
only `score` is read and one advantage is broadcast to every token, which is
exactly the contrast `train_exec_slca.sh` vs `train_exec_grpo.sh` measures.

The gold-matching metrics are still computed and logged to Weights & Biases for
comparison, but they do not enter the RL objective.
"""

from __future__ import annotations

import json
import os
import re
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

# ======================== Logging configuration ========================
import os as _os
from datetime import datetime as _datetime

_LOGGER_NAME = "reward_fn_exec_based"
logger = logging.getLogger(_LOGGER_NAME)

logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.propagate = False

_log_formatter = logging.Formatter(
    '[%(asctime)s][RewardFnExecBased][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

_log_dir = os.environ.get("REWARD_LOG_DIR", "./logs/reward_function")
_os.makedirs(_log_dir, exist_ok=True)
_log_timestamp = _datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file_path = _os.path.join(_log_dir, f'reward_fn_exec_based_{_log_timestamp}_pid{_os.getpid()}.log')

_file_handler = logging.FileHandler(filename=_log_file_path, mode='a', encoding='utf-8')
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_log_formatter)

_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.WARNING)
_stream_handler.setFormatter(_log_formatter)

logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)


# ======================== Configuration constants ========================
LLM_JUDGE_BASE_URL = os.environ.get("LLM_JUDGE_BASE_URL") or "http://127.0.0.1:8016/v1"
LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL") or "gpt-oss-120b"
LLM_JUDGE_ENABLED = os.environ.get("LLM_JUDGE_ENABLED", "true").lower() in ("true", "1", "yes")
LLM_JUDGE_TIMEOUT = int(os.environ.get("LLM_JUDGE_TIMEOUT") or "600")
LLM_JUDGE_MAX_RETRIES = int(os.environ.get("LLM_JUDGE_MAX_RETRIES") or "3")
LLM_JUDGE_RETRY_DELAY = float(os.environ.get("LLM_JUDGE_RETRY_DELAY") or "2.0")

JUDGE_FAILURE_SCORE = float(os.environ.get("JUDGE_FAILURE_SCORE") or "0.0")
NO_CALL_PENALTY = float(os.environ.get("NO_CALL_PENALTY") or "-0.5")

ENABLE_WANDB_LOGGING = os.environ.get("ENABLE_WANDB_LOGGING", "true").lower() in ("true", "1", "yes")

RESPONSE_QUALITY_SCORE_MAP = {
    "very poor": 1, "poor": 2, "acceptable": 3,
    "good": 4, "excellent": 5,
}

TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
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
        except Exception:
            pass
    return False


# ======================== Final-response extraction ========================
def _extract_final_response(solution_str: str) -> str:
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
    if not solution_str:
        return ""
    combined_pattern = re.compile(
        r"(<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>)",
        re.DOTALL,
    )
    interactions = []
    call_idx = 0
    resp_idx = 0
    for m in combined_pattern.finditer(solution_str):
        block = m.group(0)
        if block.startswith("<tool_call>"):
            call_idx += 1
            content = TOOL_CALL_PATTERN.search(block)
            interactions.append(f"[Tool Call {call_idx}]")
            interactions.append((content.group(1) if content else block).strip())
            interactions.append("")
        else:
            resp_idx += 1
            content = TOOL_RESPONSE_PATTERN.search(block)
            interactions.append(f"[Tool Response {resp_idx}]")
            interactions.append((content.group(1) if content else block).strip())
            interactions.append("")
    return "\n".join(interactions).strip()


# ======================== Format check functions ========================
def _check_tool_call_format(
    text: str,
    allowed_tools: List[str],
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
) -> FormatCheckResult:
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
                first_error_message = "JSON must be an object or an array"
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
    text_clean = THINK_PATTERN.sub("", text)
    result = _check_tool_call_format(text_clean, allowed_tools, tool_schemas)
    if result.denom <= 0:
        return 0.0, result
    ratio = float(result.valid_pair_n) / float(result.denom)
    return ratio, result


# ======================== Tool-name F1 computation ========================
def _compute_name_f1(pred_names: List[str], gold_names: List[str]) -> float:
    if not pred_names and not gold_names:
        return 1.0
    if not pred_names or not gold_names:
        return 0.0
    pred_counter = Counter(pred_names)
    gold_counter = Counter(gold_names)
    intersection = sum(min(pred_counter.get(t, 0), gold_counter.get(t, 0)) for t in set(pred_counter) | set(gold_counter))
    return (2.0 * intersection) / (len(pred_names) + len(gold_names))


# ======================== Fine-grained scores (monitoring only) ========================
def _safe_parse_args(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            obj = json.loads(args)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _compute_detailed_match_scores(
    pred_calls: List[ToolCall],
    gold_tool_calls_flat: List[Dict[str, Any]]
) -> Tuple[float, float, float]:
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


# ======================== Parallel-call score (monitoring only) ========================
def _compute_parallel_score(
    pred_tool_calls_by_block: List[List[ToolCall]],
    gold_tool_calls_nested: List[List[Dict[str, Any]]],
) -> Tuple[float, int, int]:
    gold_first_count = len(gold_tool_calls_nested[0]) if gold_tool_calls_nested else 0
    pred_first_count = len(pred_tool_calls_by_block[0]) if pred_tool_calls_by_block else 0
    parallel_score = 1.0 if gold_first_count == pred_first_count else 0.0
    return parallel_score, gold_first_count, pred_first_count


# ======================== LLM judge helpers ========================
_global_connector = None

def _get_global_connector():
    import aiohttp
    global _global_connector
    if _global_connector is None or _global_connector.closed:
        _global_connector = aiohttp.TCPConnector(limit=100, limit_per_host=50, ttl_dns_cache=300, enable_cleanup_closed=True, keepalive_timeout=60)
    return _global_connector


# ---- Toucan LLM Judge (S_summary) ----
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


# ---- DeepAgent R_succ Judge (execution-based tool reward) ----
RSUCC_CHECK_PROMPT = """You are an evaluation assistant. Please determine if the answer correctly addresses the query.

Query: {query}

Answer: {answer}

Please analyze if the answer is correct, incorrect, or unclear. Consider:
1. Does the answer directly address the query?
2. Is the answer accurate and complete?
3. Does the answer provide the requested information or solution?

Respond with one of the following:
- "Solved": The answer correctly and completely addresses the query
- "Unsolved": The answer is incorrect or does not address the query
- "Unsure": The answer is unclear or partially addresses the query

Please provide your assessment:"""

RSUCC_FALLBACK_PROMPT = """You are an evaluation assistant. Please analyze the detailed answer structure to determine if the task was solved.

Query: {query}

Answer Details: {answer}

Please examine the answer structure, tool usage, and final response to determine if the task was successfully completed.

Consider:
1. Were appropriate tools used correctly?
2. Did the agent follow a logical reasoning process?
3. Was a final answer provided that addresses the query?
4. Were there any errors or incomplete steps?

Respond with one of the following:
- "Solved": The task was successfully completed with appropriate tool usage and a correct final answer
- "Unsolved": The task was not completed due to errors, incorrect tool usage, or missing final answer
- "Unsure": The completion status is unclear due to partial information or ambiguous results

Please provide your assessment:"""


async def _call_llm_judge_async(prompt: str) -> Optional[str]:
    import aiohttp
    import asyncio
    url = LLM_JUDGE_BASE_URL.rstrip("/") + "/chat/completions"
    prompt_preview = prompt[:200].replace('\n', ' ')
    payload = {"model": LLM_JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 8192}
    headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    timeout = aiohttp.ClientTimeout(total=LLM_JUDGE_TIMEOUT, connect=60, sock_read=LLM_JUDGE_TIMEOUT)

    for attempt in range(LLM_JUDGE_MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(connector=_get_global_connector(), connector_owner=False, timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        err_body = await resp.text()
                        logger.warning(
                            f"[LLM Judge] HTTP {resp.status} (attempt {attempt+1}/{LLM_JUDGE_MAX_RETRIES+1}), "
                            f"url={url}, body={err_body[:300]}, prompt={prompt_preview}"
                        )
                        if resp.status >= 500 and attempt < LLM_JUDGE_MAX_RETRIES:
                            await asyncio.sleep(LLM_JUDGE_RETRY_DELAY * (attempt + 1))
                            continue
                        return None
                    data = await resp.json()
                    if "choices" not in data or len(data["choices"]) == 0:
                        logger.warning(f"[LLM Judge] response has no choices (attempt {attempt+1})")
                        return None
                    content = data["choices"][0]["message"]["content"]
                    if content is None:
                        finish_reason = data["choices"][0].get("finish_reason", "unknown")
                        logger.warning(f"[LLM Judge] content is None! finish_reason={finish_reason}")
                    return content
        except Exception as e:
            logger.warning(
                f"[LLM Judge] exception (attempt {attempt+1}/{LLM_JUDGE_MAX_RETRIES+1}): "
                f"{type(e).__name__}: {str(e)[:200]}"
            )
            if attempt < LLM_JUDGE_MAX_RETRIES:
                await asyncio.sleep(LLM_JUDGE_RETRY_DELAY * (attempt + 1))
    logger.warning(f"[LLM Judge] all {LLM_JUDGE_MAX_RETRIES+1} attempts failed")
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
    """S_summary: Toucan LLM judge, 5-point rating"""
    if not LLM_JUDGE_ENABLED:
        return 0.0, None, None, False, False

    if not final_response or len(final_response.strip()) < 10:
        logger.info("[ResponseQuality] final response empty or too short, low score assigned")
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


# ======================== R_succ Judge ========================
def _parse_rsucc_result(llm_response: str) -> str:
    if not llm_response:
        return "Unsure"
    text = llm_response.strip().lower()
    if "solved" in text and "unsolved" not in text:
        return "Solved"
    if "unsolved" in text:
        return "Unsolved"
    if "unsure" in text:
        return "Unsure"
    first_line = llm_response.strip().split("\n")[0].strip().lower()
    if "solved" in first_line and "unsolved" not in first_line:
        return "Solved"
    if "unsolved" in first_line:
        return "Unsolved"
    return "Unsure"


async def _compute_rsucc(
    question_content: str,
    final_response: str,
    tool_interactions: str,
) -> Tuple[float, str]:
    """R_succ: DeepAgent Solved/Unsolved LLM judge -> binary 0/1"""
    if not LLM_JUDGE_ENABLED:
        return 0.0, "Unsure"

    if not final_response or len(final_response.strip()) < 10:
        logger.info("[R_succ] final response empty or too short, R_succ=0")
        return 0.0, "Unsolved"

    # Round 1: CHECK_ANSWER_STATUS_PROMPT
    prompt1 = RSUCC_CHECK_PROMPT.format(query=question_content, answer=final_response)
    llm_response1 = await _call_llm_judge_async(prompt1)
    if llm_response1 is None:
        logger.warning("[R_succ] first-round judge call failed, R_succ=0")
        return 0.0, "Unsure"

    result1 = _parse_rsucc_result(llm_response1)
    logger.info(f"[R_succ] first-round result: {result1}")

    if result1 != "Unsure":
        r_succ = 1.0 if result1 == "Solved" else 0.0
        return r_succ, result1

    # Unsure → fallback
    raw_output = f"{tool_interactions}\n{final_response}" if tool_interactions else final_response
    answer_details = json.dumps({"final_answer": final_response, "output": raw_output}, ensure_ascii=False)
    prompt2 = RSUCC_FALLBACK_PROMPT.format(query=question_content, answer=answer_details)
    llm_response2 = await _call_llm_judge_async(prompt2)
    if llm_response2 is None:
        logger.warning("[R_succ] fallback judge call failed, R_succ=0")
        return 0.0, "Unsure"

    result2 = _parse_rsucc_result(llm_response2)
    logger.info(f"[R_succ] fallback result: {result2}")

    r_succ = 1.0 if result2 == "Solved" else 0.0
    return r_succ, result2


# ======================== Nested list format handling ========================
def _normalize_gold_tool_calls_nested(gold_tool_calls: Any) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
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
        except Exception:
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
        "reward/tool_call/r_succ": 0.0,
        "reward/tool_call/rsucc_judge_result": "error",
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
    Execution-based reward: main scoring function

    Core formula:
      process_score = R_succ (binary 0/1)  <- execution-based, not gold-matching
      summary_score = S_summary (LLM judge, 5-point rating)
      score = w_process × process_score + w_respq × summary_score

    SLCA reads process_score/summary_score for segment-level routing;
    GRPO reads only score and broadcasts a single advantage uniformly.
    """
    logger.info(f"[ComputeScore] ExecBased scoring started, data_source={data_source}")

    if not isinstance(ground_truth, dict):
        return _build_error_result(0.0, "invalid_ground_truth")

    allowed_tools = ground_truth.get("allowed_tools")
    if allowed_tools is None:
        allowed_tools = []
    elif not isinstance(allowed_tools, list):
        allowed_tools = list(allowed_tools)
    tool_schemas = ground_truth.get("tool_schemas")
    if isinstance(tool_schemas, str):
        try:
            tool_schemas = json.loads(tool_schemas)
        except Exception:
            tool_schemas = None

    if not allowed_tools:
        return _build_error_result(0.0, "missing_allowed_tools")

    question_content = ground_truth.get("question_content", "")
    if not question_content and extra_info:
        question_content = extra_info.get("input", "") or extra_info.get("question", "")

    raw_gold = ground_truth.get("gold_tool_calls") or []
    gold_nested, gold_flat = _normalize_gold_tool_calls_nested(raw_gold)

    format_score, format_result = _compute_format_score(solution_str, allowed_tools, tool_schemas)

    # Read the weight configuration
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
        # No tool calls → R_succ = 0, S_summary = 0
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
            "reward/tool_call/process_score": 0.0,
            "reward/tool_call/summary_score": 0.0,
            "reward/tool_call/weighted_score_norm": weighted_score_norm,
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
            "reward/tool_call/total_score": weighted_score_norm,
            "reward/tool_call/r_succ": 0.0,
            "reward/tool_call/rsucc_judge_result": "Unsolved",
        }

    # ========== gold-matching metrics (monitoring only, not used by RL) ==========
    name_match_score, key_match_score, value_match_score = _compute_detailed_match_scores(
        format_result.tool_calls, gold_flat
    )
    parallel_score, first_gold, first_pred = _compute_parallel_score(
        format_result.tool_calls_by_block, gold_nested
    )
    parallel_match = 1 if parallel_score == 1.0 else 0

    # gold-matching process_score (monitoring only)
    gold_process_score = (
        w_format * format_score +
        w_name * name_match_score +
        w_key * key_match_score +
        w_value * value_match_score +
        w_parallel * parallel_score
    )

    # Extract tool interactions and the final response
    tool_interactions = _extract_tool_interactions(solution_str)
    final_response = _extract_final_response(solution_str)

    logger.info(f"[ComputeScore] extracted final response: {final_response[:200] if final_response else '(empty)'}...")

    # ========== R_succ: execution-based tool reward ==========
    r_succ, rsucc_judge_result = await _compute_rsucc(
        question_content, final_response, tool_interactions
    )
    logger.info(f"[ComputeScore] ★ R_succ={r_succ:.0f} (judge={rsucc_judge_result})")

    # ========== S_summary: Toucan LLM Judge ==========
    response_quality_score, rating, _, judge_failed, parse_failed = await _compute_response_quality_score(
        question_content, tool_interactions, final_response
    )
    logger.info(f"[ComputeScore] S_summary={response_quality_score:.3f}, rating={rating}")

    # ========== Execution-based core computation ==========
    # process_score = R_succ (binary 0/1) - execution-based
    process_score = r_succ

    # summary_score = S_summary (LLM judge, 5-point rating)
    summary_score = response_quality_score

    # weighted_score (used for RL training)
    weighted_score = w_process * process_score + w_respq * summary_score

    # weighted_score_norm (normalised, used for success@)
    weighted_score_norm = weighted_score / (w_process + w_respq)

    # success@ thresholds
    success_at_07 = 1 if weighted_score_norm >= 0.7 else 0
    success_at_08 = 1 if weighted_score_norm >= 0.8 else 0
    success_at_09 = 1 if weighted_score_norm >= 0.9 else 0
    success_at_10 = 1 if weighted_score_norm >= 1.0 - 1e-9 else 0

    # tool_success@ thresholds (based on name_match_score - monitoring only)
    tool_success_at_07 = 1 if name_match_score >= 0.7 else 0
    tool_success_at_08 = 1 if name_match_score >= 0.8 else 0
    tool_success_at_09 = 1 if name_match_score >= 0.9 else 0
    tool_success_at_10 = 1 if name_match_score >= 1.0 - 1e-9 else 0

    logger.info(f"[ComputeScore] ★ ExecBased: process(R_succ)={process_score:.0f}, summary={summary_score:.4f}")
    logger.info(f"[ComputeScore] ★ weighted_score={weighted_score:.4f}, norm={weighted_score_norm:.4f}")
    logger.info(f"[ComputeScore] gold_process(monitoring)={gold_process_score:.4f}")

    denom = format_result.denom
    format_fully_passed = (denom == 0) or ((format_result.open_count == format_result.close_count) and (format_result.valid_pair_n == denom))
    format_error_type = format_result.error_type or ("none" if format_fully_passed else "partial")

    tool_match_score = name_match_score
    arg_match_score = (key_match_score + value_match_score) / 2.0

    return {
        # score = w_process × R_succ + w_respq × S_summary
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
        # gold-matching sub-scores (monitoring only, no effect on RL)
        "reward/tool_call/name_match_score": name_match_score,
        "reward/tool_call/key_match_score": key_match_score,
        "reward/tool_call/value_match_score": value_match_score,
        "reward/tool_call/parallel_score": parallel_score,
        "reward/tool_call/parallel_match": parallel_match,
        # SLCA routing: process_score = R_succ, summary_score = S_summary
        "reward/tool_call/process_score": process_score,
        "reward/tool_call/summary_score": summary_score,
        # Normalised weighted score
        "reward/tool_call/weighted_score_norm": weighted_score_norm,
        # success@ thresholds
        "reward/tool_call/success@0.7": success_at_07,
        "reward/tool_call/success@0.8": success_at_08,
        "reward/tool_call/success@0.9": success_at_09,
        "reward/tool_call/success@1.0": success_at_10,
        # tool_success@ thresholds (monitoring only)
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
        "reward/tool_call/total_score": weighted_score_norm,
        # R_succ-specific fields
        "reward/tool_call/r_succ": float(r_succ),
        "reward/tool_call/rsucc_judge_result": rsucc_judge_result,
    }
