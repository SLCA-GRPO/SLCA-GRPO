"""
ToolPO baseline reward function -- unnormalised-R_action variant.

Same contract as `reward_fn_toolpo.py` (see that file for the full mapping onto
the ToolPO paper), with two differences:

1. `R_action = sum(C(a_t^call))` is NOT divided by `|gold_flat|`, so it is a
   raw count rather than a ratio, and is exported as `toolpo_action_reward`
   instead of `toolpo_correct_ratio`. `toolpo_correct_ratio` is still reported,
   but as a diagnostic only -- it does not reach the estimator.
2. It adds run-integrity hardening around the judge: a per-process concurrency
   semaphore, an explicit prompt/context token budget, frozen formula- and
   parser-identifier constants, and structured-output constraints on the
   outcome judgement.

None of the three launchers in this directory point at this file; the ToolPO
rows of the paper were produced with `reward_fn_toolpo.py`. It ships because it
is a distinct reward definition, not a cosmetic variant. Set
`REWARD_FN_PATH=<...>/reward_fn_toolpo_sum_action.py` to use it.
"""

from __future__ import annotations

import atexit
import inspect
import json
import os
import re
import logging
import asyncio
import threading
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

# ======================== Logging configuration ========================
import os as _os
from datetime import datetime as _datetime

_LOGGER_NAME = "reward_fn_toolpo"
logger = logging.getLogger(_LOGGER_NAME)

logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.propagate = False

_log_formatter = logging.Formatter(
    '[%(asctime)s][RewardFnToolPO][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

_log_dir = _os.environ.get(
    "REWARD_LOG_DIR",
    "./logs/reward_function",
)
_os.makedirs(_log_dir, exist_ok=True)
_os.chmod(_log_dir, 0o700)
_log_timestamp = _datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file_path = _os.path.join(_log_dir, f'reward_fn_toolpo_{_log_timestamp}_pid{_os.getpid()}.log')

_file_handler = logging.FileHandler(filename=_log_file_path, mode='a', encoding='utf-8')
_os.chmod(_log_file_path, 0o600)
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_log_formatter)

_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.WARNING)
_stream_handler.setFormatter(_log_formatter)

logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)


# ======================== Configuration constants ========================
# LLM judge configuration (Toucan LLM judge + DeepAgent R_succ judge)
LLM_JUDGE_BASE_URL = os.environ.get("LLM_JUDGE_BASE_URL") or "http://127.0.0.1:8016/v1"
LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL") or "gpt-oss-120b"
LLM_JUDGE_ENABLED = os.environ.get("LLM_JUDGE_ENABLED", "true").lower() in ("true", "1", "yes")
LLM_JUDGE_TIMEOUT = int(os.environ.get("LLM_JUDGE_TIMEOUT") or "600")
LLM_JUDGE_MAX_RETRIES = int(os.environ.get("LLM_JUDGE_MAX_RETRIES") or "3")
LLM_JUDGE_MAX_TOKENS = int(os.environ.get("LLM_JUDGE_MAX_TOKENS") or "2048")
LLM_JUDGE_CONTEXT_WINDOW_TOKENS = int(
    os.environ.get("LLM_JUDGE_CONTEXT_WINDOW_TOKENS") or "16384"
)
LLM_JUDGE_PROMPT_TOKEN_BUDGET = int(
    os.environ.get("LLM_JUDGE_PROMPT_TOKEN_BUDGET") or "13312"
)
LLM_JUDGE_RETRY_DELAY = float(os.environ.get("LLM_JUDGE_RETRY_DELAY") or "2.0")
LLM_JUDGE_CONTEXT_RECOVERY_ID = (
    "vllm_tokenize_detokenize_balanced_head_tail_v1"
)
if not (
    0 < LLM_JUDGE_MAX_TOKENS
    and 0 < LLM_JUDGE_PROMPT_TOKEN_BUDGET
    and LLM_JUDGE_PROMPT_TOKEN_BUDGET + LLM_JUDGE_MAX_TOKENS
    <= LLM_JUDGE_CONTEXT_WINDOW_TOKENS
):
    raise ValueError("Invalid LLM judge prompt/context token budget")
REWARD_MAX_CONCURRENT_PER_PROCESS = int(
    os.environ.get("REWARD_MAX_CONCURRENT_PER_PROCESS") or "1"
)
if REWARD_MAX_CONCURRENT_PER_PROCESS <= 0:
    raise ValueError("REWARD_MAX_CONCURRENT_PER_PROCESS must be positive")
_reward_semaphore: Optional[asyncio.Semaphore] = None
_reward_semaphore_loop: Optional[asyncio.AbstractEventLoop] = None

# Penalty configuration
JUDGE_FAILURE_SCORE = float(os.environ.get("JUDGE_FAILURE_SCORE") or "0.0")
NO_CALL_PENALTY = float(os.environ.get("NO_CALL_PENALTY") or "-0.5")

# ToolPO action-reward contract.  These values are deliberately constants
# rather than environment overrides: a formal run must not silently change the
# scientific definition after its manifest has been written.
TOOLPO_REWARD_FORMULA_VERSION = "toolpo_eq5_eq7_toucan_v2"
TOOLPO_MATCHER_ID = "toucan_gold_greedy_one_to_one_loose_v1"
TOOLPO_OUTCOME_PARSER_ID = "deepagent_exact_single_label_v1"
TOOLPO_OUTCOME_GENERATION_CONSTRAINT_ID = (
    "vllm_structured_outputs_choice_v1"
)
TOOLPO_OUTCOME_CHOICES = ("Solved", "Unsolved", "Unsure")
TOOLPO_LAMBDA_CALL = 1.0
TOOLPO_LAMBDA_FOLD = 1.0
TOOLPO_MEMORY_FOLD_ENABLED = False

# Wandb logging switch
ENABLE_WANDB_LOGGING = os.environ.get("ENABLE_WANDB_LOGGING", "true").lower() in ("true", "1", "yes")

# Score mapping (for the Toucan LLM judge)
RESPONSE_QUALITY_SCORE_MAP = {
    "very poor": 1, "poor": 2, "acceptable": 3,
    "good": 4, "excellent": 5,
}
RESPONSE_QUALITY_PARSER_ID = "toucan_response_quality_xml_rating_v1"
RESPONSE_QUALITY_PRIMARY_GENERATION = "free_form_xml"
RESPONSE_QUALITY_PARSE_FAILURE_POLICY = (
    "one structured-choice fallback, then fail closed"
)
RESPONSE_QUALITY_FALLBACK_GENERATION_CONSTRAINT_ID = (
    "vllm_structured_outputs_choice_v1"
)
RESPONSE_QUALITY_FALLBACK_CHOICES = tuple(
    "<response><response_quality><reasoning></reasoning>"
    f"<rating>{rating}</rating></response_quality></response>"
    for rating in RESPONSE_QUALITY_SCORE_MAP
)

# Regular expressions
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


class ToolPOJudgeResult(NamedTuple):
    score: float
    label: str
    request_failed: bool
    parse_failed: bool


# ======================== Loose value matching ========================
def _loose_value_equal(pred_val: Any, gold_val: Any) -> bool:
    """Loose value-equality rule."""
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
        except (TypeError, ValueError):
            pass
    return False


# ======================== Final-response extraction ========================
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
    """Extract tool calls and tool responses from solution_str in order of appearance (DeepAgent's natural interleaving)."""
    if not solution_str:
        return ""
    # Sort by position of appearance to keep the natural call -> response interleaving
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


# ======================== Format checking ========================
def _check_tool_call_format(
    text: str,
    allowed_tools: List[str],
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
) -> FormatCheckResult:
    """Check tool-call format and return the tool calls grouped by block."""
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
                first_error_message = "tool_call content is empty"
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


# ======================== Detailed scoring ========================
def _safe_parse_args(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            obj = json.loads(args)
            return obj if isinstance(obj, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def _compute_detailed_match_scores(
    pred_calls: List[ToolCall],
    gold_tool_calls_flat: List[Dict[str, Any]]
) -> Tuple[float, float, float]:
    """
    Compute the detailed match scores

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


# ======================== Parallel-call scoring ========================
def _compute_parallel_score(
    pred_tool_calls_by_block: List[List[ToolCall]],
    gold_tool_calls_nested: List[List[Dict[str, Any]]],
) -> Tuple[float, int, int]:
    """
    Compute the parallel-call score

    Only compares whether the first step has the same number of tool calls:
    - equal: return 1.0
    - not equal: return 0.0
    """
    gold_first_count = len(gold_tool_calls_nested[0]) if gold_tool_calls_nested else 0
    pred_first_count = len(pred_tool_calls_by_block[0]) if pred_tool_calls_by_block else 0

    parallel_score = 1.0 if gold_first_count == pred_first_count else 0.0

    logger.info(f"[Parallel] gold_first={gold_first_count}, pred_first={pred_first_count}, score={parallel_score:.1f}")

    return parallel_score, gold_first_count, pred_first_count


# ======================== ToolPO binary tool correctness C(a_t^call) ========================
def _compute_binary_tool_correctness(
    pred_calls: List[ToolCall],
    gold_tool_calls_flat: List[Dict[str, Any]]
) -> Tuple[float, int, int]:
    """
    ToolPO paper C(a_t^call): binary tool-call correctness

    Matching rule: for each gold call, greedily match the best pred call:
    - exact name match AND
    - every gold key present in pred AND
    - every matched key value passes the loose match (_loose_value_equal)
    -> C=1, otherwise C=0

    Returns:
        toolpo_correct_ratio: Σ C / len(gold_flat), normalised to [0,1]
        correct_count: Σ C (number of correct calls)
        total_gold: len(gold_flat) (total number of gold calls)
    """
    if not gold_tool_calls_flat:
        # No gold calls -> nothing to verify, treat as fully correct
        return 1.0, 0, 0

    if not pred_calls:
        # gold calls present but no pred calls -> all incorrect
        return 0.0, 0, len(gold_tool_calls_flat)

    correct_count = 0
    matched_pred = set()

    for gold in gold_tool_calls_flat:
        gold_name = gold.get("name", "") if isinstance(gold, dict) else str(gold)
        gold_args = _safe_parse_args(gold.get("arguments")) if isinstance(gold, dict) else {}
        gold_keys = set(gold_args.keys())

        found_match = False

        for pred_idx, pred in enumerate(pred_calls):
            if pred_idx in matched_pred:
                continue

            # Condition 1: exact name match
            if pred.name != gold_name:
                continue

            # If gold requires no arguments, a name match is enough
            if not gold_keys:
                found_match = True
                matched_pred.add(pred_idx)
                break

            pred_keys = set(pred.arguments.keys())

            # Condition 2: every gold key is present in pred
            if not gold_keys.issubset(pred_keys):
                continue

            # Condition 3: every gold key value passes the loose match
            all_values_match = all(
                _loose_value_equal(pred.arguments.get(k), gold_args.get(k))
                for k in gold_keys
            )

            if all_values_match:
                found_match = True
                matched_pred.add(pred_idx)
                break

        if found_match:
            correct_count += 1

    total_gold = len(gold_tool_calls_flat)
    toolpo_correct_ratio = correct_count / total_gold

    logger.info(f"[ToolPO C(a_t)] correct={correct_count}/{total_gold}, ratio={toolpo_correct_ratio:.3f}")

    return toolpo_correct_ratio, correct_count, total_gold


def _toolpo_reward_fields(
    *,
    r_succ: float,
    correct_count: int,
    total_gold: int,
    reward_valid: bool,
    judge_result: str,
    judge_request_failed: bool,
    judge_parse_failed: bool,
    judge_called: int,
) -> Dict[str, Any]:
    """Build the explicit Eq. 5--7 reward interface.

    ``toolpo_correct_ratio`` remains available only for diagnostics and
    backwards-readable artifacts.  The estimator contract is the canonical
    ``toolpo_action_reward`` field.
    """
    if correct_count < 0 or total_gold < 0 or correct_count > total_gold:
        raise ValueError(
            "invalid ToolPO call counts: "
            f"correct_count={correct_count}, total_gold={total_gold}"
        )
    ratio = 1.0 if total_gold == 0 else correct_count / total_gold
    call_reward = TOOLPO_LAMBDA_CALL * correct_count
    memory_pref = 0.0
    action_reward = call_reward + TOOLPO_LAMBDA_FOLD * memory_pref
    terminal_judge_failure = judge_request_failed or judge_parse_failed
    return {
        "reward/tool_call/toolpo_reward_valid": int(reward_valid),
        "reward/tool_call/toolpo_reward_formula_version": TOOLPO_REWARD_FORMULA_VERSION,
        "reward/tool_call/toolpo_matcher_id": TOOLPO_MATCHER_ID,
        "reward/tool_call/toolpo_outcome_parser_id": TOOLPO_OUTCOME_PARSER_ID,
        "reward/tool_call/toolpo_lambda_call": TOOLPO_LAMBDA_CALL,
        "reward/tool_call/toolpo_lambda_fold": TOOLPO_LAMBDA_FOLD,
        "reward/tool_call/toolpo_memory_fold_enabled": int(TOOLPO_MEMORY_FOLD_ENABLED),
        "reward/tool_call/toolpo_r_succ": float(r_succ),
        "reward/tool_call/toolpo_call_correct_count": int(correct_count),
        "reward/tool_call/toolpo_correct_count": int(correct_count),
        "reward/tool_call/toolpo_correct_ratio": float(ratio),
        "reward/tool_call/toolpo_call_reward": float(call_reward),
        "reward/tool_call/toolpo_memory_pref": float(memory_pref),
        "reward/tool_call/toolpo_action_reward": float(action_reward),
        "reward/tool_call/toolpo_rsucc_judge_result": judge_result,
        # Legacy aggregate name is retained but now means any terminal
        # request-or-parse failure.  Explicit fields disambiguate the cause.
        "reward/tool_call/toolpo_rsucc_judge_failed": int(terminal_judge_failure),
        "reward/tool_call/toolpo_rsucc_judge_request_failed": int(judge_request_failed),
        "reward/tool_call/toolpo_rsucc_judge_parse_failed": int(judge_parse_failed),
        "reward/tool_call/toolpo_rsucc_judge_called": int(judge_called),
    }


# ======================== LLM judge helpers ========================
_global_connector = None

def _get_global_connector():
    import aiohttp
    global _global_connector
    if _global_connector is None or _global_connector.closed:
        _global_connector = aiohttp.TCPConnector(limit=100, limit_per_host=50, ttl_dns_cache=300, enable_cleanup_closed=True, keepalive_timeout=60)
    return _global_connector


# ---- Toucan LLM Judge (for SLCA evaluation) ----
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


# ---- DeepAgent R_succ Judge (for ToolPO RL training) ----
# The semantic rubric below follows the DeepAgent prompts.  The disclosed
# bare-label suffix is a Toucan-side transport adaptation: the strict
# fail-closed parser cannot safely accept prose, negations, or multiple labels.
TOOLPO_EXACT_LABEL_OUTPUT_CONTRACT = """

IMPORTANT OUTPUT CONTRACT: Your entire final response must be exactly one of
these three case-sensitive bare labels: Solved, Unsolved, Unsure. Do not include
quotation marks, punctuation, explanations, reasoning, prefixes, suffixes,
Markdown, or XML."""

TOOLPO_RSUCC_PROMPT = """You are an evaluation assistant. Please determine if the answer correctly addresses the query.

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

Please provide your assessment:

IMPORTANT OUTPUT CONTRACT: Your entire final response must be exactly one of
these three case-sensitive bare labels: Solved, Unsolved, Unsure. Do not include
quotation marks, punctuation, explanations, reasoning, prefixes, suffixes,
Markdown, or XML."""

TOOLPO_RSUCC_FALLBACK_PROMPT = """You are an evaluation assistant. Please analyze the detailed answer structure to determine if the task was solved.

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

Please provide your assessment:

IMPORTANT OUTPUT CONTRACT: Your entire final response must be exactly one of
these three case-sensitive bare labels: Solved, Unsolved, Unsure. Do not include
quotation marks, punctuation, explanations, reasoning, prefixes, suffixes,
Markdown, or XML."""


def _build_llm_judge_payload(
    prompt: str,
    *,
    outcome_choice_constraint: bool = False,
    response_quality_choice_constraint: bool = False,
) -> Dict[str, Any]:
    if outcome_choice_constraint and response_quality_choice_constraint:
        raise ValueError(
            "judge payload cannot enable outcome and response-quality "
            "choice constraints together"
        )
    payload: Dict[str, Any] = {
        "model": LLM_JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": LLM_JUDGE_MAX_TOKENS,
    }
    if outcome_choice_constraint:
        # vLLM 0.11.0 applies this constraint only with synchronous
        # scheduling.  The judge deployment and the prelaunch transport probe
        # both enforce that serving-side prerequisite.
        payload["structured_outputs"] = {
            "choice": list(TOOLPO_OUTCOME_CHOICES),
        }
    elif response_quality_choice_constraint:
        payload["structured_outputs"] = {
            "choice": list(RESPONSE_QUALITY_FALLBACK_CHOICES),
        }
    return payload


def _is_context_budget_error(status: int, body: str) -> bool:
    """Recognize only the locked vLLM context-budget HTTP 400 family."""

    if status != 400:
        return False
    try:
        payload = json.loads(body)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = str(error.get("message", "")).lower()
    max_token_error = "max_tokens" in message and (
        "at least 1" in message or "too large" in message
    )
    input_context_error = (
        "maximum context length" in message
        and "input" in message
    )
    return max_token_error or input_context_error


def _balanced_head_tail_token_ids(
    token_ids: List[int],
    budget: int,
) -> List[int]:
    """Retain a deterministic balanced head/tail slice under ``budget``."""

    if budget < 2:
        raise ValueError("Judge prompt token budget must be at least two")
    if len(token_ids) <= 2:
        return list(token_ids)
    target = min(budget, len(token_ids) - 1)
    head = (target + 1) // 2
    tail = target - head
    if tail == 0:
        return list(token_ids[:head])
    return list(token_ids[:head]) + list(token_ids[-tail:])


async def _compact_judge_prompt_async(prompt: str) -> Optional[str]:
    """Use the serving tokenizer for one content-preserving recovery."""

    import aiohttp

    base_url = LLM_JUDGE_BASE_URL.rstrip("/")
    root_url = base_url[:-3] if base_url.endswith("/v1") else base_url
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    }
    timeout = aiohttp.ClientTimeout(
        total=min(LLM_JUDGE_TIMEOUT, 120),
        connect=60,
        sock_read=min(LLM_JUDGE_TIMEOUT, 120),
    )
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                root_url + "/tokenize",
                headers=headers,
                json={
                    "model": LLM_JUDGE_MODEL,
                    "prompt": prompt,
                    "add_special_tokens": False,
                },
            ) as response:
                if response.status != 200:
                    return None
                token_payload = await response.json()
            token_ids = token_payload.get("tokens", [])
            if not token_ids or not all(
                isinstance(value, int) for value in token_ids
            ):
                return None
            compact_ids = _balanced_head_tail_token_ids(
                token_ids,
                LLM_JUDGE_PROMPT_TOKEN_BUDGET,
            )
            if len(compact_ids) >= len(token_ids):
                return None
            async with session.post(
                root_url + "/detokenize",
                headers=headers,
                json={
                    "model": LLM_JUDGE_MODEL,
                    "tokens": compact_ids,
                },
            ) as response:
                if response.status != 200:
                    return None
                text_payload = await response.json()
    except (aiohttp.ClientError, TimeoutError, ValueError, TypeError):
        return None
    compact_prompt = text_payload.get("prompt")
    if not isinstance(compact_prompt, str) or not compact_prompt:
        return None
    logger.warning(
        "[LLM Judge] context recovery compacted prompt tokens "
        f"{len(token_ids)}->{len(compact_ids)}"
    )
    return compact_prompt


async def _call_llm_judge_async(
    prompt: str,
    *,
    outcome_choice_constraint: bool = False,
    response_quality_choice_constraint: bool = False,
) -> Optional[str]:
    import aiohttp
    import asyncio
    url = LLM_JUDGE_BASE_URL.rstrip("/") + "/chat/completions"
    payload = _build_llm_judge_payload(
        prompt,
        outcome_choice_constraint=outcome_choice_constraint,
        response_quality_choice_constraint=(
            response_quality_choice_constraint
        ),
    )
    headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    timeout = aiohttp.ClientTimeout(total=LLM_JUDGE_TIMEOUT, connect=60, sock_read=LLM_JUDGE_TIMEOUT)
    context_recovery_used = False

    for attempt in range(LLM_JUDGE_MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(connector=_get_global_connector(), connector_owner=False, timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        error_body = await resp.text()
                        if (
                            not context_recovery_used
                            and _is_context_budget_error(
                                resp.status,
                                error_body,
                            )
                        ):
                            compact_prompt = (
                                await _compact_judge_prompt_async(prompt)
                            )
                            if compact_prompt is None:
                                logger.warning(
                                    "[LLM Judge] context recovery failed"
                                )
                                return None
                            prompt = compact_prompt
                            payload = _build_llm_judge_payload(
                                prompt,
                                outcome_choice_constraint=(
                                    outcome_choice_constraint
                                ),
                                response_quality_choice_constraint=(
                                    response_quality_choice_constraint
                                ),
                            )
                            context_recovery_used = True
                            continue
                        logger.warning(
                            f"[LLM Judge] HTTP {resp.status} "
                            f"(attempt {attempt+1}/"
                            f"{LLM_JUDGE_MAX_RETRIES+1})"
                        )
                        if resp.status >= 500 and attempt < LLM_JUDGE_MAX_RETRIES:
                            await asyncio.sleep(LLM_JUDGE_RETRY_DELAY * (attempt + 1))
                            continue
                        return None
                    data = await resp.json()
                    if "choices" not in data or len(data["choices"]) == 0:
                        logger.warning(
                            f"[LLM Judge] response has no choices (attempt {attempt+1}), "
                            f"data_key_count={len(data)}"
                        )
                        return None
                    content = data["choices"][0]["message"]["content"]
                    if content is None:
                        finish_reason = data["choices"][0].get("finish_reason", "unknown")
                        reasoning = data["choices"][0].get("message", {}).get("reasoning_content", "")
                        usage = data.get("usage", {})
                        logger.warning(
                            f"[LLM Judge] content is None! finish_reason={finish_reason}, "
                            f"reasoning_len={len(reasoning) if reasoning else 0}, "
                            f"usage={usage}, prompt_len={len(prompt)}"
                        )
                    return content
        except Exception as e:
            logger.warning(
                f"[LLM Judge] exception (attempt {attempt+1}/{LLM_JUDGE_MAX_RETRIES+1}): "
                f"{type(e).__name__}"
            )
            if attempt < LLM_JUDGE_MAX_RETRIES:
                await asyncio.sleep(LLM_JUDGE_RETRY_DELAY * (attempt + 1))
    logger.warning(
        f"[LLM Judge] all {LLM_JUDGE_MAX_RETRIES+1} attempts failed"
    )
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
    """Compute the final-response quality score (Toucan LLM judge, used for SLCA evaluation)."""
    if not LLM_JUDGE_ENABLED:
        return 0.0, None, None, False, False

    if not final_response or len(final_response.strip()) < 10:
        logger.info("[ResponseQuality] final response is empty or too short, assigning a low score")
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
        logger.warning(
            "[ResponseQuality] primary output failed parsing; retrying with "
            "the locked structured-choice fallback"
        )
        fallback_response = await _call_llm_judge_async(
            prompt,
            response_quality_choice_constraint=True,
        )
        if fallback_response is None:
            return JUDGE_FAILURE_SCORE, None, None, True, False
        fallback_rating = _parse_response_quality_rating(fallback_response)
        if fallback_rating is None:
            return JUDGE_FAILURE_SCORE, None, fallback_response, False, True
        return (
            _normalize_score(RESPONSE_QUALITY_SCORE_MAP[fallback_rating]),
            fallback_rating,
            fallback_response,
            False,
            False,
        )
    return _normalize_score(RESPONSE_QUALITY_SCORE_MAP[rating]), rating, llm_response, False, False


# ======================== ToolPO R_succ Judge ========================
def _parse_rsucc_result(llm_response: str) -> Tuple[str, bool]:
    """
    Parse the output of the DeepAgent Solved/Unsolved/Unsure LLM judge.

    Accepts only a complete output that is exactly one canonical label. A loose substring
    search would silently score ``not solved``, multi-label, or ambiguous text as
    success, so this Toucan adaptation uses a versioned fail-closed parser.
    """
    if not isinstance(llm_response, str):
        return "Unsure", False
    match = re.fullmatch(
        r"\s*(Solved|Unsolved|Unsure)\s*",
        llm_response,
        flags=re.IGNORECASE,
    )
    if match is None:
        return "Unsure", False
    canonical = {
        "solved": "Solved",
        "unsolved": "Unsolved",
        "unsure": "Unsure",
    }
    return canonical[match.group(1).lower()], True


async def _compute_toolpo_rsucc(
    question_content: str,
    final_response: str,
    tool_interactions: str,
) -> ToolPOJudgeResult:
    """
    ToolPO R_succ: uses the DeepAgent Solved/Unsolved LLM judge

    Flow (following DeepAgent/src/evaluate/evaluate_toolbench.py):
    1. First round uses CHECK_ANSWER_STATUS_PROMPT and parses Solved/Unsolved/Unsure
    2. If Unsure -> fall back to PARSE_ANSWER_STATUS_PROMPT
    3. Solved -> 1.0, everything else -> 0.0

    Returns:
        r_succ: 0.0 or 1.0
        judge_result: "Solved" / "Unsolved" / "Unsure"
        request_failed: True only when a judge request exhausted its retries
        parse_failed: True only when the final available response has no
            recognized Solved/Unsolved/Unsure label
    """
    if not LLM_JUDGE_ENABLED:
        return ToolPOJudgeResult(0.0, "Unsure", False, False)

    if not final_response or len(final_response.strip()) < 10:
        logger.info("[ToolPO R_succ] final response is empty or too short, R_succ=0")
        return ToolPOJudgeResult(0.0, "Unsolved", False, False)

    # Round 1: CHECK_ANSWER_STATUS_PROMPT
    prompt1 = TOOLPO_RSUCC_PROMPT.format(
        query=question_content,
        answer=final_response
    )
    llm_response1 = await _call_llm_judge_async(
        prompt1,
        outcome_choice_constraint=True,
    )
    if llm_response1 is None:
        logger.warning("[ToolPO R_succ] first-round judge call failed, R_succ=0")
        return ToolPOJudgeResult(0.0, "Unsure", True, False)

    result1, parsed1 = _parse_rsucc_result(llm_response1)
    logger.info(f"[ToolPO R_succ] first-round result: {result1}")

    if parsed1 and result1 != "Unsure":
        r_succ = 1.0 if result1 == "Solved" else 0.0
        return ToolPOJudgeResult(r_succ, result1, False, False)

    # Explicit Unsure or an unrecognized first response gets one declared
    # fallback attempt.  Only the final response determines parse validity.
    # Matches the fallback format of DeepAgent evaluate_toolbench.py check_is_solved():
    # enriched_answer_payload = json.dumps({'final_answer': final_answer_text, 'output': raw_output})
    # We use tool_interactions + final_response as the raw_output equivalent
    raw_output = f"{tool_interactions}\n{final_response}" if tool_interactions else final_response
    answer_details = json.dumps({
        "final_answer": final_response,
        "output": raw_output
    }, ensure_ascii=False)
    prompt2 = TOOLPO_RSUCC_FALLBACK_PROMPT.format(
        query=question_content,
        answer=answer_details
    )
    llm_response2 = await _call_llm_judge_async(
        prompt2,
        outcome_choice_constraint=True,
    )
    if llm_response2 is None:
        logger.warning("[ToolPO R_succ] fallback judge call failed, R_succ=0")
        return ToolPOJudgeResult(0.0, "Unsure", True, False)

    result2, parsed2 = _parse_rsucc_result(llm_response2)
    logger.info(f"[ToolPO R_succ] fallback result: {result2}")

    if not parsed2:
        logger.warning("[ToolPO R_succ] fallback output has no recognizable label, marking parse failure")
        return ToolPOJudgeResult(0.0, "Unsure", False, True)
    r_succ = 1.0 if result2 == "Solved" else 0.0
    return ToolPOJudgeResult(r_succ, result2, False, False)


# ======================== Nested-list normalisation ========================
def _normalize_gold_tool_calls_nested(gold_tool_calls: Any) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Convert to a nested list and a flat list."""
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
        except (TypeError, ValueError, json.JSONDecodeError):
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
    """Build the return payload for the error case."""
    return {
        "score": score,
        "reward/error_type": error_type,
        "reward/tool_call/judge_failed": 0,
        "reward/tool_call/judge_parse_failed": 0,
        "reward/tool_call/judge_called": 0,
        "reward/tool_call/judge_bypass": 1,
        "reward/tool_call/should_call_but_no_call": 0,
        "reward/tool_call/format_passed": 0,
        "reward/tool_call/format_error_type": "error",
        "reward/tool_call/format_score": 0.0,
        "reward/tool_call/tag_open_n": 0,
        "reward/tool_call/tag_close_n": 0,
        "reward/tool_call/tag_denom": 0,
        "reward/tool_call/tag_valid_pair_n": 0,
        "reward/tool_call/call_parseable": 0,
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
        **_toolpo_reward_fields(
            r_succ=0.0,
            correct_count=0,
            total_gold=0,
            reward_valid=False,
            judge_result="error",
            judge_request_failed=False,
            judge_parse_failed=False,
            judge_called=0,
        ),
    }


# ======================== Main scoring function ========================
async def _compute_score_unthrottled(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    ToolPO baseline main scoring function

    Returns:
    - score = R_succ (binary 0/1), used for RL training -> A_global
    - toolpo_action_reward = Σ C(a_t^call), used for A_tool
    - toolpo_correct_ratio is kept as a diagnostic field only
    - plus all unified SLCA evaluation metrics
    """
    logger.info(f"[ComputeScore] ToolPO scoring started, data_source={data_source}")

    if not isinstance(ground_truth, dict):
        return _build_error_result(0.0, "invalid_ground_truth")

    # Note: allowed_tools in the RL data may be a numpy.ndarray, so `or []` is unsafe
    allowed_tools = ground_truth.get("allowed_tools")
    if allowed_tools is None:
        allowed_tools = []
    elif not isinstance(allowed_tools, list):
        allowed_tools = list(allowed_tools)  # numpy array → list
    tool_schemas = ground_truth.get("tool_schemas")
    if isinstance(tool_schemas, str):
        try:
            tool_schemas = json.loads(tool_schemas)
        except (TypeError, ValueError, json.JSONDecodeError):
            tool_schemas = None

    if not allowed_tools:
        return _build_error_result(0.0, "missing_allowed_tools")

    question_content = ground_truth.get("question_content", "")
    if not question_content and extra_info:
        question_content = extra_info.get("input", "") or extra_info.get("question", "")

    raw_gold = ground_truth.get("gold_tool_calls") or []
    gold_nested, gold_flat = _normalize_gold_tool_calls_nested(raw_gold)

    format_score, format_result = _compute_format_score(solution_str, allowed_tools, tool_schemas)

    # Read the SLCA weight configuration (used by the evaluation metrics)
    w_process = float(os.environ.get("SLCA_WEIGHT_PROCESS", "1.0"))
    w_format = float(os.environ.get("SLCA_WEIGHT_FORMAT", "0.10"))
    w_name = float(os.environ.get("SLCA_WEIGHT_NAME", "0.25"))
    w_key = float(os.environ.get("SLCA_WEIGHT_KEY", "0.15"))
    w_value = float(os.environ.get("SLCA_WEIGHT_VALUE", "0.20"))
    w_parallel = float(os.environ.get("SLCA_WEIGHT_PARALLEL", "0.30"))
    w_respq = float(os.environ.get("SLCA_WEIGHT_RESPQ", "1.0"))

    # Should call a tool but does not
    if gold_flat and format_result.open_count == 0:
        first_gold = len(gold_nested[0]) if gold_nested else 0
        return {
            # ToolPO: R_succ = 0 (no tool was called, so the task cannot have been completed)
            "score": 0.0,
            "reward/error_type": "none",
            "reward/tool_call/judge_failed": 0,
            "reward/tool_call/judge_parse_failed": 0,
            "reward/tool_call/judge_called": 0,
            "reward/tool_call/judge_bypass": 1,
            "reward/tool_call/should_call_but_no_call": 1,
            "reward/tool_call/format_passed": 1,
            "reward/tool_call/format_error_type": "no_tool_call",
            "reward/tool_call/format_score": 0.0,
            "reward/tool_call/tag_open_n": 0,
            "reward/tool_call/tag_close_n": 0,
            "reward/tool_call/tag_denom": 0,
            "reward/tool_call/tag_valid_pair_n": 0,
            "reward/tool_call/call_parseable": 0,
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
            **_toolpo_reward_fields(
                r_succ=0.0,
                correct_count=0,
                total_gold=len(gold_flat),
                reward_valid=True,
                judge_result="Unsolved",
                judge_request_failed=False,
                judge_parse_failed=False,
                judge_called=0,
            ),
        }

    # SLCA detailed scoring
    name_match_score, key_match_score, value_match_score = _compute_detailed_match_scores(
        format_result.tool_calls, gold_flat
    )

    # Parallel-call scoring
    parallel_score, first_gold, first_pred = _compute_parallel_score(
        format_result.tool_calls_by_block, gold_nested
    )
    parallel_match = 1 if parallel_score == 1.0 else 0

    # Extract tool interactions and the final response
    tool_interactions = _extract_tool_interactions(solution_str)
    final_response = _extract_final_response(solution_str)

    logger.info(f"[ComputeScore] extracted final response: {final_response[:200] if final_response else '(empty)'}...")

    # ========== ToolPO R_succ: DeepAgent Solved/Unsolved Judge ==========
    rsucc = await _compute_toolpo_rsucc(
        question_content, final_response, tool_interactions
    )
    logger.info(
        f"[ComputeScore] ★ ToolPO R_succ={rsucc.score:.0f} "
        f"(judge={rsucc.label}, request_failed={int(rsucc.request_failed)}, "
        f"parse_failed={int(rsucc.parse_failed)})"
    )

    # ========== ToolPO C(a_t^call): binary tool correctness ==========
    toolpo_correct_ratio, toolpo_correct_count, toolpo_total_gold = _compute_binary_tool_correctness(
        format_result.tool_calls, gold_flat
    )
    logger.info(f"[ComputeScore] ★ ToolPO C(a_t): correct={toolpo_correct_count}/{toolpo_total_gold}, ratio={toolpo_correct_ratio:.3f}")

    # ========== Toucan LLM Judge: response_quality (for SLCA evaluation) ==========
    response_quality_score, rating, _, judge_failed, parse_failed = await _compute_response_quality_score(
        question_content, tool_interactions, final_response
    )
    summary_judge_called = int(
        LLM_JUDGE_ENABLED and bool(final_response and len(final_response.strip()) >= 10)
    )
    rsucc_judge_called = summary_judge_called

    logger.info(f"[ComputeScore] Toucan response_quality_score={response_quality_score:.3f}, rating={rating}")
    logger.info(f"[ComputeScore] name={name_match_score:.3f}, key={key_match_score:.3f}, value={value_match_score:.3f}")

    # ========== SLCA evaluation metrics (same as the main experiment) ==========
    # Step 1: process_score (weighted sub-scores, range [0,1])
    process_score = (
        w_format * format_score +
        w_name * name_match_score +
        w_key * key_match_score +
        w_value * value_match_score +
        w_parallel * parallel_score
    )

    # Step 2: summary_score = response_quality (range [0,1])
    summary_score = response_quality_score

    # Step 3: weighted_score (unnormalised)
    weighted_score = w_process * process_score + w_respq * summary_score

    # Step 4: weighted_score_norm (normalised, used for success@)
    weighted_score_norm = weighted_score / (w_process + w_respq)

    # Step 5: success@ thresholds
    success_at_07 = 1 if weighted_score_norm >= 0.7 else 0
    success_at_08 = 1 if weighted_score_norm >= 0.8 else 0
    success_at_09 = 1 if weighted_score_norm >= 0.9 else 0
    success_at_10 = 1 if weighted_score_norm >= 1.0 - 1e-9 else 0

    # tool_success@ thresholds (based on tool_match_score = name_match_score)
    tool_success_at_07 = 1 if name_match_score >= 0.7 else 0
    tool_success_at_08 = 1 if name_match_score >= 0.8 else 0
    tool_success_at_09 = 1 if name_match_score >= 0.9 else 0
    tool_success_at_10 = 1 if name_match_score >= 1.0 - 1e-9 else 0

    logger.info(f"[ComputeScore] ★ SLCA metrics: process={process_score:.4f}, summary={summary_score:.4f}")
    logger.info(f"[ComputeScore] ★ weighted_score_norm={weighted_score_norm:.4f}")
    logger.info(f"[ComputeScore] ★ success@: 0.7={success_at_07}, 0.8={success_at_08}, 0.9={success_at_09}, 1.0={success_at_10}")

    denom = format_result.denom
    format_fully_passed = (denom == 0) or ((format_result.open_count == format_result.close_count) and (format_result.valid_pair_n == denom))
    call_parseable = int(
        format_result.open_count > 0
        and format_result.open_count == format_result.close_count
        and format_result.valid_pair_n == denom
    )
    format_error_type = format_result.error_type or ("none" if format_fully_passed else "partial")

    # Compatibility fields
    tool_match_score = name_match_score
    arg_match_score = (key_match_score + value_match_score) / 2.0

    logger.info(f"[ComputeScore] ★ score(R_succ)={rsucc.score:.0f}, format={format_score:.3f}, name={name_match_score:.3f}, key={key_match_score:.3f}, value={value_match_score:.3f}, parallel={parallel_score:.3f}, respq={response_quality_score:.3f}")

    return {
        # For ToolPO training: score = R_succ (binary 0/1)
        "score": float(rsucc.score),
        "reward/error_type": "none",
        "reward/tool_call/judge_failed": int(judge_failed),
        "reward/tool_call/judge_parse_failed": int(parse_failed),
        "reward/tool_call/judge_called": summary_judge_called,
        "reward/tool_call/judge_bypass": 1 - summary_judge_called,
        "reward/tool_call/should_call_but_no_call": 0,
        "reward/tool_call/format_passed": int(format_fully_passed),
        "reward/tool_call/format_error_type": "none" if format_fully_passed else format_error_type,
        "reward/tool_call/format_score": format_score,
        "reward/tool_call/tag_open_n": format_result.open_count,
        "reward/tool_call/tag_close_n": format_result.close_count,
        "reward/tool_call/tag_denom": denom,
        "reward/tool_call/tag_valid_pair_n": format_result.valid_pair_n,
        "reward/tool_call/call_parseable": call_parseable,
        # Sub-scores (for wandb monitoring)
        "reward/tool_call/name_match_score": name_match_score,
        "reward/tool_call/key_match_score": key_match_score,
        "reward/tool_call/value_match_score": value_match_score,
        "reward/tool_call/parallel_score": parallel_score,
        "reward/tool_call/parallel_match": parallel_match,
        # Segment-level scores (for SLCA evaluation)
        "reward/tool_call/process_score": process_score,
        "reward/tool_call/summary_score": summary_score,
        # Normalised weighted score (for SLCA evaluation)
        "reward/tool_call/weighted_score_norm": weighted_score_norm,
        # success@ thresholds (for SLCA evaluation)
        "reward/tool_call/success@0.7": success_at_07,
        "reward/tool_call/success@0.8": success_at_08,
        "reward/tool_call/success@0.9": success_at_09,
        "reward/tool_call/success@1.0": success_at_10,
        # tool_success@ thresholds
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
        **_toolpo_reward_fields(
            r_succ=rsucc.score,
            correct_count=toolpo_correct_count,
            total_gold=toolpo_total_gold,
            reward_valid=True,
            judge_result=rsucc.label,
            judge_request_failed=rsucc.request_failed,
            judge_parse_failed=rsucc.parse_failed,
            judge_called=rsucc_judge_called,
        ),
    }


def _get_reward_semaphore() -> asyncio.Semaphore:
    """Return the semaphore bound to this RewardLoopWorker's event loop."""
    global _reward_semaphore, _reward_semaphore_loop
    loop = asyncio.get_running_loop()
    if _reward_semaphore is None or _reward_semaphore_loop is not loop:
        _reward_semaphore = asyncio.Semaphore(REWARD_MAX_CONCURRENT_PER_PROCESS)
        _reward_semaphore_loop = loop
    return _reward_semaphore


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Bound active reward computations without changing scoring semantics."""
    loop = asyncio.get_running_loop()
    queued_at = loop.time()
    semaphore = _get_reward_semaphore()
    await semaphore.acquire()
    acquired_at = loop.time()
    try:
        result = await _compute_score_unthrottled(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
    finally:
        finished_at = loop.time()
        semaphore.release()
    result["reward/runtime/gate_wait_seconds"] = acquired_at - queued_at
    result["reward/runtime/compute_seconds"] = finished_at - acquired_at
    result["reward/runtime/max_concurrent_per_process"] = REWARD_MAX_CONCURRENT_PER_PROCESS
    return result


_sync_event_loop: Optional[asyncio.AbstractEventLoop] = None
_sync_event_loop_pid: Optional[int] = None
_sync_event_loop_lock = threading.Lock()


def _get_sync_event_loop() -> asyncio.AbstractEventLoop:
    """Return one persistent event loop for the traditional VERL manager.

    ``asyncio.run`` creates and closes a loop on every reward example.  The
    reusable aiohttp connector is bound to its first loop, so the second
    example would otherwise fail with ``Event loop is closed``.  The locked
    naive manager calls this wrapper serially; the lock also fails safely if a
    future caller introduces threads.
    """
    global _global_connector
    global _reward_semaphore
    global _reward_semaphore_loop
    global _sync_event_loop
    global _sync_event_loop_pid

    current_pid = os.getpid()
    if (
        _sync_event_loop is None
        or _sync_event_loop.is_closed()
        or _sync_event_loop_pid != current_pid
    ):
        # A post-fork process must not reuse a connector or semaphore bound to
        # the parent's event loop.
        _global_connector = None
        _reward_semaphore = None
        _reward_semaphore_loop = None
        _sync_event_loop = asyncio.new_event_loop()
        _sync_event_loop_pid = current_pid
    return _sync_event_loop


def compute_score_sync(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info=None,
    **kwargs,
) -> Dict[str, Any]:
    """Run the async scorer on its persistent process-local event loop."""
    with _sync_event_loop_lock:
        loop = _get_sync_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "compute_score_sync cannot run inside an active event loop"
            )
        return loop.run_until_complete(
            compute_score(
                data_source,
                solution_str,
                ground_truth,
                extra_info,
                **kwargs,
            )
        )


def _close_sync_event_loop() -> None:
    """Close the persistent connector and loop during process shutdown."""
    loop = _sync_event_loop
    if loop is None or loop.is_closed() or loop.is_running():
        return
    connector = _global_connector
    if connector is not None and not connector.closed:
        close_result = connector.close()
        if inspect.isawaitable(close_result):
            loop.run_until_complete(close_result)
    loop.close()


atexit.register(_close_sync_event_loop)


def test_reward_function():
    """Test the ToolPO reward function."""
    def print_result(name, result):
        print(f"\n{name}")
        print(f"  Score (R_succ): {result.get('score'):.1f}")
        print(f"  toolpo_r_succ: {result.get('reward/tool_call/toolpo_r_succ'):.1f}")
        print(f"  toolpo_correct_ratio: {result.get('reward/tool_call/toolpo_correct_ratio'):.3f}")
        print(f"  toolpo_correct_count: {result.get('reward/tool_call/toolpo_correct_count')}")
        print(f"  toolpo_rsucc_judge: {result.get('reward/tool_call/toolpo_rsucc_judge_result')}")
        print("  -- SLCA evaluation metrics --")
        print(f"  weighted_score_norm: {result.get('reward/tool_call/weighted_score_norm'):.4f}")
        print(f"  process_score: {result.get('reward/tool_call/process_score'):.4f}")
        print(f"  summary_score: {result.get('reward/tool_call/summary_score'):.4f}")
        print(f"  success@: 0.7={result.get('reward/tool_call/success@0.7')}, 0.8={result.get('reward/tool_call/success@0.8')}, 0.9={result.get('reward/tool_call/success@0.9')}, 1.0={result.get('reward/tool_call/success@1.0')}")

    print("=" * 60)
    print("Testing the ToolPO reward function")
    print("=" * 60)

    # Test 1: exact match
    s1 = '''<tool_call>
{"name": "get_weather", "arguments": {"city": "Beijing"}}
</tool_call>
<tool_response>
{"temperature": 25, "weather": "sunny"}
</tool_response>
Beijing is sunny today, 25 degrees!'''
    g1 = {
        "allowed_tools": ["get_weather"],
        "gold_tool_calls": [[{"name": "get_weather", "arguments": {"city": "Beijing"}}]],
        "question_content": "What is the weather in Beijing?"
    }
    print_result("Test 1: exact match (LLM judge disabled, R_succ=0)", compute_score_sync("test", s1, g1))

    # Test 2: tool-name mismatch
    s2 = '''<tool_call>
{"name": "get_weather", "arguments": {"city": "Beijing"}}
</tool_call>
<tool_response>
{"temperature": 25}
</tool_response>
Beijing is 25 degrees'''
    g2 = {
        "allowed_tools": ["get_weather", "get_news"],
        "gold_tool_calls": [[{"name": "get_news", "arguments": {"topic": "technology"}}]],
        "question_content": "Any technology news today?"
    }
    print_result("Test 2: tool-name mismatch, C(a_t)=0", compute_score_sync("test", s2, g2))

    # Test 3: parallel calls - exact match
    s3 = '''<tool_call>
[{"name": "get_weather", "arguments": {"city": "Beijing"}}, {"name": "get_weather", "arguments": {"city": "Shanghai"}}]
</tool_call>
<tool_response>
[{"temperature": 25}, {"temperature": 28}]
</tool_response>
Beijing is 25 degrees, Shanghai is 28 degrees'''
    g3 = {
        "allowed_tools": ["get_weather"],
        "gold_tool_calls": [[{"name": "get_weather", "arguments": {"city": "Beijing"}}, {"name": "get_weather", "arguments": {"city": "Shanghai"}}]],
        "question_content": "weather"
    }
    print_result("Test 3: parallel calls, exact match, C(a_t)=1.0", compute_score_sync("test", s3, g3))

    # Test 4: partial match
    s4 = '''<tool_call>
{"name": "get_weather", "arguments": {"city": "Beijing"}}
</tool_call>
<tool_response>
{"temperature": 25}
</tool_response>
Beijing is 25 degrees'''
    g4 = {
        "allowed_tools": ["get_weather"],
        "gold_tool_calls": [[{"name": "get_weather", "arguments": {"city": "Beijing"}}, {"name": "get_weather", "arguments": {"city": "Shanghai"}}]],
        "question_content": "Beijing and Shanghai weather"
    }
    print_result("Test 4: partial match (1/2 gold), C(a_t)=0.5", compute_score_sync("test", s4, g4))

    # Test 5: should call a tool but does not
    s5 = '''I am not sure about the weather in Beijing, you can check a weather forecast.'''
    g5 = {
        "allowed_tools": ["get_weather"],
        "gold_tool_calls": [[{"name": "get_weather", "arguments": {"city": "Beijing"}}]],
        "question_content": "Beijing weather"
    }
    print_result("Test 5: should call a tool but does not", compute_score_sync("test", s5, g5))

    print("\nTests done!")


if __name__ == "__main__":
    os.environ["LLM_JUDGE_ENABLED"] = "false"
    globals()["LLM_JUDGE_ENABLED"] = False
    os.environ["SLCA_WEIGHT_PROCESS"] = "1.0"
    os.environ["SLCA_WEIGHT_RESPQ"] = "1.0"
    test_reward_function()
