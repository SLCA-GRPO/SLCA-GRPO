#!/usr/bin/env python3
"""
End-to-end smoke test for the HierR reward on two hand-written rollouts.

Sample 1 is a correct tool call, sample 2 is wrong; the test prints both score
breakdowns side by side so you can see which sub-terms react.

Exercises `compute_score` from `rl/slca_grpo/reward_fn.py`. It is an
*integration* test: it needs `aiohttp` and a reachable summary judge, because
the summary term is scored by an LLM.

    bash rl/judge/serve_llm_judge_gpt_oss.sh            # in another shell
    python tests/test_two_samples.py

Override the endpoint with LLM_JUDGE_BASE_URL / LLM_JUDGE_MODEL, or point the
test at a different reward implementation by editing the sys.path insert below.

The vendored verl also carries a pure-CPU unit test for the advantage estimator
itself, which needs no server:

    pytest verl/tests/trainer/ppo/test_slca_grpo_on_cpu.py
"""

import os
import sys
import asyncio
import json

# Set environment variables
os.environ["LLM_JUDGE_ENABLED"] = "true"
os.environ.setdefault("LLM_JUDGE_BASE_URL", "http://127.0.0.1:8016/v1")
os.environ["LLM_JUDGE_MODEL"] = "gpt-oss-120b"
os.environ["LLM_JUDGE_TIMEOUT"] = "540"

# SLCA weight configuration
os.environ["SLCA_WEIGHT_PROCESS"] = "1.0"
os.environ["SLCA_WEIGHT_FORMAT"] = "0.10"
os.environ["SLCA_WEIGHT_NAME"] = "0.25"
os.environ["SLCA_WEIGHT_KEY"] = "0.15"
os.environ["SLCA_WEIGHT_VALUE"] = "0.20"
os.environ["SLCA_WEIGHT_PARALLEL"] = "0.30"
os.environ["SLCA_WEIGHT_RESPQ"] = "1.0"

# Import the reward function (new-prompt version)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "rl", "slca_grpo"))
from reward_fn import compute_score

# ============================================================================
# Sample 1: correct tool call
# ============================================================================
SAMPLE_1_SOLUTION = """<think>
The user wants to calculate the sine of π/4 radians, but explicitly asked to convert it to degrees first, and finally square the result.
</think>
<tool_call>
[{"name": "advanced-calculator-server-radians_to_degrees", "arguments": {"radians": 0.7853981633974483}}]
</tool_call>
<tool_response>
{"degrees": 45}
</tool_response>
<think>
The angle is 45 degrees. Now I need to calculate the sine of the angle. I will use the radian value for the calculation to match the tool's requirements.
</think>
<tool_call>
[{"name": "advanced-calculator-server-sin", "arguments": {"x": 0.7853981633974483}}]
</tool_call>
<tool_response>
{"result": 0.7071067811865475}
</tool_response>
<think>
Now I have the sine value (approx 0.707). The final instruction is to raise this result to the power of 2.
</think>
<tool_call>
[{"name": "advanced-calculator-server-power", "arguments": {"base": 0.7071067811865475, "exponent": 2}}]
</tool_call>
<tool_response>
{"result": 0.5}
</tool_response>
Here is the step-by-step calculation based on your requirements:
1. Convert Radians to Degrees:
The angle $\\pi/4$ radians converts to exactly 45 degrees.
2. Calculate Sine:
The sine of the angle (sin($45^\\circ$) or sin($\\pi/4$)) is approximately 0.70710678.
3. Square the Result:
Raising the sine value to the power of 2:
$0.70710678^2 = 0.5$
The final result is 0.5."""

# ============================================================================
# Sample 2: wrong tool call (degrees_to_radians instead of radians_to_degrees)
# ============================================================================
SAMPLE_2_SOLUTION = """<think>
I'll help you calculate this step by step. First, let me convert π/4 radians to degrees, then find the sine of the original angle (π/4), and finally square the result.
</think>

<tool_call>
[{"name": "advanced-calculator-server-degrees_to_radians", "arguments": {"degrees": 45}}]
</tool_call>
<tool_response>
{"radians": 0.7853981633974483}
</tool_response>
<think>
Now let me calculate the sine of π/4 radians (which is approximately 0.7854 radians):
</think>

<tool_call>
[{"name": "advanced-calculator-server-sin", "arguments": {"x": 0.7853981633974483}}]
</tool_call>
<tool_response>
{"result": 0.7071067811865475}
</tool_response>
<think>
Now I'll square this result to get the final value:
</think>

<tool_call>
[{"name": "advanced-calculator-server-power", "arguments": {"base": 0.7071067811865475, "exponent": 2}}]
</tool_call>
<tool_response>
{"result": 0.5}
</tool_response>
Perfect! Here's the step-by-step calculation:

1. **Convert π/4 radians to degrees**:
   - π/4 radians = 45 degrees
   - (Using the calculator: 45 degrees = 0.7853981633974483 radians)

2. **Calculate the sine of π/4 radians**:
   - sin(π/4) = sin(0.7853981633974483 radians) = 0.7071067811865475

3. **Square the result**:
   - (0.7071067811865475)² = 0.5

Therefore, the final result is **0.5**.

**Note**: While the problem asked to "convert it to degrees, then raise the result to the power of 2", this interpretation would typically mean converting the angle to degrees and then squaring that degree value (45² = 2025), but the standard mathematical interpretation is to find the sine of the angle first and then square the result, which is what I calculated."""

# ============================================================================
# Ground truth (reference answer)
# ============================================================================
GROUND_TRUTH = {
    "allowed_tools": [
        "advanced-calculator-server-radians_to_degrees",
        "advanced-calculator-server-degrees_to_radians",
        "advanced-calculator-server-sin",
        "advanced-calculator-server-cos",
        "advanced-calculator-server-tan",
        "advanced-calculator-server-power",
        "advanced-calculator-server-sqrt",
        "advanced-calculator-server-add",
        "advanced-calculator-server-subtract",
        "advanced-calculator-server-multiply",
        "advanced-calculator-server-divide",
    ],
    "gold_tool_calls": [
        {"name": "advanced-calculator-server-radians_to_degrees", "arguments": {"radians": 0.7853981633974483}},
        {"name": "advanced-calculator-server-sin", "arguments": {"x": 0.7853981633974483}},
        {"name": "advanced-calculator-server-power", "arguments": {"base": 0.7071067811865475, "exponent": 2}},
    ],
    "question_content": "I need to find the sine of an angle given in radians, but first convert it to degrees, then raise the result to the power of 2. The angle in radians is π/4. Please calculate the final result.",
}


def print_result(name: str, result: dict):
    """Print the scoring result"""
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")

    print("\n[Core scores]")
    print(f"  score (weighted_score):     {result.get('score', 'N/A'):.4f}")
    print(f"  weighted_score_norm:        {result.get('reward/tool_call/weighted_score_norm', 'N/A'):.4f}")
    print(f"  process_score:              {result.get('reward/tool_call/process_score', 'N/A'):.4f}")
    print(f"  summary_score:              {result.get('reward/tool_call/summary_score', 'N/A'):.4f}")

    print("\n[Process sub-scores]")
    print(f"  format_score:               {result.get('reward/tool_call/format_score', 'N/A'):.4f}")
    print(f"  name_match_score:           {result.get('reward/tool_call/name_match_score', 'N/A'):.4f}")
    print(f"  key_match_score:            {result.get('reward/tool_call/key_match_score', 'N/A'):.4f}")
    print(f"  value_match_score:          {result.get('reward/tool_call/value_match_score', 'N/A'):.4f}")
    print(f"  parallel_score:             {result.get('reward/tool_call/parallel_score', 'N/A'):.4f}")

    print("\n[Parallel details]")
    print(f"  first_step_gold_count:      {result.get('reward/tool_call/first_step_gold_count', 'N/A')}")
    print(f"  first_step_pred_count:      {result.get('reward/tool_call/first_step_pred_count', 'N/A')}")
    print(f"  parallel_match:             {result.get('reward/tool_call/parallel_match', 'N/A')}")

    print("\n【Response Quality (LLM Judge - New Prompt)】")
    print(f"  response_quality_score:     {result.get('reward/tool_call/response_quality_score', 'N/A'):.4f}")
    print(f"  judge_failed:               {result.get('reward/tool_call/judge_failed', 'N/A')}")
    print(f"  judge_parse_failed:         {result.get('reward/tool_call/judge_parse_failed', 'N/A')}")
    print(f"  no_tool_response:           {result.get('reward/tool_call/no_tool_response', 'N/A')}")

    print("\n[Success@ thresholds]")
    print(f"  success@0.7:                {result.get('reward/tool_call/success@0.7', 'N/A')}")
    print(f"  success@0.8:                {result.get('reward/tool_call/success@0.8', 'N/A')}")
    print(f"  success@0.9:                {result.get('reward/tool_call/success@0.9', 'N/A')}")
    print(f"  success@1.0:                {result.get('reward/tool_call/success@1.0', 'N/A')}")

    print("\n[Tool Success@ thresholds]")
    print(f"  tool_success@0.7:           {result.get('reward/tool_call/tool_success@0.7', 'N/A')}")
    print(f"  tool_success@0.8:           {result.get('reward/tool_call/tool_success@0.8', 'N/A')}")
    print(f"  tool_success@0.9:           {result.get('reward/tool_call/tool_success@0.9', 'N/A')}")
    print(f"  tool_success@1.0:           {result.get('reward/tool_call/tool_success@1.0', 'N/A')}")

    print("\n[Other info]")
    print(f"  pred_tool_count:            {result.get('reward/tool_call/pred_tool_count', 'N/A')}")
    print(f"  gold_tool_count:            {result.get('reward/tool_call/gold_tool_count', 'N/A')}")
    print(f"  format_passed:              {result.get('reward/tool_call/format_passed', 'N/A')}")
    print(f"  format_error_type:          {result.get('reward/tool_call/format_error_type', 'N/A')}")
    print(f"  should_call_but_no_call:    {result.get('reward/tool_call/should_call_but_no_call', 'N/A')}")
    print(f"  total_score:                {result.get('reward/tool_call/total_score', 'N/A'):.4f}")


async def main():
    print("=" * 80)
    print("  Reward function scores for two rollout samples")
    print("  Using: rl/slca_grpo/reward_fn.py")
    print("  LLM Judge: ENABLED (New Prompt - response quality only)")
    print("=" * 80)

    print("\n[Weight configuration]")
    print(f"  SLCA_WEIGHT_PROCESS:  {os.environ.get('SLCA_WEIGHT_PROCESS')}")
    print(f"  SLCA_WEIGHT_FORMAT:   {os.environ.get('SLCA_WEIGHT_FORMAT')}")
    print(f"  SLCA_WEIGHT_NAME:     {os.environ.get('SLCA_WEIGHT_NAME')}")
    print(f"  SLCA_WEIGHT_KEY:      {os.environ.get('SLCA_WEIGHT_KEY')}")
    print(f"  SLCA_WEIGHT_VALUE:    {os.environ.get('SLCA_WEIGHT_VALUE')}")
    print(f"  SLCA_WEIGHT_PARALLEL: {os.environ.get('SLCA_WEIGHT_PARALLEL')}")
    print(f"  SLCA_WEIGHT_RESPQ:    {os.environ.get('SLCA_WEIGHT_RESPQ')}")

    print("\n" + "-" * 80)
    print("  Evaluating sample 1 (correct tool call: radians_to_degrees)...")
    print("-" * 80)

    result1 = await compute_score(
        data_source="test_sample_1",
        solution_str=SAMPLE_1_SOLUTION,
        ground_truth=GROUND_TRUTH,
        extra_info=None,
    )
    print_result("Sample 1: correct tool call (radians_to_degrees)", result1)

    print("\n" + "-" * 80)
    print("  Evaluating sample 2 (wrong tool call: degrees_to_radians)...")
    print("-" * 80)

    result2 = await compute_score(
        data_source="test_sample_2",
        solution_str=SAMPLE_2_SOLUTION,
        ground_truth=GROUND_TRUTH,
        extra_info=None,
    )
    print_result("Sample 2: wrong tool call (degrees_to_radians)", result2)

    # Comparison summary
    print("\n" + "=" * 80)
    print("  Comparison summary")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Sample1 correct':<15} {'Sample2 wrong':<15} {'Diff':<10}")
    print("-" * 70)

    metrics = [
        ("format_score", "reward/tool_call/format_score"),
        ("name_match_score", "reward/tool_call/name_match_score"),
        ("key_match_score", "reward/tool_call/key_match_score"),
        ("value_match_score", "reward/tool_call/value_match_score"),
        ("parallel_score", "reward/tool_call/parallel_score"),
        ("process_score", "reward/tool_call/process_score"),
        ("summary_score (LLM Judge)", "reward/tool_call/summary_score"),
        ("weighted_score_norm", "reward/tool_call/weighted_score_norm"),
        ("score (RL reward)", "score"),
    ]

    for name, key in metrics:
        v1 = result1.get(key, 0)
        v2 = result2.get(key, 0)
        diff = v1 - v2
        print(f"{name:<30} {v1:<15.4f} {v2:<15.4f} {diff:+.4f}")

    print("\n" + "=" * 80)
    print("  Test finished!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
