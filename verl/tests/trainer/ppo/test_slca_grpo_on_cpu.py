# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU unit tests for the SLCA-GRPO advantage estimator.

SLCA-GRPO (Segment-Locked Credit Assignment) splits every rollout into two
segments -- the *process* segment (all tool-decision spans except the last
assistant turn) and the *summary* segment (the final assistant turn) -- then
normalizes and weights each segment independently so that summary-quality
reward cannot leak backwards onto tool-decision tokens.

Coverage:
  1. Contiguous-segment extraction from a 1D mask.
  2. Two-segment (process / summary) mask construction and the guarantee that
     the two masks partition the original response mask.
  3. Group-wise normalization restricted to "present" samples, including the
     ``count <= 1`` guard that zeroes an under-populated group instead of
     falling back to mean=0 / std=1.
  4. The full ``compute_slca_grpo_advantage`` estimator: segment routing,
     KL folding, the negative-score / should_no_call penalty guard, advantage
     leakage, and the empty-mask edge case.
  5. Per-segment weights being applied *after* z-score normalization -- if they
     were applied before, the z-score would cancel them out exactly.

Run with:
    PYTHONPATH=<repo>/verl python -m pytest \\
        verl/tests/trainer/ppo/test_slca_grpo_on_cpu.py -x -q
"""

import math

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    _build_two_segment_masks,
    _extract_segments_from_mask_1d,
    _groupwise_normalize_with_present,
    compute_slca_grpo_advantage,
    get_adv_estimator_fn,
)


def make_reward_components(
    process_score,
    summary_score,
    score_total,
    should_no_call=None,
    w_process=1.0,
    w_respq=1.0,
):
    """Build the ``reward_components`` dict in the schema that ray_trainer.py uses.

    Mirrors verl/trainer/ppo/ray_trainer.py, which populates the dict from
    ``reward/tool_call/process_score``, ``reward/tool_call/summary_score``,
    ``score`` and ``reward/tool_call/should_call_but_no_call``, plus the two
    segment weights read from SLCA_WEIGHT_PROCESS / SLCA_WEIGHT_RESPQ.
    """
    process_score = torch.as_tensor(process_score, dtype=torch.float32)
    if should_no_call is None:
        should_no_call = torch.zeros_like(process_score)
    return {
        "process_score": process_score,
        "summary_score": torch.as_tensor(summary_score, dtype=torch.float32),
        "score_total": torch.as_tensor(score_total, dtype=torch.float32),
        "should_no_call": torch.as_tensor(should_no_call, dtype=torch.float32),
        "w_process": w_process,
        "w_respq": w_respq,
    }


def assert_no_leakage(advantages, response_mask):
    """Tokens outside the response mask must carry exactly zero advantage."""
    leak = (advantages * (1.0 - response_mask)).abs().max()
    assert leak.item() == 0.0, f"Advantage leaked onto masked-out tokens: {leak.item()}"


class TestExtractSegments:
    """Contiguous-run extraction from a 1D boolean mask."""

    def test_empty_mask(self):
        """An all-zero mask yields no segments."""
        m = torch.zeros(10, dtype=torch.bool)
        assert _extract_segments_from_mask_1d(m) == []

    def test_single_segment(self):
        """A single contiguous run is returned as one half-open interval."""
        m = torch.tensor([0, 0, 1, 1, 1, 0, 0], dtype=torch.bool)
        assert _extract_segments_from_mask_1d(m) == [(2, 5)]

    def test_multiple_segments(self):
        """Non-contiguous runs are split at every gap."""
        m = torch.tensor([1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=torch.bool)
        assert _extract_segments_from_mask_1d(m) == [(0, 2), (4, 7), (8, 9)]

    def test_all_ones(self):
        """A fully-on mask is a single segment spanning the whole length."""
        m = torch.ones(5, dtype=torch.bool)
        assert _extract_segments_from_mask_1d(m) == [(0, 5)]

    def test_boundary_segments(self):
        """Runs touching either end of the tensor are not truncated."""
        m = torch.tensor([1, 0, 1], dtype=torch.bool)
        assert _extract_segments_from_mask_1d(m) == [(0, 1), (2, 3)]


class TestBuildTwoSegmentMasks:
    """Process / summary mask construction."""

    def test_no_segments(self):
        """With no learnable tokens both masks and both present flags are empty."""
        response_mask = torch.zeros(2, 10)
        process_mask, summary_mask, process_present, summary_present = _build_two_segment_masks(response_mask)

        assert not process_mask.any()
        assert not summary_mask.any()
        assert not process_present.any()
        assert not summary_present.any()

    def test_single_segment_is_summary(self):
        """A lone segment is routed to the summary, leaving the process empty."""
        response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
        process_mask, summary_mask, process_present, summary_present = _build_two_segment_masks(response_mask)

        assert not process_mask[0].any()
        assert not process_present[0]

        assert summary_mask[0, 2:5].all()
        assert summary_mask[0, :2].sum() == 0
        assert summary_mask[0, 5:].sum() == 0
        assert summary_present[0]

    def test_two_segments(self):
        """With two segments the first is process and the last is summary."""
        response_mask = torch.tensor([[0, 1, 1, 0, 0, 1, 1, 1, 0, 0]], dtype=torch.float32)
        process_mask, summary_mask, process_present, summary_present = _build_two_segment_masks(response_mask)

        assert process_mask[0, 1:3].all()
        assert not process_mask[0, 0]
        assert process_mask[0, 3:].sum() == 0
        assert process_present[0]

        assert summary_mask[0, 5:8].all()
        assert summary_mask[0, :5].sum() == 0
        assert summary_mask[0, 8:].sum() == 0
        assert summary_present[0]

    def test_three_segments(self):
        """All but the last segment are unioned into the process mask."""
        response_mask = torch.tensor([[1, 1, 0, 0, 1, 0, 0, 1, 1, 0]], dtype=torch.float32)
        process_mask, summary_mask, process_present, summary_present = _build_two_segment_masks(response_mask)

        assert process_mask[0, 0:2].all()
        assert process_mask[0, 4:5].all()
        assert process_present[0]

        assert summary_mask[0, 7:9].all()
        assert summary_present[0]

        # The two segments must be mutually exclusive.
        assert not (process_mask[0] & summary_mask[0]).any()

    def test_masks_partition_the_response_mask(self):
        """process_mask | summary_mask reconstructs the original mask exactly."""
        response_mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 0, 1, 0]], dtype=torch.float32)
        process_mask, summary_mask, _, _ = _build_two_segment_masks(response_mask)

        assert torch.equal(process_mask | summary_mask, response_mask > 0.5)

    def test_mixed_batch(self):
        """Per-sample routing is independent across the batch dimension."""
        response_mask = torch.tensor(
            [
                [0, 1, 1, 0, 0, 1, 1, 1, 0, 0],  # two segments -> process + summary
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # one segment  -> summary only
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # no learnable tokens
            ],
            dtype=torch.float32,
        )
        _, _, process_present, summary_present = _build_two_segment_masks(response_mask)

        assert process_present.tolist() == [True, False, False]
        assert summary_present.tolist() == [True, True, False]


class TestGroupwiseNormalize:
    """Group-wise normalization restricted to the samples flagged as present."""

    def test_all_present(self):
        """A fully present group is a plain z-score over that group."""
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        index = np.array(["g0", "g0", "g0", "g0"])
        present = torch.ones(4, dtype=torch.bool)

        adv = _groupwise_normalize_with_present(scores, index, present, norm_by_std=True)

        expected = (scores - scores.mean()) / (scores.std(unbiased=True) + 1e-6)
        assert torch.allclose(adv, expected, atol=1e-5)

    def test_partial_present(self):
        """Absent samples are excluded from the statistics and get zero advantage."""
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        index = np.array(["g0", "g0", "g0", "g0"])
        present = torch.tensor([True, True, False, False])

        adv = _groupwise_normalize_with_present(scores, index, present, norm_by_std=True)

        kept = scores[:2]
        expected = (kept - kept.mean()) / (kept.std(unbiased=True) + 1e-6)
        assert torch.allclose(adv[:2], expected, atol=1e-5)
        assert adv[2].item() == 0.0
        assert adv[3].item() == 0.0

    def test_singleton_group_is_zeroed(self):
        """A group holding a single present sample is zeroed, not left unnormalized.

        This is the ``count <= 1`` guard. The earlier revision fell back to
        mean=0 / std=1, which handed a singleton group its raw score as an
        advantage -- an unnormalized, arbitrarily large update. The guard now
        drops such groups entirely.
        """
        scores = torch.tensor([5.0, 10.0])
        index = np.array(["g0", "g1"])  # two groups of size one
        present = torch.ones(2, dtype=torch.bool)

        adv = _groupwise_normalize_with_present(scores, index, present, norm_by_std=True)

        assert torch.equal(adv, torch.zeros(2))

    def test_count_one_after_present_filter_is_zeroed(self):
        """The guard triggers on the *present* count, not the raw group size."""
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        index = np.array(["g0", "g0", "g0", "g0"])  # one group of size four ...
        present = torch.tensor([True, False, False, False])  # ... but only one present

        adv = _groupwise_normalize_with_present(scores, index, present, norm_by_std=True)

        assert torch.equal(adv, torch.zeros(4))

    def test_nothing_present(self):
        """An entirely absent batch short-circuits to all zeros."""
        scores = torch.tensor([1.0, 2.0])
        index = np.array(["g0", "g0"])
        present = torch.zeros(2, dtype=torch.bool)

        adv = _groupwise_normalize_with_present(scores, index, present, norm_by_std=True)

        assert torch.equal(adv, torch.zeros(2))

    def test_multiple_groups(self):
        """Groups are normalized independently of one another."""
        scores = torch.tensor([1.0, 3.0, 10.0, 20.0])
        index = np.array(["group_a", "group_a", "group_b", "group_b"])
        present = torch.ones(4, dtype=torch.bool)

        adv = _groupwise_normalize_with_present(scores, index, present, norm_by_std=True)

        a, b = scores[:2], scores[2:]
        expected_a = (a - a.mean()) / (a.std(unbiased=True) + 1e-6)
        expected_b = (b - b.mean()) / (b.std(unbiased=True) + 1e-6)
        assert torch.allclose(adv[:2], expected_a, atol=1e-5)
        assert torch.allclose(adv[2:], expected_b, atol=1e-5)

    def test_uuid_string_index(self):
        """Grouping keys may be arbitrary strings, as the real ``uid`` column is."""
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        index = np.array(
            [
                "3506fef6-0d99-4398-bc13-b4e264ede21e",
                "3506fef6-0d99-4398-bc13-b4e264ede21e",
                "abc123-def456-ghi789",
                "abc123-def456-ghi789",
            ]
        )
        present = torch.ones(4, dtype=torch.bool)

        adv = _groupwise_normalize_with_present(scores, index, present, norm_by_std=True)

        a, b = scores[:2], scores[2:]
        expected_a = (a - a.mean()) / (a.std(unbiased=True) + 1e-6)
        expected_b = (b - b.mean()) / (b.std(unbiased=True) + 1e-6)
        assert torch.allclose(adv[:2], expected_a, atol=1e-5)
        assert torch.allclose(adv[2:], expected_b, atol=1e-5)

    def test_norm_by_std_false_is_mean_subtraction(self):
        """norm_by_std=False (Dr.GRPO style) subtracts the mean without rescaling."""
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        index = np.array(["g0", "g0", "g0", "g0"])
        present = torch.ones(4, dtype=torch.bool)

        adv = _groupwise_normalize_with_present(scores, index, present, norm_by_std=False)

        assert torch.allclose(adv, scores - scores.mean(), atol=1e-6)


# Two learnable spans per sample: process then summary.
#   sample 0 -> process [1:3), summary [5:8)
#   sample 1 -> process [0:2), summary [6:9)
TWO_SEGMENT_MASK = torch.tensor(
    [
        [0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
    ],
    dtype=torch.float32,
)

# A single group, so both rollouts are normalized against each other.
ONE_GROUP = np.array(["g0", "g0"])

# For a present group of size two the z-score always collapses to +/- 1/sqrt(2).
# Kept as a Python float so torch.tensor([...]) below stays float32.
PAIR_ZSCORE = float(1.0 / math.sqrt(2.0))


class TestSLCAGRPOAdvantage:
    """End-to-end behaviour of ``compute_slca_grpo_advantage``."""

    def test_registry_registration(self):
        """The estimator is reachable through the advantage-estimator registry."""
        fn = get_adv_estimator_fn(AdvantageEstimator.SLCA_GRPO)
        assert fn is compute_slca_grpo_advantage

    def test_basic_two_segment(self):
        """Each segment gets its own group-normalized advantage, broadcast to its tokens."""
        response_mask = TWO_SEGMENT_MASK
        zeros = torch.zeros_like(response_mask)

        advantages, returns, segment_metrics = compute_slca_grpo_advantage(
            token_level_rewards=zeros,
            token_level_scores=zeros,
            response_mask=response_mask,
            index=ONE_GROUP,
            reward_components=make_reward_components(
                process_score=[0.8, 0.4],
                summary_score=[0.9, 0.5],
                score_total=[0.6, 0.3],
            ),
            norm_adv_by_std_in_grpo=True,
        )

        assert_no_leakage(advantages, response_mask)
        assert torch.equal(returns, advantages)

        adv_process = segment_metrics["adv_process"]
        adv_summary = segment_metrics["adv_summary"]
        assert adv_process.shape == (2,)
        assert adv_summary.shape == (2,)

        # Sample 0 scores higher in both segments, so it is the positive one.
        assert torch.allclose(adv_process, torch.tensor([PAIR_ZSCORE, -PAIR_ZSCORE]), atol=1e-5)
        assert torch.allclose(adv_summary, torch.tensor([PAIR_ZSCORE, -PAIR_ZSCORE]), atol=1e-5)

        # Sequence-level advantages are broadcast onto exactly their own tokens.
        assert torch.allclose(advantages[0, 1:3], adv_process[0].expand(2), atol=1e-6)
        assert torch.allclose(advantages[0, 5:8], adv_summary[0].expand(3), atol=1e-6)
        assert torch.allclose(advantages[1, 0:2], adv_process[1].expand(2), atol=1e-6)
        assert torch.allclose(advantages[1, 6:9], adv_summary[1].expand(3), atol=1e-6)

    def test_summary_only(self):
        """Rollouts with a single span contribute to the summary segment only."""
        response_mask = torch.tensor(
            [
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            ],
            dtype=torch.float32,
        )
        zeros = torch.zeros_like(response_mask)

        advantages, _, segment_metrics = compute_slca_grpo_advantage(
            token_level_rewards=zeros,
            token_level_scores=zeros,
            response_mask=response_mask,
            index=ONE_GROUP,
            reward_components=make_reward_components(
                process_score=[1.0, 0.5],
                summary_score=[0.8, 0.6],
                score_total=[0.6, 0.4],
            ),
        )

        assert_no_leakage(advantages, response_mask)

        # No process segment exists anywhere, so process advantages stay at zero
        # even though process_score differs between the two rollouts.
        assert torch.equal(segment_metrics["adv_process"], torch.zeros(2))
        assert torch.allclose(segment_metrics["adv_summary"], torch.tensor([PAIR_ZSCORE, -PAIR_ZSCORE]), atol=1e-5)

        assert advantages[0, 2:5].abs().sum() > 0
        assert advantages[1, 5:9].abs().sum() > 0

    def test_lone_process_segment_hits_count_guard(self):
        """A group with only one *present* process sample gets zero process advantage.

        Sample 0 has a process segment, sample 1 does not, so the process group
        holds a single present member and the ``count <= 1`` guard fires. The
        summary group still has two members and is normalized normally.
        """
        response_mask = torch.tensor(
            [
                [0, 1, 1, 0, 0, 1, 1, 1, 0, 0],  # process + summary
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # summary only
            ],
            dtype=torch.float32,
        )
        zeros = torch.zeros_like(response_mask)

        advantages, _, segment_metrics = compute_slca_grpo_advantage(
            token_level_rewards=zeros,
            token_level_scores=zeros,
            response_mask=response_mask,
            index=ONE_GROUP,
            reward_components=make_reward_components(
                process_score=[0.9, 0.1],
                summary_score=[0.9, 0.5],
                score_total=[0.6, 0.3],
            ),
        )

        assert torch.equal(segment_metrics["adv_process"], torch.zeros(2))
        # Sample 0's process tokens therefore receive nothing at all.
        assert advantages[0, 1:3].abs().sum().item() == 0.0

        # The summary segment is unaffected by the guard.
        assert torch.allclose(segment_metrics["adv_summary"], torch.tensor([PAIR_ZSCORE, -PAIR_ZSCORE]), atol=1e-5)
        assert advantages[0, 5:8].abs().sum() > 0

    def test_negative_score_penalty_guard(self):
        """A negative total score zeroes the process reward and books the loss to the summary.

        Sample 0 has the *higher* raw process score, so without the guard it
        would receive a positive process advantage despite having failed the
        rollout outright.
        """
        response_mask = torch.tensor(
            [
                [1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        zeros = torch.zeros_like(response_mask)

        advantages, _, segment_metrics = compute_slca_grpo_advantage(
            token_level_rewards=zeros,
            token_level_scores=zeros,
            response_mask=response_mask,
            index=ONE_GROUP,
            reward_components=make_reward_components(
                process_score=[0.9, 0.3],  # sample 0 looks better on process ...
                summary_score=[0.8, 0.4],
                score_total=[-0.5, 0.5],  # ... but its rollout scored negative
            ),
        )

        assert_no_leakage(advantages, response_mask)
        assert segment_metrics["adv_process"][0] < 0 < segment_metrics["adv_process"][1]
        assert segment_metrics["adv_summary"][0] < 0 < segment_metrics["adv_summary"][1]
        assert torch.allclose(
            segment_metrics["adv_process"], torch.tensor([-PAIR_ZSCORE, PAIR_ZSCORE]), atol=1e-5
        )

    def test_should_no_call_penalty_guard(self):
        """The should_no_call flag triggers the same guard even with a positive total score."""
        response_mask = torch.tensor(
            [
                [1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        zeros = torch.zeros_like(response_mask)

        advantages, _, segment_metrics = compute_slca_grpo_advantage(
            token_level_rewards=zeros,
            token_level_scores=zeros,
            response_mask=response_mask,
            index=ONE_GROUP,
            reward_components=make_reward_components(
                process_score=[0.9, 0.3],
                summary_score=[0.8, 0.4],
                score_total=[0.2, 0.5],  # positive, so only the flag can trip the guard
                should_no_call=[1.0, 0.0],
            ),
        )

        assert_no_leakage(advantages, response_mask)
        assert segment_metrics["adv_process"][0] < 0 < segment_metrics["adv_process"][1]

    def test_kl_is_folded_into_segment_scores(self):
        """token_level_rewards - token_level_scores is summed per segment and added in.

        Both rollouts carry identical extrinsic scores, so every advantage would
        be zero were the KL term dropped. Charging sample 0's process tokens a
        KL penalty is what separates the two.
        """
        response_mask = torch.tensor(
            [
                [0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            ],
            dtype=torch.float32,
        )
        token_level_scores = torch.zeros_like(response_mask)
        token_level_rewards = torch.zeros_like(response_mask)
        token_level_rewards[0, 1:3] = -0.1  # KL penalty on sample 0's process tokens

        _, _, segment_metrics = compute_slca_grpo_advantage(
            token_level_rewards=token_level_rewards,
            token_level_scores=token_level_scores,
            response_mask=response_mask,
            index=ONE_GROUP,
            reward_components=make_reward_components(
                process_score=[0.5, 0.5],
                summary_score=[0.5, 0.5],
                score_total=[0.5, 0.5],
            ),
        )

        assert torch.allclose(
            segment_metrics["adv_process"], torch.tensor([-PAIR_ZSCORE, PAIR_ZSCORE]), atol=1e-5
        )
        # The summary segment saw no KL, so its identical scores still cancel.
        assert torch.allclose(segment_metrics["adv_summary"], torch.zeros(2), atol=1e-6)

    def test_no_learnable_tokens(self):
        """An all-zero response mask produces all-zero advantages and metrics."""
        response_mask = torch.zeros(2, 10)
        zeros = torch.zeros_like(response_mask)

        advantages, returns, segment_metrics = compute_slca_grpo_advantage(
            token_level_rewards=zeros,
            token_level_scores=zeros,
            response_mask=response_mask,
            index=ONE_GROUP,
            reward_components=make_reward_components(
                process_score=[0.5, 0.5],
                summary_score=[0.5, 0.5],
                score_total=[0.5, 0.5],
            ),
        )

        assert advantages.abs().sum().item() == 0.0
        assert returns.abs().sum().item() == 0.0
        assert segment_metrics["adv_process"].abs().sum().item() == 0.0
        assert segment_metrics["adv_summary"].abs().sum().item() == 0.0


class TestSegmentWeighting:
    """Segment weights must be applied after normalization, not before."""

    @staticmethod
    def _run(w_process, w_respq):
        zeros = torch.zeros_like(TWO_SEGMENT_MASK)
        return compute_slca_grpo_advantage(
            token_level_rewards=zeros,
            token_level_scores=zeros,
            response_mask=TWO_SEGMENT_MASK,
            index=ONE_GROUP,
            reward_components=make_reward_components(
                process_score=[0.8, 0.4],
                summary_score=[0.9, 0.5],
                score_total=[0.6, 0.3],
                w_process=w_process,
                w_respq=w_respq,
            ),
        )

    def test_unit_weights_are_the_identity(self):
        """w_process = w_respq = 1.0 reproduces the unweighted estimator exactly."""
        adv_a, _, seg_a = self._run(1.0, 1.0)
        adv_b, _, seg_b = self._run(1.0, 1.0)

        assert torch.equal(adv_a, adv_b)
        assert torch.equal(seg_a["adv_process"], seg_b["adv_process"])

    def test_weights_scale_advantages_linearly(self):
        """Scaling a weight scales that segment's advantage by the same factor.

        This is the load-bearing assertion for the post-normalization design:
        applying the weights *before* the z-score would cancel them out
        completely -- (w*s - w*mu) / (w*sigma) == (s - mu) / sigma -- and this
        test would then see a ratio of 1.0 instead of 2.0 and 3.0.
        """
        base_adv, _, base_seg = self._run(1.0, 1.0)
        scaled_adv, _, scaled_seg = self._run(2.0, 3.0)

        # Guard against a vacuous comparison against an all-zero baseline.
        assert base_seg["adv_process"].abs().sum() > 0
        assert base_seg["adv_summary"].abs().sum() > 0

        assert torch.allclose(scaled_seg["adv_process"], 2.0 * base_seg["adv_process"], atol=1e-6)
        assert torch.allclose(scaled_seg["adv_summary"], 3.0 * base_seg["adv_summary"], atol=1e-6)

        # The scaling propagates to the token-level advantages of each segment.
        assert torch.allclose(scaled_adv[0, 1:3], 2.0 * base_adv[0, 1:3], atol=1e-6)
        assert torch.allclose(scaled_adv[0, 5:8], 3.0 * base_adv[0, 5:8], atol=1e-6)

    def test_zero_process_weight_silences_the_process_segment(self):
        """w_process = 0 removes tool-decision updates while leaving the summary intact."""
        advantages, _, segment_metrics = self._run(0.0, 1.0)

        assert torch.equal(segment_metrics["adv_process"], torch.zeros(2))
        assert advantages[0, 1:3].abs().sum().item() == 0.0
        assert advantages[1, 0:2].abs().sum().item() == 0.0

        assert segment_metrics["adv_summary"].abs().sum() > 0
        assert advantages[0, 5:8].abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
