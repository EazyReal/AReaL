# SPDX-License-Identifier: Apache-2.0
"""Loss aggregation levels (ScaleRL §3.2): token / sequence / prompt.

``token_mean`` weights every token equally (long trajectories dominate);
``seq_mean`` weights every sequence equally (GRPO); ``prompt_mean`` weights
every prompt-group equally (MiniMax-M1). The engine realizes the global mean as
``Σ_mb loss_mb · weight_mb / Σ_mb weight_mb`` (see ``compute_total_loss_weight``
/ fsdp ``loss_scale``), so ``loss_fn`` and ``loss_weight_fn`` must agree on the
unit -- the pairing test below pins exactly that for all three modes.
"""

import pytest
import torch

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    InferenceEngineConfig,
    MicroBatchSpec,
    PPOActorConfig,
)
from areal.trainer.ppo.actor import _make_loss_weight_fn
from areal.utils.functional import aggregate_pg_loss

# Two prompt-groups of 2 sequences (group_size=2), T=3. Group 0 has one
# fully-masked-but-one short sequence; group 1 is full. Token counts per group
# differ (4 vs 6), so token_mean and prompt_mean diverge.
PG = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
MASK = torch.tensor(
    [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
)
# group 0: num=4 den=4 -> 1.0 ; group 1: num=18 den=6 -> 3.0
PROMPT_MEAN = 2.0  # (1.0 + 3.0) / 2
TOKEN_MEAN = 2.2  # (3 + 1 + 9 + 9) / (3 + 1 + 3 + 3) = 22 / 10


# A 2-sequence batch with different lengths, where seq_mean != token_mean:
# seq0 = 2 valid tokens of pg 2 (seq-mean 2.0); seq1 = 4 valid tokens of pg 1.
SEQ_PG = torch.tensor([[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
SEQ_MASK = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])


def test_token_mean_is_global_token_average():
    loss = aggregate_pg_loss(PG, MASK, loss_aggregation="token_mean")
    torch.testing.assert_close(loss, torch.tensor(TOKEN_MEAN))


def test_seq_mean_weights_each_sequence_equally():
    loss = aggregate_pg_loss(SEQ_PG, SEQ_MASK, loss_aggregation="seq_mean")
    torch.testing.assert_close(loss, torch.tensor(1.5))  # (2.0 + 1.0) / 2
    token = aggregate_pg_loss(SEQ_PG, SEQ_MASK, loss_aggregation="token_mean")
    torch.testing.assert_close(token, torch.tensor(8.0 / 6.0))  # length-weighted
    assert not torch.allclose(loss, token)


def test_seq_mean_packed_matches_padded():
    # seq0 = 2 tokens, seq1 = 4 tokens (variable length, the case that matters).
    pg = torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    mask = torch.ones(6)
    cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
    loss = aggregate_pg_loss(
        pg, mask, loss_aggregation="seq_mean", cu_seqlens=cu_seqlens
    )
    torch.testing.assert_close(loss, torch.tensor(1.5))


def test_prompt_mean_weights_each_group_equally_2d():
    loss = aggregate_pg_loss(PG, MASK, loss_aggregation="prompt_mean", group_size=2)
    torch.testing.assert_close(loss, torch.tensor(PROMPT_MEAN))
    # The two modes genuinely differ on this uneven batch.
    assert not torch.allclose(loss, torch.tensor(TOKEN_MEAN))


def test_prompt_mean_packed_matches_padded():
    # Same data packed: seqs of length 3 each, masked position kept in place.
    pg = PG.reshape(-1)
    mask = MASK.reshape(-1)
    cu_seqlens = torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32)
    loss = aggregate_pg_loss(
        pg, mask, loss_aggregation="prompt_mean", group_size=2, cu_seqlens=cu_seqlens
    )
    torch.testing.assert_close(loss, torch.tensor(PROMPT_MEAN))


@pytest.mark.parametrize("aggregation", ["token_mean", "seq_mean", "prompt_mean"])
def test_loss_weight_pairing_realizes_global_mean(aggregation):
    """Splitting into per-group microbatches and applying the engine's
    Σ(loss·weight)/Σweight reduction reproduces the single-batch result."""
    group_size = 2 if aggregation == "prompt_mean" else 1
    weight_fn = _make_loss_weight_fn(aggregation, group_size)

    full = aggregate_pg_loss(
        PG, MASK, loss_aggregation=aggregation, group_size=group_size
    )

    num = torch.tensor(0.0)
    den = torch.tensor(0.0)
    for s in (slice(0, 2), slice(2, 4)):  # one prompt-group per microbatch
        mb_pg, mb_mask = PG[s], MASK[s]
        loss_mb = aggregate_pg_loss(
            mb_pg, mb_mask, loss_aggregation=aggregation, group_size=group_size
        )
        w = weight_fn({"loss_mask": mb_mask})
        num = num + loss_mb * w
        den = den + w
    torch.testing.assert_close(num / den, full)


def test_denom_mask_uses_pre_rejection_count():
    # Numerator over loss_mask (2 kept tokens); denominator over denom_mask
    # (4 original tokens), so the per-token gradient is not inflated by 4/2.
    pg = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    denom_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    loss = aggregate_pg_loss(
        pg, loss_mask, loss_aggregation="token_mean", denom_mask=denom_mask
    )
    torch.testing.assert_close(loss, torch.tensor(1.0))  # 4 / 4, not 4 / 2
    without = aggregate_pg_loss(pg, loss_mask, loss_aggregation="token_mean")
    torch.testing.assert_close(without, torch.tensor(2.0))  # 4 / 2


def test_prompt_mean_group_size_one_equals_seq_mean():
    # A prompt-group of one sequence is just that sequence, so prompt_mean with
    # group_size=1 is seq_mean (NOT token_mean).
    a = aggregate_pg_loss(PG, MASK, loss_aggregation="prompt_mean", group_size=1)
    b = aggregate_pg_loss(PG, MASK, loss_aggregation="seq_mean")
    torch.testing.assert_close(a, b)


def test_prompt_mean_rejects_ragged_group_count():
    pg = torch.ones(3, 2)
    mask = torch.ones(3, 2)
    with pytest.raises(ValueError, match="not divisible by group_size"):
        aggregate_pg_loss(pg, mask, loss_aggregation="prompt_mean", group_size=2)


def test_config_derives_group_size_from_n_samples():
    # prompt_mean's group_size is the rollout's n_samples-per-prompt, derived in
    # PPOConfig.__post_init__ -- not a separate knob the user types.
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=4),
        actor=PPOActorConfig(
            loss_aggregation="prompt_mean", mb_spec=MicroBatchSpec(granularity=4)
        ),
    )
    assert cfg.actor.group_size == 4


def test_config_hand_set_group_size_cannot_silently_take_effect():
    # A mismatched actor.group_size is overwritten by gconfig.n_samples, so it can
    # never silently drive the loss grouping out of sync with the rollout.
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=4),
        actor=PPOActorConfig(
            loss_aggregation="prompt_mean",
            group_size=99,
            mb_spec=MicroBatchSpec(granularity=4),
        ),
    )
    assert cfg.actor.group_size == 4


def test_config_granularity_is_auto_bumped_for_prompt_mean():
    # granularity that isn't a multiple of n_samples is auto-bumped (not an
    # error) -- the only prompt_mean knob the user sets is loss_aggregation.
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=4),
        actor=PPOActorConfig(
            loss_aggregation="prompt_mean", mb_spec=MicroBatchSpec(granularity=2)
        ),
    )
    assert cfg.actor.mb_spec.granularity == 4  # 2 -> next multiple of 4
    # default granularity (1) is bumped to n_samples
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=8),
        actor=PPOActorConfig(loss_aggregation="prompt_mean"),
    )
    assert cfg.actor.mb_spec.granularity == 8


def test_prompt_mean_drops_under_filled_groups():
    # prompt_mean groups positionally, so it needs whole groups: under-filled
    # rollout groups are dropped at the source by raising min_valid_group_size
    # to n_samples (the partial-group fix this PR is stacked on).
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=4),
        actor=PPOActorConfig(
            loss_aggregation="prompt_mean", mb_spec=MicroBatchSpec(granularity=4)
        ),
    )
    assert cfg.rollout.min_valid_group_size == 4


def test_min_valid_group_size_cannot_exceed_n_samples():
    # An unsatisfiable threshold (every group would be dropped) fails at config
    # time, not at the first rollout.
    with pytest.raises(ValueError, match="cannot exceed gconfig.n_samples"):
        GRPOConfig(
            gconfig=GenerationHyperparameters(n_samples=4),
            rollout=InferenceEngineConfig(min_valid_group_size=5),
        )


def test_config_validation():
    # prompt_mean needs n_samples >= 2.
    with pytest.raises(ValueError, match="n_samples >= 2"):
        GRPOConfig(
            gconfig=GenerationHyperparameters(n_samples=1),
            actor=PPOActorConfig(loss_aggregation="prompt_mean"),
        )
    # An invalid loss_aggregation value is rejected at the actor level.
    with pytest.raises(ValueError, match="loss_aggregation must be"):
        PPOActorConfig(loss_aggregation="bogus")
    # token_mean (default) and seq_mean impose no group/granularity constraint.
    GRPOConfig(gconfig=GenerationHyperparameters(n_samples=1))
    GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=1),
        actor=PPOActorConfig(loss_aggregation="seq_mean"),
    )
