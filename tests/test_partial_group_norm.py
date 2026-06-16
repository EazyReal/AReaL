"""Group-level normalization with unequal/partial groups.

When a rollout group loses members (some episodes return None), its rows no
longer line up with a fixed ``group_size`` stride. Passing the actual per-group
row counts via ``group_sizes`` keeps each group's baseline within its own rows.
"""

import pytest
import torch

from areal.api.cli_args import NormConfig
from areal.utils.data import Normalization, concat_batch


def _group_norm(mean_level="group", std_level="group", group_size=4, **kw):
    return Normalization(
        NormConfig(
            mean_level=mean_level, std_level=std_level, group_size=group_size, **kw
        )
    )


def test_group_sizes_matches_positional_for_full_groups_reward_path():
    """1-D reward path (no mask): group_sizes=[4,4] equals positional stride 4."""
    torch.manual_seed(0)
    x = torch.randn(8)
    norm = _group_norm()
    torch.testing.assert_close(norm(x), norm(x, group_sizes=[4, 4]))


def test_group_sizes_matches_positional_for_full_groups_adv_path():
    """2-D masked advantage path: group_sizes=[4,4] equals positional stride 4."""
    torch.manual_seed(1)
    x = torch.randn(8, 5)
    mask = torch.ones(8, 5)
    mask[:, 4] = 0.0
    norm = _group_norm()
    torch.testing.assert_close(norm(x, mask), norm(x, mask, group_sizes=[4, 4]))


def test_trailing_partial_group_does_not_blow_up():
    """A trailing short group [4,4,3] is normalized, not left at std 0 -> /eps."""
    torch.manual_seed(2)
    x = torch.randn(11) * 3.0
    out = _group_norm()(x, group_sizes=[4, 4, 3])
    assert torch.isfinite(out).all()
    for s in (slice(0, 4), slice(4, 8), slice(8, 11)):
        torch.testing.assert_close(out[s].mean(), torch.tensor(0.0), atol=1e-5, rtol=0)


def test_mid_batch_partial_group_avoids_cross_prompt_sign_flip():
    """[4,4,3,1] is divisible by 4 yet misaligned; positional slicing flips a sign.

    Prompt C = rows 8-10 = [0,1,2]; prompt D = row 11 = 100 (outlier). Positional
    stride 4 lumps C+D into slice [8:12] (baseline 25.75), so row 10's advantage
    goes negative; the correct within-C baseline (1.0) keeps it positive.
    """
    x = torch.tensor([0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0, 3.0, 0.0, 1.0, 2.0, 100.0])
    norm = _group_norm()

    out_positional = norm(x)
    out_grouped = norm(x, group_sizes=[4, 4, 3, 1])

    torch.testing.assert_close(out_positional[:8], out_grouped[:8])
    assert out_grouped[10] > 0 and out_positional[10] < 0
    # A group of one has no within-group signal -> advantage 0.
    torch.testing.assert_close(out_grouped[11], torch.tensor(0.0), atol=1e-6, rtol=0)


@pytest.mark.parametrize("std_unbiased", [True, False])
def test_singleton_group_is_finite_and_zero(std_unbiased):
    """1-D reward path: a group of one yields a finite 0 advantage, either std."""
    torch.manual_seed(3)
    x = torch.randn(5) * 10.0
    out = _group_norm(std_unbiased=std_unbiased)(x, group_sizes=[4, 1])
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[4], torch.tensor(0.0), atol=1e-6, rtol=0)


@pytest.mark.parametrize("std_unbiased", [True, False])
def test_singleton_group_masked_path_is_finite(std_unbiased):
    """2-D masked path: a singleton group stays finite (incl. the biased-std case)."""
    torch.manual_seed(4)
    x = torch.randn(5, 6) * 10.0
    mask = torch.ones(5, 6)
    mask[:, 5] = 0.0
    out = _group_norm(std_unbiased=std_unbiased)(x, mask, group_sizes=[4, 1])
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, 5], torch.zeros(5), atol=1e-6, rtol=0)


def test_non_divisible_batch_without_group_sizes_raises():
    """Positional fallback rejects a non-divisible batch instead of mis-slicing."""
    with pytest.raises(ValueError, match="not divisible"):
        _group_norm()(torch.randn(11))


def test_group_sizes_must_sum_to_batch_size():
    with pytest.raises(ValueError, match="group_sizes sum"):
        _group_norm()(torch.randn(8), group_sizes=[4, 3])


def test_group_sizes_must_be_positive():
    with pytest.raises(ValueError, match="group_sizes must be positive"):
        _group_norm()(torch.randn(8), group_sizes=[8, 0])


def test_batch_level_ignores_group_sizes():
    """Batch-level norm never groups, so a non-divisible batch is fine."""
    out = _group_norm(mean_level="batch", std_level="batch")(torch.randn(11))
    assert torch.isfinite(out).all()


def test_concat_batch_records_partial_group_sizes():
    """concat_batch reports actual per-group row counts, feeding the normalizer."""

    def traj(k):
        return {"rewards": torch.randn(k), "attention_mask": torch.ones(k, 3)}

    _, meta = concat_batch([traj(4), traj(4), traj(3)])
    assert meta.traj_group_sizes == [4, 4, 3]
