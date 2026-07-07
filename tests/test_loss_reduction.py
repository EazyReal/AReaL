# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.distributed as dist

from areal.api import (
    LOSS_TERM_REDUCTION_SUM,
    LossReduction,
    LossTerm,
)
from areal.engine.core.train_engine import (
    compute_global_normalizers,
    scale_loss_for_reduction,
)
from areal.utils.data import MicroBatchList


def _normalizer(_data):
    return torch.tensor(1.0)


def test_mean_scaling_preserves_local_mean_order():
    reduction = LossReduction.mean(
        loss_fn=lambda: torch.tensor(2.0), normalizer_fn=_normalizer
    )
    local_normalizer = torch.tensor(3.0)
    global_normalizer = torch.tensor(12.0)
    loss = torch.tensor(2.0)

    scaled = scale_loss_for_reduction(
        loss,
        reduction,
        {"loss": local_normalizer},
        {"loss": global_normalizer},
        loss_multiplier=1.0,
    )

    torch.testing.assert_close(
        scaled, loss * local_normalizer / global_normalizer, rtol=0, atol=0
    )


def test_sum_scaling_uses_global_normalizer_directly():
    reduction = LossReduction.sum(
        loss_fn=lambda: torch.tensor(6.0), normalizer_fn=_normalizer
    )
    local_sum = torch.tensor(6.0)

    scaled = scale_loss_for_reduction(
        local_sum,
        reduction,
        {"loss": torch.tensor(3.0)},
        {"loss": torch.tensor(12.0)},
        loss_multiplier=1.0,
    )

    torch.testing.assert_close(scaled, local_sum / 12.0, rtol=0, atol=0)


def test_local_zero_normalizer_masks_nan_loss_value():
    reduction = LossReduction.mean(
        loss_fn=lambda: torch.tensor(float("nan")), normalizer_fn=_normalizer
    )

    scaled = scale_loss_for_reduction(
        torch.tensor(float("nan")),
        reduction,
        {"loss": torch.tensor(0.0)},
        {"loss": torch.tensor(12.0)},
        loss_multiplier=1.0,
    )

    torch.testing.assert_close(scaled, torch.tensor(0.0), rtol=0, atol=0)


def test_multi_term_scaling_uses_each_terms_normalizer():
    reduction = LossReduction(
        loss_fn=lambda: {
            "pg": torch.tensor(6.0),
            "kd": torch.tensor(2.0),
        },
        terms=(
            LossTerm(
                "pg", normalizer_fn=_normalizer, reduction=LOSS_TERM_REDUCTION_SUM
            ),
            LossTerm(
                "kd", normalizer_fn=_normalizer, reduction=LOSS_TERM_REDUCTION_SUM
            ),
        ),
    )

    scaled = scale_loss_for_reduction(
        {"pg": torch.tensor(6.0), "kd": torch.tensor(2.0)},
        reduction,
        {"pg": torch.tensor(3.0), "kd": torch.tensor(2.0)},
        {"pg": torch.tensor(12.0), "kd": torch.tensor(4.0)},
        loss_multiplier=1.0,
    )

    torch.testing.assert_close(scaled, torch.tensor(1.0), rtol=0, atol=0)


def test_global_normalizer_must_be_positive(monkeypatch):
    reduction = LossReduction.sum(
        loss_fn=lambda: torch.tensor(0.0),
        normalizer_fn=lambda data: data["loss_mask"].count_nonzero(),
    )
    mb_list = MicroBatchList(
        data={},
        mb_spec=None,
        mbs=[{"loss_mask": torch.zeros(2, dtype=torch.bool)}],
        group_lens=[1],
    )
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, group=None: tensor)

    with pytest.raises(RuntimeError, match="Global loss normalizers"):
        compute_global_normalizers(mb_list, reduction, dp_group=None)
