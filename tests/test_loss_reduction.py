# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.distributed as dist

from areal.api import (
    LOSS_TERM_REDUCTION_SUM,
    LossReduction,
    LossTerm,
    TrainEngine,
)
from areal.engine.core.train_engine import (
    compute_global_normalizers,
    scale_loss_for_reduction,
)
from areal.utils.data import MicroBatchList


def _normalizer(_data):
    return torch.tensor(1.0)


def _make_original_callback_engine():
    calls = []

    def stub(*_args, **_kwargs):
        return None

    def train_batch(self, input_, loss_fn, loss_weight_fn):
        calls.append(("train", input_, loss_fn, loss_weight_fn))
        return {"lr": 1.0}

    def eval_batch(self, input_, loss_fn, loss_weight_fn):
        calls.append(("eval", input_, loss_fn, loss_weight_fn))
        return torch.tensor(2.0)

    implementations = {name: stub for name in TrainEngine.__abstractmethods__}
    implementations.update(train_batch=train_batch, eval_batch=eval_batch)
    engine_type = type("OriginalCallbackEngine", (TrainEngine,), implementations)
    return engine_type(), calls


def test_original_train_engine_subclass_supports_mean_reduction_adapter():
    engine, calls = _make_original_callback_engine()

    def loss_fn():
        return torch.tensor(2.0)

    reduction = LossReduction.mean(loss_fn, _normalizer)

    train_result = engine.train_batch_with_reduction({"x": 1}, reduction)
    eval_result = engine.eval_batch_with_reduction({"x": 2}, reduction)

    assert train_result == {"lr": 1.0}
    torch.testing.assert_close(eval_result, torch.tensor(2.0))
    assert calls == [
        ("train", {"x": 1}, loss_fn, _normalizer),
        ("eval", {"x": 2}, loss_fn, _normalizer),
    ]


def test_original_train_engine_subclass_rejects_advanced_reduction():
    engine, _ = _make_original_callback_engine()
    reduction = LossReduction.sum(lambda: torch.tensor(1.0), _normalizer)

    with pytest.raises(NotImplementedError, match="original callback API"):
        engine.train_batch_with_reduction({}, reduction)


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


@pytest.mark.parametrize("normalizer", [0.0, -1.0, float("nan"), float("inf")])
def test_global_normalizer_must_be_finite_and_positive(monkeypatch, normalizer):
    reduction = LossReduction.sum(
        loss_fn=lambda: torch.tensor(0.0),
        normalizer_fn=lambda _data: torch.tensor(normalizer),
    )
    mb_list = MicroBatchList(
        data={},
        mb_spec=None,
        mbs=[{"loss_mask": torch.zeros(2, dtype=torch.bool)}],
        group_lens=[1],
    )
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, group=None: tensor)

    with pytest.raises(RuntimeError, match="finite and positive"):
        compute_global_normalizers(mb_list, reduction, dp_group=None)
