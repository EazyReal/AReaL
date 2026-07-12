# SPDX-License-Identifier: Apache-2.0

"""Contracts for loss values and their distributed normalizers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

LOSS_TERM_REDUCTION_MEAN = "mean"
LOSS_TERM_REDUCTION_SUM = "sum"
LOSS_TERM_REDUCTIONS_ALL = (
    LOSS_TERM_REDUCTION_MEAN,
    LOSS_TERM_REDUCTION_SUM,
)

LossFnOutput = torch.Tensor | Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class LossTerm:
    """One named term in a distributed loss reduction.

    ``normalizer_fn`` returns this rank's scalar contribution to the global
    normalizer for this term. ``reduction="mean"`` means the loss value is
    already divided by that local normalizer. ``reduction="sum"`` means the
    loss value is the local numerator term.
    """

    name: str
    normalizer_fn: Callable[[dict[str, Any]], torch.Tensor]
    reduction: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LossTerm.name must be non-empty.")
        if self.reduction not in LOSS_TERM_REDUCTIONS_ALL:
            raise ValueError(
                f"reduction must be one of {LOSS_TERM_REDUCTIONS_ALL}, "
                f"got {self.reduction!r}."
            )


@dataclass(frozen=True, slots=True)
class LossReduction:
    """Loss function plus the distributed reduction contract for its outputs.

    For a ``mean`` term, the engine computes
    ``local_mean * local_normalizer / global_normalizer``. For a ``sum`` term,
    the engine computes ``local_sum / global_normalizer``. If ``loss_fn``
    returns a mapping, each term consumes the value under its own name.
    """

    loss_fn: Callable[..., LossFnOutput]
    terms: tuple[LossTerm, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("LossReduction requires at least one term.")
        names = [term.name for term in self.terms]
        if len(names) != len(set(names)):
            raise ValueError(f"LossReduction term names must be unique, got {names}.")

    @classmethod
    def mean(
        cls,
        loss_fn: Callable[..., torch.Tensor],
        normalizer_fn: Callable[[dict[str, Any]], torch.Tensor],
        name: str = "loss",
    ) -> LossReduction:
        """Build a reduction for a loss value normalized within each microbatch."""
        return cls(
            loss_fn=loss_fn,
            terms=(
                LossTerm(
                    name=name,
                    normalizer_fn=normalizer_fn,
                    reduction=LOSS_TERM_REDUCTION_MEAN,
                ),
            ),
        )

    @classmethod
    def sum(
        cls,
        loss_fn: Callable[..., LossFnOutput],
        normalizer_fn: Callable[[dict[str, Any]], torch.Tensor],
        name: str = "loss",
    ) -> LossReduction:
        """Build a reduction for a local numerator term."""
        return cls(
            loss_fn=loss_fn,
            terms=(
                LossTerm(
                    name=name,
                    normalizer_fn=normalizer_fn,
                    reduction=LOSS_TERM_REDUCTION_SUM,
                ),
            ),
        )


LossWeightFn = Callable[[dict[str, Any]], torch.Tensor]
