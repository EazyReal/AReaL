# SPDX-License-Identifier: Apache-2.0

"""Policy-gradient loss aggregation and distributed normalizer contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

LOSS_AGGREGATION_TOKEN_MEAN = "token_mean"
LOSS_AGGREGATION_SEQ_MEAN = "seq_mean"
LOSS_AGGREGATION_PROMPT_MEAN = "prompt_mean"
LOSS_AGGREGATION_CONSTANT = "constant"

LossAggregationMode = Literal["token_mean", "seq_mean", "prompt_mean", "constant"]
LOSS_AGGREGATIONS_ALL = (
    LOSS_AGGREGATION_TOKEN_MEAN,
    LOSS_AGGREGATION_SEQ_MEAN,
    LOSS_AGGREGATION_PROMPT_MEAN,
    LOSS_AGGREGATION_CONSTANT,
)

GroupSizes = Sequence[int] | torch.Tensor


def _masked_loss(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, loss, 0).to(torch.float32)


def _resolve_masks(
    loss: torch.Tensor,
    loss_mask: torch.Tensor,
    denominator_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if loss.shape != loss_mask.shape:
        raise ValueError(
            f"loss_mask shape {tuple(loss_mask.shape)} must match "
            f"loss shape {tuple(loss.shape)}."
        )
    if denominator_mask is not None and loss.shape != denominator_mask.shape:
        raise ValueError(
            f"denom_mask shape {tuple(denominator_mask.shape)} must match "
            f"loss shape {tuple(loss.shape)}."
        )
    numerator_mask = loss_mask.bool()
    return numerator_mask, (
        numerator_mask if denominator_mask is None else denominator_mask.bool()
    )


def _sequence_sums(
    values: torch.Tensor, cu_seqlens: torch.Tensor | None
) -> torch.Tensor:
    """Sum token values per sequence for padded or packed inputs."""
    if cu_seqlens is None:
        if values.ndim != 2:
            raise ValueError(
                "padded policy-gradient inputs must be 2D, "
                f"got shape {tuple(values.shape)}."
            )
        return values.sum(dim=-1)

    if values.ndim != 1:
        raise ValueError(
            "packed policy-gradient inputs must be 1D, "
            f"got shape {tuple(values.shape)}."
        )
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1D, got shape {tuple(cu_seqlens.shape)}.")

    n_sequences = cu_seqlens.numel() - 1
    sequence_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(
        device=values.device, dtype=torch.long
    )
    if torch.any(sequence_lengths < 0):
        raise ValueError("cu_seqlens must be non-decreasing.")
    sequence_ids = torch.arange(n_sequences, device=values.device).repeat_interleave(
        sequence_lengths
    )
    if sequence_ids.numel() != values.numel():
        raise ValueError(
            "cu_seqlens does not describe the packed loss: "
            f"expected {sequence_ids.numel()} tokens, got {values.numel()}."
        )
    result = torch.zeros(n_sequences, dtype=values.dtype, device=values.device)
    return result.scatter_add_(0, sequence_ids, values)


def _validate_group_sizes(
    n_sequences: int, group_sizes: GroupSizes | None
) -> list[int]:
    if group_sizes is None:
        raise ValueError("group_sizes are required for explicit prompt groups.")
    if torch.is_tensor(group_sizes):
        raw_sizes = group_sizes.detach().cpu().tolist()
    else:
        raw_sizes = list(group_sizes)
    sizes = [int(size) for size in raw_sizes]
    if any(size <= 0 for size in sizes):
        raise ValueError(f"group_sizes must be positive, got {sizes}.")
    if sum(sizes) != n_sequences:
        raise ValueError(
            f"group_sizes sum to {sum(sizes)} but sequence count is {n_sequences}."
        )
    return sizes


def _group_ids(
    n_sequences: int,
    unit_size: int,
    group_sizes: GroupSizes | None,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    if group_sizes is not None:
        sizes = _validate_group_sizes(n_sequences, group_sizes)
        return (
            torch.arange(len(sizes), device=device).repeat_interleave(
                torch.tensor(sizes, device=device)
            ),
            len(sizes),
        )
    if unit_size <= 0:
        raise ValueError(f"group_size must be positive, got {unit_size}.")
    if n_sequences % unit_size != 0:
        raise ValueError(
            f"sequence count {n_sequences} is not divisible by group_size {unit_size}."
        )
    return (
        torch.arange(n_sequences, device=device) // unit_size,
        n_sequences // unit_size,
    )


def _reduce_unit_means(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    local_mean: bool,
) -> torch.Tensor:
    active = denominator > 0
    unit_means = torch.where(
        active,
        numerator / denominator.clamp_min(1),
        torch.zeros_like(numerator),
    )
    if not local_mean:
        return unit_means.sum()
    return unit_means.sum() / active.count_nonzero().clamp_min(1)


def _aggregate_units(
    loss: torch.Tensor,
    numerator_mask: torch.Tensor,
    denominator_mask: torch.Tensor,
    *,
    unit_size: int,
    group_sizes: GroupSizes | None,
    cu_seqlens: torch.Tensor | None,
    local_mean: bool,
) -> torch.Tensor:
    masked_loss = _masked_loss(loss, numerator_mask)
    sequence_numerators = _sequence_sums(masked_loss, cu_seqlens)
    sequence_denominators = _sequence_sums(
        denominator_mask.to(torch.float32), cu_seqlens
    )
    ids, n_units = _group_ids(
        sequence_numerators.numel(),
        unit_size,
        group_sizes,
        loss.device,
    )
    unit_numerators = torch.zeros(n_units, dtype=torch.float32, device=loss.device)
    unit_denominators = torch.zeros(n_units, dtype=torch.float32, device=loss.device)
    unit_numerators.scatter_add_(0, ids, sequence_numerators)
    unit_denominators.scatter_add_(0, ids, sequence_denominators)
    return _reduce_unit_means(
        unit_numerators,
        unit_denominators,
        local_mean=local_mean,
    )


@dataclass(frozen=True, slots=True)
class PolicyGradientReduction:
    """Validated local and distributed reduction policy for actor losses.

    ``aggregate(..., local_mean=True)`` returns a microbatch mean. With
    ``local_mean=False`` it returns the corresponding numerator term expected by
    the training engine's global normalizer contract. ``normalizer_fn`` returns
    the matching number of active tokens, sequences, or prompt groups.
    """

    mode: LossAggregationMode = LOSS_AGGREGATION_TOKEN_MEAN
    group_size: int = 1
    divisor: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in LOSS_AGGREGATIONS_ALL:
            raise ValueError(
                f"loss_aggregation must be one of {LOSS_AGGREGATIONS_ALL}, "
                f"got {self.mode!r}."
            )
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}.")
        if self.mode != LOSS_AGGREGATION_PROMPT_MEAN and self.group_size != 1:
            raise ValueError(
                "group_size is only valid for loss_aggregation='prompt_mean'."
            )
        if self.mode == LOSS_AGGREGATION_CONSTANT:
            if (
                self.divisor is None
                or not math.isfinite(self.divisor)
                or self.divisor <= 0
            ):
                raise ValueError(
                    "divisor must be a positive finite value for "
                    "loss_aggregation='constant'."
                )
        elif self.divisor is not None:
            raise ValueError("divisor is only valid for loss_aggregation='constant'.")

    def _require_divisor(self) -> float:
        if self.divisor is None:
            raise ValueError(
                "a positive divisor is required for loss_aggregation='constant'."
            )
        return self.divisor

    def _require_sequence_boundaries(
        self, loss_mask: torch.Tensor, cu_seqlens: torch.Tensor | None
    ) -> None:
        if cu_seqlens is None and loss_mask.ndim == 1:
            raise ValueError(
                f"loss_aggregation='{self.mode}' requires cu_seqlens for packed "
                "inputs; tree-packed training currently supports only token_mean."
            )

    def normalizer_fn(self, data: dict[str, Any]) -> torch.Tensor:
        """Return this reduction's active local denominator."""
        loss_mask = data["loss_mask"].bool()
        if self.mode == LOSS_AGGREGATION_TOKEN_MEAN:
            return loss_mask.count_nonzero()

        self._require_sequence_boundaries(loss_mask, data.get("cu_seqlens"))
        sequence_denominators = _sequence_sums(
            loss_mask.to(torch.float32), data.get("cu_seqlens")
        )
        group_sizes = (
            data.get("group_sizes")
            if self.mode == LOSS_AGGREGATION_PROMPT_MEAN
            else None
        )
        ids, n_units = _group_ids(
            sequence_denominators.numel(),
            self.group_size if self.mode == LOSS_AGGREGATION_PROMPT_MEAN else 1,
            group_sizes,
            loss_mask.device,
        )
        unit_denominators = torch.zeros(
            n_units, dtype=torch.float32, device=loss_mask.device
        )
        unit_denominators.scatter_add_(0, ids, sequence_denominators)
        return unit_denominators.count_nonzero().to(torch.float32)

    def aggregate(
        self,
        loss: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        denominator_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        group_sizes: GroupSizes | None = None,
        local_mean: bool = True,
    ) -> torch.Tensor:
        """Aggregate a token-shaped policy-gradient loss."""
        numerator_mask, denominator_mask = _resolve_masks(
            loss, loss_mask, denominator_mask
        )
        if self.mode == LOSS_AGGREGATION_TOKEN_MEAN:
            numerator = _masked_loss(loss, numerator_mask).sum()
            if not local_mean:
                return numerator
            return numerator / denominator_mask.count_nonzero().clamp_min(1)

        self._require_sequence_boundaries(loss, cu_seqlens)
        if self.mode == LOSS_AGGREGATION_CONSTANT:
            divisor = self._require_divisor()
            numerator = _masked_loss(loss, numerator_mask).sum()
            if not local_mean:
                return numerator / divisor
            active_sequences = _sequence_sums(
                denominator_mask.to(torch.float32), cu_seqlens
            ).count_nonzero()
            return numerator / (active_sequences.clamp_min(1) * divisor)

        unit_size = self.group_size if self.mode == LOSS_AGGREGATION_PROMPT_MEAN else 1
        return _aggregate_units(
            loss,
            numerator_mask,
            denominator_mask,
            unit_size=unit_size,
            group_sizes=(
                group_sizes if self.mode == LOSS_AGGREGATION_PROMPT_MEAN else None
            ),
            cu_seqlens=cu_seqlens,
            local_mean=local_mean,
        )
