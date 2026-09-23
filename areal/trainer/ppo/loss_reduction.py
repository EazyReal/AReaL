# SPDX-License-Identifier: Apache-2.0

"""Prepare batch metadata and bind pure policy-gradient reductions."""

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from areal.utils.data import TRANSPORT_DUMMY_KEY, get_batch_size
from areal.utils.functional.loss_aggregation import (
    ConstantLength,
    GroupSizes,
    LossAggregationMode,
    PolicyGradientReduction,
    PromptMean,
    SequenceMean,
    TokenMean,
)

PG_TOKEN_WEIGHTS = "_pg_token_weights"
_MODES = ("token_mean", "seq_mean", "prompt_mean", "constant")


@dataclass(frozen=True, slots=True)
class PreparedLossStep:
    """Bind loss-side batches to one optimizer step's reduction contract.

    The callable retains only configuration and small step constants. Original
    masks and prepared coefficients come from the actual microbatch, including
    its device and packed layout, when the engine invokes either callback.
    """

    bind: Callable[[dict[str, Any]], PolicyGradientReduction]

    def loss_weight(self, data: dict[str, Any]) -> torch.Tensor:
        return self.bind(data).normalizer()


def prepare_policy_gradient_batch(
    data: dict[str, Any],
    *,
    mode: LossAggregationMode,
    group_sizes: GroupSizes | None = None,
) -> torch.Tensor | None:
    """Prepare original coefficients before response-level batch splitting.

    Prompt mean adds one token-shaped internal tensor and returns the exact
    local count of active full groups. Other modes need no prepared metadata.
    Group boundaries never enter the batch dictionary or affect scheduling.
    """
    if mode not in _MODES:
        raise ValueError(f"loss_aggregation must be one of {_MODES}, got {mode!r}.")
    if mode != "prompt_mean":
        return None
    mask = data["loss_mask"].bool()
    if mask.ndim != 2:
        raise ValueError("Preparing prompt mean requires a 2D loss_mask.")
    if group_sizes is None:
        raise ValueError("group_sizes are required to prepare prompt mean.")
    if torch.is_tensor(group_sizes):
        raise TypeError("group_sizes must be a sequence of ints, not a tensor.")
    sizes = [int(size) for size in group_sizes]
    if any(size <= 0 for size in sizes):
        raise ValueError(f"group_sizes must be positive, got {sizes}.")
    if sum(sizes) != mask.shape[0]:
        raise ValueError(
            f"group_sizes sum to {sum(sizes)} but sequence count is {mask.shape[0]}."
        )
    ids = torch.arange(len(sizes), device=mask.device).repeat_interleave(
        torch.tensor(sizes, dtype=torch.long, device=mask.device),
        output_size=mask.shape[0],
    )
    counts = torch.zeros(len(sizes), dtype=torch.int64, device=mask.device)
    counts.scatter_add_(0, ids, mask.sum(dim=-1, dtype=torch.int64))
    data[PG_TOKEN_WEIGHTS] = mask.float() / counts[ids].unsqueeze(-1).clamp_min(1)
    return counts.count_nonzero()


def prepare_policy_gradient_steps(
    microbatches: Sequence[dict[str, Any]],
    *,
    mode: LossAggregationMode,
    divisor: float | None = None,
    local_active_groups: torch.Tensor | None = None,
    dp_group: dist.ProcessGroup | None = None,
    device: torch.device | str | None = None,
) -> list[PreparedLossStep]:
    """Prepare callbacks after the synchronized optimizer schedule is known.

    For prompt mean, reduce exact active-group and real-response counts over DP
    once. The realized step count K and global active groups G fix the objective
    scale K/G; each microbatch receives its step's physical-response share q.
    A real all-masked step remains in the schedule, while transport rows have
    zero share. An update with G=0 is rejected on every participating rank.

    The scheduler owns identical step counts across ranks. The collective group
    excludes CP replicas; engines retain their existing CP/DDP compensation.
    Small counts are copied to the host once here, so binding needs no tensor
    transfers or host synchronization for CPU versus accelerator batches.
    """
    if mode not in _MODES:
        raise ValueError(f"loss_aggregation must be one of {_MODES}, got {mode!r}.")
    if mode == "constant":
        if divisor is None or not math.isfinite(divisor) or divisor <= 0:
            raise ValueError("divisor must be a positive finite value.")
    elif divisor is not None:
        raise ValueError("divisor is only valid for loss_aggregation='constant'.")
    if not microbatches:
        raise ValueError("Cannot prepare an empty optimizer schedule.")

    if mode == "token_mean":
        step = PreparedLossStep(_bind_token_mean)
    elif mode == "seq_mean":
        step = PreparedLossStep(_bind_sequence_mean)
    elif mode == "constant":
        step = PreparedLossStep(
            functools.partial(_bind_constant_length, divisor=divisor)
        )
    else:
        if local_active_groups is None:
            raise ValueError("Prompt mean requires the prepared active-group count.")
        collective_device = device if device is not None else local_active_groups.device
        if dist.is_initialized() and dist.get_backend(dp_group) == "gloo":
            collective_device = "cpu"
        row_counts = torch.tensor(
            [
                0 if mb.get(TRANSPORT_DUMMY_KEY) is True else get_batch_size(mb)
                for mb in microbatches
            ],
            dtype=torch.int64,
            device=collective_device,
        )
        counts = torch.cat(
            [local_active_groups.reshape(1).to(device=collective_device), row_counts]
        )
        if dist.is_initialized():
            dist.all_reduce(counts, group=dp_group)
        active_groups, *global_rows = counts.cpu().tolist()
        if active_groups == 0:
            raise ValueError("Prompt mean requires active prompt groups in the update.")
        if any(rows <= 0 for rows in global_rows):
            raise ValueError("Every optimizer step must contain real responses.")
        step_scale = len(microbatches) / active_groups
        return [
            PreparedLossStep(
                functools.partial(
                    _bind_prompt_mean, step_scale=step_scale, global_rows=rows
                )
            )
            for rows in global_rows
        ]
    return [step] * len(microbatches)


def _bind_token_mean(data: dict[str, Any]) -> TokenMean:
    return TokenMean(data["loss_mask"])


def _bind_sequence_mean(data: dict[str, Any]) -> SequenceMean:
    return SequenceMean(data["loss_mask"], data.get("cu_seqlens"))


def _bind_constant_length(data: dict[str, Any], *, divisor: float) -> ConstantLength:
    return ConstantLength(divisor, data["loss_mask"], data.get("cu_seqlens"))


def _bind_prompt_mean(
    data: dict[str, Any], *, step_scale: float, global_rows: int
) -> PromptMean:
    weights = data.get(PG_TOKEN_WEIGHTS)
    if weights is None:
        raise ValueError("Prompt mean requires prepared token weights.")
    mask = data["loss_mask"]
    if weights.shape != mask.shape or weights.device != mask.device:
        raise ValueError(
            "Prepared token weights must match the loss mask shape/device."
        )
    rows = 0 if data.get(TRANSPORT_DUMMY_KEY) is True else get_batch_size(data)
    return PromptMean(weights, step_scale, rows / global_rows)
