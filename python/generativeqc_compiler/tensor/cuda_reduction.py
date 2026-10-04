"""Shared legality and resource planning for TensorIR CUDA reductions.

The scientific reduction remains TensorIR-owned.  This module only decides
whether the existing cooperative CUDA reduction shape is active and which
lowering provider is requested.  CUB is qualification-only: the production
default remains the generated reduction.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from math import prod
from typing import Literal

from .cuda_dtype import scalar_type
from .cuda_plan import ELEMENTWISE

ReductionProvider = Literal["generated", "cub"]
REDUCTION_PROVIDERS: tuple[ReductionProvider, ...] = ("generated", "cub")


@dataclass(frozen=True)
class StreamedReductionFusionGroup:
    """Compatible streamed reductions that share virtual producer work."""

    steps: tuple[int, ...]
    shared_virtual_steps: tuple[int, ...]
    reduction_extent: int
    output_shape: tuple[int, ...]
    dtype: str
    accumulation_dtype: str

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "steps": self.steps,
            "shared_virtual_steps": self.shared_virtual_steps,
            "reduction_extent": self.reduction_extent,
            "output_shape": self.output_shape,
            "dtype": self.dtype,
            "accumulation_dtype": self.accumulation_dtype,
        }


def reduction_extent(node: typing.Any) -> int:
    """Return the flattened reduction domain for a reduction-like TensorIR node."""

    if node.op == "reduce":
        return prod(node.inputs[0].spec.shape[axis] for axis in node.attrs["axes"])
    if node.op == "einsum":
        domains: dict[str, int] = {}
        for child, labels in zip(node.inputs, node.attrs["labels"], strict=True):
            domains.update(zip(labels, child.spec.shape, strict=True))
        return prod(
            extent
            for label, extent in domains.items()
            if label not in node.attrs["output"]
        )
    raise ValueError("reduction extent requires a TensorIR reduce or einsum node")


def cooperative_reduction_provider(
    plan: typing.Any, index: int
) -> ReductionProvider | None:
    """Return the active cooperative provider, or None for ordinary lowering."""

    step = plan.steps[index]
    reduction_like = step.node.op == "reduce" or (
        plan.schedule.streamed_gemm_reduction
        and step.node.op == "einsum"
        and step.gemm == "none"
        and any(plan.steps[child].virtual for child in step.inputs)
    )
    if (
        not plan.schedule.stream_reductions
        or step.virtual
        or not reduction_like
        or plan.target.warp_size != 32
        or reduction_extent(step.node) < plan.target.warp_size
    ):
        return None
    provider = plan.schedule.reduction_provider
    if provider not in REDUCTION_PROVIDERS:
        raise ValueError(f"unknown CUDA reduction provider {provider!r}")
    return typing.cast("ReductionProvider", provider)


def cooperative_reduction_shared_bytes(plan: typing.Any, index: int) -> int:
    """Planned shared-memory bytes for one cooperative reduction kernel.

    The generated warp-tree lowering has an exact partial-warp footprint.
    CUB TempStorage is toolkit/header implementation detail, so use one
    accumulator per thread as a conservative static bound.  PTXAS resource
    evidence remains authoritative before any performance promotion.
    """

    provider = cooperative_reduction_provider(plan, index)
    if provider is None:
        return 0
    accumulator = scalar_type(
        plan.precision_by_node[plan.steps[index].node].accumulation_dtype
    )
    if provider == "generated":
        warps = (
            plan.schedule.threads + plan.target.warp_size - 1
        ) // plan.target.warp_size
        return warps * accumulator.itemsize
    return plan.schedule.threads * accumulator.itemsize


def _virtual_accesses(
    plan: typing.Any, index: int
) -> dict[int, tuple[typing.Any, ...]]:
    """Prove equal logical accesses without implementing another index mapper.

    Equal canonical TensorIR reduction attributes and operand shapes imply equal
    root accesses. Only flat-index-preserving primitives propagate that proof.
    Other virtual primitives are opaque boundaries: their own value can be
    shared, but an ancestor behind them has no proved access. Multiple distinct
    or unproved paths to a producer exclude it, including a second occurrence
    of the same operand with different einsum labels.
    """

    step = plan.steps[index]
    node = step.node
    if node.op == "reduce":
        accesses = [("reduce", node.inputs[0].spec.shape, node.attrs["axes"])]
    else:
        topology = (
            "einsum",
            tuple(child.spec.shape for child in node.inputs),
            node.attrs["labels"],
            node.attrs["output"],
        )
        accesses = [(*topology, labels) for labels in node.attrs["labels"]]
    pending: list[tuple[int, tuple[typing.Any, ...] | None]] = list(
        zip(step.inputs, accesses, strict=True)
    )
    seen: dict[int, set[tuple[typing.Any, ...] | None]] = {}
    while pending:
        child, access = pending.pop()
        if not plan.steps[child].virtual:
            continue
        paths = seen.setdefault(child, set())
        if access in paths:
            continue
        paths.add(access)
        producer = plan.steps[child]
        preserved = producer.node.op in ELEMENTWISE or producer.node.op in (
            "cast",
            "reshape",
        )
        pending.extend(
            (operand, access if preserved else None) for operand in producer.inputs
        )
    result = {}
    for producer, paths in seen.items():
        if len(paths) == 1:
            access = next(iter(paths))
            if access is not None:
                result[producer] = access
    return result


def _independent_consumers(plan: typing.Any, steps: tuple[int, ...]) -> bool:
    """Reject sibling groups with dependencies across any materialized boundary."""

    members = set(steps)
    pending = [child for index in steps for child in plan.steps[index].inputs]
    seen: set[int] = set()
    while pending:
        child = pending.pop()
        if child in members:
            return False
        if child not in seen:
            seen.add(child)
            pending.extend(plan.steps[child].inputs)
    return True


def streamed_reduction_fusion_groups(
    plan: typing.Any,
) -> tuple[StreamedReductionFusionGroup, ...]:
    """Find compatible streamed reductions that can share virtual producer work.

    This is a schedule fact only: it does not change lowering or claim that a
    persistent kernel exists. Grouping is deliberately conservative. Two
    reductions are grouped only when they use the active cooperative lowering,
    have matching proved accesses and precision, are independent siblings, and
    share at least one virtual producer. Unproved mappings remain ungrouped.
    Virtual producers with the same consumer set are
    collapsed into one deterministic group so the emitter can later consume a
    stable fusion identity without rediscovering graph relationships.
    """

    candidates: dict[
        tuple[int, tuple[int, ...], str, str],
        list[tuple[int, dict[int, tuple[typing.Any, ...]]]],
    ] = {}
    for index, step in enumerate(plan.steps):
        if cooperative_reduction_provider(plan, index) is None:
            continue
        ancestors = _virtual_accesses(plan, index)
        if not ancestors:
            continue
        precision = plan.precision_by_node[step.node]
        signature = (
            reduction_extent(step.node),
            tuple(step.node.spec.shape),
            step.node.spec.dtype,
            precision.accumulation_dtype,
        )
        candidates.setdefault(signature, []).append((index, ancestors))

    groups: list[StreamedReductionFusionGroup] = []
    independent: dict[tuple[int, ...], bool] = {}
    for signature, members in candidates.items():
        producer_consumers: dict[tuple[int, tuple[typing.Any, ...]], set[int]] = {}
        for index, ancestors in members:
            for producer, access in ancestors.items():
                producer_consumers.setdefault((producer, access), set()).add(index)

        shared_by_steps: dict[tuple[int, ...], set[int]] = {}
        for (producer, _), consumers in producer_consumers.items():
            steps = tuple(sorted(consumers))
            if len(steps) < 2:
                continue
            if steps not in independent:
                independent[steps] = _independent_consumers(plan, steps)
            if not independent[steps]:
                continue
            shared_by_steps.setdefault(steps, set()).add(producer)

        extent, output_shape, dtype, accumulation_dtype = signature
        for steps, producers in shared_by_steps.items():
            groups.append(
                StreamedReductionFusionGroup(
                    steps=steps,
                    shared_virtual_steps=tuple(sorted(producers)),
                    reduction_extent=extent,
                    output_shape=output_shape,
                    dtype=dtype,
                    accumulation_dtype=accumulation_dtype,
                )
            )
    return tuple(
        sorted(groups, key=lambda group: (group.steps, group.shared_virtual_steps))
    )
