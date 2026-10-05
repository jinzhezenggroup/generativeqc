"""Adapt TensorIR science and physical facts to the common provider boundary.

No matrix recognition or vendor information participates in this adapter. An
einsum stays an einsum when a provider implements it using GEMM, packing, or a
generated reduction. The same adapter is usable by CPU and CUDA lowerers.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, replace
from functools import cached_property
from math import prod

from generativeqc_compiler.common.layout import DenseLayout
from generativeqc_compiler.common.lowering_contract import (
    LoweringConstraints,
    LoweringPrecision,
    OperandLayout,
)
from generativeqc_compiler.common.lowering_provider import LoweringRequest
from generativeqc_compiler.common.precision import (
    ExecutionPrecisionSchedule,
    PrecisionDirective,
)
from generativeqc_compiler.common.provenance import canonical_hash

from .ir import Node
from .precision import PrecisionSchedule, describe_precision
from .program import Program, node_hashes

if typing.TYPE_CHECKING:
    from .cuda_plan import TensorPlan


@dataclass(frozen=True)
class TensorLoweringAdapter:
    """Resolve program-wide precision and node hashes once per preparation.

    A plan may have thousands of operation occurrences. Rebuilding these maps
    per candidate would turn linear semantic projection into quadratic work.
    The adapter is owned by the preparation call and has no global cache.
    """

    program: Program

    def __post_init__(self) -> None:
        if not isinstance(self.program, Program):
            raise TypeError("tensor adapter requires Program")

    @cached_property
    def precision(self) -> PrecisionSchedule:
        return describe_precision(self.program)

    @cached_property
    def live_nodes(self) -> frozenset[Node]:
        return frozenset(self.program.live_nodes)

    @cached_property
    def hashes(self) -> dict[Node, str]:
        return node_hashes(self.program.nodes)

    @cached_property
    def directives(self) -> dict[Node, PrecisionDirective]:
        regions = dict(self.precision.execution_precision.regions)
        names = self.program.debug_names
        return {
            node: regions[names[node]]
            for node in self.live_nodes
            if names[node] in regions
        }

    def request(
        self,
        node: Node,
        *,
        backend: str,
        layouts: tuple[DenseLayout | None, ...] | None = None,
        precisions: tuple[LoweringPrecision, ...] | None = None,
        constraints: LoweringConstraints | None = None,
    ) -> LoweringRequest:
        """Project one existing logical node and supplied physical operand views.

        ``precisions`` must come from scientific admission against this original
        node; providers must not call lower_precision themselves to invent offers.
        Without variants the adapter describes only the already resolved arithmetic.
        Input views precede the output, and None denotes a virtual input expression.
        """
        if not isinstance(node, Node):
            raise TypeError("tensor request requires Program and Node")
        if node not in self.live_nodes:
            raise ValueError("tensor request node is not live in the program")
        if node.spec.dtype not in ("float32", "float64"):
            raise ValueError(
                "typed tensor provider requests currently require floating output"
            )
        values = (*node.inputs, node)
        if layouts is None:
            layouts = tuple(DenseLayout(value.spec.shape) for value in values)
        if len(layouts) != len(values):
            raise ValueError("tensor request requires input and output layouts")
        for value, layout in zip(values, layouts, strict=True):
            if layout is not None and (
                not isinstance(layout, DenseLayout) or layout.shape != value.spec.shape
            ):
                raise ValueError("tensor layout must match the logical operand shape")
        if node.op == "einsum":
            modes = (*node.attrs["labels"], node.attrs["output"])
        elif node.op == "reduce":
            axes = tuple(range(len(node.inputs[0].spec.shape)))
            modes = (
                axes,
                tuple(axis for axis in axes if axis not in node.attrs["axes"]),
            )
        else:
            # The complete existing Node hash remains authoritative for non-einsum
            # operations; these ordinals only describe corresponding view axes.
            domains: dict[typing.Any, int] = {}
            rows = []
            for value in values:
                rows.append(
                    tuple(
                        domains.setdefault((index.name, index.domain), len(domains))
                        for index in value.spec.indices
                    )
                )
            modes = tuple(rows)
        operands = tuple(
            OperandLayout(
                operand=f"input:{index}" if index < len(node.inputs) else "output",
                modes=labels,
                shape=value.spec.shape,
                strides=None if layout is None else layout.element_strides,
                access="read" if index < len(node.inputs) else "write",
                alignment=1 if layout is None else layout.alignment,
            )
            for index, (value, labels, layout) in enumerate(
                zip(values, modes, layouts, strict=True)
            )
        )
        precision = self.precision
        directive = self.directives[node]
        if precisions is None:
            precisions = (
                LoweringPrecision(
                    schedule=ExecutionPrecisionSchedule(
                        (("operation", directive),),
                        strict_audit_dtype=precision.strict_audit_dtype,
                        audit_owner=precision.audit_owner,
                        math_mode=precision.math_mode,
                    ),
                    region="operation",
                    input_dtypes=tuple(value.spec.dtype for value in node.inputs),
                    publication_dtype=node.spec.dtype,
                ),
            )
        semantics = [
            ("node_hash", self.hashes[node]),
            (
                "symmetry_identity",
                canonical_hash(
                    [
                        [
                            (symmetry.permutation, symmetry.sign)
                            for symmetry in value.spec.symmetries
                        ]
                        for value in values
                    ]
                ),
            ),
        ]
        if node.op in ("reduce", "einsum"):
            output_modes = set(modes[-1])
            extents = {
                label: extent
                for labels, value in zip(modes, values, strict=True)
                for label, extent in zip(labels, value.spec.shape, strict=True)
            }
            semantics.extend(
                (
                    ("output_elements", node.spec.size),
                    (
                        "reduction_extent",
                        prod(
                            extent
                            for label, extent in extents.items()
                            if label not in output_modes
                        ),
                    ),
                )
            )
        return LoweringRequest(
            consumer="tensor",
            operation=node.op,
            backend=backend,
            dtype=node.spec.dtype,
            accumulation_dtype=directive.accumulation_dtype,
            shape=node.spec.shape,
            semantics=tuple(semantics),
            scientific_identity=precision.source_equation,
            operands=operands,
            input_dtypes=tuple(value.spec.dtype for value in node.inputs),
            precisions=precisions,
            constraints=constraints,
            effects=(("output", "fresh-ssa-value"),),
        )


def tensor_lowering_request(
    program: Program,
    node: Node,
    *,
    backend: str,
    layouts: tuple[DenseLayout | None, ...] | None = None,
    precisions: tuple[LoweringPrecision, ...] | None = None,
    constraints: LoweringConstraints | None = None,
) -> LoweringRequest:
    """One-shot convenience adapter; reuse TensorLoweringAdapter for whole plans."""
    return TensorLoweringAdapter(program).request(
        node,
        backend=backend,
        layouts=layouts,
        precisions=precisions,
        constraints=constraints,
    )


def plan_lowering_request(
    plan: TensorPlan, index: int, adapter: TensorLoweringAdapter
) -> LoweringRequest:
    """Bind the existing node and arena views for every provider's candidate set.

    Packing benefit, provider name and selected precision do not define a new
    operation. Optional providers must reuse this request for planned sites.
    """
    step = plan.steps[index]
    node = step.node
    if node.spec.dtype == "int64":
        # Integer control operations have no floating precision variant yet.
        return LoweringRequest(
            consumer="tensor",
            operation=node.op,
            backend="cuda",
            dtype="int64",
            accumulation_dtype="int64",
            shape=node.spec.shape,
            semantics=(("program_hash", plan.program.logical_hash),),
        )
    request = adapter.request(
        node,
        backend="cuda",
        layouts=tuple(plan.steps[child].layout for child in step.inputs)
        + (step.layout,),
    )
    # The mathematical output is SSA, but the storage planner may donate a dead
    # input allocation. Preserve that execution precondition at the boundary.
    return replace(
        request,
        operands=tuple(
            replace(layout, alias_group=f"arena:{plan.steps[owner].offset}")
            if not plan.steps[owner].virtual
            else layout
            for layout, owner in zip(
                request.operands, (*step.inputs, index), strict=True
            )
        ),
        effects=(
            (
                "output",
                "donated" if step.donated_from is not None else "fresh-ssa-value",
            ),
        ),
    )
