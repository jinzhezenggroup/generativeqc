"""Fixed-shape projection of qualified layout-copy contraction portfolios.

The request remains the canonical tensor operation. This adapter emits only
execution descriptors for the existing shared native matrix owner.
"""

from __future__ import annotations

import json
import math
import typing

from generativeqc_compiler.common.lowering_provider import candidate_layout_rejection
from generativeqc_compiler.common.native_lowering import native_lowering_portfolio

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from generativeqc_compiler.common.lowering_provider import (
        LoweringCandidate,
        LoweringRequest,
    )
    from generativeqc_compiler.common.specialization import (
        CompilationIdentity,
        TargetCapabilities,
    )

FAMILY = "tensor.cuda.affine-copy.v1"


def _initializer(candidate: LoweringCandidate, *, original: bool) -> str:
    request = candidate.request
    execution = candidate.execution
    assert execution is not None
    layouts = request.operands if original else execution.layouts
    dtype = "generativeqc::runtime::PrecisionDtype::Fp64"
    operands = [
        "{2,{"
        + ",".join(map(str, row.modes))
        + "},{"
        + ",".join(map(str, row.shape))
        + "},{"
        + ",".join(map(str, row.strides or ()))
        + "},"
        + dtype
        + "}"
        for row in layouts
    ]
    precision = execution.precision
    directive = precision.directive
    m, k = layouts[0].shape
    _, n = layouts[1].shape
    semantics = dict(request.semantics)
    return (
        "generativeqc::tensor::ContractionRequest{"
        + ",".join(
            map(
                json.dumps,
                (
                    request.scientific_identity,
                    request.semantic_identity,
                    precision.identity,
                ),
            )
        )
        + ",{{"
        + ",".join(operands)
        + "}},{"
        + ",".join(
            (
                dtype,
                dtype,
                dtype,
                json.dumps(directive.qualification or ""),
                json.dumps(directive.math_mode),
            )
        )
        + "},"
        + dtype
        + ",'N','N',1,"
        + f"{m},{n},{k},"
        + float(semantics.get("alpha", 1.0)).hex()
        + ",{},"
        + float(semantics.get("beta", 0.0)).hex()
        + "}"
    )


def emit_layout_contraction_portfolio(
    request: LoweringRequest,
    candidates: Sequence[LoweringCandidate],
    target: TargetCapabilities,
    compilation: CompilationIdentity,
    *,
    name: str,
) -> str:
    """Emit an explicit copy/contract/publication factory and joint portfolio.

    Ready offers use the v1 same-storage FP64 executor and its existing generated
    or pedantic matrix backend. All dimensions are exact; capture, implicit casts,
    checked updates, triangular/aliased storage and method policy remain outside
    this executor. Production profitability is not inferred from estimated costs.
    """
    semantics = dict(request.semantics)
    if request.backend != "cuda" or request.operation != "einsum" or request.effects:
        raise ValueError("layout executor requires a plain CUDA binary einsum")
    if set(semantics) - {"alpha", "beta"} or any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in semantics.values()
    ):
        raise ValueError("layout executor requires explicit finite scalar alpha/beta")
    if len(request.operands) != 3 or request.input_dtypes != ("float64",) * (
        3 if semantics.get("beta", 0.0) != 0 else 2
    ):
        raise ValueError("layout executor requires two FP64 reads and one publication")
    a, b, c = request.operands
    if any(
        len(row.shape) != 2
        or not all(row.shape)
        or row.strides is None
        or row.triangle != "full"
        or row.alias_group is not None
        or row.alignment > 8
        for row in request.operands
    ):
        raise ValueError(
            "layout executor requires positive disjoint affine matrix views"
        )
    if (
        len({*a.modes, *b.modes, *c.modes}) != 3
        or a.modes != (c.modes[0], b.modes[0])
        or b.modes[1] != c.modes[1]
        or a.access != "read"
        or b.access != "read"
        or c.access != ("read-write" if semantics.get("beta", 0.0) != 0 else "write")
    ):
        raise ValueError(
            "layout executor does not implement the requested axes/effects"
        )
    factories = []
    for i, candidate in enumerate(candidates):
        if candidate.status != "ready":
            continue
        execution = candidate.execution
        if execution is None or not execution.layout_transfers:
            raise ValueError(
                "ready layout portfolio offers require explicit transfer execution"
            )
        precision = execution.precision
        if (
            candidate_layout_rejection(candidate) is not None
            or precision.input_dtypes != request.input_dtypes
            or precision.publication_dtype != "float64"
            or {
                precision.directive.storage_dtype,
                precision.directive.compute_dtype,
                precision.directive.accumulation_dtype,
            }
            != {"float64"}
            or precision.casts
            or precision.refinement
            or precision.audit
            or execution.capture_safe
            or (
                candidate.providers[0].name == "cublas"
                and execution.determinism == "exact-order"
            )
            or tuple(p.name for p in candidate.providers)
            not in (("generated.cuda",), ("cublas",))
            or execution.algorithm
            != (
                "affine-copy-v1-generated"
                if candidate.providers[0].name == "generated.cuda"
                else "affine-copy-v1-blas"
            )
            or any(t.implementation != FAMILY for t in execution.layout_transfers)
            or any(
                row.strides != (row.shape[1], 1) or row.alignment > 8
                for row in execution.layouts
            )
        ):
            raise ValueError("layout offer is outside the v1 executable contract")
        factories.append(
            f"case {i}: return {{"
            + json.dumps(request.identity)
            + ","
            + json.dumps(candidate.identity)
            + ","
            + _initializer(candidate, original=True)
            + ","
            + _initializer(candidate, original=False)
            + "};"
        )
    portfolio = native_lowering_portfolio(
        request,
        candidates,
        target,
        compilation,
        name=name,
        layout_transfer_executor=FAMILY,
    )
    return portfolio + (
        f"\nstatic generativeqc::tensor::LayoutContractionPlan {name}_plan(std::size_t selected){{"
        "\nswitch(selected){\n"
        + "\n".join(factories)
        + '\ndefault: throw std::invalid_argument("selected layout executor is unavailable");\n}}\n'
    )
