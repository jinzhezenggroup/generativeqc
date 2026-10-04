"""Joint native portfolio of the existing occupied-triples W TensorIR region.

This describes an existing program boundary, not a second contraction algebra.
The four boundary operands retain TensorIR axes; physical matrix subrecipes and
runtime strides are validated separately by the shared native tensor executor.
"""

from __future__ import annotations

from generativeqc_compiler.common.backend import TargetInfo
from generativeqc_compiler.common.lowering_contract import (
    CandidateExecution,
    LoweringConstraints,
    LoweringPrecision,
    OperandLayout,
)
from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
)
from generativeqc_compiler.common.native_lowering import native_lowering_portfolio
from generativeqc_compiler.common.precision import (
    CastBoundary,
    ExecutionPrecisionSchedule,
    PrecisionDirective,
)
from generativeqc_compiler.common.schedule import ScheduleTopology
from generativeqc_compiler.common.specialization import (
    CompilationIdentity,
    TargetCapabilities,
)
from generativeqc_compiler.tensor import Program
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter

from .occupied_triples import DF_TRIPLES_W_FP32_QUALIFICATION, moment_program


def emit_w_portfolio(source_identity: str) -> str:
    """Emit strict/mixed × generated/library offers for one unchanged W root.

    Dimensions identify an AOT template. Native preparation resolves workspace,
    row strides, context and algorithm availability before executing that template.
    Missing phase timing stays unknown; no speedup or precision promotion is inferred.
    """
    source = moment_program(2, 3)
    region = Program({"w": source.outputs["w"]}, provenance=source.provenance)
    adapter = TensorLoweringAdapter(source)
    inputs = sorted(
        (node for node in region.live_nodes if node.op == "input"),
        key=lambda node: node.attrs["name"],
    )
    modes = {}
    operands = []
    for name, node, access in [
        *((node.attrs["name"], node, "read") for node in inputs),
        ("w", region.outputs["w"], "write"),
    ]:
        shape = node.spec.shape
        strides, stride = [], 1
        for extent in reversed(shape):
            strides.append(stride)
            stride *= extent
        operands.append(
            OperandLayout(
                name,
                tuple(
                    modes.setdefault(index.name, len(modes))
                    for index in node.spec.indices
                ),
                shape,
                tuple(reversed(strides)),
                access="write" if access == "write" else "read",
            )
        )
    strict = LoweringPrecision(
        ExecutionPrecisionSchedule(
            (("operation", PrecisionDirective("float64", "float64", "float64")),)
        ),
        "operation",
        ("float64",) * 4,
        "float64",
    )
    mixed = LoweringPrecision(
        ExecutionPrecisionSchedule(
            (
                (
                    "operation",
                    PrecisionDirective(
                        "float32", "float32", "float32", DF_TRIPLES_W_FP32_QUALIFICATION
                    ),
                ),
            )
        ),
        "operation",
        ("float32",) * 4,
        "float64",
        casts=tuple(
            CastBoundary(
                node.attrs["name"],
                "float64",
                "float32",
                node.spec.size,
                8 * node.spec.size,
                4 * node.spec.size,
            )
            for node in inputs
        )
        + (
            CastBoundary(
                "products-to-fp64-combine", "float32", "float64", 54, 216, 432
            ),
        ),
        audit=DF_TRIPLES_W_FP32_QUALIFICATION,
    )
    request = LoweringRequest(
        "tensor",
        "program-region",
        "cuda",
        "float64",
        "float64",
        region.outputs["w"].spec.shape,
        scientific_identity=adapter.precision.source_equation,
        operands=tuple(operands),
        precisions=(strict, mixed),
        constraints=LoweringConstraints(maximum_candidates=4),
        semantics=(
            ("root_node", adapter.hashes[source.outputs["w"]]),
            ("shape-kind", "aot-template"),
        ),
        effects=(("output", "fresh-w-value"),),
    )
    target = TargetCapabilities(
        TargetInfo("cuda", "current-aot-module", 32, 1024, None)
    )
    providers = (
        ProviderDescriptor(
            "cublas",
            "library",
            "two-affine-contractions",
            version="runtime-bound-pedantic",
        ),
        ProviderDescriptor(
            "generated.cuda",
            "generated",
            "two-affine-contractions",
            version=source_identity,
        ),
    )
    candidates = tuple(
        LoweringCandidate(
            request,
            f"w-{'strict' if precision == strict else 'mixed'}-{provider.name}",
            (provider,),
            "ready",
            precision.directive.math_mode,
            provider_bytes=(96 << 20) if provider.kind == "library" else 0,
            execution=CandidateExecution(
                precision,
                "two-contractions-fp64-combine",
                request.operands,
                ScheduleTopology(
                    fusion="cast-affine-combine",
                    materialization="bounded-product-scratch",
                    reduction="provider-reproducible",
                ),
                determinism="reproducible",
                capture_safe=False,
            ),
            target=target,
        )
        for precision in (strict, mixed)
        for provider in providers
    )
    return native_lowering_portfolio(
        request,
        candidates,
        target,
        CompilationIdentity(adapter.precision.source_equation, source_identity),
        name="w_lowering",
    )
