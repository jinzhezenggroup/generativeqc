"""Canonical portfolio for the existing native symmetric density/AO product.

The scalar DAG describes one reduction summand; the established CUDA emitters
own its strict and explicitly rounded mixed implementations. Dimensions in the
emitted portfolio are template dimensions. CudaXcPlan separately binds immutable
runtime shapes and mapped AO lifetimes. No method admission or timing is inferred.
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
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.schedule import ScheduleTopology
from generativeqc_compiler.common.specialization import (
    CompilationIdentity,
    TargetCapabilities,
)

from .xc_bilinear import density_summand

QUALIFICATION = "dft.cuda.auto/density-contraction-v1"


def density_portfolio(
    tile: int,
    source: str,
) -> tuple[
    LoweringRequest,
    tuple[LoweringCandidate, ...],
    TargetCapabilities,
    CompilationIdentity,
]:
    """Project the two existing schedules at two admitted arithmetic contracts.

    Costs intentionally remain unknown. Native preparation may retain only the
    established qualified incumbent, never use these templates as benchmark
    evidence. Cast traffic is logical register conversion, not an extra buffer.
    Scientific admission is narrowed by the native method's supplied directive.
    """
    graph, root = density_summand()
    science = canonical_hash(
        {
            "graph": [
                (
                    i,
                    graph.nodes[i].operation,
                    graph.nodes[i].arguments,
                    str(graph.nodes[i].payload),
                )
                for i in graph.topological_order([root])
            ],
            "root": root.identifier,
            "reduction": "increasing-nu",
        }
    )
    strict = LoweringPrecision(
        ExecutionPrecisionSchedule(
            (("operation", PrecisionDirective("float64", "float64", "float64")),)
        ),
        "operation",
        ("float64", "float64"),
        "float64",
    )
    mixed = LoweringPrecision(
        ExecutionPrecisionSchedule(
            (
                (
                    "operation",
                    PrecisionDirective("float64", "float32", "float64", QUALIFICATION),
                ),
            )
        ),
        "operation",
        ("float64", "float64"),
        "float64",
        casts=(
            CastBoundary("summand-loads", "float64", "float32", 3, 24, 12),
            CastBoundary("product-to-reduction", "float32", "float64", 1, 4, 8),
        ),
        refinement="method-controller/strict-density-rebuild",
        audit="method-controller/final-fp64-audit",
    )
    # Representative dimensions identify this AOT template, not a resolved
    # runtime request or a cache key for an arbitrary molecular geometry.
    request = LoweringRequest(
        "dft.density_contraction",
        "symmetric-density-ao",
        "cuda",
        "float64",
        "float64",
        (2, 4, 17, 16),
        scientific_identity=science,
        semantics=(("reduction", "increasing-nu"), ("shape-kind", "aot-template")),
        operands=(
            OperandLayout("density", (0, 3, 4), (2, 16, 16), (256, 16, 1)),
            OperandLayout("ao", (1, 2, 4), (4, 17, 16), (272, 16, 1)),
            OperandLayout(
                "work", (0, 1, 2, 3), (2, 4, 17, 16), (1088, 272, 16, 1), access="write"
            ),
        ),
        precisions=(strict, mixed),
        constraints=LoweringConstraints(
            workspace_bytes=0,
            provider_bytes=0,
            additional_device_bytes=0,
            capture_required=True,
            determinism="reproducible",
            maximum_candidates=4,
        ),
    )
    target = TargetCapabilities(
        TargetInfo("cuda", "current-aot-module", 32, 1024, None)
    )
    source_identity = canonical_hash({"source": source, "tile": tile})
    compiler = CompilationIdentity(science, source_identity)
    provider = ProviderDescriptor(
        "generated.cuda", "generated", "symmetric-density-ao", version=source_identity
    )
    candidates = tuple(
        LoweringCandidate(
            request,
            f"density-{'tiled' if tiled else 'scalar'}-{'mixed' if precision == mixed else 'strict'}",
            (provider,),
            "ready",
            precision.directive.math_mode,
            execution=CandidateExecution(
                precision,
                f"tiled-{tile}" if tiled else "grid-stride-128",
                request.operands,
                ScheduleTopology(
                    tiles=(tile, tile) if tiled else (),
                    workgroup_threads=tile * tile if tiled else 128,
                    fusion="cast-symmetrize-multiply-finite-audit",
                    materialization="borrowed-panels",
                    reduction="increasing-nu",
                ),
                determinism="reproducible",
                capture_safe=True,
            ),
            target=target,
        )
        for precision in (strict, mixed)
        for tiled in (False, True)
    )
    return request, candidates, target, compiler


def emit_density_binding(tile: int, source: str) -> str:
    """Emit prepare-only shared selection and an allocation-free launch table."""
    request, candidates, target, compiler = density_portfolio(tile, source)
    metadata = native_lowering_portfolio(
        request, candidates, target, compiler, name="density_lowering"
    )
    return (
        metadata
        + r"""
// Called once per immutable full/tail or local-map shape, never in replay.
CudaXcDensityBinding prepare_density_binding(I n, I count, I spins, I work_jets,
    generativeqc::runtime::PrecisionDirective admitted, std::uint64_t expected_replays) {
  using namespace generativeqc::runtime;
  const bool strict = admitted.is_strict_fp64();
  if (admitted.math_mode != kStrictPrecisionMathMode ||
      (!strict && (admitted.storage_dtype != PrecisionDtype::Fp64 ||
                   admitted.compute_dtype != PrecisionDtype::Fp32 ||
                   admitted.accumulation_dtype != PrecisionDtype::Fp64 ||
                   admitted.qualification != "dft.cuda.auto/density-contraction-v1")))
    throw std::invalid_argument("unqualified density contraction arithmetic");
  if (n < 0 || count <= 0 || spins <= 0 || work_jets <= 0)
    throw std::invalid_argument("invalid density contraction shape");
  // Shape products are checked before the launcher can form an index or grid.
  if (lowering_multiply(lowering_multiply(lowering_multiply(spins,work_jets),count),n)
      > std::uint64_t(std::numeric_limits<I>::max()))
    throw std::overflow_error("density contraction index overflow");
  auto offers = density_lowering_candidates;
  const bool tiled = tiled_xc_admitted(n,count,spins,work_jets);
  for (std::size_t i=0; i<offers.size(); ++i) {
    if (i>=2 && strict) offers[i].rejection="precision not admitted by scientific owner";
    if (i%2 && !tiled) offers[i].rejection="shape outside qualified tiled launch domain";
  }
  const std::size_t incumbent=(strict ? 0 : 2)+(tiled ? 1 : 0);
  const auto decision=select_native_lowering(density_lowering_request,offers,
      density_lowering_target,density_lowering_compilation,expected_replays,incumbent);
  static const CudaXcDensityLauncher launchers[]{
      launch_density_product<false,false>,launch_density_product<false,true>,
      launch_density_product<true,false>,launch_density_product<true,true>};
  const auto& selected=offers[decision.selected];
  return {launchers[decision.selected], selected,
          density_lowering_precisions[selected.precision], decision.retained_incumbent};
}
"""
    )
