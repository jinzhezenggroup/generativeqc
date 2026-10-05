"""Prepare native vector-result contractions from canonical TensorIR requests.

These provider templates retain the two existing FP64 BLAS algorithms. They
offer the same semantic operation; unknown endpoint costs preserve the qualified
incumbent, never authorize switching an algorithm or precision in method code.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from generativeqc_compiler.common.backend import TargetInfo
from generativeqc_compiler.common.lowering_contract import (
    CandidateExecution,
    LoweringConstraints,
)
from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
)
from generativeqc_compiler.common.native_lowering import native_lowering_portfolio
from generativeqc_compiler.common.specialization import (
    CompilationIdentity,
    TargetCapabilities,
)

from .df_coulomb import coulomb_program
from .lowering import TensorLoweringAdapter
from .native_lowering import contraction_initializer

if TYPE_CHECKING:
    from .ir import Node


def coulomb_portfolio(
    packed: bool, output: str, source_identity: str
) -> tuple[
    TensorLoweringAdapter,
    Node,
    LoweringRequest,
    tuple[LoweringCandidate, ...],
    TargetCapabilities,
    CompilationIdentity,
]:
    """Project a representative template; native descriptors bind actual shapes."""
    program = coulomb_program(2, 3, 5, packed=packed)
    node = program.outputs[output]
    adapter = TensorLoweringAdapter(program)
    request = replace(
        adapter.request(node, backend="cuda"),
        constraints=LoweringConstraints(maximum_candidates=2),
    )
    candidates, target, compilation = vector_portfolio(request, source_identity)
    return adapter, node, request, candidates, target, compilation


def vector_portfolio(
    request: LoweringRequest, source_identity: str
) -> tuple[tuple[LoweringCandidate, ...], TargetCapabilities, CompilationIdentity]:
    """Register the same executable algorithms for any admitted vector region."""
    target = TargetCapabilities(
        TargetInfo("cuda", "current-native-module", 32, 1024, None)
    )
    assert request.scientific_identity is not None
    compilation = CompilationIdentity(request.scientific_identity, source_identity)
    provider = ProviderDescriptor(
        "cublas", "library", "vector-contraction", version="runtime-query-required"
    )
    candidates = tuple(
        LoweringCandidate(
            request,
            algorithm,
            (provider,),
            "ready",
            "fp64",
            execution=CandidateExecution(
                request.precisions[0],
                algorithm,
                request.operands,
                capture_safe=True,
            ),
            target=target,
        )
        for algorithm in ("gemv", "gemm-strided-batched-vector")
    )
    return candidates, target, compilation


def coulomb_header(source_identity: str) -> str:
    """Emit factories whose choices are backend metadata, never method flags."""
    pieces = [
        '#pragma once\n#include "tensor/cuda_vector_contraction.hpp"\n',
        "namespace generativeqc::scf::cuda_df::coulomb_lowering {\n",
    ]
    dimensions = {"batch": "batch", "ao": "nbf", "pair": "pairs", "auxiliary": "naux"}
    for packed in (False, True):
        for output in ("charge", "coulomb"):
            adapter, node, request, candidates, target, compilation = coulomb_portfolio(
                packed, output, source_identity
            )
            name = ("packed_" if packed else "dense_") + output
            pieces.append(
                native_lowering_portfolio(
                    request, candidates, target, compilation, name=name
                )
            )
            pairs = "pairs" if packed else "tensor::contraction_product(nbf,nbf)"
            descriptor = contraction_initializer(
                adapter,
                node,
                lambda index: dimensions[index.space.kind],
                transpose=("T" if output == "charge" else "N", "N"),
                extents=(
                    "batch",
                    "naux" if output == "charge" else pairs,
                    "1",
                    pairs if output == "charge" else "naux",
                ),
                coefficient="1.0",
            )
            # Packed storage uses one shared pair scratch vector; dense storage
            # exposes a true strided batch. Preserve each qualified incumbent.
            pieces.append(f"""
inline std::unique_ptr<tensor::CudaVectorContraction> {name}(
    std::size_t batch, std::size_t nbf, std::size_t naux,
    cublasHandle_t handle, cudaStream_t stream) {{
  const auto pairs = tensor::contraction_product(nbf, nbf+1)/2;
  (void)pairs;
  auto resolved = {descriptor};
  return std::make_unique<tensor::CudaVectorContraction>(
      {name}_request, {name}_candidates, {name}_target, {name}_compilation,
      resolved, {0 if packed else 1}, handle, stream);
}}
""")
    return "\n".join([*pieces, "}\n"])
