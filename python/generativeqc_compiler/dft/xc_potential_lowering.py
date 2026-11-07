"""Canonical symmetric contraction of the existing compact XC work panels."""

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    transpose,
)
from generativeqc_compiler.tensor.symmetric_product import (
    emit_symmetric_product_portfolio,
)

from .xc_lowering_profile import emit_indexed_fp64_profile


def potential_panel_program(jets: int, points: int, columns: int) -> Program:
    """Publish A^T W + W^T A; compact_panel_program owns all XC coefficients.

    Jet and point axes form a packed reduction. No density, functional or
    precision policy is re-derived here; both inputs are already materialized.
    """
    if any(type(n) is not int or n < 1 for n in (jets, points, columns)):
        raise ValueError("potential panel dimensions must be positive integers")
    j = Index("j", IndexSpace("jets", "component", jets))
    p = Index("p", IndexSpace("points", "batch", points))
    space = IndexSpace("columns", "ao", columns)
    m, n = Index("m", space), Index("n", space)
    a = input_tensor("ao", TensorSpec((j, p, m), role="input"))
    w = input_tensor("weighted", TensorSpec((j, p, n), role="input"))
    cross = einsum("jpm,jpn->mn", a, w)
    return Program({"potential": add(cross, transpose(cross, (1, 0)))})


def emit_potential_portfolio(source: str) -> str:
    """Emit bounded full/tail/local execution and the existing FP64 profile.

    Profile axes are AO columns, total grid points, work jets and tile points.
    Bounds preserve the independently qualified large-local endpoint domain;
    they are not timing estimates. Complete costs, when available, remain owned
    by the common native selector, including preparation and expected reuse.
    """
    portfolio = emit_symmetric_product_portfolio(
        potential_panel_program(4, 3, 2),
        "potential",
        canonical_hash({"schedule": source, "profile": emit_indexed_fp64_profile()}),
        name="xc_potential",
    )
    return "\n".join(  # noqa: FLY002 - generated source is reviewed line by line
        (
            '#include "tensor/cuda_symmetric_product.cuh"',
            emit_indexed_fp64_profile(),
            "#if defined(GENERATIVEQC_TEST_HOOKS)",
            'extern "C" void xc_potential_qualification_for_test(bool library, bool unavailable) {',
            "  generativeqc::tensor::symmetric_product_library_for_test=library;",
            "  generativeqc::tensor::symmetric_product_unavailable_for_test=unavailable;}",
            'extern "C" void xc_potential_indexed_qualification_for_test(bool library) {',
            "  generativeqc::tensor::symmetric_product_indexed_library_for_test=library;}",
            'extern "C" void xc_potential_indexed_gemm_for_test(bool library) {',
            "  generativeqc::tensor::symmetric_product_indexed_gemm_for_test=library;}",
            "#endif",
            "namespace generativeqc::dft::cuda_xc_detail {",
            portfolio,
            "std::unique_ptr<tensor::PreparedSymmetricProduct> prepare_potential(",
            "    const CudaXcLayout& l, cudaStream_t stream, std::size_t provider_budget, std::uint64_t expected_replays) {",
            '  if (!expected_replays) throw std::invalid_argument("potential reuse must be positive");',
            "  auto offers=xc_potential_candidates;",
            "  auto rejection=indexed_fp64_profile_rejection(l,provider_budget);",
            "  bool indexed_gemm=l.local_ao;",
            "#if defined(GENERATIVEQC_TEST_HOOKS)",
            "  if (tensor::symmetric_product_indexed_gemm_for_test) rejection={};",
            "  if (tensor::symmetric_product_indexed_library_for_test &&",
            "      !tensor::symmetric_product_indexed_gemm_for_test) indexed_gemm=false;",
            "#endif",
            "  offers[3].rejection=rejection;",
            "  return std::make_unique<tensor::CudaSymmetricProduct>(",
            "      xc_potential_request,offers,xc_potential_target,xc_potential_compilation,",
            "      l.nao,tensor::contraction_product(l.work_jets,l.tile_points),l.spins,stream,provider_budget,",
            "      l.local_ao,!l.local_ao || l.ao_map_entries!=0,indexed_gemm,expected_replays);",
            "}",
            "} // namespace generativeqc::dft::cuda_xc_detail",
        )
    )
