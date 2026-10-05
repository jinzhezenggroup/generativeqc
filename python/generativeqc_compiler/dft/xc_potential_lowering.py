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
    """Emit a prepared provider owner for bounded full/tail/local tiles."""
    portfolio = emit_symmetric_product_portfolio(
        potential_panel_program(4, 3, 2),
        "potential",
        canonical_hash({"schedule": source}),
        name="xc_potential",
    )
    return "\n".join(  # noqa: FLY002 - generated source is reviewed line by line
        (
            '#include "tensor/cuda_symmetric_product.cuh"',
            "#if defined(GENERATIVEQC_TEST_HOOKS)",
            'extern "C" void xc_potential_qualification_for_test(bool library, bool unavailable) {',
            "  generativeqc::tensor::symmetric_product_library_for_test=library;",
            "  generativeqc::tensor::symmetric_product_unavailable_for_test=unavailable;}",
            'extern "C" void xc_potential_indexed_qualification_for_test(bool library) {',
            "  generativeqc::tensor::symmetric_product_indexed_library_for_test=library;}",
            "#endif",
            "namespace generativeqc::dft::cuda_xc_detail {",
            portfolio,
            "std::unique_ptr<tensor::PreparedSymmetricProduct> prepare_potential(",
            "    const CudaXcLayout& l, cudaStream_t stream, std::size_t provider_budget) {",
            "  return std::make_unique<tensor::CudaSymmetricProduct>(",
            "      xc_potential_request,xc_potential_candidates,xc_potential_target,xc_potential_compilation,",
            "      l.nao,tensor::contraction_product(l.work_jets,l.tile_points),l.spins,stream,provider_budget,",
            "      l.local_ao,!l.local_ao || l.ao_map_entries!=0);",
            "}",
            "} // namespace generativeqc::dft::cuda_xc_detail",
        )
    )
