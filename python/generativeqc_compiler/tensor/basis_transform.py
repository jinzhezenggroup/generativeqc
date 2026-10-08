"""Canonical public/direct basis transforms for column-major SCF buffers.

TensorIR describes the row-major views of the existing device allocations.
Each view is the transpose of its mathematical column-major matrix, so the
four contractions implement D_direct = C.T @ D_public @ C and
F_public = hcore + C @ F_direct @ C.T without packing or a symmetry assumption.
"""

from __future__ import annotations

from generativeqc_compiler.common.provenance import canonical_hash

from .ir import add, einsum, input_tensor
from .lowering import TensorLoweringAdapter
from .native_lowering import (
    contraction_initializer,
    emit_contraction_region_portfolio,
)
from .program import Program
from .types import Index, IndexSpace, TensorSpec


def basis_transform_program(public: int, direct: int) -> Program:
    """Build both transform directions and the final one-electron fold."""
    if type(public) is not int or type(direct) is not int or public < 1 or direct < 1:
        raise ValueError("basis dimensions must be positive integers")
    public_space = IndexSpace("public", "ao", public)
    direct_space = IndexSpace("direct", "ao", direct)
    p, q = Index("p", public_space), Index("q", public_space)
    a, b = Index("a", direct_space), Index("b", direct_space)
    transform = input_tensor("transform", TensorSpec((a, p), role="input"))
    density = input_tensor("density", TensorSpec((p, q), role="input"))
    direct_fock = input_tensor("direct_fock", TensorSpec((a, b), role="input"))
    hcore = input_tensor("hcore", TensorSpec((p, q), role="input"))
    density_right = einsum("ap,pq->aq", transform, density)
    density_direct = einsum("ap,bp->ab", density_right, transform)
    fock_left = einsum("ab,bp->ap", direct_fock, transform)
    fock_right = einsum("ap,aq->pq", transform, fock_left)
    return Program(
        {
            "density_right": density_right,
            "density_direct": density_direct,
            "fock_left": fock_left,
            "fock_right": fock_right,
            "fock_public": add(fock_right, hcore),
        },
        provenance={
            "operation": "public_direct_basis_transform",
            "storage": "column_major",
        },
    )


_STAGES = (
    ("density_right", ("N", "N"), ("direct_nbf", "public_nbf", "public_nbf")),
    ("density_direct", ("N", "T"), ("direct_nbf", "direct_nbf", "public_nbf")),
    ("fock_left", ("N", "N"), ("direct_nbf", "public_nbf", "direct_nbf")),
    ("fock_right", ("T", "N"), ("public_nbf", "public_nbf", "direct_nbf")),
)


def emit_basis_transform_cuda() -> str:
    """Emit canonical matrix requests and a shared-provider candidate portfolio."""
    program = basis_transform_program(5, 7)
    adapter = TensorLoweringAdapter(program)
    pieces = [
        "// Generated from SCF public/direct TensorIR; do not edit.",
        "#pragma once",
        '#include "runtime/lowering_binding.hpp"',
        '#include "tensor/native_contraction.hpp"',
        "namespace generativeqc::scf::basis_transform_lowering {",
        "// Production activity, spin broadcast and shell spans need a separate owner.",
        "inline void require_packed_stage(std::int64_t public_nbf, std::int64_t direct_nbf,",
        "                                 std::size_t batch_size, std::size_t spin_count,",
        "                                 const void* active, const void* shell_spans) {",
        "  if (public_nbf <= 0 || direct_nbf <= 0 ||",
        "      public_nbf > std::numeric_limits<int>::max() ||",
        "      direct_nbf > std::numeric_limits<int>::max() ||",
        "      batch_size != 1 || spin_count != 1 || active || shell_spans)",
        '    throw std::invalid_argument("basis stage requires one unmasked packed matrix");',
        "  const auto extent = static_cast<std::size_t>(std::max(public_nbf, direct_nbf));",
        "  const auto bytes = tensor::contraction_product(",
        "      tensor::contraction_product(extent, extent), sizeof(double));",
        "  if (bytes > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))",
        '    throw std::length_error("basis stage address range overflow");',
        "}",
    ]
    for name, transpose, (m, n, k) in _STAGES:
        node = program.outputs[name]
        request = adapter.request(node, backend="cuda")
        descriptor = contraction_initializer(
            adapter,
            node,
            lambda index: (
                "public_nbf" if index.space.name == "public" else "direct_nbf"
            ),
            transpose=transpose,
            extents=("1", m, n, k),
            coefficient="1.0",
        )
        pieces.extend(
            (
                f"inline tensor::ContractionRequest {name}(std::size_t public_nbf, std::size_t direct_nbf) {{",
                f"  return {descriptor};",
                "}",
                "#ifdef __CUDACC__",
                emit_contraction_region_portfolio(
                    request,
                    canonical_hash({"descriptor": descriptor, "stage": name}),
                    name=f"basis_{name}",
                    bounded_dense=True,
                ),
                "#endif",
            )
        )
    pieces.append("} // namespace generativeqc::scf::basis_transform_lowering")
    return "\n".join(pieces) + "\n"
