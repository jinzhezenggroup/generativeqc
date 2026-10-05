"""Canonical COSX tile projections and donated exchange-matrix updates.

ESP integrals, weights and spin conventions remain with their scientific owners.
These graphs describe only the existing dense algebra, including the explicit
previous exchange matrix that is donated after its last use.
"""

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
)
from generativeqc_compiler.tensor.contraction_update import contraction_update_request
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.native_lowering import (
    contraction_initializer,
    emit_contraction_region_portfolio,
)


def cosx_matrix_program(points: int, columns: int, *, update: bool) -> Program:
    """Project AO*D or accumulate seed+AO^T*potential without symmetry assumptions."""
    if any(type(n) is not int or n < 1 for n in (points, columns)):
        raise ValueError("COSX matrix dimensions must be positive integers")
    p = Index("p", IndexSpace("points", "batch", points))
    ao = IndexSpace("columns", "ao", columns)
    m, n = Index("m", ao), Index("n", ao)
    values = input_tensor("ao", TensorSpec((p, m), role="input"))
    if update:
        potential = input_tensor("potential", TensorSpec((p, n), role="input"))
        seed = input_tensor("seed", TensorSpec((m, n), role="input"))
        result = add(seed, einsum("pm,pn->mn", values, potential))
    else:
        density = input_tensor("density", TensorSpec((m, n), role="input"))
        result = einsum("pm,mn->pn", values, density)
    return Program({"result": result})


def emit_cosx_contractions() -> str:
    """Emit fixed full/tail descriptors and shared preparation, never vendor calls."""
    pieces = [
        "// Generated from COSX TensorIR; do not edit.",
        "#pragma once",
        '#include "tensor/cuda_contraction_sites.cuh"',
        "namespace generativeqc::dft::cosx_lowering {",
    ]
    for name, update in (("projection", False), ("accumulation", True)):
        program = cosx_matrix_program(3, 2, update=update)
        root = program.outputs["result"]
        adapter = TensorLoweringAdapter(program)
        product = next(n for n in root.inputs if n.op == "einsum") if update else root
        request = (
            contraction_update_request(adapter, root, backend="cuda")
            if update
            else adapter.request(root, backend="cuda")
        )
        descriptor = contraction_initializer(
            adapter,
            product,
            lambda index: index.space.name,
            transpose=("T" if update else "N", "N"),
            extents=(
                "1",
                "columns" if update else "points",
                "columns",
                "points" if update else "columns",
            ),
            coefficient="1.0",
            beta="1.0" if update else "0.0",
            accumulation=root if update else None,
        )
        pieces.append(
            emit_contraction_region_portfolio(
                request, canonical_hash({"descriptor": descriptor}), name=name
            )
        )
        pieces.append(f"""
inline auto {name}(std::size_t points, std::size_t columns) {{
  return tensor::ContractionSite{{{name}_request,{name}_candidates,
      {name}_target,{name}_compilation,{descriptor}}};
}}
""")
    pieces.append("""
using Prepared = tensor::PreparedContractionSites<4>;
inline std::unique_ptr<Prepared> prepare(std::size_t columns, std::size_t full,
    std::size_t tail, cudaStream_t stream, std::size_t device_budget) {
  if (!columns || !full || tail > full) throw std::invalid_argument("invalid COSX tile domain");
  // Even an absent tail has an immutable descriptor; no iteration grows a cache.
  const auto short_tile = tail ? tail : full;
  return std::make_unique<Prepared>(std::array{
      projection(full,columns),accumulation(full,columns),
      projection(short_tile,columns),accumulation(short_tile,columns)},stream,device_budget);
}
} // namespace generativeqc::dft::cosx_lowering
""")
    return "\n".join(pieces)
