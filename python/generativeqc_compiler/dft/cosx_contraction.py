"""Canonical COSX tile projections and donated exchange-matrix updates.

ESP integrals and spin conventions remain with their scientific owners.
These graphs describe the existing dense algebra and weighted publication, including the explicit
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
from generativeqc_compiler.tensor.batch_scaled_contraction import (
    batch_scaled_contraction_request,
)
from generativeqc_compiler.tensor.checked_contraction import checked_contraction_request
from generativeqc_compiler.tensor.contraction_update import contraction_update_request
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.native_lowering import (
    contraction_initializer,
    emit_contraction_region_portfolio,
    projected_contraction_request,
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


def cosx_esp_program(points: int, columns: int) -> Program:
    """Apply each point's ESP matrix, then publish its weighted potential."""
    if any(type(extent) is not int or extent < 1 for extent in (points, columns)):
        raise ValueError("COSX ESP dimensions must be positive integers")
    point = Index("p", IndexSpace("points", "batch", points))
    ao = IndexSpace("columns", "ao", columns)
    row, column = Index("m", ao), Index("n", ao)
    esp = input_tensor("esp", TensorSpec((point, row, column), role="input"))
    projected = input_tensor("projected", TensorSpec((point, column), role="input"))
    weight = input_tensor("weight", TensorSpec((point,), role="input"))
    product = einsum("pmn,pn->pm", esp, projected)
    return Program({"result": einsum("p,pm->pm", weight, product)})


def cosx_jet_projection_program(points: int, columns: int) -> Program:
    """Project three spatial AO jets; the axis is a borrowed leading slice."""
    if any(type(extent) is not int or extent < 1 for extent in (points, columns)):
        raise ValueError("COSX projection dimensions must be positive integers")
    axis = Index("a", IndexSpace("axis", "batch", 3))
    point = Index("p", IndexSpace("points", "batch", points))
    ao = IndexSpace("columns", "ao", columns)
    row, column = Index("m", ao), Index("n", ao)
    jets = input_tensor("jets", TensorSpec((axis, point, row), role="input"))
    density = input_tensor("density", TensorSpec((row, column), role="input"))
    return Program({"result": einsum("apm,mn->apn", jets, density)})


def emit_cosx_derivative_contractions(update: Program, publication: Program) -> str:
    """Prepare derivative sites around the existing checked scalar programs.

    The generator supplies method-owned scalar programs; this component does
    not import the method owner or change their arithmetic. Jet-axis traversal
    belongs to the compiler, while all matrix reduction loops share the runtime.
    """
    pieces = [
        """
#if defined(__CUDACC__)
#include "tensor/cuda_contraction_sites.cuh"
namespace generativeqc::dft::cosx_derivative_lowering {
"""
    ]
    for name, program, weighted, jet in (
        ("projection", cosx_matrix_program(3, 2, update=False), False, False),
        ("jet_projection", cosx_jet_projection_program(3, 2), False, True),
        ("esp_application", cosx_esp_program(3, 2), True, False),
    ):
        root = program.outputs["result"]
        adapter = TensorLoweringAdapter(program)
        product = next(n for n in root.inputs if n.op == "einsum") if weighted else root
        fixed = (
            (adapter.request(root, backend="cuda").operands[0].modes[0],) if jet else ()
        )
        request = (
            batch_scaled_contraction_request(adapter, root, backend="cuda")
            if weighted
            else projected_contraction_request(adapter, product, fixed_modes=fixed)
        )
        request = checked_contraction_request(
            request, update, publication if weighted else None, contraction=product
        )
        descriptor = contraction_initializer(
            adapter,
            product,
            lambda index: index.space.name,
            transpose=("N", "N"),
            extents=("points", "columns", "1", "columns")
            if weighted
            else ("1", "points", "columns", "columns"),
            coefficient="1.0",
            fixed_modes=fixed,
            batch_scale=root if weighted else None,
            checked_update=update,
            checked_publication=publication if weighted else None,
        )
        pieces.append(
            emit_contraction_region_portfolio(
                request,
                canonical_hash({"descriptor": descriptor}),
                name=name,
            )
        )
        scale = (
            ",tensor::ContractionOperand::dense({0},{points},runtime::PrecisionDtype::Fp64)"
            if weighted
            else ""
        )
        pieces.append(f"""
inline auto {name}(std::size_t points, std::size_t columns) {{
  return tensor::ContractionSite{{{name}_request,{name}_candidates,
      {name}_target,{name}_compilation,{descriptor}{scale}}};
}}
""")
    pieces.append(f"""
template <bool Weighted> struct ScalarStep {{
  static constexpr std::string_view update_identity = "{update.logical_hash}";
  static constexpr std::string_view publication_identity = Weighted ? "{publication.logical_hash}" : "";
  __device__ static bool update(double a, double b, double& value) noexcept {{
    return generated_cosx_derivative::accumulate_projection(a,b,value);
  }}
  __device__ static bool publish(double weight, double value, double& output) noexcept {{
    return generated_cosx_derivative::scale(weight,value,output);
  }}
}};
using MolecularPrepared = tensor::PreparedContractionSites<2>;
using PointPrepared = tensor::PreparedContractionSites<6>;
inline auto prepare_molecular(std::size_t columns, std::size_t full, std::size_t tail,
                              cudaStream_t stream) {{
  if (!columns || !full || tail > full) throw std::invalid_argument("invalid derivative tile domain");
  return std::make_unique<MolecularPrepared>(std::array{{
      projection(full,columns),projection(tail ? tail : full,columns)}},stream,0);
}}
inline auto prepare_point(std::size_t columns, std::size_t full, std::size_t tail,
                          cudaStream_t stream) {{
  if (!columns || !full || tail > full) throw std::invalid_argument("invalid derivative tile domain");
  const auto short_tile = tail ? tail : full;
  return std::make_unique<PointPrepared>(std::array{{
      projection(full,columns),projection(short_tile,columns),
      jet_projection(full,columns),jet_projection(short_tile,columns),
      esp_application(full,columns),esp_application(short_tile,columns)}},stream,0);
}}
template <std::size_t Sites>
inline void project(tensor::PreparedContractionSites<Sites>& plan, bool tail,
                    cudaStream_t stream, const double* ao, const double* density,
                    double* output, int* error) {{
  plan.template execute_checked<ScalarStep<false>>(tail ? 1 : 0,stream,ao,density,output,error);
}}
inline void project_jets(PointPrepared& plan, bool tail, cudaStream_t stream,
                         const double* jets, const double* density, double* output, int* error) {{
  const auto slot = tail ? 3 : 2;
  const auto stride = plan.diagnostics()[slot].resolved.output_elements();
  // Three original leading axis slices, without broadcast packing or allocation.
  for (std::size_t axis = 0; axis < 3; ++axis)
    plan.execute_checked<ScalarStep<false>>(slot,stream,jets+axis*stride,density,
                                           output+axis*stride,error);
}}
inline void apply_esp(PointPrepared& plan, bool tail, cudaStream_t stream,
                      const double* esp, const double* projected, const double* weights,
                      double* output, int* error) {{
  plan.execute_checked<ScalarStep<true>>(tail ? 5 : 4,stream,esp,projected,output,error,weights);
}}
}} // namespace generativeqc::dft::cosx_derivative_lowering
#endif
""")
    return "\n".join(pieces)


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
    program = cosx_esp_program(3, 2)
    root = program.outputs["result"]
    adapter = TensorLoweringAdapter(program)
    product = next(value for value in root.inputs if value.op == "einsum")
    request = batch_scaled_contraction_request(adapter, root, backend="cuda")
    descriptor = contraction_initializer(
        adapter,
        product,
        lambda index: index.space.name,
        transpose=("N", "N"),
        extents=("points", "columns", "1", "columns"),
        coefficient="1.0",
        batch_scale=root,
    )
    pieces.append(
        emit_contraction_region_portfolio(
            request, canonical_hash({"descriptor": descriptor}), name="esp_application"
        )
    )
    pieces.append(f"""
inline auto esp_application(std::size_t points, std::size_t columns) {{
  return tensor::ContractionSite{{esp_application_request,esp_application_candidates,
      esp_application_target,esp_application_compilation,{descriptor},
      tensor::ContractionOperand::dense({{0}},{{points}},runtime::PrecisionDtype::Fp64)}};
}}
""")
    pieces.append("""
using Prepared = tensor::PreparedContractionSites<6>;
inline std::unique_ptr<Prepared> prepare(std::size_t columns, std::size_t full,
    std::size_t tail, cudaStream_t stream, std::size_t device_budget) {
  if (!columns || !full || tail > full) throw std::invalid_argument("invalid COSX tile domain");
  // Even an absent tail has an immutable descriptor; no iteration grows a cache.
  const auto short_tile = tail ? tail : full;
  return std::make_unique<Prepared>(std::array{
      projection(full,columns),accumulation(full,columns),
      projection(short_tile,columns),accumulation(short_tile,columns),
      esp_application(full,columns),esp_application(short_tile,columns)},stream,device_budget);
}
} // namespace generativeqc::dft::cosx_lowering
""")
    return "\n".join(pieces)
