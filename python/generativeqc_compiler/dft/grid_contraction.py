"""Canonical TensorIR boundary for packed grid density/orbital projections.

AO jets have one contiguous [jet, point, active AO] panel. Density matrices and
occupation-weighted orbital factors are already canonicalized/gathered by their
source owner. Both consumers request the same binary contraction, with no
precision, approximation, or provider decision encoded in the scientific graph.
"""

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
)
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.native_lowering import (
    contraction_initializer,
    emit_contraction_region_portfolio,
)


def grid_panel_program(jets: int, points: int, active: int, width: int) -> Program:
    """Project all requested contiguous jets; coefficient panels are broadcast.

    Flattening jet/point free axes describes one matrix view without packing or
    duplicating coefficients. The surrounding owner skips empty scientific tiles
    and retains feature-mask offsets, spin provenance and orbital tail widths.
    """
    dimensions = (jets, points, active, width)
    if any(type(n) is not int or n < 1 for n in dimensions):
        raise ValueError("grid panel dimensions must be positive integers")
    j, p, u, v = (
        Index(label, IndexSpace(name, kind, extent))
        for label, name, kind, extent in zip(
            ("j", "p", "u", "v"),
            ("jets", "points", "active", "width"),
            ("component", "batch", "ao", "matrix"),
            dimensions,
            strict=True,
        )
    )
    ao = input_tensor("ao_jets", TensorSpec((j, p, u), role="input"))
    coefficients = input_tensor("coefficients", TensorSpec((u, v), role="input"))
    return Program({"projected": einsum("jpu,uv->jpv", ao, coefficients)})


def emit_grid_contraction() -> str:
    """Emit symbolic descriptors and one prepare-only shared provider binding."""
    program = grid_panel_program(4, 2, 3, 5)
    node = program.outputs["projected"]
    adapter = TensorLoweringAdapter(program)
    request = adapter.request(node, backend="cuda")
    descriptor = contraction_initializer(
        adapter,
        node,
        lambda index: index.space.name,
        transpose=("N", "N"),
        extents=("1", "jets*points", "width", "active"),
        coefficient="1.0",
    )
    portfolio = emit_contraction_region_portfolio(
        request,
        canonical_hash({"descriptor": descriptor, "domain": "bounded-packed-v1"}),
        name="grid_panel_lowering",
        bounded_dense=True,
    )
    return "\n".join(
        (
            '#include "tensor/native_contraction.hpp"',
            "namespace generativeqc::dft::generated {",
            "inline tensor::ContractionRequest grid_panel_descriptor(std::size_t jets,",
            "    std::size_t points, std::size_t active, std::size_t width) {",
            "  (void)tensor::contraction_product(jets,points); return "
            + descriptor
            + ";}",
            "} // namespace generativeqc::dft::generated",
            "#ifdef __CUDACC__",
            '#include "tensor/cuda_contraction_selection.cuh"',
            "namespace generativeqc::dft::generated {",
            portfolio,
            "inline std::unique_ptr<tensor::PreparedBoundedContraction> prepare_grid_panel(",
            "    std::size_t jets, std::size_t points, std::size_t active, std::size_t width,",
            "    cudaStream_t stream, std::size_t budget) {",
            "  return std::make_unique<tensor::PreparedBoundedContraction>(",
            "      grid_panel_lowering_request,grid_panel_lowering_candidates,",
            "      grid_panel_lowering_target,grid_panel_lowering_compilation,",
            "      grid_panel_descriptor(jets,points,active,width),stream,budget);}",
            "} // namespace generativeqc::dft::generated\n#endif\n",
        )
    )
