"""Prepare streamed metric views without changing their vector recipes."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from generativeqc_compiler.common.lowering_contract import LoweringConstraints
from generativeqc_compiler.common.native_lowering import native_lowering_portfolio

from .contraction_update import contraction_update_request
from .df_coulomb_metric import metric_program
from .lowering import TensorLoweringAdapter
from .native_lowering import contraction_initializer
from .vector_lowering import vector_portfolio

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_provider import LoweringRequest

    from .ir import Node

OPERATIONS = ("metric_charge", "metric_potential", "metric_project", "metric_rotate")


def metric_request(
    operation: str, *, backend: str
) -> tuple[TensorLoweringAdapter, Node, Node, LoweringRequest]:
    """Bind representative physical padding and explicit seed donation."""
    program = metric_program(5, 3, operation)
    root = program.outputs["result"]
    adapter = TensorLoweringAdapter(program)
    if operation == "metric_charge":
        product = root.inputs[1]
        request = contraction_update_request(adapter, root, backend=backend)
    else:
        product = root
        request = adapter.request(root, backend=backend)
    # A column subpanel of the original square root retains its full row stride.
    # Padding belongs to the same operand's physical view, not scientific identity.
    if operation == "metric_potential":
        request = replace(
            request,
            operands=(
                replace(request.operands[0], strides=(5, 1)),
                *request.operands[1:],
            ),
        )
    return (
        adapter,
        root,
        product,
        replace(request, constraints=LoweringConstraints(maximum_candidates=2)),
    )


def metric_header(source_identity: str) -> str:
    """Emit fixed full/tail shape factories; no preparation occurs in a tile loop."""
    pieces = ["namespace generativeqc::scf::cuda_df::coulomb_lowering {"]
    dimensions = {"auxiliary": "naux", "eigendirection": "naux", "panel": "panel"}
    for operation in OPERATIONS:
        adapter, root, product, request = metric_request(operation, backend="cuda")
        candidates, target, compilation = vector_portfolio(request, source_identity)
        pieces.append(
            native_lowering_portfolio(
                request, candidates, target, compilation, name=operation
            )
        )
        descriptor = contraction_initializer(
            adapter,
            product,
            lambda index: dimensions[index.space.name],
            transpose=("N" if operation == "metric_project" else "T", "N"),
            extents=(
                "1",
                "panel" if operation == "metric_potential" else "naux",
                "1",
                "panel" if operation == "metric_charge" else "naux",
            ),
            coefficient="1.0",
            row_axes=(1, 1, 1),
            leading_dimensions=("naux", "1", "1"),
            beta="1.0" if operation == "metric_charge" else "0.0",
            accumulation=root if operation == "metric_charge" else None,
        )
        pieces.append(f"""
inline std::unique_ptr<tensor::CudaVectorContraction> {operation}(
    std::size_t naux, std::size_t panel, cublasHandle_t handle, cudaStream_t stream) {{
  if (!panel || panel > naux) throw std::invalid_argument("invalid metric panel extent");
  auto resolved = {descriptor};
  return std::make_unique<tensor::CudaVectorContraction>(
      {operation}_request, {operation}_candidates, {operation}_target,
      {operation}_compilation, resolved, 0, handle, stream);
}}
""")
    return "\n".join([*pieces, "}\n"])
