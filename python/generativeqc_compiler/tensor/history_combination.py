"""Retained CPU/CUDA schedules for the canonical DIIS history contraction.

The caller supplies a chronological read-only history view. Ring indexing,
retirement, solve success and final publication remain runtime responsibilities;
this lowerer owns only the ordered coefficient application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from generativeqc_compiler.common.lowering_contract import (
    CandidateExecution,
    LoweringConstraints,
)
from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    ProviderDescriptor,
)
from generativeqc_compiler.common.schedule import ScheduleTopology

from .lowering import TensorLoweringAdapter
from .optimize import prepare_for_backend
from .scf import diis_extrapolation_program

if TYPE_CHECKING:
    from .program import Program


def history_combination_candidate(program: Program, backend: str) -> LoweringCandidate:
    """Bind one strict-FP64 incumbent to the existing scientific equation.

    Both physical schedules visit history in increasing chronological order.
    CPU borrows pointer rows; CUDA borrows an indexed element view, so neither
    requires packing, provider workspace or an additional launch.
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("unsupported DIIS history backend")
    if tuple(program.outputs) != ("effective_fock",):
        raise ValueError("DIIS history combination requires effective_fock")
    root = program.outputs["effective_fock"]
    if (
        root.op != "einsum"
        or len(root.inputs) != 2
        or len(root.inputs[0].spec.shape) != 5
    ):
        raise ValueError("DIIS history combination topology changed")
    batch, history, spins, rows, columns = root.inputs[0].spec.shape
    expected = prepare_for_backend(
        diis_extrapolation_program(batch, history, rows, spin_count=spins),
        backend,
        preserve_reduction_order=True,
    )
    # Compare the complete live equation, including index domains and dtypes;
    # shape-only recognition would admit transposed or mixed-precision graphs.
    if rows != columns or program.logical_hash != expected.logical_hash:
        raise ValueError("DIIS history combination requires its canonical topology")
    request = TensorLoweringAdapter(program).request(
        root,
        backend=backend,
        layouts=(None, None, None),
        constraints=LoweringConstraints(determinism="exact-order"),
    )
    if (
        len(request.precisions) != 1
        or not request.precisions[0].schedule.is_strict_fp64
    ):
        raise ValueError("DIIS history combination requires strict FP64")
    algorithm = (
        "ordered-pointer-rows" if backend == "cpu" else "ordered-indexed-element"
    )
    return LoweringCandidate(
        request,
        algorithm,
        (
            ProviderDescriptor(
                "generated." + backend, "generated", algorithm, version="v1"
            ),
        ),
        "ready",
        request.precisions[0].directive.math_mode,
        execution=CandidateExecution(
            request.precisions[0],
            algorithm,
            request.operands,
            ScheduleTopology(
                materialization="borrowed-history-view",
                reduction="increasing-history-native-expression",
            ),
            determinism="exact-order",
            capture_safe=backend == "cuda",
        ),
    )


def emit_history_combination(program: Program, backend: str) -> str:
    """Emit the selected retained schedule without changing its FMA policy."""
    selected = history_combination_candidate(program, backend)
    if selected.implementation == "ordered-pointer-rows":
        return r"""template <class History>
inline void diis_extrapolate(double* output, const History& fock_history,
                             const double* coefficients, std::size_t history_size,
                             std::size_t vector_size) {
  for (std::size_t element = 0; element < vector_size; ++element) output[element] = 0.0;
  // Keep the historical reduction order over the chronological history.
  for (std::size_t i = 0; i < history_size; ++i)
    for (std::size_t element = 0; element < vector_size; ++element)
      output[element] += coefficients[i] * fock_history[i][element];
}"""
    if selected.implementation == "ordered-indexed-element":
        return r"""template <class HistoryElement>
#ifdef __CUDACC__
__host__ __device__
#endif
inline double ordered_history_combination(const double* coefficients,
                                          std::size_t history_size,
                                          const HistoryElement& history) {
  double value = 0.0;
  for (std::size_t item = 0; item < history_size; ++item)
    value += coefficients[item] * history[item];
  return value;
}"""
    raise ValueError("unqualified DIIS history combination schedule")
