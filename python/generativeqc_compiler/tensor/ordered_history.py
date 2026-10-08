"""Closed stateless ordered-history algebra below method policy.

Scalar equations are ordinary TensorIR Programs. The finite phase/recurrence
manifest supplies their iteration dependencies; this is not a general loop IR,
an alternate scalar algebra, a mixer controller, or an AD-capable solve node.
Physical history buffers are borrowed through a tentative read-only ring overlay.
Only scratch Gram/RHS/mixed values are written. Methods retain publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from generativeqc_compiler.common.provenance import canonical_hash

from .ir import Node, add, divide, input_tensor, multiply, sqrt
from .program import Program
from .types import TensorSpec

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_provider import LoweringCandidate


@dataclass(frozen=True)
class OrderedHistoryProgram:
    """Scientific scalar stages and their closed ordered recurrence contract."""

    stages: tuple[tuple[str, Program], ...]
    phases: tuple[str, ...] = ("gram-rhs", "lower-cholesky-solve", "seeded-correction")
    ring: str = "h=min(k,m);new=(k-1)%m;slot(j)=(k-h+j)%m;tentative-overlay"
    dot: str = "positive-zero-seed;increasing-component;check-each-update"
    gram: str = "full-row-column;weights-after-dot;diagonal-square-after-weight"
    solve: str = "lower-row-column-inner;forward-increasing;backward-decreasing"
    solve_checks: str = "after-inner-fold;positive-pivot;after-division"
    correction: str = "damped-seed;increasing-history-subtract;check-final"
    publication: str = "partial-scratch;method-owned-persistent-commit"

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "schema": "generativeqc.tensor.ordered-history.v1",
                **{
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if name != "stages"
                },
                "stages": [(name, stage.logical_hash) for name, stage in self.stages],
            }
        )

    def stage(self, name: str) -> Program:
        return dict(self.stages)[name]


def ordered_history_program() -> OrderedHistoryProgram:
    """Build the one canonical algebra; no backend names or method constants."""

    def scalar(name: str) -> Node:
        return input_tensor(name, TensorSpec((), dtype="float64", role="input"))

    accumulator, left, right = (scalar(n) for n in ("accumulator", "left", "right"))
    omega, coefficient, update = (scalar(n) for n in ("omega", "coefficient", "update"))
    row_weight, column_weight, overlap = (
        scalar(n) for n in ("row_weight", "column_weight", "overlap")
    )
    value, diagonal, omega_zero = (
        scalar(n) for n in ("value", "diagonal", "omega_zero")
    )
    current, damping, residual = (scalar(n) for n in ("current", "damping", "residual"))
    expressions = (
        ("dot_update", add(accumulator, multiply(left, right))),
        ("weighted_rhs", multiply(omega, value)),
        ("weighted_gram", multiply(multiply(row_weight, column_weight), overlap)),
        ("diagonal_shift", add(value, multiply(omega_zero, omega_zero))),
        (
            "subtract_product",
            add(accumulator, multiply(left, right), coefficients=(1, -1)),
        ),
        ("divide", divide(value, diagonal)),
        ("sqrt", sqrt(value)),
        ("seed", add(current, multiply(damping, residual))),
        (
            "correction",
            add(
                accumulator,
                multiply(multiply(omega, coefficient), update),
                coefficients=(1, -1),
            ),
        ),
    )
    return OrderedHistoryProgram(
        tuple((name, Program({name: root})) for name, root in expressions)
    )


def recognize_ordered_history(program: OrderedHistoryProgram) -> OrderedHistoryProgram:
    """The retained implementation admits exactly its complete ordered graph."""
    if (
        not isinstance(program, OrderedHistoryProgram)
        or program.identity != ordered_history_program().identity
    ):
        raise ValueError(
            "ordered history requires its canonical complete scalar and recurrence graph"
        )
    return program


@dataclass(frozen=True)
class RetainedHistorySchedule:
    """The two incumbent schedules, including different error continuations.

    No maximum capacity is introduced below the method's existing admission.
    FP64 source expressions retain the enclosing compiler's FMA policy.
    """

    name: str
    program_identity: str
    index_type: str
    zero: str
    one: str
    math_namespace: str
    leading_dimension: str
    row_distribution: str
    weight_failure: str
    dot_failure: str
    final_store: str
    arithmetic: str = "fp64-native-expression-retained-contraction-policy"

    @property
    def identity(self) -> str:
        return canonical_hash(self.__dict__)


def retained_history_schedule(
    program: OrderedHistoryProgram, name: str
) -> RetainedHistorySchedule:
    recognize_ordered_history(program)
    common = (name, program.identity)
    if name == "compact-cpu":
        return RetainedHistorySchedule(
            *common,
            "std::size_t",
            "0u",
            "1u",
            "std::",
            "dimension",
            "serial",
            "short-circuit",
            "return",
            "check-before-store",
        )
    if name == "capacity-cuda":
        return RetainedHistorySchedule(
            *common,
            "std::int64_t",
            "0",
            "1",
            "",
            "leading_dimension",
            "thread-strided",
            "flag-continue",
            "flag-break-dot",
            "store-before-check",
        )
    raise ValueError("unqualified ordered-history schedule")


def require_history_schedule(
    program: OrderedHistoryProgram, schedule: RetainedHistorySchedule
) -> None:
    if not isinstance(
        schedule, RetainedHistorySchedule
    ) or schedule != retained_history_schedule(program, schedule.name):
        raise ValueError("ordered history requires its admitted retained schedule")


def ordered_history_candidate(
    program: OrderedHistoryProgram,
    name: str,
    *,
    vector_size: int = 3,
    history_count: int = 2,
    capacity: int = 4,
) -> LoweringCandidate:
    """Select the only qualified implementation for an admitted physical view.

    Default extents identify the shape-independent native template. Runtime
    method admission binds actual extents/strides before execution; no source
    specializes to the representative sizes. All buffers are borrowed, and
    this ownership cutover adds no workspace/provider storage or launches.
    """
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
    from generativeqc_compiler.common.precision import (
        ExecutionPrecisionSchedule,
        PrecisionDirective,
    )
    from generativeqc_compiler.common.schedule import ScheduleTopology

    schedule = retained_history_schedule(program, name)
    if (
        any(
            type(n) is not int or n <= 0 for n in (vector_size, history_count, capacity)
        )
        or history_count > capacity
    ):
        raise ValueError(
            "history candidate requires positive dimensions and h <= capacity"
        )
    backend = "cpu" if name == "compact-cpu" else "cuda"
    d, h, m = vector_size, history_count, capacity
    ld = h if backend == "cpu" else m
    operands = (
        OperandLayout("df_history", (2, 0), (m, d), (d, 1)),
        OperandLayout("u_history", (2, 0), (m, d), (d, 1)),
        OperandLayout("weights", (2,), (m,), (1,)),
        *(
            OperandLayout(role, (0,), (d,), (1,))
            for role in ("tentative_df", "tentative_u", "residual", "current")
        ),
        *(
            OperandLayout(role, (), (), ())
            for role in ("tentative_weight", "damping", "omega_zero")
        ),
        # Distinct row/column modes describe the full Gram, not a diagonal view.
        OperandLayout("beta", (1, 3), (h, h), (ld, 1), access="write"),
        OperandLayout("coefficients", (1,), (h,), (1,), access="write"),
        OperandLayout("mixed", (0,), (d,), (1,), access="write"),
    )
    inputs = ("float64",) * sum(op.access != "write" for op in operands)
    precision = LoweringPrecision(
        ExecutionPrecisionSchedule(
            (("ordered-history", PrecisionDirective("float64", "float64", "float64")),)
        ),
        "ordered-history",
        inputs,
        "float64",
    )
    request = LoweringRequest(
        consumer="tensor.ordered_history",
        operation="ordered-history",
        backend=backend,
        dtype="float64",
        accumulation_dtype="float64",
        shape=(d, h, m),
        scientific_identity=program.identity,
        operands=operands,
        input_dtypes=inputs,
        precisions=(precision,),
        semantics=(
            ("ring", program.ring),
            ("dot", program.dot),
            ("gram", program.gram),
            ("solve", program.solve),
            ("correction", program.correction),
            (
                "binding_domain",
                "runtime-positive-dimensions-h<=capacity-checked-native-storage",
            ),
            (
                "stage_graphs",
                canonical_hash([(n, p.logical_hash) for n, p in program.stages]),
            ),
        ),
        effects=(
            ("publication", program.publication),
            ("weight_failure", schedule.weight_failure),
            ("dot_failure", schedule.dot_failure),
            ("final_store", schedule.final_store),
            ("solve_checks", program.solve_checks),
            ("arithmetic", schedule.arithmetic),
        ),
        constraints=LoweringConstraints(
            workspace_bytes=0,
            provider_bytes=0,
            additional_device_bytes=0,
            host_bytes=0,
            determinism="exact-order",
            capture_required=backend == "cuda",
        ),
    )
    return LoweringCandidate(
        request,
        name,
        (
            ProviderDescriptor(
                "generated." + backend, "generated", name, version="ordered-history-v1"
            ),
        ),
        "ready",
        precision.directive.math_mode,
        provenance=(("retained_schedule", schedule.identity),),
        execution=CandidateExecution(
            precision,
            name,
            operands,
            ScheduleTopology(
                workgroup_threads=256 if backend == "cuda" else None,
                fusion="incumbent-mixer-phases",
                materialization="borrowed-scratch",
                reduction="increasing-component-and-history",
                residency="resident",
            ),
            determinism="exact-order",
            capture_safe=backend == "cuda",
        ),
    )


def require_history_candidate(
    program: OrderedHistoryProgram, candidate: LoweringCandidate
) -> RetainedHistorySchedule:
    """Candidate selection controls emission, including the complete ABI/effects."""
    from generativeqc_compiler.common.lowering_provider import LoweringCandidate

    if (
        not isinstance(candidate, LoweringCandidate)
        or len(candidate.request.shape) != 3
    ):
        raise ValueError("history emission requires its admitted candidate")
    d, h, m = candidate.request.shape
    if candidate != ordered_history_candidate(
        program, candidate.implementation, vector_size=d, history_count=h, capacity=m
    ):
        raise ValueError("history emission requires its admitted candidate")
    assert candidate.execution is not None
    return retained_history_schedule(program, candidate.execution.algorithm)


def chronological_slots(iteration: int, capacity: int) -> tuple[int, tuple[int, ...]]:
    """Pure diagnostic binding; native method guards own production admission."""
    if (
        type(iteration) is not int
        or type(capacity) is not int
        or not (1 <= iteration < 2**64 - 1)
        or capacity <= 0
    ):
        raise ValueError(
            "ordered history requires positive admitted iteration/capacity"
        )
    count = min(iteration, capacity)
    return (iteration - 1) % capacity, tuple(
        (iteration - count + j) % capacity for j in range(count)
    )
