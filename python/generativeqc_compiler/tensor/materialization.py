"""Consumer-aware, opt-in dense TensorIR materialization planning.

The exact index algebra and contraction-tree search remain owned by TensorIR.
This pass proves single-use contraction substitution, not sparsity, factorization
of opaque inputs, physical aliasing, or endpoint profitability. Ordinary
preparation does not enable it.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from fractions import Fraction

from generativeqc_compiler.common.storage import (
    BufferOp,
    BufferValue,
    MemoryEffect,
    analyze_lifetimes,
)

from .complexity import (
    _reassociate_node,
    analyze_complexity,
    node_complexity,
)
from .ir import Node, _infer
from .program import Program
from .types import checked_size

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

RULE = "single-consumer-einsum"
RESOURCE_SCOPE = (
    "Dense immutable SSA values in dependency order, inclusive producer/last-reader "
    "lifetimes; inputs remain caller-owned and outputs remain live through return. "
    "The peak includes all logical input bytes plus live compiler-owned values. "
    "Materialized bytes retain every non-input value. No physical aliases, provider "
    "workspace, transfers, detached output copies or allocator overhead are inferred; "
    "backend and complete-endpoint admission remain required."
)


@dataclass(frozen=True)
class MaterializationFootprint:
    """Symbolic work plus two explicit logical dense storage models."""

    max_storage_degree: int
    max_work_degree: int
    max_materialized_degree: int
    concrete_work: int
    storage_monomials: tuple[str, ...]
    work_monomials: tuple[str, ...]
    input_bytes: int
    output_bytes: int
    materialized_bytes: int
    peak_live_bytes: int

    def to_payload(self) -> dict[str, typing.Any]:
        """Return data-only identities, resource scope and planning evidence."""
        return {**vars(self), "scope": RESOURCE_SCOPE}


def materialization_footprint(program: Program) -> MaterializationFootprint:
    """Reuse shared lifetime legality; do not turn a logical peak into an ABI."""
    if not isinstance(program, Program):
        raise TypeError("materialization footprint requires a TensorIR Program")
    nodes = program.dependency_order
    inputs = tuple(node for node in nodes if node.op == "input")
    outputs = tuple(dict.fromkeys(program.outputs.values()))
    owned = tuple(node for node in nodes if node.op != "input")
    lifetimes = analyze_lifetimes(
        tuple(
            BufferValue(
                node,
                node.spec.size * node.spec.itemsize,
                "logical",
                compiler_owned=node.op != "input",
            )
            for node in nodes
        ),
        tuple(
            BufferOp(node, node.inputs, (node,), MemoryEffect.EXPLICIT)
            for node in owned
        ),
        inputs=inputs,
        outputs=outputs,
    )
    # Inputs are retained externally. Charge every distinct logical input once;
    # physical input aliasing is neither assumed nor used to obtain savings.
    input_bytes = sum(node.spec.size * node.spec.itemsize for node in inputs)
    events: dict[int, int] = {}
    for item in lifetimes.ranges:
        if not item.compiler_owned:
            continue
        events[item.first_phase] = events.get(item.first_phase, 0) + item.bytes
        events[item.last_phase + 1] = events.get(item.last_phase + 1, 0) - item.bytes
    live = peak = 0
    for phase in sorted(events):
        live += events[phase]
        peak = max(peak, live)
    report = analyze_complexity(program)
    return MaterializationFootprint(
        report.max_storage_degree,
        report.max_work_degree,
        max((node_complexity(node).storage.degree for node in owned), default=0),
        sum(node_complexity(node).concrete_work for node in nodes),
        tuple(sorted({node_complexity(node).storage.notation for node in owned})),
        tuple(
            sorted(
                {
                    node_complexity(node).work.notation
                    for node in owned
                    if node_complexity(node).concrete_work
                }
            )
        ),
        input_bytes,
        sum(node.spec.size * node.spec.itemsize for node in outputs),
        sum(node.spec.size * node.spec.itemsize for node in owned),
        input_bytes + peak,
    )


@dataclass(frozen=True)
class MaterializationDecision:
    """Rule request and proof/rejection bound to exact mathematical identities."""

    producer: str
    consumers: tuple[str, ...]
    rule: str
    status: str
    reason: str
    before: MaterializationFootprint
    candidate: MaterializationFootprint | None = None
    candidate_hash: str | None = None

    def to_payload(self) -> dict[str, typing.Any]:
        """Return data-only identities, resource scope and planning evidence."""
        return {
            "producer": self.producer,
            "consumers": list(self.consumers),
            "rule": self.rule,
            "status": self.status,
            "reason": self.reason,
            "before": self.before.to_payload(),
            "candidate": None
            if self.candidate is None
            else self.candidate.to_payload(),
            "candidate_hash": self.candidate_hash,
        }


@dataclass(frozen=True)
class MaterializationPlan:
    """A finite compiler plan; backend resource/scientific qualification is separate."""

    source_hash: str
    program: Program
    before: MaterializationFootprint
    after: MaterializationFootprint
    decisions: tuple[MaterializationDecision, ...]
    truncated: int

    def to_payload(self) -> dict[str, typing.Any]:
        """Return data-only identities, resource scope and planning evidence."""
        return {
            "schema": "generativeqc.tensor.consumer-materialization.v1",
            "source_hash": self.source_hash,
            "selected_hash": self.program.logical_hash,
            "before": self.before.to_payload(),
            "after": self.after.to_payload(),
            "decisions": [item.to_payload() for item in self.decisions],
            "truncated": self.truncated,
            "runtime_endpoint_selection_proven": False,
        }


def _substitute(producer: Node, consumer: Node) -> Node:
    """Alpha-rename private summed labels before exact rational substitution."""
    position = consumer.inputs.index(producer)
    bindings = dict(
        zip(producer.attrs["output"], consumer.attrs["labels"][position], strict=True)
    )
    next_label = 1 + max(
        (label for labels in consumer.attrs["labels"] for label in labels),
        default=-1,
    )
    for labels in producer.attrs["labels"]:
        for label in labels:
            if label not in bindings:
                bindings[label] = next_label
                next_label += 1
    inputs = (
        consumer.inputs[:position] + producer.inputs + consumer.inputs[position + 1 :]
    )
    labels = (
        consumer.attrs["labels"][:position]
        + tuple(
            tuple(bindings[label] for label in axis)
            for axis in producer.attrs["labels"]
        )
        + consumer.attrs["labels"][position + 1 :]
    )
    # Node validation requires canonical first-occurrence labels.
    canonical = {
        label: i
        for i, label in enumerate(
            dict.fromkeys(label for axis in labels for label in axis)
        )
    }
    coefficient = Fraction(*producer.attrs["coefficient"]) * Fraction(
        *consumer.attrs["coefficient"]
    )
    attrs = {
        "labels": tuple(tuple(canonical[label] for label in axis) for axis in labels),
        "output": tuple(canonical[label] for label in consumer.attrs["output"]),
        "coefficient": (coefficient.numerator, coefficient.denominator),
    }
    spec = _infer("einsum", inputs, attrs, consumer.spec)
    return Node("einsum", inputs, spec, tuple(attrs.items()))


def _replace(program: Program, old: Node, new: Node) -> Program:
    replacements: dict[Node, Node] = {}
    for node in program.live_nodes:
        inputs = tuple(replacements[child] for child in node.inputs)
        replacements[node] = (
            new
            if node is old
            else node
            if inputs == node.inputs
            else Node(
                node.op,
                inputs,
                _infer(node.op, inputs, node.attrs, node.spec),
                node.attributes,
            )
        )
    # Definitions are inspection-only, never execution roots. Preserve them so
    # the source artifact remains inspectable, without making dead work live.
    return Program(
        {name: replacements[node] for name, node in program.outputs.items()},
        program.definitions,
        program.provenance,
    )


def _contexts(program: Program) -> tuple[tuple[Node, tuple[Node, ...]], ...]:
    """Bind every high-rank value to its current immutable reader edges."""
    nodes = program.live_nodes
    readers: dict[Node, list[Node]] = {node: [] for node in nodes}
    for node in nodes:
        for child in node.inputs:
            readers[child].append(node)
    return tuple(
        (node, tuple(readers[node]))
        for node in nodes
        if node_complexity(node).storage.degree >= 4
    )


def plan_materializations(
    program: Program,
    *,
    allow_reassociation: bool = False,
    available_rules: tuple[str, ...] = (RULE,),
    max_operands: int = 6,
    max_candidates: int = 32,
    max_materialized_bytes: int | None = None,
    max_intermediate_axes: Mapping[str, int] | None = None,
) -> MaterializationPlan:
    """Select proven lower-storage single-use contraction plans.

    Only immutable FP64 einsums without repeated labels are substituted.
    Multi-reader, retained-output and view/opaque consumers remain materialized.
    Algebraic reassociation is explicit opt-in, including factor distribution.
    A precision-execution contract is a barrier, even with that opt-in.

    Available rules represent compiler capabilities, not vendor names. Opaque
    rank-four sources and nonlinear producers request a factorized-source rule
    with a separability/derivative contract; no factorization is invented. The
    same bound source/consumer request is usable by future provider adapters.
    """
    if not isinstance(program, Program):
        raise TypeError("materialization planning requires a TensorIR Program")
    if type(allow_reassociation) is not bool:
        raise TypeError("allow_reassociation must be Boolean")
    if type(max_operands) is not int or not 3 <= max_operands <= 8:
        raise ValueError("max_operands must lie in [3, 8]")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if (
        not isinstance(available_rules, tuple)
        or any(rule != RULE for rule in available_rules)
        or len(set(available_rules)) != len(available_rules)
    ):
        raise ValueError("unrecognized or duplicate materialization rule")
    if max_materialized_bytes is not None:
        checked_size(max_materialized_bytes, "materialization byte budget")
    from .types import SPACE_KINDS

    limits = dict(max_intermediate_axes or {})
    if any(
        kind not in SPACE_KINDS or type(limit) is not int or limit < 0
        for kind, limit in limits.items()
    ):
        raise ValueError(
            "intermediate axis limits require known kinds and nonnegative integers"
        )
    before = materialization_footprint(program)
    current = program
    decisions: list[MaterializationDecision] = []
    # A producer rejected earlier may become eligible after its reader changes.
    # Bound attempts over current value/reader contexts, never stale node lists.
    visited: set[tuple[Node, tuple[Node, ...]]] = set()
    for _ in range(max_candidates):
        targets = tuple(
            context for context in _contexts(current) if context not in visited
        )
        if not targets:
            break
        producer, reader_edges = targets[0]
        visited.add(targets[0])
        names = current.debug_names
        users = tuple(dict.fromkeys(reader_edges))
        edges = len(reader_edges)
        baseline = materialization_footprint(current)
        rule = RULE
        reason = ""
        status = "retained"
        candidate = None
        candidate_hash = None
        if producer in current.outputs.values():
            reason = "retained-output: the caller explicitly requests the dense tensor"
        elif producer.op != "einsum":
            rule = "factorized-source"
            status = "requires-provider"
            reason = (
                "No exact contraction factors are visible. Supply a compiler-visible "
                "factor/provider with index, consumer, alias, derivative and resource "
                "contracts; dense MP2/CC cotangents do not establish separability."
            )
        elif edges != 1:
            reason = "shared-value: repeated edges or multiple consumers would duplicate work"
        elif users[0].op != "einsum":
            reason = "consumer-boundary: view, nonlinear or opaque consumer is not proven substitutable"
        elif not allow_reassociation:
            reason = "reduction-order: explicit algebraic reassociation permission is required"
        elif "precision_execution" in current.provenance:
            reason = "precision-execution: arithmetic contracts for new intermediates are undefined"
        elif RULE not in available_rules:
            reason = "unsupported-provider: no available compiler rule can execute this request"
        elif any(node.spec.dtype != "float64" for node in (producer, users[0])):
            reason = (
                "precision-domain: only real FP64 contraction substitution is admitted"
            )
        elif any(
            len(labels) != len(set(labels))
            for node in (producer, users[0])
            for labels in node.attrs["labels"]
        ):
            reason = "diagonal-domain: repeated-label contraction substitution is not admitted"
        elif len(producer.inputs) + len(users[0].inputs) - 1 > max_operands:
            reason = (
                "search-bound: combined contraction exceeds the finite operand limit"
            )
        else:
            consumer = users[0]
            fused = _substitute(producer, consumer)
            root = _reassociate_node(
                fused,
                max_operands=max_operands,
                max_intermediate_axes=limits,
                require_degree_reduction=False,
            )
            proposed = _replace(current, consumer, root)
            candidate = materialization_footprint(proposed)
            candidate_hash = proposed.logical_hash
            if candidate.max_work_degree > baseline.max_work_degree or (
                candidate.concrete_work > baseline.concrete_work
            ):
                reason = "work-regression: lower storage does not justify additional scientific work"
            elif candidate.materialized_bytes >= baseline.materialized_bytes or (
                candidate.peak_live_bytes > baseline.peak_live_bytes
            ):
                reason = "storage-regression: no strict materialized-byte reduction with nonincreasing live peak"
            elif max_materialized_bytes is not None and (
                candidate.materialized_bytes > max_materialized_bytes
            ):
                reason = "resource-budget: candidate exceeds the logical materialization byte budget"
            else:
                status = "selected"
                reason = (
                    "Immutable SSA, one reader edge, private dummy-index renaming and "
                    "exact rational coefficients prove algebraic equivalence. Public "
                    "outputs, input roles and differentiability are preserved; no "
                    "physical alias or external factorization is assumed."
                )
                current = proposed
        decisions.append(
            MaterializationDecision(
                names[producer],
                tuple(names[user] for user in users),
                rule,
                status,
                reason,
                baseline,
                candidate,
                candidate_hash,
            )
        )
    return MaterializationPlan(
        program.logical_hash,
        current,
        before,
        materialization_footprint(current),
        tuple(decisions),
        sum(context not in visited for context in _contexts(current)),
    )
