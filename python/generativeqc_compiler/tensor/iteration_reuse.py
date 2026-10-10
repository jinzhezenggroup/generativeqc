"""Backend-neutral dependency proof for reuse between iterative evaluations.

This is a schedule analysis over existing immutable TensorIR, not a cache owner
or a new solver. Callers must hold every declared invariant input immutable for
one execution epoch. Changing reference/geometry/basis/parameters ends that epoch;
matching shapes or pointer addresses are never a validity proof. Native owners
remain responsible for retained-storage admission, failure and publication.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from generativeqc_compiler.common.liveness import EffectKind
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.resources import checked_bytes

from .optimize import _tensor_effect
from .program import Program, node_hashes

if TYPE_CHECKING:
    from .ir import Node


@dataclass(frozen=True)
class IterationReusePlan:
    """Proven invariant operations in the original deterministic graph order.

    Every retained intermediate requires distinct live storage throughout replay.
    ``retained_bytes`` is their logical numeric capacity, not a complete endpoint
    peak: native planners must also charge inputs, dynamic scratch, outputs and
    preparation overlap. Pure TensorIR views are logical values; a backend that
    aliases their storage must retain the entire underlying allocation lease.
    """

    invariant_nodes: tuple[Node, ...]
    dynamic_nodes: tuple[Node, ...]
    input_dependencies: tuple[tuple[Node, tuple[str, ...]], ...]
    invariant_inputs: tuple[str, ...]
    retained_bytes: int
    identity: str


def invariant_frontier(program: Program, plan: IterationReusePlan) -> tuple[Node, ...]:
    """Return immutable values read by dynamic operations or published outputs.

    All invariant operations still execute during preparation. Only values that
    cross its boundary need persistent storage; interior preparation temporaries
    can share the caller's existing scratch. This is a liveness cut over the
    existing purity/dependency proof, not a new validity or mathematical proof.
    """
    operations = {node for node in program.live_nodes if node.op != "input"}
    invariant = set(plan.invariant_nodes)
    dynamic = set(plan.dynamic_nodes)
    if invariant & dynamic or invariant | dynamic != operations:
        raise ValueError(
            "invariant frontier requires this program's complete reuse plan"
        )
    readers = set(program.outputs.values()) | {
        source for node in plan.dynamic_nodes for source in node.inputs
    }
    return tuple(node for node in plan.invariant_nodes if node in readers)


def analyze_iteration_reuse(
    program: Program,
    *,
    invariant_inputs: Sequence[str],
    effects: Mapping[str, EffectKind] | None = None,
) -> IterationReusePlan:
    """Classify transitive dependencies before any CPU/CUDA lowering.

    No shape, method name, numerical value or backend is used to infer reuse.
    Effects may conservatively downgrade an existing primitive's PURE proof;
    unknown operations can never be promoted to PURE by this argument. A node
    depending on an opaque/effectful operation stays dynamic even when all of its
    named inputs are invariant. The analysis neither reorders nor drops nodes.
    """
    if not isinstance(program, Program):
        raise TypeError("iteration reuse requires a TensorIR Program")
    if isinstance(invariant_inputs, (str, bytes)) or not isinstance(
        invariant_inputs, Sequence
    ):
        raise TypeError("invariant inputs must be a sequence of names")
    names = tuple(invariant_inputs)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("invariant input names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("duplicate invariant input")
    names = tuple(sorted(names))
    if effects is None:
        effects = {}
    if not isinstance(effects, Mapping) or any(
        not isinstance(op, str) or not isinstance(effect, EffectKind)
        for op, effect in effects.items()
    ):
        raise TypeError("iteration effects must map operation names to EffectKind")
    nodes = program.live_nodes
    available_inputs = {node.attrs["name"] for node in nodes if node.op == "input"}
    if set(names) - available_inputs:
        raise ValueError("invariant input is absent from the live program")
    declared = set(names)
    dependencies: dict[Node, frozenset[str]] = {}
    reusable: dict[Node, bool] = {}
    invariant, dynamic = [], []
    hashes = node_hashes(nodes)
    effective = []
    retained = 0
    for node in nodes:
        proof = _tensor_effect(node)
        effect = effects.get(node.op, proof)
        pure = proof is EffectKind.PURE and effect is EffectKind.PURE
        effective.append((hashes[node], pure))
        if node.op == "input":
            dependencies[node] = frozenset((node.attrs["name"],))
            reusable[node] = pure and node.attrs["name"] in declared
            continue
        dependencies[node] = frozenset().union(
            *(dependencies[source] for source in node.inputs)
        )
        reusable[node] = pure and all(reusable[source] for source in node.inputs)
        if reusable[node]:
            invariant.append(node)
            retained = checked_bytes(
                retained + node.spec.size * node.spec.itemsize,
                "iteration invariant retained bytes",
            )
        else:
            dynamic.append(node)
    identity = canonical_hash(
        {
            "schema": "generativeqc.tensor.iteration-reuse.v1",
            "program": program.logical_hash,
            "invariant_inputs": names,
            "pure_proofs": effective,
        }
    )
    return IterationReusePlan(
        tuple(invariant),
        tuple(dynamic),
        tuple((node, tuple(sorted(dependencies[node]))) for node in nodes),
        names,
        retained,
        identity,
    )
