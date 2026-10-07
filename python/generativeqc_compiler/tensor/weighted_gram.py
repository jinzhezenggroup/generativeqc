"""Recognize and lower the existing SCF weighted-Gram TensorIR regions.

This module adds no scientific operation. The ternary einsum and its optional
weight multiply in tensor.scf remain authoritative. The closed incumbent
schedules retain different physical layouts, rounding, checks and publication;
a generic binary contraction or symmetric cross product cannot replace them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from generativeqc_compiler.common.provenance import canonical_hash

from .ir import Node, add, input_tensor, multiply
from .lowering import TensorLoweringAdapter
from .native_lowering import weighted_gram_candidate as retained_candidate
from .optimize import prepare_for_backend
from .program import Program
from .scf import density_program, weighted_density_program
from .types import TensorSpec

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_provider import LoweringCandidate

Schedule = Literal[
    "occupied-cpu", "occupied-cuda", "column-scaled-cpu", "checked-pair-cuda"
]


def template_hash(program: Program) -> str:
    """Preserve the existing shape-independent SCF AOT equation identity."""

    def signature(node: Node) -> dict:
        attrs = {
            key: node.attrs[key]
            for key in ("coefficient", "labels", "output")
            if key in node.attrs
        }
        if node.op == "input":
            attrs["name"] = node.attrs["name"]
        return {
            "op": node.op,
            "kinds": tuple(i.space.kind for i in node.spec.indices),
            "inputs": tuple(signature(value) for value in node.inputs),
            "attrs": attrs,
        }

    return canonical_hash(
        {
            "outputs": {
                name: signature(node) for name, node in sorted(program.outputs.items())
            }
        }
    )


@dataclass(frozen=True)
class WeightedGram:
    """Recognition metadata referencing original nodes, never new equations."""

    adapter: TensorLoweringAdapter
    output: str
    contraction: Node
    coefficients: Node
    occupations: Node
    energies: Node | None

    @property
    def weighted(self) -> bool:
        return self.energies is not None


def recognize_weighted_gram(program: Program, output: str) -> WeightedGram:
    """Admit only the original shared-coefficient, unit FP64 SCF topology."""
    if output not in ("density", "weighted_density") or tuple(program.outputs) != (
        output,
    ):
        raise ValueError("weighted Gram requires its one expected SCF output")
    if any(node.spec.dtype != "float64" for node in program.live_nodes):
        raise ValueError("weighted Gram retained schedules require float64 IR")
    root = program.outputs[output]
    kinds = lambda node: tuple(i.space.kind for i in node.spec.indices)
    if (
        root.op != "einsum"
        or root.attrs.get("coefficient") != (1, 1)
        or root.attrs.get("labels") != ((0, 1, 2, 3), (0, 1, 3), (0, 1, 4, 3))
        or root.attrs.get("output") != (0, 1, 2, 4)
        or kinds(root) != ("batch", "spin", "ao", "ao")
        or len(root.inputs) != 3
    ):
        raise ValueError("weighted Gram contraction topology changed")
    coefficients, weights, right = root.inputs
    if (
        coefficients is not right
        or coefficients.op != "input"
        or coefficients.attrs["name"] != "coefficients"
        or kinds(coefficients) != ("batch", "spin", "ao", "orbital")
        or kinds(weights) != ("batch", "spin", "orbital")
    ):
        raise ValueError("weighted Gram coefficient layout or alias topology changed")
    energies = None
    occupations = weights
    if output == "weighted_density":
        if weights.op != "multiply" or len(weights.inputs) != 2:
            raise ValueError("weighted Gram weight topology changed")
        inputs = {n.attrs.get("name"): n for n in weights.inputs if n.op == "input"}
        if set(inputs) != {"occupations", "orbital_energies"}:
            raise ValueError("weighted Gram energy-weight operands changed")
        occupations, energies = inputs["occupations"], inputs["orbital_energies"]
        if any(n.spec.indices != weights.spec.indices for n in (occupations, energies)):
            raise ValueError("weighted Gram energy-weight layout changed")
    elif weights.op != "input" or weights.attrs["name"] != "occupations":
        raise ValueError("weighted Gram occupation topology changed")
    # Domain names/populations, not just their kinds, must agree at every use.
    domains = {}
    for value, labels in zip(root.inputs, root.attrs["labels"], strict=True):
        for label, index in zip(labels, value.spec.indices, strict=True):
            if domains.setdefault(label, index.space) != index.space:
                raise ValueError("weighted Gram crosses scientific index domains")
    if any(
        index.space != domains[label]
        for index, label in zip(root.spec.indices, root.attrs["output"], strict=True)
    ):
        raise ValueError("weighted Gram output layout changed")
    adapter = TensorLoweringAdapter(program)
    if any(not d == adapter.directives[root] for d in adapter.directives.values()):
        raise ValueError("weighted Gram cannot change intermediate precision")
    return WeightedGram(adapter, output, root, coefficients, occupations, energies)


def require_candidate(
    region: WeightedGram, candidate: LoweringCandidate, schedule: Schedule
) -> None:
    """Reject stale, forged or mismatched candidate records before emission."""
    if candidate != retained_candidate(region, schedule):
        raise ValueError(
            "weighted Gram emission requires its admitted retained candidate"
        )


def canonical_regions(
    backend: str, *, orbital_count: int = 2
) -> tuple[WeightedGram, WeightedGram]:
    def region(program: Program, output: str) -> WeightedGram:
        return recognize_weighted_gram(
            prepare_for_backend(program, backend, preserve_reduction_order=True), output
        )

    return (
        region(
            density_program(1, 3, spin_count=2, orbital_count=orbital_count), "density"
        ),
        region(
            weighted_density_program(1, 3, spin_count=2, orbital_count=orbital_count),
            "weighted_density",
        ),
    )


def require_pair(plain: WeightedGram, weighted: WeightedGram) -> None:
    """Bind paired execution to the same coefficient and occupation inputs."""
    if (
        plain.weighted
        or not weighted.weighted
        or any(
            plain.adapter.hashes[a] != weighted.adapter.hashes[b]
            for a, b in (
                (plain.coefficients, weighted.coefficients),
                (plain.occupations, weighted.occupations),
            )
        )
    ):
        raise ValueError("paired weighted Gram requires matching original inputs")


def scalar_programs(plain: WeightedGram, weighted: WeightedGram) -> dict[str, Program]:
    """Scalarize the recognized ternary product and weight multiply stages.

    Parent hashes bind the factorization proof. Stage input/output role names
    retain the established generated scalar hashes and exact emitted bodies.
    These are ordinary TensorIR programs for a schedule, not independent method
    equations. No stage may be requested without first recognizing both roots.
    """
    require_pair(plain, weighted)

    def scalar(name: str, source: Node) -> Node:
        return input_tensor(name, TensorSpec((), dtype=source.spec.dtype, role="input"))

    c = scalar("coefficient", plain.coefficients)
    f = scalar("weight", plain.occupations)
    weighted_c = scalar("weighted_coefficient", plain.coefficients)
    accumulator = scalar("accumulator", plain.contraction)
    occupation = scalar("occupation", weighted.occupations)
    assert weighted.energies is not None
    epsilon = scalar("eigenvalue", weighted.energies)
    stages = {
        "energy_weight": multiply(occupation, epsilon),
        "weighted_coefficient": multiply(c, f),
        "contribution": multiply(weighted_c, c),
        "updated": add(accumulator, multiply(weighted_c, c)),
    }
    return {
        name: Program(
            {name: root},
            provenance={
                "source": "tensor.scf weighted-Gram schedule scalarization",
                "density_node": plain.adapter.hashes[plain.contraction],
                "weighted_density_node": weighted.adapter.hashes[weighted.contraction],
                "stage": name,
            },
        )
        for name, root in stages.items()
    }


def checked_scalar_programs(backend: str) -> dict[str, Program]:
    plain, weighted = canonical_regions(backend, orbital_count=3)
    schedule = "column-scaled-cpu" if backend == "cpu" else "checked-pair-cuda"
    for region in (plain, weighted):
        require_candidate(region, retained_candidate(region, schedule), schedule)
    return {
        name: prepare_for_backend(program, backend)
        for name, program in scalar_programs(plain, weighted).items()
    }


def density_template_hash() -> str:
    return template_hash(canonical_regions("cuda")[0].adapter.program)


def weighted_density_template_hash() -> str:
    return template_hash(canonical_regions("cuda")[1].adapter.program)
