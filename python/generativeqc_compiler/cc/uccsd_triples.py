"""Canonical real spin-orbital (T) tiles, lowered before numerical execution.

The connected/disconnected seeds use the general spin-orbital definition also
implemented by PySCF 2.14.0 ``gccsd_t.py``. Independent occupied and virtual
cyclic sums give the nine-term antisymmetrizer; no restricted spin factors are
borrowed. Only one canonical spin pattern is evaluated per channel. Its exact
ordered-spin multiplicity is 1/36 for aaa/bbb and 1/4 for aab/abb.

Bindings describe slices of existing UCCSD feeds rather than embedding physical
tile positions in the graph. Equal-shaped tiles therefore share one artifact,
and their input cotangents can later be scattered through the same bindings.
This module imports no runtime/oracle and requires no molecular T3 input/output.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import comb

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    broadcast,
    constant,
    divide,
    einsum,
    input_tensor,
    transpose_program,
)

from .uccsd_layout import UCCSDLayout

CHANNELS = ("aaa", "aab", "abb", "bbb")
_CYCLIC = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
_DOUBLES = {
    "aaaa": ("t2aa", (0, 1, 2, 3), 1),
    "bbbb": ("t2bb", (0, 1, 2, 3), 1),
    "abab": ("t2ab", (0, 1, 2, 3), 1),
    "baab": ("t2ab", (1, 0, 2, 3), -1),
    "abba": ("t2ab", (0, 1, 3, 2), -1),
    "baba": ("t2ab", (1, 0, 3, 2), 1),
}


@dataclass(frozen=True)
class TriplesOperand:
    """One position-independent input binding to a full, canonical source feed.

    Transpose ``source`` by ``axes``, then select the tile coordinates associated
    with each external label in ``labels``. Contracted e/m coordinates remain
    whole. Signs are part of the graph, not an implicit feed normalization.
    """

    name: str
    source: str
    axes: tuple[int, ...]
    labels: str


@dataclass(frozen=True)
class TriplesTileProgram:
    """Immutable scalar tile graph and its explicit input-coordinate bindings."""

    program: Program
    operands: tuple[TriplesOperand, ...]
    channel: str
    shape: tuple[int, ...]


def _resolve(name: str, spins: str) -> tuple[str, tuple[int, ...], int] | None:
    """Resolve physical seeds to the five amplitudes and existing UCCSD feeds."""
    if name == "t2":
        return _DOUBLES.get(spins)
    if name in ("t1", "fov"):
        if spins[0] != spins[1]:
            return None
        source = f"t1{spins[0]}" if name == "t1" else f"fov_{spins}"
        return source, (0, 1), 1
    if sorted(spins[:2]) != sorted(spins[2:]):
        return None
    domains, axes = {
        "gvvvo": ("ovvv", (3, 2, 1, 0)),
        "govoo": ("ooov", (2, 3, 0, 1)),
        "gvvoo": ("oovv", (2, 3, 0, 1)),
    }[name]
    inverse = tuple(axes.index(position) for position in range(4))
    source_spins = "".join(spins[position] for position in inverse)
    return f"g{domains}_{source_spins}", axes, 1


def channel_shape(layout: UCCSDLayout, channel: str) -> tuple[int, ...]:
    """Return canonical occupied/virtual populations for one conserved channel."""
    if not isinstance(layout, UCCSDLayout):
        raise TypeError("triples require an explicit UCCSD layout")
    if channel not in CHANNELS:
        raise ValueError("unknown unrestricted triples spin channel")
    return tuple(getattr(layout, domain + spin) for domain in "ov" for spin in channel)


def channel_possible(layout: UCCSDLayout, channel: str) -> bool:
    """Require enough distinct same-spin orbitals for a physical triple."""
    channel_shape(layout, channel)
    return all(
        getattr(layout, domain + spin) >= channel.count(spin)
        for domain in "ov"
        for spin in "ab"
    )


def _build_tile_program(
    layout: UCCSDLayout,
    channel: str,
    shape: tuple[int, ...],
    *,
    resolvents: bool = False,
) -> TriplesTileProgram:
    """Lower one scalar (T) tile with independent alpha/beta populations.

    ``shape`` specifies the six local i/j/k/a/b/c lengths, not physical offsets.
    Zero/oversized/non-integer axes are rejected. The caller owns coordinate
    coverage, finite denominator admission and buffer/execution budgets.
    """
    full_shape = channel_shape(layout, channel)
    if (
        not isinstance(shape, tuple)
        or len(shape) != 6
        or any(
            type(size) is not int or not 0 < size <= maximum
            for size, maximum in zip(shape, full_shape, strict=True)
        )
    ):
        raise ValueError(
            "triples tile shape must contain six positive bounded integers"
        )
    indices = {
        label: Index(
            label,
            IndexSpace(
                f"tile_{domain}{spin}_{label}",
                "occupied" if domain == "o" else "virtual",
                size,
                "alpha" if spin == "a" else "beta",
            ),
        )
        for label, domain, spin, size in zip(
            "ijkabc", "ooovvv", channel * 2, shape, strict=True
        )
    }
    output_spins = dict(zip("ijkabc", channel * 2, strict=True))
    nodes: dict[TriplesOperand, Node] = {}

    def operand(
        name: str, labels: str, assignment: dict[str, str]
    ) -> tuple[Node, int] | None:
        resolved = _resolve(name, "".join(assignment[label] for label in labels))
        if resolved is None:
            return None
        source, axes, sign = resolved
        binding = TriplesOperand("", source, axes, labels)
        if binding not in nodes:
            bound_indices = []
            for label in labels:
                if label in indices:
                    bound_indices.append(indices[label])
                else:
                    domain = "v" if label == "e" else "o"
                    spin = assignment[label]
                    bound_indices.append(
                        Index(
                            label,
                            IndexSpace(
                                domain + spin,
                                "virtual" if domain == "v" else "occupied",
                                getattr(layout, domain + spin),
                                "alpha" if spin == "a" else "beta",
                            ),
                        )
                    )
            nodes[binding] = input_tensor(
                f"tile_operand_{len(nodes)}",
                TensorSpec(
                    tuple(bound_indices),
                    representation="spin_orbital",
                    role="parameter",
                    differentiable=True,
                ),
            )
        return nodes[binding], sign

    connected_terms, disconnected_terms = [], []
    for occupied, virtual in product(_CYCLIC, repeat=2):
        permutation = str.maketrans(
            {
                **dict(zip("ijk", ("ijk"[axis] for axis in occupied), strict=True)),
                **dict(zip("abc", ("abc"[axis] for axis in virtual), strict=True)),
            }
        )
        for connected, coefficient, equation, sources in (
            (True, 1, "jkae,bcei", ("t2", "gvvvo")),
            (True, -1, "imbc,majk", ("t2", "govoo")),
            (False, 1, "ia,bcjk", ("t1", "gvvoo")),
            (False, 1, "ia,jkbc", ("fov", "t2")),
        ):
            labels = equation.translate(permutation).split(",")
            contracted = sorted(set("".join(labels)) - set(indices))
            for spins in product("ab", repeat=len(contracted)):
                assignment = {
                    **output_spins,
                    **dict(zip(contracted, spins, strict=True)),
                }
                resolved = [
                    operand(source, axes, assignment)
                    for source, axes in zip(sources, labels, strict=True)
                ]
                if any(value is None for value in resolved):
                    continue
                values = [value for value in resolved if value is not None]
                term = einsum(
                    ",".join(labels) + "->ijkabc",
                    *(value[0] for value in values),
                    coefficient=coefficient * values[0][1] * values[1][1],
                )
                (connected_terms if connected else disconnected_terms).append(term)
    output_indices = tuple(indices.values())
    zero = broadcast(constant(0), output_indices, ())
    connected = add(*connected_terms) if connected_terms else zero
    disconnected = add(*disconnected_terms) if disconnected_terms else zero
    denominators = []
    for position, label in enumerate("ijkabc"):
        domain = "o" if position < 3 else "v"
        binding = TriplesOperand("", f"eps_{domain}_{output_spins[label]}", (0,), label)
        node = input_tensor(
            f"tile_operand_{len(nodes)}",
            TensorSpec(
                (indices[label],),
                representation="spin_orbital",
                role="parameter",
                differentiable=True,
            ),
        )
        nodes[binding] = node
        denominators.append(broadcast(node, output_indices, (position,)))
    denominator = add(*denominators, coefficients=(1, 1, 1, -1, -1, -1))
    factor = Fraction(comb(3, channel.count("a")) ** 2, 36)
    resolved_left = divide(connected, denominator)
    complete = add(connected, disconnected)
    energy = einsum(
        "ijkabc,ijkabc->",
        resolved_left,
        complete,
        coefficient=factor,
    )
    outputs = {"triples_energy": energy}
    if resolvents:
        outputs.update(x=resolved_left, y=divide(complete, denominator))
    program = Program(
        outputs,
        provenance={
            "method": "UCCSD(T)",
            "spin_channel": channel,
            "spin_multiplicity": (factor.numerator, factor.denominator),
            "inventory_version": 1,
            "contract": "real canonical spin-orbital nine-term antisymmetrizer",
        },
    )
    live_inputs = {
        node.attrs["name"] for node in program.live_nodes if node.op == "input"
    }
    return TriplesTileProgram(
        program,
        tuple(
            TriplesOperand(
                node.attrs["name"], binding.source, binding.axes, binding.labels
            )
            for binding, node in nodes.items()
            if node.attrs["name"] in live_inputs
        ),
        channel,
        shape,
    )


def build_tile_program(
    layout: UCCSDLayout, channel: str, shape: tuple[int, ...]
) -> TriplesTileProgram:
    """Lower the canonical nine-term scalar tile without physical offsets.

    Six local occupied/virtual lengths must be positive and within their
    actual spin populations. Coordinate coverage, finite denominator gates
    and complete endpoint work/storage admission remain caller responsibilities.
    """
    return _build_tile_program(layout, channel, shape)


def build_tile_response_program(
    layout: UCCSDLayout,
    channel: str,
    shape: tuple[int, ...],
    *,
    max_elements: int = 1_000_000,
) -> TriplesTileProgram:
    """Differentiate the actual bounded (T) tile with respect to every input.

    Seed ``bar_triples_energy`` with FP64 one. Each ``bar_<operand.name>``
    is a full-Frobenius cotangent of the explicitly gathered operand, not a
    packed excitation dual or symmetry-folded integral weight. The caller
    reverses the same transpose/coordinate bindings and sums overlapping
    contributions. Orbital-energy bars are explicit: they must eventually
    chain to the appropriate spin Fock diagonal, not be discarded or treated
    as fixed parameters when forming physical response.
    """
    tile = build_tile_program(layout, channel, shape)
    derivative = transpose_program(
        tile.program,
        ("triples_energy",),
        inputs=tuple(operand.name for operand in tile.operands),
        max_elements=max_elements,
    )
    return TriplesTileProgram(
        Program(
            {
                "triples_energy": tile.program.outputs["triples_energy"],
                **derivative.program.outputs,
            },
            provenance={
                **tile.program.provenance,
                "response_primal_hash": tile.program.logical_hash,
                "response_derivative_hash": derivative.derivative_hash,
                "response_contract": "all tile input bars, including physical denominators",
            },
        ),
        tile.operands,
        channel,
        shape,
    )
