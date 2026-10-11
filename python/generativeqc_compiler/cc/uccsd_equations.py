"""Real-FP64 UCCSD energy and physical residuals in shared TensorIR.

Spin summation is performed while building the graph, not by embedding five
amplitude blocks in a dense spin-orbital array. The same graph is consumable by
the CPU, CUDA, JVP and VJP owners. There is no reference, solver or backend policy
here. Integrals are real antisymmetrized physicists' integrals <pq||rs>.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from itertools import product

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    broadcast,
    constant,
    einsum,
    input_tensor,
    transpose,
)

from .uccsd_layout import UCCSDLayout

Term = tuple[Fraction | int, str, tuple[str, ...]]
SpinBlocks = dict[tuple[str, ...], Node]
INTEGRAL_DOMAINS = ("oooo", "ooov", "oovv", "ovov", "ovvv", "vvvv")
VERSION = 1


def _permutations(term: Term, *, occupied: bool, virtual: bool) -> tuple[Term, ...]:
    """Expand (1-Pij)/(1-Pab), including independent mixed-spin sectors."""
    coefficient, equation, operands = term
    source, target = equation.split("->")
    terms = []
    for swap_occ, swap_vir in product((False, True), repeat=2):
        if (swap_occ and not occupied) or (swap_vir and not virtual):
            continue
        permutation = {}
        if swap_occ:
            permutation.update({"i": "j", "j": "i"})
        if swap_vir:
            permutation.update({"a": "b", "b": "a"})
        permuted = source.translate(str.maketrans(permutation))
        terms.append(
            (
                coefficient * (-1) ** (swap_occ + swap_vir),
                f"{permuted}->{target}",
                operands,
            )
        )
    return tuple(terms)


def _conserves_spin(spins: tuple[str, ...]) -> bool:
    middle = len(spins) // 2
    return sorted(spins[:middle]) == sorted(spins[middle:])


class _SpinLowering:
    """Resolve contracted spin labels into existing typed einsum primitives."""

    def __init__(self, layout: UCCSDLayout) -> None:
        self.spaces = {
            (domain, spin): IndexSpace(
                f"{domain}{spin}",
                "occupied" if domain == "o" else "virtual",
                population,
                "alpha" if spin == "a" else "beta",
            )
            for (domain, spin), population in zip(
                (("o", "a"), ("o", "b"), ("v", "a"), ("v", "b")),
                (layout.oa, layout.ob, layout.va, layout.vb),
                strict=True,
            )
        }
        self.tensors: dict[str, SpinBlocks] = {}

    def spec(self, domains: str, spins: tuple[str, ...]) -> TensorSpec:
        return TensorSpec(
            tuple(
                Index(f"p{axis}", self.spaces[(domain, spin)])
                for axis, (domain, spin) in enumerate(zip(domains, spins, strict=True))
            ),
            representation="spin_orbital",
            role="parameter",
            differentiable=True,
        )

    def inputs(self, name: str, domains: str) -> None:
        self.tensors[name] = {
            spins: input_tensor(f"{name}_{''.join(spins)}", self.spec(domains, spins))
            for spins in product("ab", repeat=len(domains))
            if _conserves_spin(spins)
        }

    def amplitudes(self, layout: UCCSDLayout) -> None:
        """Resolve all four mixed permutations to the single stored ab block."""
        nodes = {
            block.name: input_tensor(
                block.name, replace(block.spec, role="parameter", differentiable=True)
            )
            for block in layout.blocks
        }
        self.tensors["t1"] = {("a", "a"): nodes["t1a"], ("b", "b"): nodes["t1b"]}
        mixed = nodes["t2ab"]
        self.tensors["t2"] = {
            ("a", "a", "a", "a"): nodes["t2aa"],
            ("b", "b", "b", "b"): nodes["t2bb"],
            ("a", "b", "a", "b"): mixed,
            ("b", "a", "a", "b"): add(
                transpose(mixed, (1, 0, 2, 3)), coefficients=(-1,)
            ),
            ("a", "b", "b", "a"): add(
                transpose(mixed, (0, 1, 3, 2)), coefficients=(-1,)
            ),
            ("b", "a", "b", "a"): transpose(mixed, (1, 0, 3, 2)),
        }

    def define(self, name: str, domains: str, terms: tuple[Term, ...]) -> None:
        """Eliminate forbidden spin assignments before constructing any node."""
        blocks = {}
        for output_spins in product("ab", repeat=len(domains)):
            if not _conserves_spin(output_spins):
                continue
            contributions = []
            for coefficient, equation, operands in terms:
                sources, target = equation.split("->")
                labels = sources.split(",")
                contracted = sorted(set(sources.replace(",", "")) - set(target))
                for summed_spins in product("ab", repeat=len(contracted)):
                    assignment = dict(zip(target, output_spins, strict=True))
                    assignment.update(zip(contracted, summed_spins, strict=True))
                    nodes = [
                        self.tensors[operand].get(
                            tuple(assignment[label] for label in axes)
                        )
                        for operand, axes in zip(operands, labels, strict=True)
                    ]
                    if all(node is not None for node in nodes):
                        contributions.append(
                            einsum(
                                equation,
                                *(node for node in nodes if node is not None),
                                coefficient=coefficient,
                            )
                        )
            spec = self.spec(domains, output_spins)
            blocks[output_spins] = (
                add(*contributions)
                if contributions
                else broadcast(constant(0), spec.indices, ())
            )
        self.tensors[name] = blocks


def build_program(oa: int, ob: int, va: int, vb: int) -> Program:
    """Build energy and five unpreconditioned projected UCCSD residuals.

    Populations are independent nonnegative integer dimensions, including empty
    spin blocks. The caller owns UHF-domain admission and canonical denominators.
    Keeping the full occupied/virtual Fock matrices in these equations makes
    residuals physical, rather than denominator-divided Jacobi proposals.
    """
    layout = UCCSDLayout(oa, ob, va, vb)
    lowering = _SpinLowering(layout)
    lowering.amplitudes(layout)
    for name, domains in (("foo", "oo"), ("fov", "ov"), ("fvv", "vv")):
        lowering.inputs(name, domains)
    for domains in INTEGRAL_DOMAINS:
        lowering.inputs(f"g{domains}", domains)
    half, quarter = Fraction(1, 2), Fraction(1, 4)
    disconnected: tuple[Term, ...] = (
        (1, "ia,jb->ijab", ("t1", "t1")),
        (-1, "ib,ja->ijab", ("t1", "t1")),
    )
    for name, factor in (("tau", 1), ("tau_half", half)):
        lowering.define(
            name,
            "oovv",
            ((1, "ijab->ijab", ("t2",)),)
            + tuple(
                (coefficient * factor, equation, operands)
                for coefficient, equation, operands in disconnected
            ),
        )
    lowering.define(
        "Fvv",
        "vv",
        (
            (1, "ae->ae", ("fvv",)),
            (-half, "me,ma->ae", ("fov", "t1")),
            (1, "mf,mafe->ae", ("t1", "govvv")),
            (-half, "mnaf,mnef->ae", ("tau_half", "goovv")),
        ),
    )
    lowering.define(
        "Foo",
        "oo",
        (
            (1, "mi->mi", ("foo",)),
            (half, "me,ie->mi", ("fov", "t1")),
            (1, "ne,mnie->mi", ("t1", "gooov")),
            (half, "inef,mnef->mi", ("tau_half", "goovv")),
        ),
    )
    lowering.define(
        "Fov",
        "ov",
        (
            (1, "me->me", ("fov",)),
            (1, "nf,mnef->me", ("t1", "goovv")),
        ),
    )
    lowering.define(
        "Woooo",
        "oooo",
        (
            (1, "mnij->mnij", ("goooo",)),
            (1, "je,mnie->mnij", ("t1", "gooov")),
            (-1, "ie,mnje->mnij", ("t1", "gooov")),
            (quarter, "ijef,mnef->mnij", ("tau", "goovv")),
        ),
    )
    lowering.define(
        "Wvvvv",
        "vvvv",
        (
            (1, "abef->abef", ("gvvvv",)),
            (-1, "ma,mbef->abef", ("t1", "govvv")),
            (1, "mb,maef->abef", ("t1", "govvv")),
            (quarter, "mnab,mnef->abef", ("tau", "goovv")),
        ),
    )
    lowering.define(
        "Wovvo",
        "ovvo",
        (
            (-1, "mbje->mbej", ("govov",)),
            (1, "jf,mbef->mbej", ("t1", "govvv")),
            (1, "nb,mnje->mbej", ("t1", "gooov")),
            (-half, "jnfb,mnef->mbej", ("t2", "goovv")),
            (-1, "jf,nb,mnef->mbej", ("t1", "t1", "goovv")),
        ),
    )
    lowering.define(
        "Fvv_doubles",
        "vv",
        (
            (1, "be->be", ("Fvv",)),
            (-half, "mb,me->be", ("t1", "Fov")),
        ),
    )
    lowering.define(
        "Foo_doubles",
        "oo",
        (
            (1, "mj->mj", ("Foo",)),
            (half, "je,me->mj", ("t1", "Fov")),
        ),
    )
    lowering.define(
        "r1",
        "ov",
        (
            (1, "ia->ia", ("fov",)),
            (1, "ie,ae->ia", ("t1", "Fvv")),
            (-1, "ma,mi->ia", ("t1", "Foo")),
            (1, "imae,me->ia", ("t2", "Fov")),
            (-1, "nf,naif->ia", ("t1", "govov")),
            (-half, "imef,maef->ia", ("t2", "govvv")),
            (-half, "mnae,mnie->ia", ("t2", "gooov")),
        ),
    )
    doubles: tuple[Term, ...] = (
        (1, "ijab->ijab", ("goovv",)),
        (half, "mnab,mnij->ijab", ("tau", "Woooo")),
        (half, "ijef,abef->ijab", ("tau", "Wvvvv")),
    )
    for term, occupied, virtual in (
        ((1, "ijae,be->ijab", ("t2", "Fvv_doubles")), False, True),
        ((-1, "imab,mj->ijab", ("t2", "Foo_doubles")), True, False),
        ((1, "imae,mbej->ijab", ("t2", "Wovvo")), True, True),
        ((1, "ie,ma,mbje->ijab", ("t1", "t1", "govov")), True, True),
        ((1, "ie,jeba->ijab", ("t1", "govvv")), True, False),
        ((-1, "ma,ijmb->ijab", ("t1", "gooov")), False, True),
    ):
        doubles += _permutations(term, occupied=occupied, virtual=virtual)
    lowering.define("r2", "oovv", doubles)
    lowering.define(
        "energy",
        "",
        (
            (1, "ia,ia->", ("fov", "t1")),
            (quarter, "ijab,ijab->", ("goovv", "t2")),
            (half, "ijab,ia,jb->", ("goovv", "t1", "t1")),
        ),
    )
    outputs = {"correlation_energy": lowering.tensors["energy"][()]}
    for block in layout.blocks:
        spins = tuple(
            "a" if index.space.spin == "alpha" else "b" for index in block.spec.indices
        )
        outputs[block.name.replace("t", "r", 1)] = lowering.tensors[
            block.name[:2].replace("t", "r", 1)
        ][spins]
    return Program(outputs, provenance={"method": "UCCSD", "version": VERSION})
