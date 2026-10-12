"""Shared unrestricted raw-Hamiltonian, normal-ordering and orbital pullbacks.

Both native backends consume one primal binding and generic TensorIR AD. Raw
chemists' integrals and independent spin frames are not relaxed AO densities by
themselves; runtime consumers qualify CC/Lambda, shared UHF response and physical
producer identities before exposing relaxed cotangents or nuclear derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from generativeqc_compiler.cc.uccsd_equations import build_program
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES, UCCSDLayout
from generativeqc_compiler.cc.uccsd_response import RESIDUALS
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    JVPProgram,
    Node,
    Program,
    Symmetry,
    TensorSpec,
    VJPProgram,
    add,
    constant,
    einsum,
    input_tensor,
    linearize,
    reshape,
    slice_tensor,
    transpose,
    transpose_program,
)
from generativeqc_compiler.tensor.ad_program import _rebuild

RAW_INPUTS = ("h_a", "h_b", "g_aa", "g_ab", "g_bb", "rotation_a", "rotation_b")


@dataclass(frozen=True)
class UCCSDHamiltonianPrograms:
    """Fixed-occupation primal, raw-input/rotation bars and Fov rotation JVP."""

    primal: Program
    pullback: VJPProgram
    orbital_jvp: JVPProgram
    parameter_names: tuple[str, ...]


def build_raw_hamiltonian(
    layout: UCCSDLayout, *, max_elements: int = 1_000_000
) -> UCCSDHamiltonianPrograms:
    """Rotate raw spin matrices, build normal ordering, and bind every CC input.

    The mixed Coulomb tensor has alpha owners on its first pair and beta owners
    on its second. It is not an independent beta-alpha tensor and does not have
    fixed-owner pair-exchange symmetry. Rotation matrices need not be orthogonal
    during AD, so a later overlap-transport chain can use their symmetric bars.
    """
    if not isinstance(layout, UCCSDLayout):
        raise TypeError("raw unrestricted Hamiltonian requires UCCSDLayout")
    counts = {"a": layout.oa + layout.va, "b": layout.ob + layout.vb}
    occupied = {"a": layout.oa, "b": layout.ob}
    spaces = {
        spin: IndexSpace(
            f"full_{spin}", "orbital", counts[spin], "alpha" if spin == "a" else "beta"
        )
        for spin in "ab"
    }

    def spec(
        spins: str,
        symmetries: tuple[Symmetry, ...] = (),
        *,
        role: Literal["parameter", "constant"] = "parameter",
    ) -> TensorSpec:
        return TensorSpec(
            tuple(
                Index(f"axis_{axis}", spaces[spin]) for axis, spin in enumerate(spins)
            ),
            symmetries=symmetries,
            representation="spin_orbital",
            role=role,
            differentiable=role == "parameter",
        )

    rotations = {
        spin: input_tensor(f"rotation_{spin}", spec(spin * 2)) for spin in "ab"
    }
    raw_h = {
        spin: input_tensor(f"h_{spin}", spec(spin * 2, (Symmetry((1, 0)),)))
        for spin in "ab"
    }
    raw_g = {
        pair: input_tensor(
            f"g_{pair}",
            spec(
                pair[0] * 2 + pair[1] * 2,
                tuple(
                    Symmetry(axes)
                    for axes in (
                        ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1))
                        if pair[0] == pair[1]
                        else ((1, 0, 2, 3), (0, 1, 3, 2))
                    )
                ),
            ),
        )
        for pair in ("aa", "ab", "bb")
    }
    rotated_h = {
        spin: einsum(
            "pv,vq->pq",
            einsum("up,uv->pv", rotations[spin], raw_h[spin]),
            rotations[spin],
        )
        for spin in "ab"
    }
    rotated_g = {}
    for pair, value in raw_g.items():
        owners = pair[0] * 2 + pair[1] * 2
        transformed = einsum("up,uvwx->pvwx", rotations[owners[0]], value)
        for equation, spin in zip(
            ("vq,pvwx->pqwx", "wr,pqwx->pqrx", "xs,pqrx->pqrs"),
            owners[1:],
            strict=True,
        ):
            transformed = einsum(equation, rotations[spin], transformed)
        rotated_g[pair] = transformed
    densities = {
        spin: constant(
            tuple(
                1 if row == column and row < occupied[spin] else 0
                for row in range(counts[spin])
                for column in range(counts[spin])
            ),
            spec(spin * 2, role="constant"),
        )
        for spin in "ab"
    }

    def coulomb(first: str, second: str) -> Node:
        if first + second == "ba":
            return transpose(rotated_g["ab"], (2, 3, 0, 1))
        return rotated_g[first + second]

    fock = {
        spin: add(
            rotated_h[spin],
            *(
                einsum("pqrs,rs->pq", coulomb(spin, source), densities[source])
                for source in "ab"
            ),
            einsum(
                "prqs,rs->pq",
                rotated_g[spin * 2],
                densities[spin],
                coefficient=-1,
            ),
        )
        for spin in "ab"
    }
    reference_energy = add(
        *(
            einsum(
                "pq,pq->",
                densities[spin],
                add(rotated_h[spin], fock[spin]),
                coefficient=Fraction(1, 2),
            )
            for spin in "ab"
        )
    )
    cc = build_program(layout.oa, layout.ob, layout.va, layout.vb)
    names = tuple(
        sorted(
            {
                node.attrs["name"]
                for node in cc.nodes
                if node.op == "input" and node.attrs["name"] not in BLOCK_NAMES
            }
        )
    )
    outputs = {}
    for name in names:
        domains, owners = name[1:].split("_")
        if name.startswith("f"):
            full = fock[owners[0]]
        else:
            terms, coefficients = [], []
            if owners[0] == owners[2] and owners[1] == owners[3]:
                terms.append(transpose(coulomb(owners[0], owners[1]), (0, 2, 1, 3)))
                coefficients.append(1)
            if owners[0] == owners[3] and owners[1] == owners[2]:
                terms.append(transpose(coulomb(owners[0], owners[1]), (0, 2, 3, 1)))
                coefficients.append(-1)
            if not terms:
                raise ValueError("canonical CC input violates spin conservation")
            full = add(*terms, coefficients=coefficients)
        ranges = tuple(
            (0, occupied[spin]) if domain == "o" else (occupied[spin], counts[spin])
            for domain, spin in zip(domains, owners, strict=True)
        )
        outputs[name] = slice_tensor(full, ranges)
    outputs["reference_electronic_energy"] = reference_energy
    primal = Program(
        outputs,
        provenance={
            "model": "unrestricted raw chemists Hamiltonian to canonical CC inputs",
            "occupied": occupied,
            "counts": counts,
            "normal_ordering": "Fock after independent spin rotations; unit occupations",
            "metric": "full Frobenius raw-input and general rotation cotangents",
        },
    )
    pullback = transpose_program(
        primal,
        (*names, "reference_electronic_energy"),
        inputs=("h_a", "h_b", "g_aa", "g_ab", "g_bb", "rotation_a", "rotation_b"),
        max_elements=max_elements,
    )
    orbital_jvp = linearize(
        primal,
        ("rotation_a", "rotation_b"),
        outputs=("fov_aa", "fov_bb"),
        max_elements=max_elements,
    )
    return UCCSDHamiltonianPrograms(primal, pullback, orbital_jvp, names)


@dataclass(frozen=True)
class UCCSDHamiltonianResponsePrograms:
    """One composed CC/HF/UHF-constraint Lagrangian and its raw-input bars.

    Seed correlation energy with one, physical CC residuals with U lambda/w,
    reference electronic energy with zero or one, and Fov with the shared UHF
    adjoint. The overlap bars use general-frame metric transport dU=-dS_MO/2.
    The UHF response convention is C'=C exp(-K), K_ia=x and K_ai=-x, so its
    orbital RHS is +antisym(G_U)_ia, not RCCSD's assumed sign or spin factors.
    """

    primal: Program
    pullback: VJPProgram
    weights: Program
    equation_hash: str
    binding_hash: str


def build_lagrangian_programs(
    layout: UCCSDLayout,
    hamiltonian: UCCSDHamiltonianPrograms,
    *,
    max_elements: int = 1_000_000,
) -> UCCSDHamiltonianResponsePrograms:
    """Compose the existing CC DAG with physical inputs before one shared VJP.

    Same-shape typed views identify full-frame slices with the CC graph's
    occupied/virtual index spaces. Generic DAG rebuilding and AD accumulate
    all aliases; no CC equation, normal-ordering derivative, or dense amplitude
    Jacobian is reimplemented. This avoids a separate complete CC VJP/replay
    for each canonical independent graph input.
    """
    if not isinstance(layout, UCCSDLayout) or not isinstance(
        hamiltonian, UCCSDHamiltonianPrograms
    ):
        raise TypeError("UCCSD physical response requires a layout and Hamiltonian")
    cc = build_program(layout.oa, layout.ob, layout.va, layout.vb)
    replacements = {}
    for node in cc.nodes:
        if node.op != "input" or node.attrs["name"] in BLOCK_NAMES:
            continue
        binding = hamiltonian.primal.outputs[node.attrs["name"]]
        if binding.spec.shape != node.spec.shape:
            raise ValueError("UCCSD physical binding populations disagree")
        replacements[node] = reshape(binding, node.spec.indices)
    bound = _rebuild(cc, replacements)
    primal = Program(
        {
            **bound.outputs,
            **{
                name: hamiltonian.primal.outputs[name]
                for name in ("reference_electronic_energy", "fov_aa", "fov_bb")
            },
        },
        provenance={
            "model": "physical unrestricted CC/HF/UHF-constraint Lagrangian",
            "cc_equation": cc.logical_hash,
            "hamiltonian_binding": hamiltonian.primal.logical_hash,
        },
    )
    pullback = transpose_program(
        primal,
        (
            "correlation_energy",
            *RESIDUALS,
            "reference_electronic_energy",
            "fov_aa",
            "fov_bb",
        ),
        inputs=RAW_INPUTS,
        max_elements=max_elements,
    )
    outputs = {
        name: pullback.program.outputs[f"bar_{name}"]
        for name in RAW_INPUTS
        if not name.startswith("rotation")
    }
    occupied = {"a": layout.oa, "b": layout.ob}
    counts = {"a": layout.oa + layout.va, "b": layout.ob + layout.vb}
    for spin in "ab":
        gradient = pullback.program.outputs[f"bar_rotation_{spin}"]
        stationarity = add(gradient, transpose(gradient, (1, 0)), coefficients=(1, -1))
        outputs[f"rotation_{spin}"] = gradient
        outputs[f"stationarity_{spin}"] = stationarity
        outputs[f"overlap_{spin}"] = add(
            gradient,
            transpose(gradient, (1, 0)),
            coefficients=(Fraction(-1, 4), Fraction(-1, 4)),
        )
        outputs[f"orbital_rhs_{spin}"] = slice_tensor(
            stationarity, ((0, occupied[spin]), (occupied[spin], counts[spin]))
        )
    weights = Program(
        outputs,
        provenance={
            "physical_lagrangian_pullback": pullback.derivative_hash,
            "overlap": "-sym(G_U)/2 for each spin, then accumulate physical AO binding",
            "orbital_rhs": "+antisym(G_U)_ia for shared UHF C exp(-K) convention",
        },
    )
    return UCCSDHamiltonianResponsePrograms(
        primal, pullback, weights, cc.logical_hash, hamiltonian.primal.logical_hash
    )
