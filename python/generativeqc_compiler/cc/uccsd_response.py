"""UCCSD amplitude and Hamiltonian derivatives from the shared physical DAG.

These programs differentiate fixed-amplitude energy/residual equations, never
iterations, DIIS or shifted proposal denominators. The runtime response owner
supplies the physical packed primal/dual maps and qualifies stationary states.
"""

from __future__ import annotations

from dataclasses import dataclass

from generativeqc_compiler.cc.uccsd_equations import build_program
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES, UCCSDLayout
from generativeqc_compiler.tensor import (
    JVPProgram,
    Program,
    VJPProgram,
    linearize,
    transpose_program,
)

RESIDUALS = tuple(name.replace("t", "r", 1) for name in BLOCK_NAMES)


@dataclass(frozen=True)
class UCCSDResponsePrograms:
    """Shared CPU/CUDA RHS, J*, J and fresh Lagrangian-stationarity actions.

    Amplitude directions expand with U. Residual cotangents lift with U/w,
    where w=4 for aa/bb doubles and w=1 otherwise. Returned amplitude bars
    pull back with U*, not ordinary amplitude packing. Physical Lambda is
    the unweighted excitation dual; its full arrays expand with U, without
    the one-quarter residual-seed factor.
    """

    primal: Program
    energy_vjp: VJPProgram
    residual_vjp: VJPProgram
    residual_jvp: JVPProgram
    stationarity_vjp: VJPProgram

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Canonical independent Fock/integral inputs, excluding amplitudes."""
        return tuple(
            sorted(
                {
                    node.attrs["name"]
                    for node in self.primal.nodes
                    if node.op == "input" and node.attrs["name"] not in BLOCK_NAMES
                }
            )
        )


def build_response_programs(
    layout: UCCSDLayout, *, max_elements: int = 1_000_000
) -> UCCSDResponsePrograms:
    """Generate exact typed derivatives for all five unrestricted blocks."""
    if not isinstance(layout, UCCSDLayout):
        raise TypeError("UCCSD response requires UCCSDLayout")
    primal = build_program(layout.oa, layout.ob, layout.va, layout.vb)
    return UCCSDResponsePrograms(
        primal,
        transpose_program(
            primal,
            ("correlation_energy",),
            inputs=BLOCK_NAMES,
            max_elements=max_elements,
        ),
        transpose_program(
            primal, RESIDUALS, inputs=BLOCK_NAMES, max_elements=max_elements
        ),
        linearize(primal, BLOCK_NAMES, outputs=RESIDUALS, max_elements=max_elements),
        transpose_program(
            primal,
            ("correlation_energy", *RESIDUALS),
            inputs=BLOCK_NAMES,
            max_elements=max_elements,
        ),
    )


def build_parameter_vjp(
    programs: UCCSDResponsePrograms, parameter: str, *, max_elements: int = 1_000_000
) -> VJPProgram:
    """Generate one fixed-orbital correlation-Lagrangian parameter cotangent.

    Seed energy with one and residuals with the full physical-dual lift.
    Several redundant/permuted inputs can depend on one physical Hamiltonian
    perturbation; upstream consumers must accumulate all canonical bindings.
    This contains neither HF/reference energy, normal-ordering pullback nor
    orbital/nuclear response, and is not a relaxed density or force by itself.
    """
    if not isinstance(programs, UCCSDResponsePrograms):
        raise TypeError("UCCSD parameter VJP requires response programs")
    if not isinstance(parameter, str) or parameter not in programs.parameter_names:
        raise ValueError("unknown UCCSD Fock/integral parameter")
    return transpose_program(
        programs.primal,
        ("correlation_energy", *RESIDUALS),
        inputs=(parameter,),
        max_elements=max_elements,
    )
