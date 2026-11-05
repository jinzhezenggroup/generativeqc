"""Shared AO density-fitting value map and its unrelaxed raw-source VJP.

The metric spectral rule remains owned by the existing matrix-function owner.
This map treats its symmetric inverse square root as an independent argument;
runtime callers compose its cotangent with that exact fixed-rank metric rule.
"""

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
    transpose_program,
)


def programs() -> tuple[Program, Program]:
    """Return representative-shape programs with runtime AO/auxiliary extents."""
    orbital = IndexSpace("full_a", "orbital", 6, "alpha")
    auxiliary = IndexSpace("df_auxiliary", "batch", 3)
    ao_indices = tuple(Index(name, orbital) for name in ("mu", "nu"))
    aux_indices = tuple(Index(name, auxiliary) for name in ("source", "whitened"))
    raw = input_tensor(
        "three_center",
        TensorSpec(
            (*ao_indices, aux_indices[0]), role="parameter", differentiable=True
        ),
    )
    root = input_tensor(
        "inverse_root", TensorSpec(aux_indices, role="parameter", differentiable=True)
    )
    factors = einsum("mnP,PQ->mnQ", raw, root)
    primal = Program(
        {"eri": einsum("mnQ,klQ->mnkl", factors, factors)},
        provenance={
            "model": "AO Coulomb-metric fitted ERI",
            "layout": "mu,nu,auxiliary",
        },
    )
    return primal, transpose_program(primal, ("eri",)).program


def factor_programs() -> tuple[Program, Program]:
    """Transform and reverse three-index factors without an AO rank-four value.

    Each spin supplies its own immutable physical coefficient matrix. Keeping
    both orbital projections explicit makes the reverse include both orbital
    legs as well as the raw three-center and independent metric-root inputs.
    """
    orbital = IndexSpace("full_a", "orbital", 6, "alpha")
    auxiliary = IndexSpace("df_auxiliary", "batch", 3)
    mu, nu, orbital_index = (Index(name, orbital) for name in ("mu", "nu", "p"))
    source, whitened = (Index(name, auxiliary) for name in ("source", "whitened"))
    raw = input_tensor(
        "three_center",
        TensorSpec((mu, nu, source), role="parameter", differentiable=True),
    )
    root = input_tensor(
        "inverse_root",
        TensorSpec((source, whitened), role="parameter", differentiable=True),
    )
    coefficients = input_tensor(
        "coefficients",
        TensorSpec((mu, orbital_index), role="parameter", differentiable=True),
    )
    first = einsum("mnP,nq->mqP", raw, coefficients)
    transformed = einsum("mp,mqP->pqP", coefficients, first)
    primal = Program(
        {"factors": einsum("pqP,PQ->pqQ", transformed, root)},
        provenance={
            "model": "spin-labelled three-index Coulomb-metric fitted interaction",
            "layout": "orbital,orbital,auxiliary",
            "orbital_response": "both supplied coefficient legs",
        },
    )
    return primal, transpose_program(primal, ("factors",)).program


def factor_block_program() -> Program:
    """Reconstruct only a requested rectangular chemist block from pair panels."""
    left = Index("left", IndexSpace("full_a", "orbital", 6, "alpha"))
    right = Index("right", IndexSpace("full_b", "orbital", 6, "beta"))
    auxiliary = Index("auxiliary", IndexSpace("df_auxiliary", "batch", 3))
    lhs = input_tensor("left_factors", TensorSpec((left, auxiliary), role="input"))
    rhs = input_tensor("right_factors", TensorSpec((right, auxiliary), role="input"))
    return Program({"block": einsum("pQ,rQ->pr", lhs, rhs)})
