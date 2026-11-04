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
