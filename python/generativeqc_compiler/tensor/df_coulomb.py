"""Resident density-fitting Coulomb algebra in the existing TensorIR.

The packed pair input is the caller's symmetric pair contraction: its density
contains D(mu,nu)+D(nu,mu) off diagonal and D(mu,mu) on diagonal. Packing and
scattering stay explicit in their existing owner; this program never changes
the fitting metric, cutoff, raw source, or AO-pair convention.
"""

from __future__ import annotations

from .ir import einsum, input_tensor
from .program import Program
from .types import Index, IndexSpace, TensorSpec


def coulomb_program(batch: int, nbf: int, naux: int, *, packed: bool) -> Program:
    """Return charge=B:D and J=B*charge for explicit resident tensor storage."""
    if any(type(n) is not int or n <= 0 for n in (batch, nbf, naux)):
        raise ValueError("positive resident Coulomb dimensions required")
    b = Index("b", IndexSpace("batch", "batch", batch))
    q = Index("q", IndexSpace("auxiliary", "auxiliary", naux))
    if packed:
        p = Index("p", IndexSpace("pair", "pair", nbf * (nbf + 1) // 2))
        pairs = (p,)
        labels = "p"
    else:
        ao = IndexSpace("ao", "ao", nbf)
        pairs = (Index("i", ao), Index("j", ao))
        labels = "ij"
    values = input_tensor("factor", TensorSpec((b, *pairs, q), role="input"))
    density = input_tensor("density", TensorSpec((b, *pairs), role="input"))
    charge = einsum(f"b{labels}q,b{labels}->bq", values, density)
    coulomb = einsum(f"b{labels}q,bq->b{labels}", values, charge)
    return Program({"charge": charge, "coulomb": coulomb})
