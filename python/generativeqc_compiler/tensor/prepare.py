"""Shared production preparation before TensorIR backend lowering.

The preparation boundary is intentionally backend-light: it owns exact/canonical
whole-program rewrites and strict symbolic-degree contraction reassociation,
while concrete storage/layout/kernel scheduling remains with the selected backend.
"""

from __future__ import annotations

import typing

from .optimize import optimize
from .program import Program

_BACKENDS = frozenset(("cpu", "cuda"))
SCHEMA = "generativeqc.tensor.production-preparation.v1"


def prepare_for_backend(
    program: Program,
    *,
    backend: str,
    requested_outputs: typing.Any = None,
    preserve_contraction_order: bool = False,
) -> Program:
    """Prepare one production TensorIR program for CPU or CUDA lowering.

    Exact canonical optimization always runs. Strict symbolic-degree contraction
    reassociation runs by default for both production backends when no explicit
    precision-execution contract is attached. Callers may preserve the original
    contraction order for bitwise/order-sensitive auditing.

    The result remains an ordinary :class:`Program`; existing specialized
    emitters can migrate incrementally without adopting a second execution IR.
    """

    if not isinstance(program, Program):
        raise TypeError("production preparation requires a TensorIR Program")
    if backend not in _BACKENDS:
        raise ValueError("production preparation backend must be cpu or cuda")
    if type(preserve_contraction_order) is not bool:
        raise TypeError("preserve_contraction_order must be a Boolean")

    source_hash = program.logical_hash
    precision_locked = program.provenance.get("precision_execution") is not None
    reassociate = not preserve_contraction_order and not precision_locked
    prepared = optimize(
        program,
        requested_outputs=requested_outputs,
        reassociate_contractions=reassociate,
    )
    preparation = {
        "schema": SCHEMA,
        "backend": backend,
        "source_logical_hash": source_hash,
        "prepared_logical_hash": prepared.logical_hash,
        "strict_degree_reassociation": reassociate,
        "precision_locked": precision_locked,
        "preserve_contraction_order": preserve_contraction_order,
    }
    return Program(
        dict(prepared.outputs),
        provenance={
            **prepared.provenance,
            "production_preparation": preparation,
        },
    )
