"""Compatibility for independent GPU4PySCF comparisons on explicit grids.

This module is benchmark-only. It must not become a native DFT dependency or
change either engine's coordinates, weights, density cutoff or functional.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


def _ordered_blocks(original: Any) -> Any:
    """Keep zero-AO blocks in a bound or unbound GPU4PySCF iterator.

    GPU4PySCF 1.8.1 fills full-grid density arrays in yielded-block order. Its
    default iterator skips empty AO blocks, shifting densities relative to the
    original weights and leaving the tail uninitialized. Strict ordering keeps
    the offsets, but downstream AO scaling rejects zero-dimensional kernels.

    Represent an empty block by one exactly zero AO row mapped to AO zero. This
    contributes zero density and potential while retaining all point offsets,
    including a partial final tile and higher AO derivative components. The
    adapter is local to this numerical-integrator instance and idempotent.
    """
    import cupy as cp

    if getattr(original, "_generativeqc_preserves_grid_order", False):
        return original
    parameters = signature(original)
    if "strict_grid_order" not in parameters.parameters:
        raise RuntimeError("GPU4PySCF block_loop lacks the strict-grid-order contract")

    @wraps(original)
    def blocks(*args: Any, **kwargs: Any) -> Any:
        # Binding also handles callers that pass strict_grid_order positionally.
        bound = parameters.bind(*args, **kwargs)
        bound.arguments["strict_grid_order"] = True
        for ao, indices, weights, coords in original(*bound.args, **bound.kwargs):
            if len(indices) == 0:
                shape = list(ao.shape)
                shape[-2] = 1
                # The 1.8.1 strict empty-block branch leaves the singleton
                # component axis in LDA, although nonempty LDA AO is 2-D.
                if bound.arguments.get("deriv", 0) == 0 and len(shape) == 3:
                    shape = shape[1:]
                yield (
                    cp.zeros(shape, dtype=ao.dtype),
                    cp.zeros(1, dtype=cp.int32),
                    weights,
                    coords,
                )
            else:
                yield ao, indices, weights, coords

    blocks._generativeqc_preserves_grid_order = True  # type: ignore[attr-defined]
    return blocks


def preserve_reference_grid_order(engine: Any) -> None:
    """Adapt one independent SCF integrator without changing its quadrature."""
    engine._numint.block_loop = _ordered_blocks(engine._numint.block_loop)


@contextmanager
def reference_grid_order() -> Iterator[None]:
    """Also cover fresh integrators constructed by analytic grid response.

    GPU4PySCF 1.8.1 get_exc_full_response creates a new NumInt rather than using
    the SCF instance. Scope its iterator compatibility to this reference call,
    restoring the class even when SCF/forces fail. Different concurrent reference
    policies must run in separate processes, as for the VV10 density domain.
    """
    from gpu4pyscf.dft import numint

    original = numint.NumInt.block_loop
    try:
        numint.NumInt.block_loop = _ordered_blocks(original)
        yield
    finally:
        numint.NumInt.block_loop = original
