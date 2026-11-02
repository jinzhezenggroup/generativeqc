"""Resolve qualified CUDA SCF defaults without changing explicit caller controls."""

from functools import lru_cache
from typing import Any

from generativeqc_compiler.dft.grid import GridSpec

from . import _native
from .basis import BasisSet


@lru_cache(maxsize=1)
def _standard_models() -> frozenset[str]:
    """Complete electronic models; ingredient labels cannot admit custom mixtures."""
    from generativeqc_compiler.method import resolve_method

    return frozenset(
        resolve_method(name, spin="unpolarized").identity
        for name in ("PBE", "PBE0", "B3LYP", "R2SCAN", "M06-2X")
    )


def default_scf_controls(calculator: Any) -> dict[str, float]:
    """Query the selected native policy while preserving explicit caller choices."""
    from ._scf_defaults_native import default_controls

    controls = default_controls(calculator)
    return (
        _compatibility_default_scf_controls(calculator)
        if controls is None
        else controls
    )


def _compatibility_default_scf_controls(calculator: Any) -> dict[str, float]:
    """Keep older libraries, CPU, correlated and unqualified owners strict.

    Default convergence is separate from FP64 arithmetic. The native owner
    still proves its final physical state before forces or warm caches escape.
    Explicit tighter tolerances never inherit the relaxed stationarity ceiling.
    """
    controls = {
        "energy_tolerance": 1e-10,
        "density_tolerance": 1e-8,
        "screening_tolerance": 1e-12,
    }
    query = getattr(
        getattr(calculator, "_library", None),
        "generativeqc_standard_scf_convergence_version_v1",
        None,
    )
    if (
        getattr(calculator, "_device_name", None) != "cuda"
        or query is None
        or query() != 1
    ):
        return controls
    basis = getattr(calculator, "_basis", None)
    if isinstance(basis, BasisSet) and any(
        element.ecp_core_electrons for element in basis.elements
    ):
        return controls
    method = getattr(calculator, "_method", None)
    options = getattr(calculator, "_ks_options", None)
    admitted = method == _native.METHOD_RHF
    if options is not None:
        admitted = (
            options.xc_schedule == "device_fused"
            and options.grid == GridSpec.treutler(level=3)
            and options.method_ir.identity in _standard_models()
        )
    if admitted:
        controls.update(energy_tolerance=1e-9, density_tolerance=1e-5)
    return controls
