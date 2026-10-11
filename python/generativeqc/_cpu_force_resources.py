"""Numeric capacity contract for the bounded CPU stationary force consumer.

The native SCF owner is reserved by resources_ks. This inventory covers the
additional AO owner, snapshot/export copies, derivative provider and generated
consumer staging. Python/compiler objects, loaded code, BLAS/runtime internals
and allocator overhead are excluded; this is not a process RSS bound.
"""

import typing

from generativeqc_compiler.common.resources import checked_bytes
from generativeqc_compiler.integral.ecp_policy import (
    REFINED_POLAR_POINTS,
    REFINED_RADIAL_POINTS,
)
from generativeqc_compiler.integral.first_derivative_schedule import (
    DISPATCH_ROWS,
    DISPATCH_WIDTH,
)
from generativeqc_compiler.integral.weighted_eri_inputs import PRIMITIVE_RANGE_RECORD

from .basis import BasisSet

CPU_FORCE_HOST_CAP = 256 << 20
CPU_MIXED_DF_INTEGRAL_TERMS = 512
CPU_DF_FORCE_PRIMITIVE_CAP = 32_000_000
CPU_DIRECT_SEMILOCAL_FORCE_METHODS = frozenset(
    ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks")
)


def qualified_all_electron_basis(basis: typing.Any) -> bool:
    """Bound direct CPU semilocal promotion to all-electron s/p/d bases."""
    if isinstance(basis, BasisSet):
        if any(element.ecp_core_electrons for element in basis.elements):
            return False
        shells = (shell for element in basis.elements for shell in element.shells)
    else:
        try:
            shells = iter(basis)
        except TypeError:
            return False
    return all(shell.angular_momentum <= 2 for shell in shells)


def qualified_direct_semilocal_context(calculator: typing.Any) -> bool:
    """One admission predicate for the optional all-electron CPU force owner."""
    from . import _native

    return (
        calculator._device_name == "cpu"
        and calculator._method_name in CPU_DIRECT_SEMILOCAL_FORCE_METHODS
        and calculator._density_fitting_mode == _native.DENSITY_FITTING_NONE
        and calculator._automatic_libxc_name is None
        and calculator._dispersion_method_ir is None
        and calculator._ks_options is not None
        and calculator._ks_options.coefficients == (1.0, 1.0, 0.0)
        and calculator._ks_options.execution_plan.nonlocal_correlation is None
        and qualified_all_electron_basis(calculator._basis)
    )


def qualified_basis(basis: typing.Any) -> bool:
    """CPU promotion is specific to s/p/d ECP records, including their fragments."""
    return (
        isinstance(basis, BasisSet)
        and any(element.ecp_core_electrons for element in basis.elements)
        and all(
            shell.angular_momentum <= 2
            for element in basis.elements
            for shell in element.shells
        )
    )


def cpu_force_inventory(
    basis: typing.Any,
    *,
    grid_points: int,
    ecp_terms: int,
    nonlocal_correlation: bool = False,
    tile_points: int = 256,
    primitive_tile: int = 128,
    integral_terms: int = 32,
    range_exchange_sources: int = 0,
    native_fitted_integrals: bool = False,
) -> dict[str, int]:
    """Conservative simultaneous numeric capacities; no scientific execution.

    Snapshot factor eight covers native source/state copies, the wire export,
    immutable copies, decoded grid and identity validation. XC factor 128
    covers AO jets (at most 10), immutable/BLAS copies, feature coefficients,
    generated pullback input/output and the previous live tile. Tensor and
    grid adapters each independently enforce an 8 MiB arena cap. Sum phases
    rather than relying on Python's timing of releasing previous locals.
    """
    n, a, p = basis.nao, basis.natom, basis.nprimitive
    if type(range_exchange_sources) is not int or not 0 <= range_exchange_sources <= 2:
        raise ValueError("CPU force range-exchange source count must be 0, 1, or 2")
    maximum_angular = 3 if native_fitted_integrals else 2
    if any(shell.angular_momentum > maximum_angular for shell in basis.shells):
        raise NotImplementedError(
            f"CPU stationary force resources cover angular momentum <= {maximum_angular}"
        )
    maximum_aos = 64 if native_fitted_integrals and not ecp_terms else 16
    if n > maximum_aos or a > 8 or p > 128 or ecp_terms > 128:
        raise ValueError("CPU public force dense-export domain exceeded")
    if grid_points > 1_000_000:
        raise ValueError("CPU public force grid point work budget exceeded")
    packed = 3 * a + 2 * p + 16 * n
    snapshot_values = 256 + 8 * a + 20 * n * n + 8 * n + packed
    snapshot_values += 6 * grid_points + 5 * ecp_terms
    # CPU ECP provider consumes the shared generated refined-grid policy.
    # Both value/derivative grids coexist; one radial shell owns AO samples.
    sphere = 2 * REFINED_POLAR_POINTS**2
    ecp = (
        32 * n * n * (1 + 3 * a)
        + 32 * n * (sphere + 16)
        + 384 * (sphere + REFINED_RADIAL_POINTS + n + p + ecp_terms)
        + 4096
        if ecp_terms
        else 0
    )
    nonlocal_force = (
        8 * (40 * grid_points + n * n + 3 * a) if nonlocal_correlation else 0
    )

    # One operator-frozen generic owner, independent of component count.
    # 256 KiB covers six Dual3 Hermite arrays, the order-12 simplex, moment
    # seeds, geometry and reduction stack. Tuple/orbit staging is below.
    range_provider_bytes = range_exchange_sources * ((256 << 10) + 8 * (12 + 3 * a))
    # Keep Direct s/p packaged-AOT plans alongside their generic fallback.
    # Mixed fitted consumers always use the component-independent owner.
    if not native_fitted_integrals and all(
        shell.angular_momentum <= 1 for shell in basis.shells
    ):
        angular = {shell.angular_momentum for shell in basis.shells}
        component_labels = sum((value + 1) * (value + 2) // 2 for value in angular)
        range_provider_bytes += (
            range_exchange_sources
            * component_labels**4
            * (
                primitive_tile * PRIMITIVE_RANGE_RECORD.size
                + 3 * 13 * 8
                + PRIMITIVE_RANGE_RECORD.size
            )
        )
    blocks = {
        "snapshot_and_ao": 8 * 8 * snapshot_values + 1024 * (1 + a + n + p),
        "xc_tiles": 8 * (128 * tile_points * n + 256 * tile_points + 64 * n * n),
        "integral_staging": 8 * (40 * primitive_tile + 128 * integral_terms + 256 * a),
        "range_exchange_providers": range_provider_bytes,
        # Include construction copies and the retained immutable metadata; the
        # conservative full s/p/d table also covers smaller component domains.
        "component_dispatch": 0
        if native_fitted_integrals
        else 3 * 8 * (DISPATCH_ROWS * DISPATCH_WIDTH + 3 * n + 46),
        "tensor_and_grid_arenas": 32 << 20,
        "ecp_provider": ecp,
        "ecp_export_and_contraction": 8 * (24 * a * n * n + 24 * a * integral_terms),
        "nonlocal_force": nonlocal_force,
    }
    return {
        key: checked_bytes(value, "CPU force " + key) for key, value in blocks.items()
    }
