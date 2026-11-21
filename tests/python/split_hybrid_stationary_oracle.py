"""Independent Libxc XC values and chain rule for the stationary split domain."""

from __future__ import annotations

import numpy as np
import pytest
from r2scan_stationary_oracle import _activation, _floor
from test_wb97mv_native_grid import _assemble, _fields

_EXCHANGE_CUTOFF = 1e-15
_TAU_LIMIT = 1e-20
_RELATIVE_FLOOR = 2.0**-46
_COMPONENTS = {
    "m06-2x": ("HYB_MGGA_X_M06_2X", "MGGA_C_M06_2X", 1e-12),
    "mn15": ("HYB_MGGA_X_MN15", "MGGA_C_MN15", 1e-15),
}


def variational_split_hybrid_grid(
    fixture: dict, density: np.ndarray, method: str
) -> tuple[float, np.ndarray]:
    """Use upstream semilocal components at the smooth physical work inputs."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    if libxc.__version__ != "7.0.0":
        pytest.skip("requires the pinned upstream split-hybrid work formulas")
    exchange, correlation, cutoff = _COMPONENTS[method]
    physical = _fields(fixture["jets"], density)
    if len(density) == 1:
        physical = np.repeat(0.5 * physical, 2, axis=0)
    assert np.min(physical[:, (0, 4)]) >= -1e-15
    physical[:, 0] = np.maximum(0, physical[:, 0])
    kinetic_work, kinetic_slope = _floor(physical[:, 4], _TAU_LIMIT)
    energy = np.zeros(physical.shape[-1])
    potential = np.zeros_like(physical)

    for spin in range(2):
        active = physical[spin, 0] > _EXCHANGE_CUTOFF
        if not np.any(active):
            continue
        work = np.zeros((2, 5, np.count_nonzero(active)))
        work[0] = physical[spin][:, active]
        work[0, 4] = kinetic_work[spin, active]
        work[1, 0] = _EXCHANGE_CUTOFF
        work[1, 4] = _TAU_LIMIT
        exc, (vrho, vsigma, _, vtau), *_ = libxc.eval_xc(
            exchange, work, spin=1, deriv=1
        )
        weight, slope = _activation(work[0, 0], _EXCHANGE_CUTOFF)
        raw_energy = exc * work[:, 0].sum(axis=0)
        energy[active] += weight * raw_energy
        potential[spin, 0, active] += weight * vrho[:, 0] + slope * raw_energy
        potential[spin, 1:4][:, active] += 2 * weight * vsigma[:, 0] * work[0, 1:4]
        potential[spin, 4, active] += weight * vtau[:, 0] * kinetic_slope[spin, active]

    total = physical[:, 0].sum(axis=0)
    active = total > cutoff
    if np.any(active):
        work = physical[:, :, active].copy()
        limit = 2 * cutoff + _RELATIVE_FLOOR * total[active]
        rho = work[:, 0].copy()
        work[:, 0], rho_slope = _floor(rho, limit)
        work[:, 4] = kinetic_work[:, active]
        exc, (vrho, vsigma, _, vtau), *_ = libxc.eval_xc(
            correlation, work, spin=1, deriv=1
        )
        weight, slope = _activation(total[active], cutoff)
        ratio = total[active] / work[:, 0].sum(axis=0)
        physical_energy = exc * total[active]
        energy[active] += weight * physical_energy
        # The minority floor depends on the total density. Euler homogeneity
        # of the smooth floor gives the partial with respect to its limit.
        excess = vrho.T - exc
        floor_slope = (work[:, 0] - rho * rho_slope) / limit
        pullback = rho_slope * excess + _RELATIVE_FLOOR * (floor_slope * excess).sum(
            axis=0
        )
        potential[:, 0, active] += slope * physical_energy + weight * (
            exc + ratio * pullback
        )
        for spin in range(2):
            potential[spin, 1:4][:, active] += (
                weight
                * ratio
                * (
                    2 * vsigma[:, 2 * spin] * work[spin, 1:4]
                    + vsigma[:, 1] * work[1 - spin, 1:4]
                )
            )
        potential[:, 4, active] += weight * ratio * vtau.T * kinetic_slope[:, active]

    if len(density) == 1:
        potential = np.mean(potential, axis=0, keepdims=True)
    weights = fixture["weights"]
    return float(weights @ energy), _assemble(fixture["jets"], potential * weights)
