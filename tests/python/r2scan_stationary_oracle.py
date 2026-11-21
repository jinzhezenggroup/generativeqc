"""Test-only Libxc energy and explicit chain rule for the smooth point policy."""

from __future__ import annotations

import numpy as np
import pytest
from test_wb97mv_native_grid import _assemble, _fields


def _floor(value: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray]:
    position = np.clip(value / limit - 0.5, 0, 1)
    work = np.maximum(value, limit * (1 + 0.5 * position**3 * (2 - position)))
    slope = position**2 * (3 - 2 * position)
    return work, slope


def _activation(value: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    position = np.clip(value / cutoff - 1, 0, 1)
    return (
        position**3 * (10 + position * (-15 + 6 * position)),
        30 * position**2 * (1 - position) ** 2 / cutoff,
    )


def variational_r2scan_grid(
    fixture: dict, density: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return weighted XC energy and potential without native or compiler AD.

    Libxc supplies E/vxc at the actual work inputs, not at shifted molecular
    geometries. This test oracle explicitly differentiates the work projection.
    """
    libxc = pytest.importorskip("pyscf.dft.libxc")
    if libxc.__version__ != "7.0.0":
        pytest.skip("requires the pinned upstream r2SCAN work formulas")
    physical = _fields(fixture["jets"], density)
    spins = len(density)
    if spins == 1:
        physical = np.repeat(0.5 * physical, 2, axis=0)
    density_fields = physical[:, 0]
    assert np.min(density_fields) >= -1e-15
    assert np.min(physical[:, 4]) >= -1e-15
    density_fields[:] = np.maximum(0, density_fields)
    kinetic_work, kinetic_slope = _floor(physical[:, 4], 1e-20)
    energy = np.zeros(physical.shape[-1])
    potential = np.zeros_like(physical)

    exchange_cutoff = 1e-11
    for spin in range(2):
        active = density_fields[spin] > exchange_cutoff
        if not np.any(active):
            continue
        work = np.zeros((2, 5, np.count_nonzero(active)))
        work[0] = physical[spin][:, active]
        work[0, 4] = kinetic_work[spin, active]
        excitation, partials, *_ = libxc.eval_xc("MGGA_X_R2SCAN", work, spin=1, deriv=1)
        local, gradient, _, kinetic = partials
        activation, slope = _activation(work[0, 0], exchange_cutoff)
        raw_energy = excitation * (work[0, 0] + exchange_cutoff)
        energy[active] += activation * raw_energy
        potential[spin, 0, active] += activation * local[:, 0] + slope * raw_energy
        potential[spin, 1:4][:, active] += (
            2 * activation * gradient[:, 0] * work[0, 1:4]
        )
        potential[spin, 4, active] += (
            activation * kinetic[:, 0] * kinetic_slope[spin, active]
        )

    correlation_cutoff = 1e-15
    total = density_fields.sum(axis=0)
    active = total > correlation_cutoff
    if np.any(active):
        work = physical[:, :, active].copy()
        work[:, 0], density_slope = _floor(work[:, 0], correlation_cutoff)
        work[:, 4] = kinetic_work[:, active]
        excitation, partials, *_ = libxc.eval_xc("MGGA_C_R2SCAN", work, spin=1, deriv=1)
        local, gradient, _, kinetic = partials
        work_total = work[:, 0].sum(axis=0)
        raw_energy = excitation * work_total
        ratio = total[active] / work_total
        activation, slope = _activation(total[active], correlation_cutoff)
        energy[active] += activation * ratio * raw_energy
        potential[:, 0, active] += slope * ratio * raw_energy + activation * (
            ratio * local.T * density_slope + excitation * (1 - ratio * density_slope)
        )
        for spin in range(2):
            potential[spin, 1:4][:, active] += (
                activation
                * ratio
                * (
                    2 * gradient[:, 2 * spin] * work[spin, 1:4]
                    + gradient[:, 1] * work[1 - spin, 1:4]
                )
            )
        potential[:, 4, active] += (
            activation * ratio * kinetic.T * kinetic_slope[:, active]
        )
    if spins == 1:
        potential = np.mean(potential, axis=0, keepdims=True)
    weights = fixture["weights"]
    return float(weights @ energy), _assemble(fixture["jets"], potential * weights)
