"""Opt-in complete CUDA qualification of the 26-row KTBM worker package.

Use a library and native driver built with exactly the KTBM registrations, and
run through a finite Slurm allocation. The ordinary Libxc point and public KS
oracles remain independent of the parameterized compiler path.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.xc.automatic_semilocal import automatic_functional_code
from generativeqc_compiler.xc.parameter_cuda_package import parameterized_cuda_package
from test_libxc_automatic_cuda_ks import (
    test_public_automatic_cuda_ks_lifecycle_matches_independent_pyscf as _check_ks,
)
from test_libxc_automatic_cuda_xc import (
    _independent_xc,
    native_xc_driver,
)
from test_libxc_automatic_cuda_xc import (
    test_automatic_cuda_aot_point_layout_matches_independent_libxc as _check_points,
)
from test_libxc_automatic_cuda_xc import (
    test_automatic_cuda_invalid_point_is_reported as _check_invalid,
)
from test_parameterized_xc import RECORDS

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ("native_xc_driver",)
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_PARAMETERIZED_XC_CUDA_TEST") != "1",
    reason="requires the KTBM CUDA package and a finite Slurm allocation",
)


@pytest.mark.parametrize("name", tuple(RECORDS))
@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("tile_points", (0, 16))
def test_parameterized_points_match_independent_libxc(
    native_xc_driver: Path, name: str, spins: int, tile_points: int
) -> None:
    """All coefficient rows retain ordinary/batched layouts and domain errors."""
    _check_points(native_xc_driver, name, spins, tile_points)
    _check_invalid(native_xc_driver, name, spins, tile_points)


@pytest.mark.parametrize("name", tuple(RECORDS))
def test_general_worker_retains_equal_spin_fallback(
    native_xc_driver: Path, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicitly disabling the equal-spin optimization still uses the same row."""
    monkeypatch.setenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", "0")
    _check_points(native_xc_driver, name, 1, 16)


@pytest.mark.parametrize("name", tuple(RECORDS))
@pytest.mark.parametrize("spins", (1, 2))
def test_parameterized_fixed_density_fock_and_identity(
    native_xc_driver: Path, name: str, spins: int
) -> None:
    """Cross-shard host linkage authenticates the row and executes full AO/XC/Fock."""
    _, providers = parameterized_cuda_package(tuple(RECORDS))
    completed = subprocess.run(
        [
            str(native_xc_driver),
            "--automatic-xc-json",
            str(automatic_functional_code(name)),
            str(spins),
            "0.08",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    record = json.loads(completed.stdout)
    assert record["expression_identity"] == providers[name].expression_identity
    assert record["source_identity"] == providers[name].source_identity
    energy, potential = _independent_xc(name, record, 0.08)
    assert abs(record["energy"] - energy) < 2e-10
    np.testing.assert_allclose(record["potential"], potential, rtol=2e-9, atol=2e-10)
    assert record["energy"] == record["replay_energy"]
    np.testing.assert_array_equal(record["potential"], record["replay_potential"])
    derivative = (record["plus_energy"] - record["minus_energy"]) / (2 * record["step"])
    assert (
        abs(derivative - np.asarray(record["potential"]) @ record["direction"]) < 2e-8
    )
    tiles = (record["point_count"] + record["tile_points"] - 1) // record["tile_points"]
    assert record["evaluations"] == 4
    assert record["potential_calls"] == 4 * spins * tiles
    assert record["potential_summands"] == 4 * 6 * record["point_count"] * 4 * spins


@pytest.mark.parametrize("name", tuple(RECORDS))
@pytest.mark.parametrize("spins", (1, 2))
def test_parameterized_public_ks_lifecycle(name: str, spins: int) -> None:
    """Cold/warm/displaced SCF includes independent density, Fock and energy gates."""
    _, providers = parameterized_cuda_package(tuple(RECORDS))
    _check_ks(name, spins, "selector", provider=providers[name])
