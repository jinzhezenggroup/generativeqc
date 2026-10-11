"""Opt-in same-source CPU/CUDA inward derivative gates, not SCF promotion."""

from __future__ import annotations

import os
from typing import Any

import pytest
from test_libxc_physical_kinetic_predicate import (
    test_decimal_rank_one_fock_is_independent_inward_energy_derivative as assert_inward,
)

pytest_plugins = ("test_libxc_automatic_point_native",)

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_XC_CUDA_TEST") != "1",
    reason="requires finite Slurm allocation and opt-in point compilation",
)


@pytest.mark.parametrize("scale", (1.0, 2.0))
def test_cpu_cuda_decimal_tie_uses_independent_inward_energy(
    native_point: tuple[Any, Any, Any], scale: float
) -> None:
    """Compile actual point lowering on both backends against the energy oracle."""
    point, library, _ = native_point
    if point.name != "MGGA_C_TPSS":
        pytest.skip("this independent boundary reproducer is TPSS correlation")
    assert_inward(library, scale)
