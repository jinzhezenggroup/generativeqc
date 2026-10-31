"""Device-free negative controls for cross-functional CUDA acceptance gates."""

import numpy as np
import pytest
from test_cross_functional_force_ao_cuda import (
    _assert_force_endpoint,
    _assert_physical_residual,
)


def test_valid_force_endpoint_acceptance() -> None:
    _assert_force_endpoint(-1.0, np.zeros((3, 3)), 3)


@pytest.mark.parametrize("shape", [(), (3,), (3, 1), (1, 3), (0, 3), (3, 3, 1)])
def test_force_endpoint_rejects_broadcastable_and_wrong_shapes(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(AssertionError, match="shape"):
        _assert_force_endpoint(-1.0, np.zeros(shape), 3)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_force_endpoint_rejects_nonfinite_energy_and_forces(value: float) -> None:
    with pytest.raises(AssertionError, match="energy"):
        _assert_force_endpoint(value, np.zeros((3, 3)), 3)
    forces = np.zeros((3, 3))
    forces[1, 2] = value
    with pytest.raises(AssertionError, match="forces must be finite"):
        _assert_force_endpoint(-1.0, forces, 3)


@pytest.mark.parametrize("residual", [0.0, 1e-12, 1e-9])
def test_physical_residual_accepts_inclusive_boundary(residual: float) -> None:
    _assert_physical_residual(residual)


@pytest.mark.parametrize(
    "residual", [float("nan"), float("inf"), -float("inf"), -1e-12, 1.01e-9]
)
def test_physical_residual_rejects_invalid_values(residual: float) -> None:
    with pytest.raises(AssertionError, match="physical residual"):
        _assert_physical_residual(residual)
