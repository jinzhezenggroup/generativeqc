"""Device-free validation for the complete public CPU Hessian benchmark driver."""

from types import SimpleNamespace

import numpy as np
import pytest

from tools import benchmark_cpu_hessian as benchmark


def test_cpu_hessian_receipt_preserves_every_timing_sample():
    values = [1.3, 7.2, 0.8]
    result = benchmark._summary(values)
    assert result["seconds"] == values
    assert result["n"] == 3
    assert result["median_seconds"] == 1.3
    assert result["max_seconds"] == 7.2
    with pytest.raises(ValueError, match="nonnegative"):
        benchmark._summary([1.0, float("nan")])
    with pytest.raises(ValueError, match="nonempty"):
        benchmark._summary([])


def test_cpu_hessian_moved_geometry_does_not_change_source():
    atoms, _, direction = benchmark._fixture("h2")
    original = tuple(atoms)
    moved = benchmark._moved(atoms, direction, 0.02)
    assert tuple(atoms) == original
    assert moved[0][0] == atoms[0][0]
    np.testing.assert_allclose(
        np.asarray([row[1] for row in moved])
        - np.asarray([row[1] for row in atoms]),
        0.02 * direction,
        atol=1e-15,
    )
    with pytest.raises(ValueError, match="displacement"):
        benchmark._moved(atoms, direction, 0.0)


def test_cpu_hessian_raw_matrix_is_not_symmetrized():
    direction = np.asarray([[0.25, -0.5, 0.75]])
    # Intentionally asymmetric to confirm that the driver records RAW symmetry.
    matrix = np.asarray(
        [
            [1.0, 0.4, 0.0],
            [-0.2, 2.0, 0.1],
            [0.0, 0.1, 3.0],
        ]
    )
    vector = matrix @ direction.ravel()
    hvp = SimpleNamespace(value=vector.reshape(1, 3))
    hessian = SimpleNamespace(matrix=matrix)
    result = benchmark._validate_pair(hvp, hessian, direction)
    assert result["hvp_full_max_abs_error"] < 1e-15
    assert result["raw_symmetry_max_abs_error"] == pytest.approx(0.6)
    assert result["posthoc_symmetrization"] is False
    with pytest.raises(ArithmeticError, match="disagree"):
        benchmark._validate_pair(
            SimpleNamespace(value=np.ones_like(vector)), hessian, direction
        )
