"""Gate the generated spectral action against displaced eigensystems, not AD."""

from __future__ import annotations

import math
import os
import subprocess
import typing

import numpy as np
import pytest


def _native(
    eigenvalues: np.ndarray,
    frame: np.ndarray,
    threshold: float,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    path = os.environ.get("GENERATIVEQC_CUDA_PSEUDOINVERSE_SECOND_PROBE")
    if not path:
        pytest.skip("requires the native second spectral response probe")
    assert os.environ.get("SLURM_JOB_ID"), "real device tests require finite srun"
    data = np.concatenate(
        (eigenvalues, frame.ravel(order="F"), left.ravel(), right.ravel())
    )
    process = subprocess.run(
        [path],
        input=f"{eigenvalues.size} {threshold:.17g}\n" + " ".join(map(str, data)),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    return np.fromstring(process.stdout, sep=" ").reshape(frame.shape)


def _inverse(metric: np.ndarray, threshold: float, rank: int) -> np.ndarray:
    """Independently diagonalize the perturbed metric and apply the SCF cutoff."""
    values, vectors = np.linalg.eigh(metric)
    retained = values > threshold * values[-1]
    assert np.count_nonzero(retained) == rank
    orthogonal = vectors[:, retained]
    return (orthogonal / values[retained]) @ orthogonal.T


@pytest.mark.parametrize(
    "spectrum,rank",
    [
        ([0.12, 0.12, 0.77, 0.77, 1.51], 3),
        ([-2e-16, 0.18, 0.76, 0.76 + 2e-14, 1.49], 3),
        ([0.51, 0.51, 0.77, 0.77, 1.51], 5),
        ([0.0, 0.07, 0.19, 0.19, 1.64], 1),
    ],
    ids=("two-degenerate-clusters", "near-degenerate", "full-rank", "rank-one"),
)
def test_fixed_rank_second_spectral_action(
    spectrum: list[float], rank: int, record_property: typing.Callable
) -> None:
    rng = np.random.default_rng(99137)
    threshold = 0.25
    values = np.array(spectrum)
    size = len(values)
    frame, _ = np.linalg.qr(rng.normal(size=(size, size)))
    left, right, seed = (
        (direction + direction.T) * 0.09
        for direction in rng.normal(size=(3, size, size))
    )
    metric = (frame * values) @ frame.T
    actual = _native(values, frame, threshold, left, right)
    estimates = [
        sum(
            sign_left
            * sign_right
            * _inverse(
                metric + step * (sign_left * left + sign_right * right), threshold, rank
            )
            for sign_left in (1, -1)
            for sign_right in (1, -1)
        )
        / (4 * step**2)
        for step in (0.002, 0.001)
    ]
    reference = (4 * estimates[1] - estimates[0]) / 3
    record_property("max_second_action_error", float(np.max(abs(actual - reference))))
    np.testing.assert_allclose(actual, reference, atol=3e-8, rtol=2e-7)
    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_allclose(
        actual, _native(values, frame, threshold, right, left), atol=3e-14, rtol=2e-14
    )
    # The Hessian needs the differentiated pullback, not just a forward action.
    reverse = _native(values, frame, threshold, left, seed)
    np.testing.assert_allclose(
        np.sum(seed * actual), np.sum(right * reverse), atol=4e-14
    )
    if rank < size:
        inverse = _inverse(metric, threshold, rank)
        naive = inverse @ (left @ inverse @ right + right @ inverse @ left) @ inverse
        assert np.max(abs(reference - naive)) > 1e-2


@pytest.mark.parametrize("exponent", [-400, 600])
def test_second_action_keeps_seeded_products_in_fp64_range(exponent: int) -> None:
    values = np.array([math.ldexp(1.0, exponent)])
    direction = np.array([[math.ldexp(1.0, 3 * exponent // 2)]])
    np.testing.assert_array_equal(
        _native(values, np.eye(1), 0.25, direction, direction), np.array([[2.0]])
    )
