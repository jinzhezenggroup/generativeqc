"""Independent algebra, scientific oracle and lifetime gates for the prototype."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc.response_solver import GMRESOptions, solve
from generativeqc_compiler.cc.lambda_equations import build_lambda_programs
from generativeqc_compiler.cc.pair_coordinates import RestrictedPairCoordinates

from benchmarks.cc_lambda_multisecant import (
    Coordinates,
    converge,
    lambda_rhs,
    transpose_action,
)
from benchmarks.cc_lambda_multisecant_native import NativeGMRES
from tools.generativeqc_cc.multisecant import (
    SecantHistory,
    fit_multisecant,
    recover_jacobi_input,
)
from tools.generativeqc_cc.oracle import DeterminantOracle, dense_feeds, random_case

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_full_rank_fit_and_transpose_against_dense_pseudoinverse() -> None:
    rng = np.random.default_rng(192)
    s, y = rng.normal(size=(17, 5)), rng.normal(size=(17, 5))
    h0 = rng.uniform(0.2, 1.4, 17)
    inverse = fit_multisecant(h0, s, y, max_bytes=1 << 20)
    # Independent dense expression is permitted in a tiny algebra oracle only.
    expected = np.diag(h0) + (s - h0[:, None] * y) @ np.linalg.pinv(y)
    vector = rng.normal(size=17)
    np.testing.assert_allclose(inverse.forward(vector), expected @ vector, atol=2e-14)
    np.testing.assert_allclose(inverse.apply(vector), expected.T @ vector, atol=2e-14)
    assert np.linalg.norm(expected @ vector - expected.T @ vector) > 0.1
    np.testing.assert_allclose(
        np.column_stack([inverse.forward(col) for col in y.T]), s, atol=2e-14
    )
    assert inverse.rank == 5
    s.fill(0)
    y.fill(0)
    h0.fill(1)
    np.testing.assert_allclose(inverse.apply(vector), expected.T @ vector, atol=2e-14)


def test_svd_filters_dependence_and_handles_disparate_column_scales() -> None:
    rng = np.random.default_rng(44)
    j = rng.normal(size=(8, 8)) + 5 * np.eye(8)
    y = rng.normal(size=(8, 2))
    y = np.column_stack((y, y[:, 0], 2 * y[:, 1], np.zeros(8)))
    s = np.linalg.solve(j, y)
    scales = np.array([1e-10, 1, 2e-9, 1e-3, 1])
    inverse = fit_multisecant(
        np.full(8, 0.2), s * scales, y * scales, max_bytes=1 << 20
    )
    assert inverse.rank == 2
    np.testing.assert_allclose(
        np.column_stack([inverse.forward(col) for col in y.T]), s, atol=2e-14
    )
    y[:, 2] += rng.normal(scale=1e-12, size=8)
    near = fit_multisecant(
        np.full(8, 0.2),
        np.linalg.solve(j, y),
        y,
        relative_cutoff=1e-8,
        max_bytes=1 << 20,
    )
    assert near.rank == 2


def test_history_recovery_coordinates_precision_epoch_and_lifetime() -> None:
    coordinates = Coordinates(2, 3)
    n = coordinates.dimension
    rng = np.random.default_rng(718)
    t, r = rng.normal(size=(2, n))
    d = -rng.uniform(0.4, 2, n)
    damping = 0.2
    image = t + (1 - damping) * r / d
    np.testing.assert_allclose(
        recover_jacobi_input(image, r, d, damping), t, atol=5e-16
    )
    history = SecantHistory(n, 4, max_bytes=1 << 20)
    for step in range(8):
        history.append_jacobi(
            image + step, r + step, d, damping, epoch="reference:fp64", precision="fp64"
        )
    assert history.samples == 5
    assert history.retained_bytes <= history.capacity_bytes
    s, y = history.differences(epoch="reference:fp64", rank=4)
    np.testing.assert_allclose(
        s, np.tile((1 - (1 - damping) / d)[:, None], (1, 4)), atol=4e-15
    )
    np.testing.assert_allclose(y, 1, atol=2e-15)
    image.fill(123)
    r.fill(456)
    s2, y2 = history.differences(epoch="reference:fp64", rank=4)
    np.testing.assert_array_equal(s2, s)
    np.testing.assert_array_equal(y2, y)
    history.append_jacobi(
        image, r, d * 2, damping, epoch="reference:fp64", precision="fp64"
    )
    assert history.samples == 1
    history.append_jacobi(
        image, r, d * 2, damping + 0.1, epoch="reference:fp64", precision="fp64"
    )
    assert history.samples == 1
    # Non-strict arithmetic is rejected even if its storage was upcast to FP64.
    history.append_pair(t, r, epoch="reference:mixed", precision="fp32")
    assert history.samples == 0
    history.append_pair(t, r, epoch="reference:fp64-next", precision="fp64")
    assert history.samples == 1
    history.append_pair(t, r, epoch="different-reference", precision="fp64")
    assert history.samples == 1
    with pytest.raises(ValueError, match="stale"):
        history.differences(epoch="reference:fp64-next", rank=4)
    with pytest.raises(ValueError, match="FP64"):
        history.append_pair(
            t.astype(np.float32), r, epoch="different-reference", precision="fp64"
        )
    # Euclidean weighted coordinates reproduce the full dense Frobenius metric.
    u = rng.normal(size=n)
    dense_t, dense_u = coordinates.unpack(t), coordinates.unpack(u)
    np.testing.assert_allclose(
        t @ u,
        sum(np.sum(a * b) for a, b in zip(dense_t, dense_u, strict=True)),
        atol=2e-14,
    )
    covector = tuple(rng.normal(size=a.shape) for a in dense_t)
    np.testing.assert_allclose(
        coordinates.project_cotangent(*covector) @ t,
        sum(np.sum(a * b) for a, b in zip(covector, dense_t, strict=True)),
        atol=2e-14,
    )
    native_row = np.empty(n)
    native_row[:6] = dense_t[0].ravel()
    # Fill through all dense elements; both members of an orbit write the same
    # value. This tests the adapter without using its own permutation table.
    for flat, value in enumerate(dense_t[1].ravel()):
        native_row[6 + RestrictedPairCoordinates(2, 3).orbit(flat)[1]] = value
    np.testing.assert_allclose(coordinates.from_diis_packed(native_row), t, atol=2e-15)
    assert not np.allclose(native_row * coordinates.weights, t)


def test_explicit_budget_empty_history_and_invalid_inputs() -> None:
    h0, s, y = np.ones(8), np.eye(8)[:, :4], np.eye(8)[:, :4]
    for budget, rank, reason in ((64, 4, "workspace_limit"), (1 << 20, 0, "disabled")):
        p = fit_multisecant(h0, s, y, max_rank=rank, max_bytes=budget)
        assert p.rank == 0 and p.reason == reason
        np.testing.assert_array_equal(p.apply(h0), h0)
        assert p.retained_bytes <= budget
    empty = fit_multisecant(h0, np.empty((8, 0)), np.empty((8, 0)), max_bytes=1024)
    assert empty.rank == 0
    zeros = fit_multisecant(h0, s, y * 0, max_bytes=1 << 20)
    assert zeros.rank == 0 and zeros.reason == "negligible_history"
    refused = SecantHistory(8, 6, max_bytes=64)
    refused.append_pair(h0, h0, epoch="strict", precision="fp64")
    assert not refused.admitted and refused.retained_bytes == 0
    for invalid in (
        np.full_like(y, np.nan),
        np.full_like(y, np.inf),
        y.astype(np.float32),
    ):
        with pytest.raises(ValueError, match="FP64"):
            fit_multisecant(h0, s, invalid, max_bytes=1 << 20)
    with pytest.raises(ValueError, match="denominator"):
        recover_jacobi_input(h0, h0, h0 * 0, 0.0)
    with pytest.raises(ValueError, match="shape"):
        fit_multisecant(h0, s[:-1], y, max_bytes=1 << 20)


def test_nonsymmetric_exact_operator_remains_authoritative() -> None:
    rng = np.random.default_rng(916)
    n = 12
    diagonal = np.linspace(1.0, 3.0, n)
    u, v = rng.normal(size=(2, n))
    j = np.diag(diagonal) + 0.2 * np.outer(u, v)
    # The rank-one inverse correction has row space D^-1 v. Training on this
    # subspace makes the low-rank inverse exact despite nonsymmetry.
    y = (v / diagonal)[:, None]
    s = np.linalg.solve(j, y)
    fitted = fit_multisecant(1 / diagonal, s, y, max_bytes=1 << 20)
    baseline = fit_multisecant(1 / diagonal, s, y, max_rank=0, max_bytes=1 << 20)
    b = rng.normal(size=n)
    operator = SimpleNamespace(dimension=n, apply=lambda x: j.T @ x)
    options = GMRESOptions(atol=1e-12, rtol=0.0, true_residual_every=30)
    first = solve(
        operator, b, options=options, preconditioner=baseline, raise_on_failure=True
    )
    second = solve(
        operator, b, options=options, preconditioner=fitted, raise_on_failure=True
    )
    assert second.operator_actions < first.operator_actions
    expected = np.linalg.solve(j.T, b)
    np.testing.assert_allclose(second.solution, expected, atol=2e-13)
    # A poor history can alter the path but cannot bypass the true residual.
    poor = fit_multisecant(1 / diagonal, s * 0.03, y, max_bytes=1 << 20)
    third = solve(
        operator, b, options=options, preconditioner=poor, raise_on_failure=True
    )
    np.testing.assert_allclose(third.solution, expected, atol=2e-11)


def test_native_bridge_true_residual_and_callback_failure(
    tmp_path: Path,
    native_cxx: NativeCxx,
) -> None:
    root = Path(__file__).resolve().parents[2]
    compiler = replace(native_cxx, cache=shutil.which("sccache") or native_cxx.cache)
    subprocess.run(
        [compiler.cache, "--version"], check=True, capture_output=True, timeout=10
    )
    library = compiler.build_shared(
        [
            root / "src/response/native_gmres.cpp",
            root / "benchmarks/cc_lambda_multisecant_gmres.cpp",
        ],
        tmp_path / "gmres.so",
        compile_args=(
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "src"),
        ),
    )
    gmres = NativeGMRES(library)
    j = np.array([[2.0, 0.4], [-0.2, 1.5]])
    rhs = np.array([0.3, -0.7])
    result = gmres.solve(lambda x: j.T @ x, lambda x: x / np.diag(j), rhs)
    np.testing.assert_allclose(result.solution, np.linalg.solve(j.T, rhs), atol=1e-13)
    assert np.linalg.norm(j.T @ result.solution - rhs) <= 1e-12

    def fail(_: np.ndarray) -> np.ndarray:
        raise ValueError("injected callback error")

    with pytest.raises(RuntimeError, match="callback failed") as failure:
        gmres.solve(fail, lambda x: x, rhs)
    assert isinstance(failure.value.__cause__, ValueError)
    with pytest.raises(RuntimeError, match="callback failed"):
        gmres.solve(lambda x: j.T @ x, fail, rhs)
    with pytest.raises(ValueError, match="RHS"):
        gmres.solve(lambda x: x, lambda x: x, rhs.astype(complex))
    with pytest.raises(RuntimeError, match="callback failed"):
        gmres.solve(lambda x: x.astype(complex), lambda x: x, rhs)


@pytest.mark.parametrize("seed", [148, 157])
def test_generated_cc_history_and_lambda_vs_determinant_oracle(seed: int) -> None:
    f, g, t1, t2 = random_case(2, 2, seed)
    f *= 0.1
    f += np.diag([-1.3, -0.8, 0.5, 1.0])
    g *= 0.3
    coordinates = Coordinates(2, 2)
    d1 = np.diag(f)[:2, None] - np.diag(f)[None, 2:]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    d = coordinates.pack(d1, d2) / coordinates.weights
    programs = build_lambda_programs(2, 2)
    feeds = dense_feeds(f, g, t1 * 0, t2 * 0)
    feeds, history, _ = converge(
        feeds, d, coordinates, programs, epoch="determinant", damping=0.15
    )
    oracle = DeterminantOracle(f, g, 2)
    point = coordinates.pack(feeds["t1"], feeds["t2"])
    energy, one, two = oracle.evaluate_full(*coordinates.unpack(point))
    assert np.linalg.norm(coordinates.pack(one, two)) < 1e-10
    assert np.isfinite(energy)
    # Five-point differentiation of independent determinant exponentials.
    # Tiny dense J is an oracle only, never an input to the preconditioner.
    jacobian, gradient = np.empty((point.size, point.size)), np.empty(point.size)
    h = 1e-3
    for k in range(point.size):
        direction = np.eye(point.size)[k] * h
        values = [
            oracle.evaluate_full(*coordinates.unpack(point + shift * direction))
            for shift in (-2, -1, 1, 2)
        ]
        coefficients = np.array([1, -8, 8, -1]) / (12 * h)
        gradient[k] = sum(
            c * value[0] for c, value in zip(coefficients, values, strict=True)
        )
        jacobian[:, k] = sum(
            c * coordinates.pack(*value[1:])
            for c, value in zip(coefficients, values, strict=True)
        )
    rhs = lambda_rhs(programs, feeds, coordinates)
    np.testing.assert_allclose(rhs, -gradient, atol=2e-12)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=point.size)
    np.testing.assert_allclose(
        transpose_action(programs, feeds, coordinates, direction),
        jacobian.T @ direction,
        atol=2e-11,
    )
    expected = np.linalg.solve(jacobian.T, -gradient)
    for rank in (0, 4, 6):
        s, y = history.differences(epoch="determinant", rank=rank)
        preconditioner = fit_multisecant(-1 / d, s, y, max_rank=rank, max_bytes=1 << 20)
        operator = SimpleNamespace(
            dimension=point.size,
            apply=lambda x: transpose_action(programs, feeds, coordinates, x),
        )
        result = solve(
            operator,
            rhs,
            options=GMRESOptions(atol=1e-12, rtol=0),
            preconditioner=preconditioner,
            raise_on_failure=True,
        )
        np.testing.assert_allclose(result.solution, expected, atol=2e-11)
        assert np.linalg.norm(jacobian.T @ result.solution + gradient) < 1e-10
