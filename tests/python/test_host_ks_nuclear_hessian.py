"""Native stationary assembly, not installed/heavy-basis/TS qualification."""

from __future__ import annotations

import ctypes as ct
import subprocess
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from test_host_ks_nuclear_response import (
    CENTERS,
    METHOD,
    MOTION,
    POINTER,
    _frame,
    _owner,
    _pointer,
)
from test_host_ks_nuclear_response import native as native_fixture

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def native(tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any) -> Any:
    """Reuse the genuine native SCF/verified-state fixture, with no oracle seed."""
    builder = cast("Any", native_fixture)
    loaded = builder.__wrapped__(tmp_path_factory, required_native_cxx, policy_library)
    loaded.host_ks_hessian_bind.argtypes = [ct.c_void_p, ct.c_int, ct.c_uint]
    loaded.host_ks_hessian_apply.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        POINTER,
        ct.c_int,
    ]
    loaded.host_ks_hessian_resources.argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
    loaded.host_ks_hessian_oracle_grid.argtypes = [
        POINTER,
        ct.c_uint,
        ct.c_uint,
        POINTER,
        POINTER,
        ct.c_size_t,
    ]
    loaded.host_ks_hessian_batch_bind.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.c_size_t,
        ct.c_size_t,
    ]
    loaded.host_ks_hessian_batch_apply.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_int,
    ]
    return loaded


def _apply(
    native: Any,
    handle: Any,
    right: np.ndarray,
    iterations: int = 100,
    workspace: int = 16 << 20,
    failure: int = 0,
) -> tuple:
    output, diagnostic = np.full(2 + 2 * right.size, -781.25), np.full(5, -781.25)
    status = native.host_ks_hessian_apply(
        handle,
        _pointer(right),
        right.size,
        _pointer(output),
        output.size,
        iterations,
        workspace,
        _pointer(diagnostic),
        failure,
    )
    return status, output, diagnostic


def test_native_stationary_assembly_smoke(native: Any) -> None:
    with _owner(native, 1, 0) as (handle, _):
        assert native.host_ks_hessian_bind(handle, 0, 0) == 0, (
            native.host_ks_nuclear_error().decode()
        )
        result = _apply(native, handle, MOTION[:2].copy())
        assert result[0] == 0, native.host_ks_nuclear_error().decode()
        assert np.isfinite(result[1]).all()
        assert result[2][0] < 2e-11 and result[2][4] < 2e-9


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("bj", [0, 1])
def test_independently_batched_complete_raw_columns_and_fallback(
    native: Any, spins: int, bj: int
) -> None:
    with _owner(native, spins, 2) as (handle, _):
        dimension = 3 * (spins + 1)
        fields = 2 + 2 * dimension
        assert native.host_ks_hessian_bind(handle, bj, 0) == 0
        expected = []
        for direction in np.eye(dimension):
            status, values, _ = _apply(native, handle, direction)
            assert status == 0, native.host_ks_nuclear_error().decode()
            expected.append(values)
        expected = np.stack(expected)
        assert native.host_ks_hessian_batch_bind(handle, bj, 0, 4) == 1
        assert native.host_ks_hessian_batch_bind(handle, bj, 1 << 20, 4) == 4
        actual = np.full_like(expected, -781.25)
        for first in range(0, dimension, 4):
            count = min(4, dimension - first)
            residuals = np.empty(count)
            status = native.host_ks_hessian_batch_apply(
                handle,
                first,
                count,
                _pointer(actual[first : first + count]),
                200,
                _pointer(residuals),
                0,
            )
            assert status == 0, native.host_ks_nuclear_error().decode()
            assert np.all(residuals <= 1e-12)
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-10)
        raw = actual[:, 2 + dimension :].T
        np.testing.assert_allclose(raw, raw.T, atol=2e-10, rtol=2e-10)
        failed = np.full((4, fields), -781.25)
        residuals = np.empty(4)
        assert (
            native.host_ks_hessian_batch_apply(
                handle, 0, 4, _pointer(failed), 1, _pointer(residuals), 0
            )
            == 1
        )
        np.testing.assert_array_equal(failed, -781.25)
        assert (
            native.host_ks_hessian_batch_apply(
                handle, 0, 4, _pointer(failed), 200, _pointer(residuals), 2
            )
            == -1
        )
        np.testing.assert_array_equal(failed, -781.25)
        assert (
            native.host_ks_hessian_batch_apply(
                handle, 0, 4, _pointer(failed), 200, _pointer(residuals), 0
            )
            == 0
        ), native.host_ks_nuclear_error().decode()
        np.testing.assert_allclose(failed, expected[:4], atol=3e-10, rtol=3e-10)


def _energy(
    native: Any,
    spins: int,
    residual: int,
    centers: np.ndarray,
    points: int,
    density: np.ndarray,
    bj: bool,
) -> float:
    """Independent reconverged libcint/Libxc energy on the exact primal grid.

    The native helper generates only the undisplaced grid definition: no native
    SCF, gradient, response or Hessian feeds this independent energy evaluation.
    Nuclear moves and finite differences are confined to this validation oracle.
    """
    from generativeqc_compiler.method._generated_parameters import d3_parameters

    gto, dft = pytest.importorskip("pyscf.gto"), pytest.importorskip("pyscf.dft")
    labels = (["Na0"] if residual else ["H0"]) + [
        f"H{index}" for index in range(1, len(centers))
    ]
    basis = {label: [[0, [1.2, 1]], [0, [0.35, 1]]] for label in labels}
    potentials = {}
    if residual:
        channels = []
        for channel in range(-1, 4 if residual == 2 else 3):
            powers = [[] for _ in range(5)]
            for power in range(5) if residual == 2 else (2,):
                powers[power] = [[0.67, -0.012], [1.23, 0.0081]]
            channels.append([channel, powers])
        potentials = {"Na0": [10, channels]}
    molecule = gto.M(
        atom=list(zip(labels, centers, strict=True)),
        basis=basis,
        ecp=potentials,
        unit="Bohr",
        spin=spins - 1,
        verbose=0,
    )
    coordinates, weights = np.empty((points, 3)), np.empty(points)
    assert (
        native.host_ks_hessian_oracle_grid(
            _pointer(np.ascontiguousarray(centers)),
            spins,
            residual,
            _pointer(coordinates),
            _pointer(weights),
            points,
        )
        == 0
    )
    reference = (dft.RKS if spins == 1 else dft.UKS)(molecule)
    reference.xc = METHOD
    reference.grids.coords, reference.grids.weights = coordinates, weights
    reference.conv_tol = 1e-14
    reference.conv_tol_grad = 2e-12
    reference.max_cycle = 200
    reference.kernel(dm0=density[0] if spins == 1 else density)
    assert reference.converged
    energy = float(reference.e_tot)
    if bj:
        interface = pytest.importorskip("dftd3.interface")
        fit = d3_parameters("B3LYP-D3(BJ)")
        parameter = interface.RationalDampingParam(
            **{key: fit[key] for key in ("s6", "s8", "a1", "a2", "s9")}
        )
        model = interface.DispersionModel(
            np.array([11 if residual else 1] + [1] * (len(centers) - 1)), centers
        )
        model.set_realspace_cutoff(disp2=1e6, disp3=1e6, cn=1e6)
        energy += float(model.get_dispersion(parameter, grad=False)["energy"])
    return energy


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("residual", [0, 1, 2])
def test_stationary_columns_against_independent_reconverged_energies(
    native: Any,
    spins: int,
    residual: int,
    tmp_path: Path,
) -> None:
    """Freeze energy2e-9, gradient3e-7/2e-6, HVP3e-6/2e-5 gates before execution."""
    centers, right = CENTERS[: spins + 1].copy(), MOTION[: spins + 1].copy()
    bj = residual == 1
    started = time.perf_counter()
    with _owner(native, spins, residual) as (handle, sizes):
        density, _, _, _ = _frame(native, handle, sizes)
        assert native.host_ks_hessian_bind(handle, bj, 0) == 0, (
            native.host_ks_nuclear_error().decode()
        )
        prepared = time.perf_counter()
        status, actual, diagnostics = _apply(native, handle, right)
        finished = time.perf_counter()
        assert status == 0, native.host_ks_nuclear_error().decode()
        baseline = _energy(native, spins, residual, centers, sizes[3], density, bj)
        np.testing.assert_allclose(actual[0], baseline, atol=2e-9, rtol=0)
        np.testing.assert_allclose(
            actual[1], actual[2 : 2 + right.size] @ right.ravel(), atol=2e-13, rtol=0
        )
        steps = np.array([3e-3, 1e-3, 3e-4])
        corners = np.empty((len(steps), centers.size, 4))
        for index, step in enumerate(steps):
            for coordinate, unit in enumerate(np.eye(centers.size)):
                left = unit.reshape(centers.shape)
                for corner, (left_sign, right_sign) in enumerate(
                    ((1, 1), (1, -1), (-1, 1), (-1, -1))
                ):
                    displaced = centers + step * (left_sign * left + right_sign * right)
                    corners[index, coordinate, corner] = _energy(
                        native, spins, residual, displaced, sizes[3], density, bj
                    )
        gradient = (
            corners[:, :, 0] + corners[:, :, 1] - corners[:, :, 2] - corners[:, :, 3]
        ) / (4 * steps[:, None])
        hvp = (
            corners[:, :, 0] - corners[:, :, 1] - corners[:, :, 2] + corners[:, :, 3]
        ) / (4 * steps[:, None] ** 2)
        np.testing.assert_allclose(
            actual[2 : 2 + centers.size], gradient[-1], atol=3e-7, rtol=2e-6
        )
        np.testing.assert_allclose(
            actual[2 + centers.size :], hvp[-1], atol=3e-6, rtol=2e-5
        )
        errors = np.max(
            np.abs(np.concatenate((gradient, hvp), axis=1) - actual[2:]), axis=1
        )
        assert errors[-1] < max(errors[0] / 8, 5e-8)
        work = np.zeros(12, dtype=np.uintp)
        native.host_ks_hessian_resources(
            handle, work.ctypes.data_as(ct.POINTER(ct.c_size_t))
        )
        assert work[2] + work[3] <= work[1]
        np.savez(
            tmp_path / "stationary-reconverged-energy.npz",
            spins=spins,
            residual=residual,
            bj=bj,
            centers=centers,
            right=right,
            native=actual,
            diagnostics=diagnostics,
            steps=steps,
            corners=corners,
            errors=errors,
            baseline=baseline,
            work=work,
            preparation_seconds=prepared - started,
            prepared_column_seconds=finished - prepared,
            fixture_scf_through_column_seconds=finished - started,
        )


def test_raw_stationary_symmetry_translation_bilinearity_and_d3_once(
    native: Any, tmp_path: Path
) -> None:
    """Raw matrix2e-8 gates: no symmetrization or translational projection repair."""
    with _owner(native, 2, 1) as (handle, _):
        assert native.host_ks_hessian_bind(handle, 0, 0) == 0
        right = MOTION.copy()
        plain = _apply(native, handle, right)
        assert plain[0] == 0, native.host_ks_nuclear_error().decode()
        dimension = right.size
        matrix = []
        for unit in np.eye(dimension):
            column = _apply(native, handle, unit.reshape(right.shape))
            assert column[0] == 0, native.host_ks_nuclear_error().decode()
            matrix.append(column[1][2 + dimension :])
        matrix = np.column_stack(matrix)
        np.testing.assert_allclose(matrix, matrix.T, atol=2e-8, rtol=0)
        np.testing.assert_allclose(
            matrix @ np.tile(np.eye(3), (len(right), 1)), 0, atol=2e-8, rtol=0
        )
        np.testing.assert_allclose(
            matrix @ right.ravel(), plain[1][2 + dimension :], atol=2e-8, rtol=0
        )
        second = np.random.default_rng(2844).normal(size=right.shape) * 0.1
        other = _apply(native, handle, second)
        assert other[0] == 0
        np.testing.assert_allclose(
            second.ravel() @ plain[1][2 + dimension :],
            right.ravel() @ other[1][2 + dimension :],
            atol=2e-8,
            rtol=0,
        )
        assert native.host_ks_hessian_bind(handle, 1, 0) == 0
        with_d3 = _apply(native, handle, right)
        assert with_d3[0] == 0
        from generativeqc.d3_hessian import NativeD3SecondOrder
        from generativeqc_compiler.method._generated_parameters import d3_parameters
        from generativeqc_compiler.method.dispersion import D3Spec

        independent_schedule = NativeD3SecondOrder(
            D3Spec(**d3_parameters("B3LYP-D3(BJ)")), [11, 1, 1], CENTERS, cache=tmp_path
        )
        np.testing.assert_allclose(
            with_d3[1][2 + dimension :] - plain[1][2 + dimension :],
            independent_schedule.hvp(right).value.ravel(),
            atol=2e-11,
            rtol=0,
        )
        np.savez(
            tmp_path / "stationary-raw-invariants.npz",
            matrix=matrix,
            right=right,
            plain=plain[1],
            with_d3=with_d3[1],
            second=second,
            second_column=other[1],
        )


def test_native_stationary_executable_without_python(native: Any) -> None:
    result = subprocess.run(
        [str(native._hessian_executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env={
            "PATH": "/no-python",
            "PYTHONPATH": "/no-python",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    assert "native stationary RKS/UKS columns" in result.stdout


@pytest.mark.parametrize("spins", [1, 2])
def test_failed_reservations_response_and_input_publication_recover(
    native: Any, spins: int
) -> None:
    with _owner(native, spins, 1) as (handle, _):
        assert native.host_ks_hessian_bind(handle, 1, 0) == 0
        right = MOTION[: spins + 1].copy()
        baseline = _apply(native, handle, right)
        assert baseline[0] == 0, native.host_ks_nuclear_error().decode()
        for refusal in range(1, 8):
            assert native.host_ks_hessian_bind(handle, 1, refusal) != 0
        for iterations, workspace in ((0, 16 << 20), (100, 1)):
            failure = _apply(
                native, handle, right, iterations=iterations, workspace=workspace
            )
            assert failure[0] != 0
            np.testing.assert_array_equal(failure[1], -781.25)
        for invalid in (right.ravel()[:-1], np.full_like(right, np.nan)):
            failed = _apply(native, handle, invalid)
            assert failed[0] != 0
            np.testing.assert_array_equal(failed[1], -781.25)
        output = np.full_like(baseline[1], -781.25)
        diagnostic = np.full(5, -781.25)
        assert (
            native.host_ks_hessian_apply(
                handle,
                None,
                right.size,
                _pointer(output),
                output.size,
                100,
                16 << 20,
                _pointer(diagnostic),
                0,
            )
            != 0
        )
        np.testing.assert_array_equal(output, -781.25)
        for failure_mode in (1, 2):
            failed = _apply(native, handle, right.copy(), failure=failure_mode)
            assert failed[0] != 0
            np.testing.assert_array_equal(failed[1], -781.25)
        for borrowed in (0, 3, 4, 5, 6):
            pointer = native.host_ks_nuclear_borrowed(handle, borrowed)
            saved = pointer[0]
            pointer[0] += 0.125
            try:
                failed = _apply(native, handle, right)
                assert failed[0] != 0, (
                    f"changed stationary owner {borrowed} was admitted"
                )
                np.testing.assert_array_equal(failed[1], -781.25)
            finally:
                pointer[0] = saved
        storage = np.empty_like(baseline[1])
        storage[: right.size] = right.ravel()
        assert (
            native.host_ks_hessian_apply(
                handle,
                _pointer(storage),
                right.size,
                _pointer(storage),
                storage.size,
                100,
                16 << 20,
                _pointer(diagnostic),
                0,
            )
            == 0
        ), native.host_ks_nuclear_error().decode()
        np.testing.assert_allclose(storage, baseline[1], atol=2e-12, rtol=0)
        np.testing.assert_allclose(
            _apply(native, handle, right)[1], baseline[1], atol=2e-12, rtol=0
        )
