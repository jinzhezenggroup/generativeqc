"""Independent displaced-SCF controls for composed native nuclear D/W response."""

from __future__ import annotations

import ctypes as ct
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
CENTERS = np.array([[0, 0, 0], [0.15, -0.08, 1.6], [1.8, 0.25, 0.3]])
MOTION = np.array([[0.12, -0.09, 0.07], [-0.03, 0.06, -0.1], [0.04, -0.02, 0.08]])
METHOD = ".2*HF+.08*LDA_X+.72*GGA_X_B88+.19*LDA_C_VWN_RPA+.81*GGA_C_LYP"


def _pointer(values: np.ndarray) -> Any:
    return values.ctypes.data_as(POINTER)


@pytest.fixture(scope="module")
def native(tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any) -> Any:
    directory = tmp_path_factory.mktemp("host-nuclear-response")
    core = Path(policy_library._name).resolve()
    cmake = shutil.which("cmake")
    assert cmake is not None, "native nuclear response requires CMake"
    (directory / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(native_nuclear_response LANGUAGES CXX)\n"
        f'set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        f'include("{ROOT}/cmake/GenerativeQCGenerated.cmake")\n'
        f'include("{ROOT}/cmake/GenerativeQCHostHessianAot.cmake")\n'
        "add_library(core SHARED IMPORTED GLOBAL)\n"
        f'set_target_properties(core PROPERTIES IMPORTED_LOCATION "{core}" '
        f'INTERFACE_INCLUDE_DIRECTORIES "{ROOT}/include")\n'
        'generativeqc_add_host_hessian_aot(leaves "'
        f'{ROOT}/manifests/host_hessian_aot_b3lyp.json")\n'
        "generativeqc_add_host_ks_response(response leaves core)\n"
        f'add_library(probe SHARED "{ROOT}/tests/native/host_ks_nuclear_response_probe.cpp")\n'
        f'target_include_directories(probe PRIVATE "{ROOT}/include")\n'
        "target_link_libraries(probe PRIVATE response)\n"
        f'add_executable(standalone "{ROOT}/tests/native/host_ks_nuclear_response_probe.cpp")\n'
        "target_compile_definitions(standalone PRIVATE GENERATIVEQC_NUCLEAR_RESPONSE_STANDALONE=1)\n"
        "target_link_libraries(standalone PRIVATE response)\n"
        f'add_executable(hessian_standalone "{ROOT}/tests/native/host_ks_nuclear_response_probe.cpp")\n'
        "target_compile_definitions(hessian_standalone PRIVATE GENERATIVEQC_NUCLEAR_HESSIAN_STANDALONE=1)\n"
        "target_link_libraries(hessian_standalone PRIVATE response)\n"
    )
    build = directory / "build"
    for command, timeout in (
        (
            [
                cmake,
                "-S",
                str(directory),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                "-DCMAKE_CXX_FLAGS=-Wall -Wextra -Werror",
                "-DCMAKE_CXX_COMPILER=" + required_native_cxx.compiler,
                "-DCMAKE_CXX_COMPILER_LAUNCHER=" + required_native_cxx.cache,
            ],
            180,
        ),
        ([cmake, "--build", str(build), "--parallel", "2", "--verbose"], 1200),
    ):
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        )
        assert result.returncode == 0, result.stdout + result.stderr
    loaded = cast("Any", ct.CDLL(str(build / "libprobe.so")))
    loaded.host_ks_nuclear_error.restype = ct.c_char_p
    loaded.host_ks_nuclear_point.argtypes = [POINTER, POINTER, POINTER]
    loaded.host_ks_nuclear_create.argtypes = [
        POINTER,
        ct.c_uint,
        ct.c_uint,
        ct.c_size_t,
    ]
    loaded.host_ks_nuclear_create.restype = ct.c_void_p
    loaded.host_ks_nuclear_destroy.argtypes = [ct.c_void_p]
    loaded.host_ks_nuclear_sizes.argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
    loaded.host_ks_nuclear_frame.argtypes = [ct.c_void_p] + [POINTER] * 4
    loaded.host_ks_nuclear_source.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        POINTER,
    ]
    loaded.host_ks_nuclear_solve.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        POINTER,
        ct.c_size_t,
        ct.c_size_t,
        POINTER,
    ]
    loaded.host_ks_nuclear_batch_plan.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.POINTER(ct.c_size_t),
    ]
    loaded.host_ks_nuclear_batch_solve.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.POINTER(ct.c_size_t),
    ]
    loaded.host_ks_nuclear_bind.argtypes = [ct.c_void_p, ct.c_uint]
    loaded.host_ks_nuclear_failure.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_int]
    loaded.host_ks_nuclear_validations.argtypes = [ct.c_void_p]
    loaded.host_ks_nuclear_validations.restype = ct.c_size_t
    loaded.host_ks_nuclear_borrowed.argtypes = [ct.c_void_p, ct.c_uint]
    loaded.host_ks_nuclear_borrowed.restype = POINTER
    loaded.host_ks_nuclear_panel.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_uint,
    ]
    loaded.host_ks_nuclear_cache.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.POINTER(ct.c_size_t),
    ]
    loaded.host_ks_nuclear_cache_work.argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
    loaded._test_executable = build / "standalone"
    loaded._hessian_executable = build / "hessian_standalone"
    return loaded


@pytest.mark.parametrize("spins", [1, 2])
def test_coordinate_panel_composed_admission_and_source_parity(
    native: Any, spins: int
) -> None:
    """The optional bank must fit inside response+GMRES, not just its own cap."""
    with _owner(native, spins, 0) as (handle, sizes):
        motion = np.ascontiguousarray(MOTION[: spins + 1])
        expected = _source(native, handle, sizes, motion)
        assert expected[0] == 0
        for panel_bytes, response_bytes in (
            (0, 0),
            (1, 0),
            (64 << 20, sizes[4] + sizes[5]),
        ):
            assert (
                native.host_ks_nuclear_panel(handle, panel_bytes, response_bytes, 0)
                == 0
            )
            unchanged = (ct.c_size_t * 7)()
            native.host_ks_nuclear_sizes(handle, unchanged)
            assert tuple(unchanged) == sizes
            actual = _source(native, handle, sizes, motion)
            assert actual[0] == 0, native.host_ks_nuclear_error().decode()
            for result, reference in zip(actual[1:], expected[1:], strict=True):
                np.testing.assert_array_equal(result, reference)
        assert native.host_ks_nuclear_panel(handle, 64 << 20, 0, 0) == 1
        admitted = (ct.c_size_t * 7)()
        native.host_ks_nuclear_sizes(handle, admitted)
        assert admitted[4] > sizes[4]
        assert _source(native, handle, sizes, motion)[0] != 0
        assert native.host_ks_nuclear_panel(handle, 0, 0, 1) == 1
        actual = _source(native, handle, sizes, motion)
        assert actual[0] == 0, native.host_ks_nuclear_error().decode()
        for result, reference in zip(actual[1:], expected[1:], strict=True):
            np.testing.assert_allclose(result, reference, atol=3e-13, rtol=2e-11)


@pytest.mark.parametrize("spins", [1, 2])
def test_exact_response_caches_composed_budget_and_response_parity(
    native: Any, spins: int
) -> None:
    """XC/AO reuse fits response+GMRES and preserves actual solved D'/W' bytes."""
    with _owner(native, spins, 0) as (handle, sizes):
        motion = np.ascontiguousarray(MOTION[: spins + 1])
        expected = _solve(native, handle, sizes, motion)
        assert expected[0] == 0, native.host_ks_nuclear_error().decode()
        bytes_per_point = (12 if spins == 1 else 41) * 8
        xc_full = sizes[3] * bytes_per_point
        basis_bytes = (7 * (spins + 1) + 16 * sizes[0]) * 8
        ao_point_bytes = (4 * sizes[0] + 3) * 8
        for extra in (
            0,
            256 * bytes_per_point,
            xc_full,
            xc_full + basis_bytes + 256 * ao_point_bytes,
            xc_full + basis_bytes + sizes[3] * ao_point_bytes,
        ):
            cap = sizes[4] + sizes[5] + extra
            plan = (ct.c_size_t * 5)()
            admitted = native.host_ks_nuclear_cache(handle, cap, plan)
            assert admitted == int(extra > 0), native.host_ks_nuclear_error().decode()
            assert plan[0] <= extra
            assert plan[1] == sizes[4] + plan[0]
            assert plan[1] + plan[2] <= cap
            assert bool(plan[3]) == bool(plan[4]) == (extra > xc_full)
            work = (ct.c_size_t * 7)()
            native.host_ks_nuclear_cache_work(handle, work)
            assert tuple(work) == (0,) * 7
            actual = _solve(native, handle, sizes, motion)
            assert actual[0] == 0, native.host_ks_nuclear_error().decode()
            for value, reference in zip(actual[1:3], expected[1:3], strict=True):
                np.testing.assert_array_equal(value, reference)
            native.host_ks_nuclear_cache_work(handle, work)
            assert work[0] == work[1] + work[2] == work[3] * sizes[3]
            assert bool(work[2]) == bool(extra)
            assert work[5] + work[6] == work[3] * sizes[3]
            assert bool(work[6]) == bool(plan[4])
            if extra:
                assert work[1] < work[0]


def test_subnormal_cartesian_gram_against_independent_point(native: Any) -> None:
    """Relative raw-sigma roundoff must not reject physical Cartesian gradients."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    rho = np.array([0.3, 0.1])
    first = np.array([1.0, 0.3, -0.2]) * 1e-154
    gradient = np.array([first, 0.13 * first])
    sigma = [
        sum(float(left) * float(right) for left, right in zip(one, two, strict=True))
        for one, two in (
            (gradient[0], gradient[0]),
            (gradient[0], gradient[1]),
            (gradient[1], gradient[1]),
        )
    ]
    bound = np.sqrt(sigma[0]) * np.sqrt(sigma[2])
    assert sigma[1] > bound * (1 + 16 * np.finfo(float).eps)
    output = np.full(9, -781.25)
    assert (
        native.host_ks_nuclear_point(
            _pointer(rho), _pointer(gradient), _pointer(output)
        )
        == 0
    )
    fields = np.column_stack([rho, gradient])[:, :, None]
    energy, potential, _, _ = libxc.eval_xc(METHOD, fields, spin=1, deriv=1)
    expected_gradient = np.array(
        [
            2 * potential[1][0, 0] * gradient[0] + potential[1][0, 1] * gradient[1],
            2 * potential[1][0, 2] * gradient[1] + potential[1][0, 1] * gradient[0],
        ]
    )
    np.testing.assert_allclose(
        output[:3],
        np.r_[energy[0] * rho.sum(), potential[0][0]],
        atol=2e-13,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        output[3:].reshape(2, 3), expected_gradient, atol=1e-168, rtol=3e-12
    )
    for invalid in (np.array([-0.3, 0.1]), np.array([np.nan, 0.1])):
        output.fill(-781.25)
        assert (
            native.host_ks_nuclear_point(
                _pointer(invalid), _pointer(gradient), _pointer(output)
            )
            != 0
        )
        np.testing.assert_array_equal(output, -781.25)


@contextmanager
def _owner(
    native: Any, spins: int, residual: int, centers: np.ndarray | None = None
) -> Any:
    if centers is None:
        centers = CENTERS[: spins + 1]
    centers = np.ascontiguousarray(centers, dtype=float)
    handle = native.host_ks_nuclear_create(_pointer(centers), spins, residual, 256)
    assert handle, native.host_ks_nuclear_error().decode()
    try:
        sizes = (ct.c_size_t * 7)()
        native.host_ks_nuclear_sizes(handle, sizes)
        yield handle, tuple(sizes)
    finally:
        native.host_ks_nuclear_destroy(handle)


def _frame(native: Any, handle: Any, sizes: tuple) -> tuple:
    basis, _, spins, points = sizes[:4]
    density = np.zeros((spins, basis, basis))
    weighted = np.zeros_like(density)
    coordinates, weights = np.empty((points, 3)), np.empty(points)
    native.host_ks_nuclear_frame(
        handle, *map(_pointer, (density, weighted, coordinates, weights))
    )
    return density, weighted, coordinates, weights


def _source(native: Any, handle: Any, sizes: tuple, motion: np.ndarray) -> tuple:
    overlap = np.full((sizes[0], sizes[0]), -781.25)
    fock = np.full((sizes[2], sizes[0], sizes[0]), -781.25)
    status = native.host_ks_nuclear_source(
        handle, _pointer(motion), motion.size, _pointer(overlap), _pointer(fock)
    )
    return status, overlap, fock


def _solve(
    native: Any,
    handle: Any,
    sizes: tuple,
    motion: np.ndarray,
    iterations: int = 100,
    workspace: int = 16 << 20,
) -> tuple:
    density = np.full((sizes[2], sizes[0], sizes[0]), -781.25)
    weighted, residual = np.full_like(density, -781.25), np.zeros(3)
    status = native.host_ks_nuclear_solve(
        handle,
        _pointer(motion),
        motion.size,
        _pointer(density),
        _pointer(weighted),
        iterations,
        workspace,
        _pointer(residual),
    )
    return status, density, weighted, residual


def _batch_solve(
    native: Any, handle: Any, sizes: tuple, motions: np.ndarray, iterations: int = 200
) -> tuple:
    matrices = (len(motions), sizes[2], sizes[0], sizes[0])
    density = np.full(matrices, -781.25)
    weighted = np.full_like(density, -781.25)
    residuals = np.empty(len(motions))
    work = (ct.c_size_t * (len(motions) + 4))()
    status = native.host_ks_nuclear_batch_solve(
        handle,
        _pointer(motions),
        len(motions),
        _pointer(density),
        _pointer(weighted),
        iterations,
        _pointer(residuals),
        work,
    )
    return status, density, weighted, residuals, np.asarray(work)


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_independent_nuclear_batch_budget_oracle_path_and_recovery(
    native: Any, spins: int, count: int
) -> None:
    with _owner(native, spins, 2) as (handle, sizes):
        motions = (
            np.random.default_rng(2844 + count).normal(size=(count, spins + 1, 3))
            * 0.05
        )
        expected = [_solve(native, handle, sizes, motion) for motion in motions]
        assert all(result[0] == 0 for result in expected)
        actual = _batch_solve(native, handle, sizes, motions)
        assert actual[0] == 0, native.host_ks_nuclear_error().decode()
        np.testing.assert_allclose(
            actual[1], np.stack([part[1] for part in expected]), atol=3e-11
        )
        np.testing.assert_allclose(
            actual[2], np.stack([part[2] for part in expected]), atol=3e-11
        )
        assert np.all(actual[3] <= 1e-12)
        assert actual[4][count] == actual[4][count + 1] == count
        plan = (ct.c_size_t * 3)()
        assert native.host_ks_nuclear_batch_plan(handle, count, plan) == 0
        total = plan[0] + plan[1]
        assert native.host_ks_nuclear_panel(handle, 0, total - 1, 0) == 0
        refused = _batch_solve(native, handle, sizes, motions)
        assert refused[0] == 1 and refused[4][count:].sum() == 0
        np.testing.assert_array_equal(refused[1], -781.25)
        np.testing.assert_array_equal(refused[2], -781.25)
        assert native.host_ks_nuclear_panel(handle, 0, total, 0) == 0
        recovered = _batch_solve(native, handle, sizes, motions)
        assert recovered[0] == 0, native.host_ks_nuclear_error().decode()
        np.testing.assert_array_equal(recovered[1], actual[1])
        np.testing.assert_array_equal(recovered[2], actual[2])
        native.host_ks_nuclear_failure(handle, 1, 0)
        failed = _batch_solve(native, handle, sizes, motions)
        assert failed[0] == -1
        np.testing.assert_array_equal(failed[1], -781.25)
        np.testing.assert_array_equal(failed[2], -781.25)
        native.host_ks_nuclear_failure(handle, 0, 0)
        assert _batch_solve(native, handle, sizes, motions)[0] == 0
        failed = _batch_solve(native, handle, sizes, motions, iterations=1)
        assert failed[0] == 1
        np.testing.assert_array_equal(failed[1], -781.25)
        np.testing.assert_array_equal(failed[2], -781.25)


def _oracle(
    spins: int,
    residual: int,
    centers: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    density: np.ndarray,
) -> tuple:
    """libcint primal operators and fully reconverged independent KS D/W."""
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
    reference = (dft.RKS if spins == 1 else dft.UKS)(molecule)
    reference.xc = METHOD
    reference.grids.coords = points
    reference.grids.weights = weights
    reference.conv_tol = 1e-14
    reference.conv_tol_grad = 2e-12
    reference.max_cycle = 200
    frozen = reference.get_hcore() + reference.get_veff(
        dm=density[0] if spins == 1 else density
    )
    reference.kernel(dm0=density[0] if spins == 1 else density)
    assert reference.converged
    relaxed = np.asarray(reference.make_rdm1()).reshape(density.shape)
    coefficients = np.asarray(reference.mo_coeff).reshape(density.shape)
    occupations = np.asarray(reference.mo_occ).reshape(spins, -1)
    energies = np.asarray(reference.mo_energy).reshape(spins, -1)
    weighted = np.array(
        [
            (current * (occupations[index] * energies[index])) @ current.T
            for index, current in enumerate(coefficients)
        ]
    )
    return (
        molecule.intor("int1e_ovlp"),
        np.asarray(frozen).reshape(density.shape),
        relaxed,
        weighted,
    )


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("residual", [0, 1, 2])
def test_actual_nuclear_sources_and_density_weights(
    native: Any,
    spins: int,
    residual: int,
    tmp_path: Path,
    record_testsuite_property: Any,
) -> None:
    centers, motion = CENTERS[: spins + 1].copy(), MOTION[: spins + 1].copy()
    with _owner(native, spins, residual) as (handle, sizes):
        density, weighted, points, grid_weights = _frame(native, handle, sizes)
        source = _source(native, handle, sizes, motion)
        assert source[0] == 0, native.host_ks_nuclear_error().decode()
        solved = _solve(native, handle, sizes, motion)
        assert solved[0] == 0, native.host_ks_nuclear_error().decode()
        assert solved[3][0] < 2e-11
        baseline = _oracle(spins, residual, centers, points, grid_weights, density)
        np.testing.assert_allclose(density, baseline[2], atol=2e-8, rtol=2e-8)
        np.testing.assert_allclose(weighted, baseline[3], atol=2e-8, rtol=2e-8)
        actual = [source[1], source[2], solved[1], solved[2]]
        errors, primals = [], []
        steps = [1e-3, 3e-4, 1e-4]
        for step in steps:
            displaced = []
            for sign in (1, -1):
                moved = centers + sign * step * motion
                with _owner(native, spins, residual, moved) as (other, other_sizes):
                    _, _, shifted_points, shifted_weights = _frame(
                        native, other, other_sizes
                    )
                displaced.append(
                    _oracle(
                        spins, residual, moved, shifted_points, shifted_weights, density
                    )
                )
            numerical = [
                (positive - negative) / (2 * step)
                for positive, negative in zip(*displaced, strict=True)
            ]
            errors.append(
                [
                    float(np.max(np.abs(value - expected)))
                    for value, expected in zip(actual, numerical, strict=True)
                ]
            )
            primals.append(displaced)
        np.savez_compressed(
            tmp_path / "nuclear-response.npz",
            centers=centers,
            motion=motion,
            steps=steps,
            errors=errors,
            overlap=actual[0],
            frozen=actual[1],
            density=actual[2],
            weighted=actual[3],
            primal_overlap=np.array([[part[0] for part in pair] for pair in primals]),
            primal_fock=np.array([[part[1] for part in pair] for pair in primals]),
            primal_density=np.array([[part[2] for part in pair] for pair in primals]),
            primal_weighted=np.array([[part[3] for part in pair] for pair in primals]),
        )
        for value, expected in zip(actual, numerical, strict=True):
            np.testing.assert_allclose(value, expected, atol=3e-7, rtol=2e-6)
        record_testsuite_property(
            f"nuclear_s{spins}_ecp{residual}",
            json.dumps(
                {"sizes": sizes, "errors": errors, "residual": solved[3].tolist()}
            ),
        )


@pytest.mark.parametrize("spins", [1, 2])
def test_refusal_publication_and_recovery(native: Any, spins: int) -> None:
    motion = MOTION[: spins + 1].copy()
    with _owner(native, spins, 2) as (handle, sizes):
        expected = _source(native, handle, sizes, motion)
        assert expected[0] == 0
        validations = native.host_ks_nuclear_validations(handle)
        for position in (1, validations // 2, validations):
            native.host_ks_nuclear_failure(handle, position, 0)
            refused = _source(native, handle, sizes, motion)
            assert refused[0] != 0
            np.testing.assert_array_equal(refused[1], -781.25)
            np.testing.assert_array_equal(refused[2], -781.25)
            native.host_ks_nuclear_failure(handle, 0, 0)
            recovered = _source(native, handle, sizes, motion)
            np.testing.assert_array_equal(recovered[1], expected[1])
            np.testing.assert_array_equal(recovered[2], expected[2])
        for refusal in range(1, 7):
            assert native.host_ks_nuclear_bind(handle, refusal) != 0
        for iterations, workspace in ((0, 16 << 20), (1, 1)):
            refused = _solve(native, handle, sizes, motion, iterations, workspace)
            assert refused[0] != 0
            np.testing.assert_array_equal(refused[1], -781.25)
            np.testing.assert_array_equal(refused[2], -781.25)
        assert _solve(native, handle, sizes, motion)[0] == 0


@pytest.mark.parametrize("spins", [1, 2])
def test_translation_linearity_aliases_and_stale_state(native: Any, spins: int) -> None:
    motion = MOTION[: spins + 1].copy()
    with _owner(native, spins, 2) as (handle, sizes):
        expected = _source(native, handle, sizes, motion)
        second_motion = np.flip(motion, axis=1).copy() * 0.7
        second = _source(native, handle, sizes, second_motion)
        combined = _source(native, handle, sizes, 1.3 * motion - 0.4 * second_motion)
        for index in (1, 2):
            np.testing.assert_allclose(
                combined[index],
                1.3 * expected[index] - 0.4 * second[index],
                atol=2e-12,
                rtol=2e-11,
            )
        translated = _source(
            native,
            handle,
            sizes,
            np.broadcast_to([0.13, -0.07, 0.09], motion.shape).copy(),
        )
        np.testing.assert_allclose(translated[1], 0, atol=2e-12)
        np.testing.assert_allclose(translated[2], 0, atol=2e-12)
        matrix, matrices = sizes[0] ** 2, sizes[2] * sizes[0] ** 2
        alias = np.full(matrix + matrices, -781.25)
        alias[: motion.size] = motion.ravel()
        assert (
            native.host_ks_nuclear_source(
                handle,
                _pointer(alias),
                motion.size,
                _pointer(alias),
                _pointer(alias[matrix:]),
            )
            == 0
        )
        np.testing.assert_array_equal(
            alias[:matrix].reshape(expected[1].shape), expected[1]
        )
        np.testing.assert_array_equal(
            alias[matrix:].reshape(expected[2].shape), expected[2]
        )
        unchanged = np.full(matrices, -781.25)
        for owner in range(3):
            assert (
                native.host_ks_nuclear_source(
                    handle,
                    _pointer(motion),
                    motion.size,
                    native.host_ks_nuclear_borrowed(handle, owner),
                    _pointer(unchanged),
                )
                != 0
            )
            np.testing.assert_array_equal(unchanged, -781.25)
        alias.fill(-781.25)
        assert (
            native.host_ks_nuclear_source(
                handle, _pointer(motion), motion.size, _pointer(alias), _pointer(alias)
            )
            != 0
        )
        np.testing.assert_array_equal(alias, -781.25)
        solved = _solve(native, handle, sizes, motion)
        alias = np.full(2 * matrices, -781.25)
        alias[: motion.size] = motion.ravel()
        diagnostics = np.zeros(3)
        assert (
            native.host_ks_nuclear_solve(
                handle,
                _pointer(alias),
                motion.size,
                _pointer(alias),
                _pointer(alias[matrices:]),
                100,
                16 << 20,
                _pointer(diagnostics),
            )
            == 0
        )
        np.testing.assert_array_equal(
            alias[:matrices].reshape(solved[1].shape), solved[1]
        )
        np.testing.assert_array_equal(
            alias[matrices:].reshape(solved[2].shape), solved[2]
        )
        native.host_ks_nuclear_failure(handle, 0, 1)
        refused = _solve(native, handle, sizes, motion)
        assert refused[0] != 0
        np.testing.assert_array_equal(refused[1], -781.25)
        np.testing.assert_array_equal(refused[2], -781.25)
        native.host_ks_nuclear_failure(handle, 0, 0)
        assert _solve(native, handle, sizes, motion)[0] == 0


def test_composed_native_response_without_python(native: Any, tmp_path: Path) -> None:
    empty = tmp_path / "no-python"
    empty.mkdir()
    result = subprocess.run(
        [str(native._test_executable)],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
        env={
            **os.environ,
            "PATH": str(empty),
            "PYTHONPATH": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    assert "RKS/UKS, empty/analytic/grid ECP, no Python owner" in result.stdout
