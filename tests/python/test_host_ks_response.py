"""Native molecular CPKS with independent Libxc/ERI action and solve oracles."""

from __future__ import annotations

import ctypes as ct
import json
import shutil
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.method.host_hessian_aot import (
    emit_registry,
    parse_inventory,
    programs,
)
from generativeqc_compiler.method.spec import resolve_method

from tools.generate_host_ks_cpks import generate

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
SIZE_POINTER = ct.POINTER(ct.c_size_t)
METHOD = ".08*LDA_X+.72*GGA_X_B88+.19*LDA_C_VWN_RPA+.81*GGA_C_LYP"


def _pointer(values: np.ndarray) -> POINTER:
    return values.ctypes.data_as(POINTER)


@pytest.fixture(scope="module")
def native(tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any) -> Any:
    """Link new leaves/consumer to ordinary CPU SCF, without rebuilding its core."""
    directory = tmp_path_factory.mktemp("host-ks-response")
    payload = json.loads((ROOT / "manifests/host_hessian_aot_b3lyp.json").read_text())
    payload["arrays"] = ["xc", "coefficient_response"]
    inventory = parse_inventory(payload)
    sources = []
    for entry in programs(inventory):
        source = directory / (entry.name + ".cpp")
        source.write_text(entry.emit())
        sources.append(source)
    registry = directory / "registry.cpp"
    registry.write_text(emit_registry(inventory))
    (directory / "host_ks_cpks_maps.hpp").write_text(generate())
    core = Path(policy_library._name).resolve()
    compiler = replace(
        required_native_cxx, cache=shutil.which("sccache") or required_native_cxx.cache
    )
    library = compiler.build_shared(
        [
            *sources,
            registry,
            ROOT / "src/dft/host_xc_response.cpp",
            ROOT / "src/dft/host_ks_response.cpp",
            ROOT / "src/dft/host_ks_perturbation.cpp",
            ROOT / "src/response/native_gmres.cpp",
            ROOT / "tests/native/host_ks_response_probe.cpp",
        ],
        directory / "response.so",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(directory),
        ],
        link_args=[str(core), "-Wl,-rpath," + str(core.parent)],
    )
    loaded = ct.CDLL(str(library))
    loaded.host_ks_error.restype = ct.c_char_p
    loaded.host_ks_create.argtypes = [
        ct.c_uint,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_char_p,
        ct.c_int,
    ]
    loaded.host_ks_create.restype = ct.c_void_p
    loaded.host_ks_destroy.argtypes = [ct.c_void_p]
    loaded.host_ks_rebind.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_char_p]
    loaded.host_ks_rebind.restype = ct.c_int
    loaded.host_ks_sizes.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_ks_frame.argtypes = [ct.c_void_p] + [POINTER] * 8
    loaded.host_ks_failure.argtypes = [ct.c_void_p, ct.c_int]
    loaded.host_ks_apply.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
    ]
    loaded.host_ks_apply.restype = ct.c_int
    loaded.host_ks_alias_frame.argtypes = [ct.c_void_p, POINTER]
    loaded.host_ks_alias_frame.restype = ct.c_int
    loaded.host_ks_solve.argtypes = [
        ct.c_void_p,
        POINTER,
        POINTER,
        ct.c_size_t,
        ct.c_size_t,
        POINTER,
        SIZE_POINTER,
    ]
    loaded.host_ks_solve.restype = ct.c_int
    loaded.host_ks_density.argtypes = [ct.c_void_p, POINTER, POINTER]
    loaded.host_ks_perturbation_bind.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t]
    loaded.host_ks_perturbation_solve.argtypes = (
        [ct.c_void_p] + [POINTER] * 4 + [ct.c_size_t] * 2 + [POINTER, SIZE_POINTER]
    )
    loaded.host_ks_batch_resources.argtypes = [ct.c_void_p, ct.c_size_t, SIZE_POINTER]
    loaded.host_ks_batch_apply.argtypes = [
        ct.c_void_p,
        POINTER,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
    ]
    loaded.host_ks_batch_solve.argtypes = [
        ct.c_void_p,
        POINTER,
        POINTER,
        ct.c_size_t,
        ct.c_size_t,
        POINTER,
        SIZE_POINTER,
    ]
    return loaded


@contextmanager
def _owner(
    native: Any,
    spins: int,
    tile: int = 256,
    budget: int = 4 << 20,
    mutation: int = 0,
    bad_method: bool = False,
) -> Any:
    method = resolve_method("B3LYP", spin="polarized" if spins == 2 else "unpolarized")
    handle = native.host_ks_create(
        spins,
        tile,
        budget,
        b"absent" if bad_method else method.identity.encode(),
        mutation,
    )
    if not handle:
        raise RuntimeError(native.host_ks_error().decode())
    try:
        sizes = (ct.c_size_t * 6)()
        native.host_ks_sizes(handle, sizes)
        yield handle, tuple(sizes)
    finally:
        native.host_ks_destroy(handle)


def _apply(
    native: Any, handle: int, vector: np.ndarray, output: np.ndarray | None = None
) -> np.ndarray:
    vector = np.ascontiguousarray(vector, dtype=np.float64)
    if output is None:
        output = np.empty_like(vector)
    status = native.host_ks_apply(
        handle, _pointer(vector), vector.size, _pointer(output), output.size
    )
    if status:
        raise RuntimeError(native.host_ks_error().decode())
    return output


def _frame(native: Any, handle: int, sizes: tuple) -> tuple:
    basis, points, spins, *_ = sizes
    arrays = (
        np.empty((spins, basis, basis)),
        np.empty((spins, basis, basis)),
        np.empty((spins, basis)),
        np.empty((basis, basis)),
        np.empty((basis, basis)),
        np.empty((points, 3)),
        np.empty(points),
        np.empty(4),
    )
    native.host_ks_frame(handle, *map(_pointer, arrays))
    return arrays


def _solve(
    native: Any,
    handle: int,
    rhs: np.ndarray,
    iterations: int = 150,
    workspace: int = 1 << 20,
    output: np.ndarray | None = None,
) -> tuple:
    rhs = np.ascontiguousarray(rhs, dtype=np.float64)
    if output is None:
        output = np.full_like(rhs, -9182.5)
    diagnostic = np.empty(2)
    work = (ct.c_size_t * 12)()
    status = native.host_ks_solve(
        handle,
        _pointer(rhs),
        _pointer(output),
        iterations,
        workspace,
        _pointer(diagnostic),
        work,
    )
    if status < 0:
        raise RuntimeError(native.host_ks_error().decode())
    return status, output, diagnostic, tuple(work)


def _batch_resources(native: Any, handle: int, count: int) -> tuple[int, ...]:
    sizes = (ct.c_size_t * 5)()
    if native.host_ks_batch_resources(handle, count, sizes):
        raise RuntimeError(native.host_ks_error().decode())
    return tuple(sizes)


def _batch_apply(
    native: Any,
    handle: int,
    vectors: np.ndarray,
    output: np.ndarray | None = None,
    workspace: np.ndarray | None = None,
) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float64)
    if output is None:
        output = np.full_like(vectors, -9182.5)
    if workspace is None:
        workspace = np.empty(_batch_resources(native, handle, len(vectors))[3] // 8)
    if native.host_ks_batch_apply(
        handle,
        _pointer(vectors),
        _pointer(output),
        len(vectors),
        _pointer(workspace),
        workspace.size,
    ):
        raise RuntimeError(native.host_ks_error().decode())
    return output


def _batch_solve(
    native: Any,
    handle: int,
    rhs: np.ndarray,
    iterations: int = 200,
    output: np.ndarray | None = None,
) -> tuple:
    rhs = np.ascontiguousarray(rhs, dtype=np.float64)
    if output is None:
        output = np.full_like(rhs, -9182.5)
    residuals = np.empty(len(rhs))
    work = (ct.c_size_t * (4 * len(rhs) + 4))()
    if native.host_ks_batch_solve(
        handle,
        _pointer(rhs),
        _pointer(output),
        len(rhs),
        iterations,
        _pointer(residuals),
        work,
    ):
        raise RuntimeError(native.host_ks_error().decode())
    return output, residuals, np.asarray(work)


def _oracle(spins: int, points: np.ndarray, weights: np.ndarray) -> tuple:
    """Independently normalized Gaussian integrals and first XC potentials."""
    gto = pytest.importorskip("pyscf.gto")
    dft = pytest.importorskip("pyscf.dft")
    atoms = [("H", (0, 0, 0)), ("H", (0.15, -0.08, 1.6))]
    if spins == 2:
        atoms.append(("H", (1.8, 0.25, 0.3)))
    molecule = gto.M(
        atom=atoms,
        unit="Bohr",
        spin=spins - 1,
        basis={"H": [[0, [1.2, 1.0]], [0, [0.35, 1.0]]]},
        verbose=0,
    )
    ao = dft.numint.eval_ao(molecule, points, deriv=1)
    eri = molecule.intor("int2e")

    def fock(density: np.ndarray) -> np.ndarray:
        value, gradient = ao[0], ao[1:]
        rho = np.einsum("pi,sij,pj->sp", value, density, value)
        grad_rho = 2 * np.einsum("kpi,sij,pj->skp", gradient, density, value)
        ingredients = np.concatenate((rho[:, None], grad_rho), axis=1)
        _, potentials, _, _ = dft.libxc.eval_xc(
            METHOD,
            ingredients if spins == 2 else ingredients[0],
            spin=spins - 1,
            deriv=1,
        )
        coulomb = np.einsum("ijkl,kl->ij", eri, density.sum(axis=0))
        result = np.empty_like(density)
        for spin in range(spins):
            scalar = potentials[0] if spins == 1 else potentials[0][:, spin]
            vector = (
                2 * potentials[1] * grad_rho[0]
                if spins == 1
                else 2 * potentials[1][:, 2 * spin] * grad_rho[spin]
                + potentials[1][:, 1] * grad_rho[1 - spin]
            )
            xc = np.einsum("p,pi,pj->ij", weights * scalar, value, value)
            grad = np.einsum("p,kp,kpi,pj->ij", weights, vector, gradient, value)
            exchange = np.einsum("ikjl,kl->ij", eri, density[spin])
            result[spin] = (
                coulomb - (0.1 if spins == 1 else 0.2) * exchange + xc + grad + grad.T
            )
        return result

    reference = (dft.RKS if spins == 1 else dft.UKS)(molecule)
    reference.xc = ".2*HF+" + METHOD
    reference.grids.coords = points
    reference.grids.weights = weights
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-10
    reference.max_cycle = 150
    reference.kernel()
    assert reference.converged
    return molecule, fock, reference


@pytest.mark.parametrize("spins", [1, 2])
def test_native_molecular_action_and_coupled_solve_against_independent_oracle(
    native: Any,
    spins: int,
    record_testsuite_property: Any,
) -> None:
    with _owner(native, spins) as (handle, sizes):
        basis, _, _, dimension, *occupations = sizes
        density, coefficients, energies, overlap, hcore, points, weights, summary = (
            _frame(native, handle, sizes)
        )
        molecule, independent_fock, independent_scf = _oracle(spins, points, weights)
        reference_density = np.asarray(independent_scf.make_rdm1()).reshape(
            density.shape
        )
        np.testing.assert_allclose(density, reference_density, atol=3e-9, rtol=3e-9)
        assert abs(summary[0] - independent_scf.e_tot) < 3e-9
        np.testing.assert_allclose(overlap, molecule.intor("int1e_ovlp"), atol=2e-13)
        np.testing.assert_allclose(
            hcore, molecule.intor("int1e_kin") + molecule.intor("int1e_nuc"), atol=2e-12
        )
        assert summary[1] < 1e-10 and summary[2] > 0 and summary[3] >= summary[2]
        physical = independent_fock(density) + hcore
        for spin in range(spins):
            np.testing.assert_allclose(
                coefficients[spin].T @ physical[spin] @ coefficients[spin],
                np.diag(energies[spin]),
                atol=3e-9,
            )
        generator = np.random.default_rng(2845 + spins)
        vector = generator.normal(size=dimension) * 0.03

        def reference_action(vector: np.ndarray, step: float) -> np.ndarray:
            delta_density = np.zeros_like(density)
            expected = np.empty(dimension)
            offset = 0
            for spin in range(spins):
                occupied = occupations[spin]
                virtual = basis - occupied
                block = vector[offset : offset + occupied * virtual].reshape(
                    occupied, virtual
                )
                occupied_coefficients = coefficients[spin, :, :occupied]
                virtual_coefficients = coefficients[spin, :, occupied:]
                transition = occupied_coefficients @ block @ virtual_coefficients.T
                delta_density[spin] = (2 if spins == 1 else 1) * (
                    transition + transition.T
                )
                gaps = energies[spin, occupied:] - energies[spin, :occupied, None]
                expected[offset : offset + occupied * virtual] = (gaps * block).ravel()
                offset += occupied * virtual
            derivative = sum(
                coefficient * independent_fock(density + shift * step * delta_density)
                for shift, coefficient in ((-2, 1), (-1, -8), (1, 8), (2, -1))
            ) / (12 * step)
            offset = 0
            for spin in range(spins):
                occupied = occupations[spin]
                count = occupied * (basis - occupied)
                block = (
                    coefficients[spin, :, :occupied].T
                    @ derivative[spin]
                    @ coefficients[spin, :, occupied:]
                )
                expected[offset : offset + count] += block.ravel()
                offset += count
            return expected

        actual = _apply(native, handle, vector)
        errors = []
        for step in (1e-3, 3e-4, 1e-4):
            reference = reference_action(vector, step)
            errors.append(float(np.max(np.abs(actual - reference))))
            np.testing.assert_allclose(actual, reference, atol=2e-8, rtol=3e-7)
        matrix = np.column_stack(
            [_apply(native, handle, column) for column in np.eye(dimension)]
        )
        np.testing.assert_allclose(matrix, matrix.T, atol=3e-12)
        independent_matrix = np.column_stack(
            [reference_action(column, 1e-4) for column in np.eye(dimension)]
        )
        np.testing.assert_allclose(matrix, independent_matrix, atol=2e-8, rtol=3e-7)
        rhs = generator.normal(size=dimension) * 0.02
        status, solution, residual, work = _solve(native, handle, rhs)
        assert status in (0, 1)
        np.testing.assert_allclose(
            solution, np.linalg.solve(matrix, rhs), atol=2e-10, rtol=3e-10
        )
        np.testing.assert_allclose(
            solution, np.linalg.solve(independent_matrix, rhs), atol=2e-8, rtol=3e-7
        )
        assert np.linalg.norm(matrix @ solution - rhs) < 2e-11
        assert residual[1] < 2e-11
        assert 0 < work[4] <= work[3] <= 1 << 20
        assert work[7] >= work[8] > 0 and work[9] >= sizes[1]
        directions = generator.normal(size=(4, dimension)) * 0.03
        batched = _batch_apply(native, handle, directions)
        for direction, actual_batch in zip(directions, batched, strict=True):
            np.testing.assert_array_equal(
                actual_batch, _apply(native, handle, direction)
            )
        np.testing.assert_allclose(
            batched, directions @ independent_matrix.T, atol=2e-8, rtol=3e-7
        )
        batched_solution, batched_residuals, batched_work = _batch_solve(
            native, handle, directions
        )
        np.testing.assert_allclose(
            batched_solution,
            np.linalg.solve(independent_matrix, directions.T).T,
            atol=2e-8,
            rtol=3e-7,
        )
        assert np.all(batched_residuals <= 1e-12)
        assert np.all(batched_work[:16:4] <= 1)
        assert batched_work[-3] == batched_work[-2] == batched_work[2:16:4].sum()
        assert batched_work[-1] == batched_work[-3] * sizes[1]
        tiles = (sizes[1] + 255) // 256
        assert batched_work[-4] < batched_work[-3] * tiles
        prefix = "host_ks_" + ("rks" if spins == 1 else "uks") + "_"
        for name, value in {
            "basis": basis,
            "grid_points": sizes[1],
            "dimension": dimension,
            "native_scf_iterations": int(summary[2]),
            "native_scf_fock_builds": int(summary[3]),
            "scf_energy_error": abs(summary[0] - independent_scf.e_tot),
            "scf_density_maxabs": float(np.max(np.abs(density - reference_density))),
            "action_three_step_maxabs": errors,
            "matrix_maxabs": float(np.max(np.abs(matrix - independent_matrix))),
            "matrix_asymmetry": float(np.max(np.abs(matrix - matrix.T))),
            "solve_relative_residual": float(residual[1]),
            "response_work_census": work,
            "batch_response_work_census": batched_work.tolist(),
            "batch_absolute_true_residuals": batched_residuals.tolist(),
        }.items():
            record_testsuite_property(prefix + name, value)


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_batched_orbital_actions_bounds_alias_and_failures(
    native: Any, spins: int, count: int
) -> None:
    with _owner(native, spins, tile=137) as (handle, sizes):
        dimension = sizes[3]
        vectors = (
            np.random.default_rng(2848 + count).normal(size=(count, dimension)) * 0.02
        )
        expected = np.stack([_apply(native, handle, vector) for vector in vectors])
        resource = _batch_resources(native, handle, count)
        workspace = np.full(resource[3] // 8 + 2, -8127.25)
        guarded = np.full(vectors.size + 2, -8127.25)
        output = guarded[1:-1].reshape(vectors.shape)
        _batch_apply(native, handle, vectors, output, workspace[1:-1])
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(guarded[[0, -1]], -8127.25)
        np.testing.assert_array_equal(workspace[[0, -1]], -8127.25)
        inplace = vectors.copy()
        _batch_apply(native, handle, inplace, inplace, workspace[1:-1])
        np.testing.assert_array_equal(inplace, expected)
        if count > 1:
            output.fill(-9182.5)
            with pytest.raises(RuntimeError, match="workspace"):
                _batch_apply(native, handle, vectors, output, workspace[1:-2])
            np.testing.assert_array_equal(output, -9182.5)
            alias = np.empty(max(workspace.size, vectors.size + 1))
            alias[: vectors.size] = vectors.ravel()
            with pytest.raises(RuntimeError, match="aliases"):
                _batch_apply(
                    native,
                    handle,
                    alias[: vectors.size].reshape(vectors.shape),
                    output,
                    alias[: resource[3] // 8],
                )
            np.testing.assert_array_equal(output, -9182.5)
        for failure in (1, 2, 3, 4, 5, 6, 7):
            native.host_ks_failure(handle, failure)
            output.fill(-9182.5)
            with pytest.raises(RuntimeError):
                _batch_apply(native, handle, vectors, output, workspace[1:-1])
            np.testing.assert_array_equal(output, -9182.5)
            native.host_ks_failure(handle, 0)
        np.testing.assert_array_equal(_batch_apply(native, handle, vectors), expected)
        vectors[-1, -1] = np.nan
        with pytest.raises(RuntimeError, match="nonfinite"):
            _batch_apply(native, handle, vectors, output, workspace[1:-1])
        np.testing.assert_array_equal(output, -9182.5)


@pytest.mark.parametrize("spins", [1, 2])
def test_batched_solve_composed_budget_failure_and_recovery(
    native: Any, spins: int
) -> None:
    with _owner(native, spins) as (handle, sizes):
        rhs = np.random.default_rng(2844).normal(size=(4, sizes[3])) * 0.01
        expected, residuals, _ = _batch_solve(native, handle, rhs)
        assert np.all(residuals <= 1e-12)
        resource = _batch_resources(native, handle, 4)
        total = sum(resource[:4])
        method = resolve_method(
            "B3LYP", spin="polarized" if spins == 2 else "unpolarized"
        ).identity.encode()
        assert native.host_ks_rebind(handle, total - 1, method) == 0
        refused, _, work = _batch_solve(native, handle, rhs)
        np.testing.assert_array_equal(work[:16:4], 2)
        np.testing.assert_array_equal(refused, -9182.5)
        assert work[-4:].sum() == 0
        assert native.host_ks_rebind(handle, total, method) == 0
        actual, residuals, _ = _batch_solve(native, handle, rhs)
        np.testing.assert_array_equal(actual, expected)
        assert np.all(residuals <= 1e-12)
        failed, _, work = _batch_solve(native, handle, rhs, iterations=1)
        assert np.any(work[:16:4] > 1)
        np.testing.assert_array_equal(failed, -9182.5)
        native.host_ks_failure(handle, 4)
        failed.fill(-9182.5)
        with pytest.raises(RuntimeError, match="reenter"):
            _batch_solve(native, handle, rhs, output=failed)
        np.testing.assert_array_equal(failed, -9182.5)
        native.host_ks_failure(handle, 0)
        actual, _, _ = _batch_solve(native, handle, rhs)
        np.testing.assert_array_equal(actual, expected)


def test_failure_lifetime_reentrancy_alias_and_nonconvergence_recovery(
    native: Any,
) -> None:
    with _owner(native, 2) as (handle, sizes):
        vector = np.linspace(0.01, 0.03, sizes[3])
        expected = _apply(native, handle, vector)
        for failure in (1, 2, 3, 4, 5, 6, 7):
            native.host_ks_failure(handle, failure)
            output = np.full_like(vector, -9182.5)
            with pytest.raises(RuntimeError):
                _apply(native, handle, vector, output)
            np.testing.assert_array_equal(output, -9182.5)
            with pytest.raises(RuntimeError):
                _solve(native, handle, vector, output=output)
            np.testing.assert_array_equal(output, -9182.5)
            native.host_ks_failure(handle, 0)
            np.testing.assert_allclose(
                _apply(native, handle, vector), expected, atol=2e-14
            )
        assert native.host_ks_alias_frame(handle, _pointer(vector)) == -1
        assert "aliases" in native.host_ks_error().decode()
        for invalid in (vector[:-1], np.full_like(vector, np.nan)):
            output = np.full_like(vector, -9182.5)
            with pytest.raises(RuntimeError):
                _apply(native, handle, invalid, output)
            np.testing.assert_array_equal(output, -9182.5)
        output = vector.copy()
        np.testing.assert_allclose(
            _apply(native, handle, output, output), expected, atol=2e-14
        )
        for iterations, workspace, reason in ((1, 1 << 20, 3), (150, 1, 2)):
            status, output, _, _ = _solve(native, handle, vector, iterations, workspace)
            assert status == reason
            np.testing.assert_array_equal(output, -9182.5)
        status, solution, *_ = _solve(native, handle, vector)
        assert status in (0, 1)
        np.testing.assert_allclose(_apply(native, handle, solution), vector, atol=2e-11)


@pytest.mark.parametrize("mutation", range(1, 9))
def test_model_grid_and_frame_mismatches_are_not_admitted(
    native: Any, mutation: int
) -> None:
    with pytest.raises(RuntimeError), _owner(native, 1, mutation=mutation):
        pytest.fail("invalid model/frame must not authorize native CPKS")


def test_missing_method_and_owned_numeric_budget_fail_before_actions(
    native: Any,
) -> None:
    for settings in ({"budget": 1}, {"bad_method": True}):
        with pytest.raises(RuntimeError), _owner(native, 1, **settings):
            pytest.fail("missing closure or capacity must not start response work")


def test_solve_budget_composes_action_arrays_and_native_gmres(native: Any) -> None:
    with _owner(native, 1) as (handle, sizes):
        rhs = np.linspace(0.01, 0.03, sizes[3])
        status, expected, _, work = _solve(native, handle, rhs)
        assert status in (0, 1)
        base, krylov = work[5] + work[6], work[3]
        method = resolve_method("B3LYP", spin="unpolarized").identity.encode()
        for budget in (base, base + krylov - 1):
            assert native.host_ks_rebind(handle, budget, method) == 0
            status, output, _, refused_work = _solve(native, handle, rhs)
            assert status == 2 and refused_work[4] == 0
            assert refused_work[7] == refused_work[11] == 0
            np.testing.assert_array_equal(output, -9182.5)
        assert native.host_ks_rebind(handle, base - 1, method) == -1
        assert "budget" in native.host_ks_error().decode()
        assert native.host_ks_rebind(handle, base + krylov, method) == 0
        status, actual, _, work = _solve(native, handle, rhs)
        assert status in (0, 1)
        assert work[4] <= krylov
        np.testing.assert_allclose(actual, expected, atol=2e-14)


def _perturbation(
    native: Any,
    handle: int,
    overlap: np.ndarray,
    frozen: np.ndarray,
    iterations: int = 150,
    workspace: int = 1 << 20,
    density: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple:
    """Call the staged native metric/field consumer without an oracle input."""
    overlap = np.ascontiguousarray(overlap, dtype=np.float64)
    frozen = np.ascontiguousarray(frozen, dtype=np.float64)
    if density is None:
        density = np.full_like(frozen, -9182.5)
    if weights is None:
        weights = np.full_like(frozen, -9182.5)
    diagnostic = np.empty(3)
    work = (ct.c_size_t * 10)()
    status = native.host_ks_perturbation_solve(
        handle,
        _pointer(overlap),
        _pointer(frozen),
        _pointer(density),
        _pointer(weights),
        iterations,
        workspace,
        _pointer(diagnostic),
        work,
    )
    if status < 0:
        raise RuntimeError(native.host_ks_error().decode())
    return status, density, weights, diagnostic, tuple(work)


def _directions(basis: int, spins: int) -> tuple:
    random = np.random.default_rng(5112026)
    metric = random.normal(size=(basis, basis)) * 0.01
    frozen = random.normal(size=(spins, basis, basis)) * 0.03
    return metric + metric.T, frozen + frozen.transpose(0, 2, 1)


@pytest.mark.parametrize("spins", [1, 2])
def test_native_metric_field_density_and_full_weights_against_displaced_scf(
    native: Any,
    spins: int,
    record_testsuite_property: Any,
) -> None:
    """Abstract S'/field control, not nuclear integrals or a Hessian endpoint."""
    with _owner(native, spins) as (handle, sizes):
        basis = sizes[0]
        density, _, _, overlap, hcore, points, grid_weights, _ = _frame(
            native, handle, sizes
        )
        metric, frozen = _directions(basis, spins)
        assert native.host_ks_perturbation_bind(handle, 8 << 20, 1 << 30) == 0
        status, actual_density, actual_weights, residual, work = _perturbation(
            native,
            handle,
            metric,
            frozen,
        )
        assert status == 0
        assert residual[1] < 2e-11 and residual[2] < 2e-11
        assert 0 < work[5] <= work[3]
        assert work[0] + work[2] + work[3] <= 8 << 20
        assert work[1] < work[0] and work[4] <= 1 << 30
        np.testing.assert_allclose(
            np.einsum("sij,ji->s", actual_density, overlap)
            + np.einsum("sij,ji->s", density, metric),
            0,
            atol=2e-11,
        )
        molecule, independent_fock, reference = _oracle(spins, points, grid_weights)
        density_action = np.empty_like(density)
        assert (
            native.host_ks_density(
                handle,
                _pointer(actual_density),
                _pointer(density_action),
            )
            == 0
        )
        action_error = []
        density_errors, weight_errors = [], []
        for step in (1e-4, 3e-4):
            numerical_action = (
                independent_fock(density + step * actual_density)
                - independent_fock(density - step * actual_density)
            ) / (2 * step)
            action_error.append(
                float(np.max(np.abs(density_action - numerical_action)))
            )
            np.testing.assert_allclose(
                density_action, numerical_action, atol=3e-8, rtol=3e-7
            )
            displaced = []
            for direction in (step, -step):
                oracle = type(reference)(molecule)
                oracle.xc = reference.xc
                oracle.grids.coords = points
                oracle.grids.weights = grid_weights
                oracle.conv_tol = 1e-14
                oracle.conv_tol_grad = 2e-12
                oracle.max_cycle = 200
                oracle.get_ovlp = lambda *args, value=overlap + direction * metric: (
                    value
                )
                field = hcore + direction * frozen
                oracle.get_hcore = (
                    lambda *args, value=field[0] if spins == 1 else field: value
                )
                if spins == 2:
                    oracle.energy_elec = lambda dm, h1e, vhf: (
                        np.einsum("sij,sji->", h1e, dm) + vhf.ecoul + vhf.exc,
                        vhf.ecoul + vhf.exc,
                    )
                oracle.kernel()
                assert oracle.converged
                coefficients = np.asarray(oracle.mo_coeff).reshape(spins, basis, basis)
                energies = np.asarray(oracle.mo_energy).reshape(spins, basis)
                occupations = np.asarray(oracle.mo_occ).reshape(spins, basis)
                weighted = np.einsum(
                    "smi,si,sni->smn",
                    coefficients,
                    energies * occupations,
                    coefficients,
                )
                displaced.append(
                    (np.asarray(oracle.make_rdm1()).reshape(density.shape), weighted)
                )
            expected_density, expected_weights = (
                (displaced[0][index] - displaced[1][index]) / (2 * step)
                for index in range(2)
            )
            density_errors.append(
                float(np.max(np.abs(actual_density - expected_density)))
            )
            weight_errors.append(
                float(np.max(np.abs(actual_weights - expected_weights)))
            )
            np.testing.assert_allclose(
                actual_density, expected_density, atol=3e-8, rtol=3e-7
            )
            np.testing.assert_allclose(
                actual_weights, expected_weights, atol=3e-8, rtol=3e-7
            )
        prefix = "host_ks_perturbation_" + ("rks_" if spins == 1 else "uks_")
        for name, value in {
            "density_two_step_maxabs": density_errors,
            "weights_two_step_maxabs": weight_errors,
            "density_action_two_step_maxabs": action_error,
            "physical_relative_residual": float(residual[1]),
            "resources_and_work": work,
        }.items():
            record_testsuite_property(prefix + name, value)


def test_native_perturbation_failures_are_transactional_and_recover(
    native: Any,
) -> None:
    with _owner(native, 2) as (handle, sizes):
        metric, frozen = _directions(sizes[0], 2)
        assert native.host_ks_perturbation_bind(handle, 8 << 20, 1 << 30) == 0
        _, expected_density, expected_weights, _, work = _perturbation(
            native, handle, metric, frozen
        )
        for failure in (1, 2, 3, 4, 5, 6, 7, 8):
            native.host_ks_failure(handle, failure)
            density, weights = (
                np.full_like(frozen, -9182.5),
                np.full_like(frozen, -9182.5),
            )
            with pytest.raises(RuntimeError):
                _perturbation(
                    native, handle, metric, frozen, density=density, weights=weights
                )
            np.testing.assert_array_equal(density, -9182.5)
            np.testing.assert_array_equal(weights, -9182.5)
            native.host_ks_failure(handle, 0)
            recovered = _perturbation(native, handle, metric, frozen)
            assert recovered[0] == 0
            np.testing.assert_allclose(recovered[1], expected_density, atol=2e-14)
            np.testing.assert_allclose(recovered[2], expected_weights, atol=2e-14)
        for bad_metric in (
            np.full_like(metric, np.nan),
            metric + np.tril(np.ones_like(metric), -1),
        ):
            with pytest.raises(RuntimeError):
                _perturbation(native, handle, bad_metric, frozen)
        shared = np.full_like(frozen, -9182.5)
        with pytest.raises(RuntimeError, match="alias"):
            _perturbation(
                native, handle, metric, frozen, density=shared, weights=shared
            )
        np.testing.assert_array_equal(shared, -9182.5)
        for iterations, workspace, reason in ((1, 1 << 20, 2), (150, 1, 1)):
            result = _perturbation(
                native, handle, metric, frozen, iterations, workspace
            )
            assert result[0] == reason
            np.testing.assert_array_equal(result[1], -9182.5)
            np.testing.assert_array_equal(result[2], -9182.5)
        assert native.host_ks_perturbation_bind(handle, 8 << 20, work[4] - 1) == -1
        base = work[0] + work[2]
        assert native.host_ks_perturbation_bind(handle, base - 1, work[4]) == -1
        assert (
            native.host_ks_perturbation_bind(handle, base + work[3] - 1, work[4]) == 0
        )
        refused = _perturbation(native, handle, metric, frozen)
        assert refused[0] == 1 and refused[4][8] == refused[4][9] == 0
        assert native.host_ks_perturbation_bind(handle, base + work[3], work[4]) == 0
        aliased_input = frozen.copy()
        recovered = _perturbation(
            native, handle, metric, aliased_input, density=aliased_input
        )
        assert recovered[0] == 0
        np.testing.assert_allclose(recovered[1], expected_density, atol=2e-14)
        np.testing.assert_allclose(recovered[2], expected_weights, atol=2e-14)
