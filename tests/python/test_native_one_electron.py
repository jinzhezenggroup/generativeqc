"""Native normalized H/S source: independent libcint values, Slurm CUDA only.

The probe uses the actual C++ System normalizer and AoBasis, not Python-built AO
records. This qualifies T+effective-charge attraction only, not ECP projectors,
JK/DF response, a stationary Hessian or a complete CLI endpoint.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest


def _fixture(spherical: bool = False, removal: bool = False) -> dict:
    random = np.random.default_rng(8173)
    shells = [
        (0, 0, [[0.8, 0.7], [0.23, -0.15], [0.09, 0.3]]),
        (0, 3, [[0.6, 0.8], [0.19, -0.2]]),
        (1, 1, [[0.9, 0.6], [0.31, 0.25]]),
        (2, 2, [[0.7, 0.7], [0.27, -0.11]]),
    ]
    nao = sum(
        2 * angular + 1 if spherical else (angular + 1) * (angular + 2) // 2
        for _, angular, _ in shells
    )
    panels = random.normal(scale=0.07, size=(4, nao, nao))
    panels += panels.transpose(0, 2, 1)
    return {
        "centers": np.array([[0.1, -0.2, 0.3], [1.2, 0.7, -0.4], [-0.3, 0.5, 1.7]]),
        "numbers": np.array([8, 6, 2]),
        "cores": np.array([2, 2, 0]) if removal else np.zeros(3, dtype=int),
        "shells": shells,
        "spherical": spherical,
        "nao": nao,
        "weights": panels,
        "direction": random.normal(scale=0.15, size=(3, 3)),
    }


def _write_fixture(
    fixture: dict, path: Path, *, tile: int = 64, reject: int = 0
) -> None:
    with path.open("wb") as output:
        np.array(
            [
                len(fixture["centers"]),
                len(fixture["shells"]),
                fixture["spherical"],
                tile,
                reject,
            ],
            dtype=np.uint64,
        ).tofile(output)
        np.column_stack(
            (fixture["numbers"], fixture["cores"], fixture["centers"])
        ).astype(np.float64).tofile(output)
        for atom, angular, primitives in fixture["shells"]:
            np.array([atom, angular, len(primitives)], dtype=np.uint64).tofile(output)
            np.array(primitives, dtype=np.float64).tofile(output)
        np.asarray(fixture["direction"], dtype=np.float64).tofile(output)
        np.asarray(fixture["weights"], dtype=np.float64).tofile(output)


def _native(fixture: dict, tmp_path: Path, *, tile: int = 64, reject: int = 0) -> tuple:
    executable = os.environ.get("GENERATIVEQC_CUDA_ONE_ELECTRON_PROBE")
    if not executable:
        pytest.skip("requires a native one-electron probe and allocated Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    source, result = tmp_path / "source.bin", tmp_path / "result.bin"
    _write_fixture(fixture, source, tile=tile, reject=reject)
    process = subprocess.run(
        [executable, str(source), str(result)],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env={
            **os.environ,
            "PYTHONHOME": str(tmp_path / "missing"),
            "PYTHONPATH": str(tmp_path / "missing"),
        },
    )
    receipt = json.loads(process.stdout)
    if reject:
        assert receipt == {"prepare_rejected": True}
        assert not result.exists()
        return (), receipt
    assert receipt["lifecycle"]
    values = np.fromfile(result)
    matrix, atoms = fixture["nao"] ** 2, len(fixture["centers"])
    assert values.size == 4 * matrix + 6 * atoms
    return (
        values[: 4 * matrix].reshape(4, fixture["nao"], fixture["nao"]),
        values[4 * matrix :].reshape(2, atoms, 3),
    ), receipt


def _oracle(fixture: dict) -> tuple:
    gto = pytest.importorskip("pyscf.gto")
    from pyscf import lib

    lib.num_threads(1)
    labels = fixture.get("labels", ("O0", "C1", "He2"))
    basis = {name: [] for name in labels}
    for atom, angular, radial in fixture["shells"]:
        basis[labels[atom]].append([angular, *radial])
    molecule = gto.M(
        atom=list(zip(basis, fixture["centers"], strict=True)),
        basis=basis,
        cart=not fixture["spherical"],
        unit="Bohr",
        spin=int(sum(fixture["numbers"]) % 2),
        verbose=0,
    )
    assert molecule.nao == fixture["nao"]
    normalization = np.sqrt(molecule.intor("int1e_ovlp").diagonal())
    scale = normalization[:, None] * normalization[None, :]

    def matrices(centers: np.ndarray) -> np.ndarray:
        displaced = molecule.set_geom_(centers, unit="Bohr", inplace=False)
        core = displaced.intor("int1e_kin")
        for center, charge in zip(
            centers, fixture["numbers"] - fixture["cores"], strict=True
        ):
            with displaced.with_rinv_origin(center):
                core -= charge * displaced.intor("int1e_rinv")
        return np.array([core, displaced.intor("int1e_ovlp")]) / scale

    def energy(left: np.ndarray, right: float) -> float:
        core, overlap = matrices(
            fixture["centers"] + left + right * fixture["direction"]
        )
        weights = fixture["weights"]
        return np.sum(
            (weights[0] + right * weights[1]) * core
            + (weights[2] + right * weights[3]) * overlap
        )

    return matrices, energy


def _derivatives(fixture: dict, energy: object, step: float) -> np.ndarray:
    """Fourth-order Cartesian/mixed *value* differences, no production AD."""
    stencil = ((-2, 1 / 12), (-1, -2 / 3), (1, 2 / 3), (2, -1 / 12))
    result = np.zeros((2, *fixture["centers"].shape))
    for atom, axis in np.ndindex(fixture["centers"].shape):
        left = np.zeros_like(fixture["centers"])
        left[atom, axis] = step
        for displacement, coefficient in stencil:
            result[0, atom, axis] += coefficient * energy(displacement * left, 0) / step
            for motion, right_coefficient in stencil:
                result[1, atom, axis] += (
                    coefficient
                    * right_coefficient
                    * energy(displacement * left, motion * step)
                    / step**2
                )
    return result


def test_host_one_electron_admission_without_cuda_initialization() -> None:
    executable = os.environ.get("GENERATIVEQC_CUDA_ONE_ELECTRON_PROBE")
    if not executable:
        pytest.skip("requires the built native host admission probe")
    process = subprocess.run(
        [executable, "host"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert json.loads(process.stdout) == {"host_admission": True}


@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("removal", [False, True])
def test_native_all_ordered_spdf_pairs_against_independent_displaced_values(
    spherical: bool, removal: bool, tmp_path: Path
) -> None:
    fixture = _fixture(spherical, removal)
    (matrices, vectors), receipt = _native(fixture, tmp_path)
    primal, energy = _oracle(fixture)
    expected = primal(fixture["centers"])
    np.testing.assert_allclose(matrices[:2], expected, atol=2e-11, rtol=2e-11)
    step = 0.001
    expected_direction = sum(
        coefficient
        * primal(fixture["centers"] + motion * step * fixture["direction"])
        / step
        for motion, coefficient in (
            (-2, 1 / 12),
            (-1, -2 / 3),
            (1, 2 / 3),
            (2, -1 / 12),
        )
    )
    np.testing.assert_allclose(matrices[2:], expected_direction, atol=3e-10, rtol=2e-9)
    first = _derivatives(fixture, energy, 0.004)
    second = _derivatives(fixture, energy, 0.002)
    np.testing.assert_allclose(vectors[0], second[0], atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(vectors[1], second[1], atol=3e-7, rtol=2e-7)
    np.testing.assert_allclose(first, second, atol=1e-6, rtol=2e-7)
    np.testing.assert_allclose(vectors.sum(axis=1), 0, atol=2e-11, rtol=0)
    assert receipt["jobs"] > 48 and receipt["components"] > 1200


@pytest.mark.parametrize("spherical", [False, True])
def test_native_radial_tiles_and_rigid_translation(
    spherical: bool, tmp_path: Path
) -> None:
    fixture = _fixture(spherical)
    (matrices, vectors), first = _native(fixture, tmp_path, tile=1)
    (other_matrices, other_vectors), second = _native(fixture, tmp_path, tile=3)
    np.testing.assert_allclose(matrices, other_matrices, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(vectors, other_vectors, atol=3e-12, rtol=3e-12)
    assert first["components"] == second["components"]
    assert first["records"] == second["records"]
    assert first["projections"] > second["projections"]
    fixture["direction"][:] = [0.3, -0.4, 0.2]
    fixture["weights"][[1, 3]] = 0
    (translated, response), _ = _native(fixture, tmp_path, tile=64)
    np.testing.assert_allclose(translated[:2], matrices[:2], atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(translated[2:], 0, atol=3e-12, rtol=0)
    np.testing.assert_allclose(response[1], 0, atol=3e-12, rtol=0)


@pytest.mark.parametrize("reject", range(1, 7))
def test_native_rejects_malformed_normalized_source(
    reject: int, tmp_path: Path
) -> None:
    _native(_fixture(True, True), tmp_path, reject=reject)


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("fitted", [False, True])
def test_actual_native_ks_h_s_source_and_owner_lifetime(
    spins: int, fitted: bool
) -> None:
    executable = os.environ.get("GENERATIVEQC_KS_WB97MV_GRID_PROBE")
    if not executable:
        pytest.skip("requires the actual native KS molecular probe and Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    process = subprocess.run(
        [
            executable,
            "rks" if spins == 1 else "uks",
            "df-one-electron" if fitted else "one-electron",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "PYTHONHOME": "/nonexistent", "PYTHONPATH": "/nonexistent"},
    )
    result = json.loads(process.stdout)
    atoms = result["atoms"]
    radial = [
        [3.425250914, 0.1543289673],
        [0.6239137298, 0.5353281423],
        [0.168855404, 0.4446345422],
    ]
    fixture = {
        "centers": np.array(result["basis"][: 3 * atoms]).reshape(atoms, 3),
        "numbers": np.ones(atoms, dtype=int),
        "cores": np.zeros(atoms, dtype=int),
        "shells": [(atom, 0, radial) for atom in range(atoms)],
        "labels": tuple(f"H{atom}" for atom in range(atoms)),
        "spherical": False,
        "nao": result["nao"],
        "direction": np.array(result["integral_motion"]).reshape(atoms, 3),
        "weights": np.array(result["integral_weights"]).reshape(4, atoms, atoms),
    }
    primal, energy = _oracle(fixture)
    matrices = np.array(result["integral_matrices"]).reshape(4, atoms, atoms)
    vectors = np.array(result["integral_vectors"]).reshape(2, atoms, 3)
    np.testing.assert_allclose(
        matrices[:2], primal(fixture["centers"]), atol=2e-11, rtol=2e-11
    )
    np.testing.assert_allclose(
        vectors, _derivatives(fixture, energy, 0.002), atol=3e-7, rtol=2e-7
    )
    assert result["revoked_without_publication"] and result["integral_device_bytes"] > 0
