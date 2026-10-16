"""Slurm-only native J/K response against independent displaced libcint values.

Finite differences are an oracle only. The native probe constructs the actual
normalized C++ System/AoBasis and executes without Python/compiler discovery.
This is component qualification, not a stationary Hessian endpoint benchmark.
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

PARTS = ("exact", "primary", "correction", "coulomb", "short", "long")


def _fixture(spins: int = 2, spherical: bool = False) -> dict:
    random = np.random.default_rng(1729)
    shells = [
        (0, 0, [[0.8, 0.7], [0.23, -0.15], [0.09, 0.3]]),
        (1, 1, [[0.9, 0.6], [0.31, 0.25]]),
    ]
    panels = random.normal(scale=0.17, size=(2, spins, 4, 4))
    panels += panels.transpose(0, 1, 3, 2)
    return {
        "centers": np.array([[0.1, -0.2, 0.3], [1.2, 0.7, -0.4]]),
        "numbers": np.array([8, 6]),
        "shells": shells,
        "spherical": spherical,
        "nao": 4,
        "spins": spins,
        "panels": panels,
        "direction": random.normal(scale=0.15, size=(2, 3)),
    }


def _write_fixture(
    fixture: dict, path: Path, *, part: str = "exact", tile: int = 64, reject: int = 0
) -> None:
    with path.open("wb") as output:
        np.array(
            [
                len(fixture["centers"]),
                len(fixture["shells"]),
                fixture["spherical"],
                fixture["spins"],
                PARTS.index(part),
                tile,
                reject,
            ],
            dtype=np.uint64,
        ).tofile(output)
        np.column_stack(
            (fixture["numbers"], np.zeros(len(fixture["centers"])), fixture["centers"])
        ).astype(np.float64).tofile(output)
        for atom, angular, radial in fixture["shells"]:
            np.array([atom, angular, len(radial)], dtype=np.uint64).tofile(output)
            np.array(radial, dtype=np.float64).tofile(output)
        np.asarray(fixture["direction"], dtype=np.float64).tofile(output)
        np.asarray(fixture["panels"], dtype=np.float64).tofile(output)


def _native(
    fixture: dict, path: Path, *, part: str = "exact", tile: int = 64, reject: int = 0
) -> tuple:
    executable = os.environ.get("GENERATIVEQC_CUDA_WB97MV_NUCLEAR_JK_PROBE")
    if not executable:
        pytest.skip("requires native J/K probe and allocated Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    source, result = path / "source.bin", path / "output.bin"
    result.unlink(missing_ok=True)
    _write_fixture(fixture, source, part=part, tile=tile, reject=reject)
    completed = subprocess.run(
        [executable, str(source), str(result)],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
        env={**os.environ, "PYTHONHOME": "/nonexistent", "PYTHONPATH": "/nonexistent"},
    )
    receipt = json.loads(completed.stdout)
    if reject:
        assert receipt == {"prepare_rejected": True} and not result.exists()
        return (), receipt
    assert receipt["lifecycle"]
    values = np.fromfile(result)
    panel = fixture["spins"] * fixture["nao"] ** 2
    assert values.size == 1 + 2 * panel + 6 * len(fixture["centers"])
    return (
        values[0],
        values[1 : 1 + 2 * panel].reshape(
            2, fixture["spins"], fixture["nao"], fixture["nao"]
        ),
        values[1 + 2 * panel :].reshape(2, *fixture["centers"].shape),
    ), receipt


def _oracle(fixture: dict, part: str) -> tuple:
    gto = pytest.importorskip("pyscf.gto")
    from pyscf import lib

    lib.num_threads(1)
    labels = tuple(fixture.get("labels", ("O0", "C1")))
    basis = {name: [] for name in labels}
    for atom, angular, radial in fixture["shells"]:
        basis[labels[atom]].append([angular, *radial])
    molecule = gto.M(
        atom=list(zip(labels, fixture["centers"], strict=True)),
        basis=basis,
        cart=not fixture["spherical"],
        unit="Bohr",
        spin=int(sum(fixture["numbers"]) % 2),
        verbose=0,
    )
    assert molecule.nao == fixture["nao"]
    normalization = np.sqrt(molecule.intor("int1e_ovlp").diagonal())
    scale = np.einsum(
        "i,j,k,l->ijkl", normalization, normalization, normalization, normalization
    )
    spins = fixture["spins"]

    def primal(centers: np.ndarray, density: np.ndarray) -> tuple:
        displaced = molecule.set_geom_(centers, unit="Bohr", inplace=False)
        full = displaced.intor("int2e", aosym="s1") / scale
        with displaced.with_range_coulomb(0.3):
            long = displaced.intor("int2e", aosym="s1") / scale
        coulomb = (
            full if part in ("exact", "primary", "coulomb") else np.zeros_like(full)
        )
        exchange = {
            "exact": 0.15 * full + 0.85 * long,
            "primary": 0.15 * full,
            "correction": 0.85 * long,
            "coulomb": np.zeros_like(full),
            "short": 0.15 * (full - long),
            "long": long,
        }[part]
        fock = np.broadcast_to(
            np.einsum("cd,abcd->ab", density.sum(axis=0), coulomb), density.shape
        ).copy()
        fock -= (0.5 if spins == 1 else 1.0) * np.einsum(
            "scd,acbd->sab", density, exchange
        )
        energy = 0.5 * np.einsum("sab,sab->", density, fock)
        return float(energy), fock

    def energy(left: np.ndarray, right: float) -> float:
        return primal(
            fixture["centers"] + left + right * fixture["direction"],
            fixture["panels"][0] + right * fixture["panels"][1],
        )[0]

    return primal, energy


def _derivatives(fixture: dict, energy: object, step: float) -> np.ndarray:
    """Fourth-order mixed displaced values, independent of compiler AD."""
    stencil = ((-2, 1 / 12), (-1, -2 / 3), (1, 2 / 3), (2, -1 / 12))
    result = np.zeros((2, *fixture["centers"].shape))
    for atom, axis in np.ndindex(fixture["centers"].shape):
        left = np.zeros_like(fixture["centers"])
        left[atom, axis] = step
        for displacement, coefficient in stencil:
            result[0, atom, axis] += coefficient * energy(displacement * left, 0) / step
            for motion, right in stencil:
                result[1, atom, axis] += (
                    coefficient
                    * right
                    * energy(displacement * left, motion * step)
                    / step**2
                )
    return result


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("part", PARTS)
@pytest.mark.parametrize("spherical", (False, True))
def test_native_jk_against_independent_displaced_integrals(
    spins: int, part: str, spherical: bool, tmp_path: Path
) -> None:
    fixture = _fixture(spins, spherical)
    (energy, fock, vectors), receipt = _native(fixture, tmp_path, part=part)
    primal, objective = _oracle(fixture, part)
    expected_energy, expected_fock = primal(fixture["centers"], fixture["panels"][0])
    assert energy == pytest.approx(expected_energy, abs=3e-12, rel=3e-12)
    np.testing.assert_allclose(fock[0], expected_fock, atol=3e-12, rtol=3e-12)
    stencil = ((-2, 1 / 12), (-1, -2 / 3), (1, 2 / 3), (2, -1 / 12))
    expected_motion = sum(
        weight
        * primal(
            fixture["centers"] + offset * 0.001 * fixture["direction"],
            fixture["panels"][0],
        )[1]
        / 0.001
        for offset, weight in stencil
    )
    np.testing.assert_allclose(fock[1], expected_motion, atol=5e-11, rtol=3e-9)
    first, second = (
        _derivatives(fixture, objective, 0.004),
        _derivatives(fixture, objective, 0.002),
    )
    np.testing.assert_allclose(vectors[0], second[0], atol=3e-10, rtol=2e-9)
    np.testing.assert_allclose(vectors[1], second[1], atol=3e-8, rtol=2e-7)
    np.testing.assert_allclose(first, second, atol=1e-7, rtol=3e-7)
    np.testing.assert_allclose(vectors.sum(axis=1), 0, atol=2e-12, rtol=0)
    assert receipt["groups"] == (32 if part == "exact" else 16)
    assert receipt["components"] == (13122 if part == "exact" else 6561)


@pytest.mark.parametrize("spins", (1, 2))
def test_native_jk_cache_reuse_and_translation(spins: int, tmp_path: Path) -> None:
    fixture = _fixture(spins, True)
    (energy, fock, vectors), first = _native(fixture, tmp_path, tile=1)
    (other_energy, other_fock, other_vectors), second = _native(
        fixture, tmp_path, tile=3
    )
    assert energy == pytest.approx(other_energy, abs=2e-12)
    np.testing.assert_allclose(fock, other_fock, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(vectors, other_vectors, atol=3e-12, rtol=3e-12)
    assert (
        first["components"] == second["components"]
        and first["records"] == second["records"]
    )
    assert (
        first["projections"] == second["projections"]
        and first["reductions"] > second["reductions"]
    )
    fixture["direction"][:] = [0.3, -0.4, 0.2]
    fixture["panels"][1] = 0
    (translated_energy, translated_fock, translated_vectors), _ = _native(
        fixture, tmp_path
    )
    assert translated_energy == pytest.approx(energy, abs=2e-12)
    np.testing.assert_allclose(translated_fock[1], 0, atol=3e-12, rtol=0)
    np.testing.assert_allclose(translated_vectors[1], 0, atol=3e-12, rtol=0)


@pytest.mark.parametrize("reject", range(1, 7))
def test_native_jk_rejects_malformed_packed_basis(reject: int, tmp_path: Path) -> None:
    _native(_fixture(), tmp_path, reject=reject)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("fitted", (False, True))
def test_actual_native_ks_jk_source(spins: int, fitted: bool) -> None:
    executable = os.environ.get("GENERATIVEQC_KS_WB97MV_GRID_PROBE")
    if not executable:
        pytest.skip("requires actual native KS probe and allocated Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    completed = subprocess.run(
        [
            executable,
            "rks" if spins == 1 else "uks",
            "df-nuclear-jk" if fitted else "nuclear-jk",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
        env={**os.environ, "PYTHONHOME": "/nonexistent", "PYTHONPATH": "/nonexistent"},
    )
    result = json.loads(completed.stdout)
    atoms, nao = result["atoms"], result["nao"]
    radial = [
        [3.425250914, 0.1543289673],
        [0.6239137298, 0.5353281423],
        [0.168855404, 0.4446345422],
    ]
    fixture = {
        "centers": np.array(result["basis"][: 3 * atoms]).reshape(atoms, 3),
        "numbers": np.ones(atoms, dtype=int),
        "labels": tuple(f"H{atom}" for atom in range(atoms)),
        "shells": [(atom, 0, radial) for atom in range(atoms)],
        "nao": nao,
        "spins": spins,
        "spherical": False,
        "panels": np.array([result["density"], result["direction"]]).reshape(
            2, spins, nao, nao
        ),
        "direction": np.array(result["jk_motion"]).reshape(atoms, 3),
    }
    primal, objective = _oracle(fixture, "correction" if fitted else "exact")
    energy, fock = primal(fixture["centers"], fixture["panels"][0])
    assert result["jk_energy"][0] == pytest.approx(energy, abs=2e-11, rel=2e-11)
    np.testing.assert_allclose(
        np.array(result["jk_matrices"]).reshape(2, spins, nao, nao)[0],
        fock,
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        np.array(result["jk_vectors"]).reshape(2, atoms, 3),
        _derivatives(fixture, objective, 0.002),
        atol=2e-7,
        rtol=2e-7,
    )
    assert result["jk_device_bytes"] > 0 and result["revoked_without_publication"]
