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

PARTS = ("exact", "primary", "correction", "coulomb", "short", "long", "full_hybrid")


def _fixture(spins: int = 2, spherical: bool = False, angular: int = 1) -> dict:
    random = np.random.default_rng(1729)
    shells = [
        (0, 0, [[0.8, 0.7], [0.23, -0.15], [0.09, 0.3]]),
        (1, angular, [[0.9, 0.6], [0.31, 0.25]]),
    ]
    nao = 1 + (2 * angular + 1 if spherical else (angular + 1) * (angular + 2) // 2)
    panels = random.normal(scale=0.17, size=(2, spins, nao, nao))
    panels += panels.transpose(0, 1, 3, 2)
    return {
        "centers": np.array([[0.1, -0.2, 0.3], [1.2, 0.7, -0.4]]),
        "numbers": np.array([8, 6]),
        "shells": shells,
        "spherical": spherical,
        "nao": nao,
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
        if part == "full_hybrid":
            np.array([fixture.get("exchange_fraction", 0.25)], dtype=np.float64).tofile(
                output
            )
        if part == "exact" and "range_exchange" in fixture:
            np.asarray(fixture["range_exchange"], dtype=np.float64).tofile(output)


def _native(
    fixture: dict,
    path: Path,
    *,
    part: str = "exact",
    tile: int = 64,
    reject: int = 0,
    cache_mode: str = "resident",
) -> tuple:
    executable = os.environ.get("GENERATIVEQC_CUDA_WB97MV_NUCLEAR_JK_PROBE")
    if not executable:
        pytest.skip("requires native J/K probe and allocated Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    source, result = path / "source.bin", path / "output.bin"
    result.unlink(missing_ok=True)
    _write_fixture(fixture, source, part=part, tile=tile, reject=reject)
    completed = subprocess.run(
        [executable, str(source), str(result), cache_mode],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        env={**os.environ, "PYTHONHOME": "/nonexistent", "PYTHONPATH": "/nonexistent"},
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    if cache_mode in {"screened-table", "indexed-table"}:
        assert receipt == {"screened_table": True} and not result.exists()
        return (), receipt
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


def test_screened_frozen_jk_is_private_and_bounded(tmp_path: Path) -> None:
    fixture = _fixture(spins=1)
    fixture["centers"][1, 0] += 18.0
    fixture["panels"][1] = 0
    _native(fixture, tmp_path, part="full_hybrid", cache_mode="screened-table")


@pytest.mark.parametrize("primitives,p_centers", ((1, 2), (3, 2), (3, 3)))
def test_indexed_frozen_jk_matches_dense_replay(
    tmp_path: Path, primitives: int, p_centers: int
) -> None:
    """Cover short and two-chunk radial folds, including three distinct p centers."""
    fixture = _fixture(spins=1)
    shells = [(0, int(p_centers == 3), 0.8), (1, 1, 0.7), (2, 1, 0.6), (3, 0, 0.9)]
    nao = sum(3 if angular else 1 for _, angular, _ in shells)
    fixture.update(
        numbers=np.array([1, 8, 8, 1]),
        centers=np.array(
            [[0.0, 0.0, 0.0], [1.3, -0.2, 0.5], [0.4, 1.6, -0.9], [20, 0, 0]]
        ),
        shells=[
            (
                atom,
                angular,
                [
                    [exponent + 0.3 * index, (-0.4) ** index]
                    for index in range(primitives)
                ],
            )
            for atom, angular, exponent in shells
        ],
        nao=nao,
        panels=np.stack([0.25 * np.eye(nao) + 0.05, np.zeros((nao, nao))]),
        direction=np.zeros((4, 3)),
    )
    _native(fixture, tmp_path, part="full_hybrid", cache_mode="indexed-table")


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
        short_fraction, correction, omega = fixture.get(
            "range_exchange", (0.15, 0.85, 0.3)
        )
        with displaced.with_range_coulomb(omega):
            long = displaced.intor("int2e", aosym="s1") / scale
        coulomb = (
            full
            if part in ("exact", "primary", "coulomb", "full_hybrid")
            else np.zeros_like(full)
        )
        exchange = {
            "exact": short_fraction * full + correction * long,
            "primary": 0.15 * full,
            "correction": 0.85 * long,
            "coulomb": np.zeros_like(full),
            "short": 0.15 * (full - long),
            "long": long,
            "full_hybrid": fixture.get("exchange_fraction", 0.25) * full,
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
    values, receipt = _native(fixture, tmp_path, part=part)
    _qualify(fixture, values, part)
    assert receipt["groups"] == (32 if part == "exact" else 16)
    assert 0 < receipt["components"] <= (13122 if part == "exact" else 6561)


def _qualify(fixture: dict, values: tuple, part: str) -> None:
    """Retain the independent libcint value and displacement acceptance gates."""
    energy, fock, vectors = values
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


@pytest.mark.parametrize("spins", (1, 2))
def test_native_jk_runtime_omega_batch_and_scalar(spins: int, tmp_path: Path) -> None:
    """Compare both launch contracts against the same signed, non-WB97M-V ERIs."""
    fixture = _fixture(spins, spherical=True)
    fixture["range_exchange"] = (0.65, -0.45, 0.33)
    batched, batch_work = _native(fixture, tmp_path)
    scalar, scalar_work = _native(fixture, tmp_path, cache_mode="unbatched")
    assert batch_work["batch_bytes"] > 0
    assert scalar_work["batch_bytes"] == 0
    _qualify(fixture, batched, "exact")
    _qualify(fixture, scalar, "exact")
    for actual, expected in zip(batched, scalar, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize(("angular", "part"), ((0, "full_hybrid"), (1, "coulomb")))
def test_native_jk_retained_window_budget_fallback(
    angular: int, part: str, tmp_path: Path
) -> None:
    """Retention and immediate replay share the same canonical Direct sources."""
    fixture = _fixture(spins=1, angular=angular)
    retained, retained_work = _native(
        fixture, tmp_path, part=part, cache_mode="fresh-retained"
    )
    streamed, streamed_work = _native(
        fixture, tmp_path, part=part, cache_mode="fresh-streamed"
    )
    assert retained_work["retained_source_bytes"] > 0
    assert streamed_work["retained_source_bytes"] == 0
    assert (
        retained_work["bytes"] - streamed_work["bytes"]
        == retained_work["retained_source_bytes"]
    )
    assert retained_work["streaming_tiles"] == streamed_work["streaming_tiles"] > 1
    assert retained_work["components"] == streamed_work["components"]
    for actual, expected in zip(retained, streamed, strict=True):
        np.testing.assert_array_equal(actual, expected)
    _qualify(fixture, retained, part)


@pytest.mark.parametrize(
    ("radials", "angular"), ((1, 0), (2, 1), (3, 0), (3, 1), (4, 1))
)
def test_native_jk_full_range_radial_pair(
    radials: int, angular: int, tmp_path: Path
) -> None:
    """Compare short, two-chunk, and over-cap primitive lists with libcint."""
    fixture = _fixture(spins=1, angular=angular)
    basis = [[1.05, 0.8], [0.41, -0.15], [0.17, 0.25], [0.06, -0.04]]
    fixture["shells"] = [(0, 0, basis[:radials]), (1, angular, basis[:radials])]
    full, full_work = _native(
        fixture, tmp_path, part="full_hybrid", tile=64, cache_mode="fresh-retained"
    )
    split, split_work = _native(
        fixture, tmp_path, part="full_hybrid", tile=32, cache_mode="fresh-retained"
    )
    assert full_work["components"] == split_work["components"]
    assert full_work["records"] == split_work["records"]
    for actual, expected in zip(full, split, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
    _qualify(fixture, full, "full_hybrid")


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("tile", (2, 64))
def test_native_jk_batched_sources_and_ordered_budget_fallback(
    spins: int, tile: int, tmp_path: Path
) -> None:
    """Independent ERIs qualify batched chunks and both resident fallbacks."""
    fixture = _fixture(spins, spherical=True)
    outputs = {}
    receipts = {}
    for mode in (
        "resident",
        "batch-chunked",
        "unbatched",
        "batch-budgeted",
        "fresh-batch-budgeted",
        "projection-budgeted",
        "tiled",
        "resident-cap-tiled",
        "resident-cap-fallback",
        "streamed",
    ):
        outputs[mode], receipts[mode] = _native(
            fixture, tmp_path, tile=tile, cache_mode=mode
        )
        _qualify(fixture, outputs[mode], "exact")
    assert receipts["resident"]["batch_bytes"] > 0
    assert receipts["batch-chunked"]["batch_bytes"] > 0
    # Raw radial data may determine the shared scratch size for both routes.
    assert (
        receipts["batch-chunked"]["batch_bytes"] <= receipts["resident"]["batch_bytes"]
    )
    assert (
        receipts["batch-chunked"]["projection_launches"]
        > receipts["resident"]["projection_launches"]
    )
    for actual, expected in zip(
        outputs["resident"], outputs["batch-chunked"], strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    for mode in ("unbatched", "batch-budgeted", "projection-budgeted"):
        assert receipts[mode]["batch_bytes"] == 0
        assert receipts[mode]["resident_bytes"] > 0
        for actual, expected in zip(outputs["resident"], outputs[mode], strict=True):
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
    assert (
        receipts["resident"]["integral_launches"]
        < receipts["unbatched"]["integral_launches"]
    )
    for field in ("projection_launches", "reduction_launches"):
        assert receipts["resident"][field] < receipts["unbatched"][field]
    for mode in ("tiled", "resident-cap-tiled", "fresh-batch-budgeted"):
        receipt = receipts[mode]
        assert receipt["streaming_tiles"] > 1
        assert receipt["source_groups"] == receipts["resident"]["source_groups"]
        assert receipt["components"] == receipts["resident"]["components"]
        assert receipt["components"] < receipts["streamed"]["components"]
        assert receipt["integral_launches"] < receipts["streamed"]["integral_launches"]
        for actual, expected in zip(outputs[mode], outputs["resident"], strict=True):
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
    fallback = receipts["resident-cap-fallback"]
    assert fallback["streaming_tiles"] == fallback["resident_bytes"] == 0
    assert fallback["components"] == receipts["streamed"]["components"]


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("spherical", (False, True))
@pytest.mark.parametrize("cache_mode", ("resident", "streamed", "budgeted"))
def test_native_jk_residency_and_budget_fallback_against_d_integrals(
    spins: int, spherical: bool, cache_mode: str, tmp_path: Path
) -> None:
    """Exercise larger Cartesian/spherical closure and the exact-budget fallback."""
    fixture = _fixture(spins, spherical, angular=2)
    values, receipt = _native(fixture, tmp_path, cache_mode=cache_mode)
    assert bool(receipt["resident_bytes"]) == (cache_mode == "resident")
    _qualify(fixture, values, "exact")


@pytest.mark.parametrize("spins,spherical", ((1, False), (2, True)))
def test_native_jk_four_distinct_centers_against_independent_integrals(
    spins: int, spherical: bool, tmp_path: Path
) -> None:
    """Symmetry reuse must permute Cartesian components and all derivative legs."""
    fixture = _fixture(spins, spherical, angular=2)
    fixture.update(
        labels=("O0", "C1", "N2", "H3"),
        numbers=np.array([8, 6, 7, 1]),
        centers=np.array(
            [[0.1, -0.2, 0.3], [1.2, 0.7, -0.4], [-0.3, 1.1, 0.8], [0.7, -0.8, 1.3]]
        ),
        shells=[
            (0, 0, [[0.8, 1.0]]),
            (1, 1, [[0.7, 1.0]]),
            (2, 2, [[0.6, 1.0]]),
            (3, 0, [[0.9, 1.0]]),
        ],
        nao=10 if spherical else 11,
    )
    random = np.random.default_rng(2718)
    panels = random.normal(scale=0.17, size=(2, spins, fixture["nao"], fixture["nao"]))
    fixture["panels"] = panels + panels.transpose(0, 1, 3, 2)
    fixture["direction"] = random.normal(scale=0.15, size=(4, 3))
    values, _ = _native(fixture, tmp_path)
    _qualify(fixture, values, "exact")


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
