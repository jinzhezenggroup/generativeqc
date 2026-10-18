"""Independent physical J/K and displaced-eigenproblem response oracles.

Python/PySCF are test oracles only. The native molecular subprocess executes
SCF, authenticated physical Fock reconstruction/action and coupled CPKS with
all interpreter and compiler discovery paths unavailable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.method.wb97mv_fock_response import fock_maps
from generativeqc_compiler.tensor import execute
from test_wb97mv_native_grid import _independent_potential, _signed_density_libxc


@pytest.mark.parametrize("spins", (1, 2))
def test_compiler_fock_maps_against_signed_density_integrals(spins: int) -> None:
    """Nonsymmetric whitening distinguishes W^T W from a guessed W W^T."""
    random = np.random.default_rng(825 + spins)
    raw = random.normal(size=(2, 2, 3))
    raw = (raw + raw.transpose(1, 0, 2)) / 2
    whitening = random.normal(size=(3, 3))
    inverse = whitening.T @ whitening
    density = np.array([[[0.3, 0.2], [0.2, -0.4]]] * spins)
    eri = np.einsum("ijp,pq,klq->ijkl", raw, inverse, raw)
    coulomb = np.einsum("ijkl,skl->ij", eri, density)
    exchange = np.einsum("ikjl,skl->sij", eri, density)
    expected = coulomb - 0.15 / (2 if spins == 1 else 1) * exchange
    programs = fock_maps()
    prefix = "rks" if spins == 1 else "uks"
    computed = execute(programs["inverse"], {"whitening": whitening}).outputs["inverse"]
    np.testing.assert_allclose(computed, inverse, atol=2e-15)
    for kind, feeds, output in (
        ("exact", {"coulomb": coulomb, "exchange": exchange}, "primary"),
        (
            "fitted",
            {
                "density": density,
                "three_center": raw,
                "inverse": inverse,
                "bar_energy": np.array(1.0),
            },
            "bar_density",
        ),
    ):
        actual = execute(programs[f"{prefix}_{kind}"], feeds).outputs[output]
        np.testing.assert_allclose(actual, expected, atol=3e-14)
    hcore = random.normal(size=(2, 2))
    grid, long_range = random.normal(size=(2, spins, 2, 2))
    assembled = execute(
        programs[f"{prefix}_assemble"],
        {"primary": expected, "grid": grid, "long_range": long_range, "hcore": hcore},
    ).outputs["fock"]
    np.testing.assert_allclose(
        assembled,
        expected + grid + hcore - 0.85 / (2 if spins == 1 else 1) * long_range,
        atol=3e-14,
    )


def test_fock_aot_generation_is_build_only(tmp_path: Path) -> None:
    """The build does not require a runtime installation, GPU or reference code."""
    root = Path(__file__).resolve().parents[2]
    script = """
import runpy, sys
class Deny:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'generativeqc', 'cupy', 'torch', 'pyscf'}:
            raise RuntimeError('forbidden build-time runtime import: ' + fullname)
sys.meta_path.insert(0, Deny())
sys.argv = ['generate_wb97mv_fock_response.py', '--output', sys.argv[1]]
runpy.run_path('tools/generate_wb97mv_fock_response.py', run_name='__main__')
"""
    outputs = [tmp_path / name for name in ("first.cuh", "second.cuh")]
    for output in outputs:
        subprocess.run(
            [sys.executable, "-c", script, str(output)],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def _native(spins: int, fitted: bool) -> dict:
    executable = os.environ.get("GENERATIVEQC_KS_WB97MV_GRID_PROBE")
    if not executable:
        pytest.skip(
            "set GENERATIVEQC_KS_WB97MV_GRID_PROBE for native physical Fock/CPKS"
        )
    path = Path(executable).resolve()
    assert path.is_file()
    process = subprocess.run(
        [str(path), "rks" if spins == 1 else "uks", "df-fock" if fitted else "fock"],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env=dict(
            os.environ,
            PATH="/nonexistent",
            PYTHONHOME="/nonexistent",
            PYTHONPATH="/nonexistent",
        ),
    )
    return json.loads(process.stdout)


def _sources(result: dict, fitted: bool) -> tuple:
    """Construct raw molecular integrals and Gaussian jets without native AD."""
    gto = pytest.importorskip("pyscf.gto")
    df = pytest.importorskip("pyscf.df")
    atoms, primitives, nao, spins = (
        result[name] for name in ("atoms", "primitives", "nao", "spins")
    )
    packed = np.array(result["basis"])
    centers = packed[: 3 * atoms].reshape(atoms, 3)
    contractions = packed[3 * atoms : 3 * atoms + 2 * primitives].reshape(primitives, 2)
    records = packed[3 * atoms + 2 * primitives :].reshape(nao, 16)
    points = np.array(result["points"]).reshape(-1, 3)
    jets = np.zeros((4, len(points), nao))
    for orbital, record in enumerate(records):
        atom, begin, count, terms = record[:4].astype(int)
        assert terms == 1 and np.all(record[4:7] == 0)
        displacement = points - centers[atom]
        for exponent, weight in contractions[begin : begin + count]:
            gaussian = (
                record[7] * weight * np.exp(-exponent * np.sum(displacement**2, axis=1))
            )
            jets[0, :, orbital] += gaussian
            jets[1:, :, orbital] -= 2 * exponent * displacement.T * gaussian
    fixture = {
        "spins": spins,
        "jets": jets,
        "points": points,
        "weights": np.array(result["weights"]),
    }
    molecule = gto.M(
        atom=[("H", position) for position in centers],
        unit="Bohr",
        basis={
            "H": [
                [
                    0,
                    [3.425250914, 0.1543289673],
                    [0.6239137298, 0.5353281423],
                    [0.168855404, 0.4446345422],
                ]
            ]
        },
        spin=spins - 1,
        verbose=0,
    )
    eri = molecule.intor("int2e", aosym="s1")
    if fitted:
        raw = df.incore.aux_e2(molecule, molecule, aosym="s1")
        metric = molecule.intor("int2c2e")
        assert np.linalg.eigvalsh(metric).min() > 1e-6
        eri = np.einsum("ijp,pq,klq->ijkl", raw, np.linalg.inv(metric), raw)
    with molecule.with_range_coulomb(0.3):
        long_range = molecule.intor("int2e", aosym="s1")
    np.testing.assert_allclose(
        np.array(result["hcore"]).reshape(nao, nao),
        molecule.intor("int1e_kin") + molecule.intor("int1e_nuc"),
        atol=3e-12,
    )
    np.testing.assert_allclose(
        np.array(result["overlap"]).reshape(nao, nao),
        molecule.intor("int1e_ovlp"),
        atol=3e-13,
    )
    return fixture, eri, long_range


def _jk(density: np.ndarray, eri: np.ndarray, long_range: np.ndarray) -> np.ndarray:
    """Exact LR is a correction to full-range exact or fitted short exchange."""
    occupation = 2 if len(density) == 1 else 1
    return (
        np.einsum("ijkl,skl->ij", eri, density)
        - 0.15 / occupation * np.einsum("ikjl,skl->sij", eri, density)
        - 0.85 / occupation * np.einsum("ikjl,skl->sij", long_range, density)
    )


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("fitted", (False, True))
def test_live_physical_fock_and_cpks_against_independent_oracles(
    spins: int, fitted: bool
) -> None:
    result = _native(spins, fitted)
    assert result["revoked_without_publication"]
    nao = result["nao"]
    shape = (spins, nao, nao)
    density, direction, fock, frozen, actual, actual_weighted = (
        np.array(result[name]).reshape(shape)
        for name in (
            "density",
            "direction",
            "fock",
            "direction",
            "cpks_density",
            "cpks_weighted",
        )
    )
    fixture, eri, long_range = _sources(result, fitted)
    hcore = np.array(result["hcore"]).reshape(nao, nao)
    overlap = np.array(result["overlap"]).reshape(nao, nao)
    pairs = [
        (spin, row, column)
        for spin in range(spins)
        for row in range(nao)
        for column in range(row, nao)
    ]

    def pack(matrix: np.ndarray) -> np.ndarray:
        return np.array([matrix[index] for index in pairs])

    def unpack(vector: np.ndarray) -> np.ndarray:
        matrix = np.zeros(shape)
        for (spin, row, column), value in zip(pairs, vector, strict=True):
            matrix[spin, row, column] = matrix[spin, column, row] = value
        return matrix

    eigen = pytest.importorskip("scipy.linalg").eigh
    occupation = 2 if spins == 1 else 1
    counts = np.rint(np.einsum("sij,ji->s", density, overlap) / occupation).astype(int)

    def projector(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        states, weighted = [], []
        for matrix, count in zip(matrices, counts, strict=True):
            energies, vectors = eigen(matrix, overlap)
            occupied = vectors[:, :count]
            states.append(occupation * occupied @ occupied.T)
            weighted.append(occupation * (occupied * energies[:count]) @ occupied.T)
        return np.array(states), np.array(weighted)

    basis = [unpack(vector) for vector in np.eye(len(pairs))]
    with _signed_density_libxc() as code:

        def potential(value: np.ndarray) -> np.ndarray:
            return _jk(value, eri, long_range) + _independent_potential(
                fixture, value, libxc_code=code
            )

        np.testing.assert_allclose(
            fock, hcore + potential(density), atol=3e-9, rtol=3e-9
        )
        for step in (1e-3, 3e-4):
            expected = (
                potential(density + step * direction)
                - potential(density - step * direction)
            ) / (2 * step)
            np.testing.assert_allclose(
                np.array(result["fock_response"]).reshape(shape),
                expected,
                atol=3e-9,
                rtol=2e-6,
            )
        errors = []
        for step in (1e-4, 3e-5):
            kernel = np.column_stack(
                [
                    pack(
                        (
                            potential(density + step * vector)
                            - potential(density - step * vector)
                        )
                        / (2 * step)
                    )
                    for vector in basis
                ]
            )
            susceptibility = np.column_stack(
                [
                    pack(
                        (
                            projector(fock + step * vector)[0]
                            - projector(fock - step * vector)[0]
                        )
                        / (2 * step)
                    )
                    for vector in basis
                ]
            )
            tangent = np.linalg.solve(
                np.eye(len(pairs)) - susceptibility @ kernel,
                susceptibility @ pack(frozen),
            )
            total_fock = frozen + unpack(kernel @ tangent)
            weighted = (
                projector(fock + step * total_fock)[1]
                - projector(fock - step * total_fock)[1]
            ) / (2 * step)
            np.testing.assert_allclose(actual, unpack(tangent), atol=3e-8, rtol=2e-6)
            np.testing.assert_allclose(actual_weighted, weighted, atol=3e-8, rtol=2e-6)
            errors.append(
                max(
                    np.max(np.abs(actual - unpack(tangent))),
                    np.max(np.abs(actual_weighted - weighted)),
                )
            )
        assert errors[1] <= max(2e-10, errors[0] * 0.4)
    assert result["capacity_alias_finite_gates"]
    assert result["fock_actions"] > 0 and result["cpks_actions"] > 0
    assert (
        result["reference_eris"] > 0
        and result["response_eris"] > result["reference_eris"]
    )
    assert 0 < result["fock_device_bytes"] <= 512 << 20
    assert (
        0
        < result["affine_action_terms"]
        <= result["affine_reference_terms"]
        <= 100_000_000
    )
