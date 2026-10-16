"""Slurm-only native residual ECP AD against independent Cartesian libcint.

The independent analytic gradient retains AO and potential-center incidences.
Displaced energies validate that oracle; displaced analytic gradients validate
moving-density HVPs. None of these reference calculations is a production path.
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


def _fixture(domain: str = "sp", spherical: bool = False) -> dict:
    centers = np.array([[0.1, -0.2, 0.3], [0.5, 0.2, 2.1]])
    atoms = [(11, 10), (1, 0)]
    radial = [[0.8, 0.8], [0.23, -0.15]]
    if domain == "sp":
        shells = [(0, 0, radial), (0, 1, [[0.45, 1]]), (1, 0, [[0.7, 1]])]
    elif domain == "df":
        shells = [(0, 2, radial), (1, 3, [[0.7, 0.8], [0.22, -0.1]])]
    else:
        assert domain == "multi"
        centers = np.vstack((centers, [-0.7, 0.6, -1.2]))
        atoms = [(11, 10), (11, 10), (1, 0)]
        shells = [(0, 0, radial), (1, 1, [[0.6, 1]]), (2, 0, [[0.7, 1]])]
    terms = [
        [0, -1, 0, 0.9, 0.25],
        [0, -1, 2, 0.4, -0.1],
        [0, 0, 1, 0.8, 0.3],
        [0, 1, 2, 0.65, -0.4],
        [0, 2, 3, 0.7, 0.12],
        [0, 3, 4, 0.55, -0.08],
    ]
    if domain == "multi":
        terms += [
            [1, channel, power, exponent * 1.2, coefficient * -0.7]
            for _, channel, power, exponent, coefficient in terms.copy()
        ]
    nao = sum(
        2 * angular + 1 if spherical else (angular + 1) * (angular + 2) // 2
        for _, angular, _ in shells
    )
    random = np.random.default_rng(731)
    density = random.normal(scale=0.03, size=(2, nao, nao))
    density += density.transpose(0, 2, 1)
    return {
        "centers": centers,
        "atoms": atoms,
        "shells": shells,
        "terms": terms,
        "spherical": spherical,
        "nao": nao,
        "density": density,
        "direction": random.normal(scale=0.15, size=centers.shape),
    }


def _write_fixture(
    fixture: dict, path: Path, *, reject: int = 0, exercise: bool = False
) -> None:
    centers = [atom for atom, (_, core) in enumerate(fixture["atoms"]) if core]
    with path.open("wb") as output:
        np.array(
            [
                len(fixture["atoms"]),
                len(fixture["shells"]),
                fixture["spherical"],
                len(fixture["terms"]),
                len(centers),
                reject,
                exercise,
            ],
            dtype=np.uint64,
        ).tofile(output)
        np.array(
            [
                [*atom, *position]
                for atom, position in zip(
                    fixture["atoms"], fixture["centers"], strict=True
                )
            ],
            dtype=np.float64,
        ).tofile(output)
        for atom, angular, radial in fixture["shells"]:
            np.array([atom, angular, len(radial)], dtype=np.uint64).tofile(output)
            np.array(radial, dtype=np.float64).tofile(output)
        np.array(fixture["terms"], dtype=np.float64).tofile(output)
        np.array(centers, dtype=np.uint32).tofile(output)
        np.asarray(fixture["direction"], dtype=np.float64).tofile(output)
        np.asarray(fixture["density"], dtype=np.float64).tofile(output)


def _native(
    fixture: dict, path: Path, *, reject: int = 0, exercise: bool = False
) -> tuple:
    executable = os.environ.get("GENERATIVEQC_CUDA_ECP_NUCLEAR_PROBE")
    if not executable:
        pytest.skip("requires native residual ECP probe and a Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    source, target = path / "source.bin", path / "result.bin"
    target.unlink(missing_ok=True)
    _write_fixture(fixture, source, reject=reject, exercise=exercise)
    completed = subprocess.run(
        [executable, str(source), str(target)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env={
            **os.environ,
            "PATH": "/nonexistent",
            "PYTHONHOME": "/nonexistent",
            "PYTHONPATH": "/nonexistent",
        },
    )
    receipt = json.loads(completed.stdout)
    if reject:
        assert receipt == {
            "prepare_rejected" if reject <= 7 else "action_rejected": True
        }
        assert not target.exists()
        return (), receipt
    assert receipt["lifecycle"]
    matrix, coordinates = fixture["nao"] ** 2, fixture["direction"].size
    values = np.fromfile(target)
    assert values.size == 4 * matrix + 4 * coordinates
    return {
        "matrix": values[: 2 * matrix].reshape(2, fixture["nao"], fixture["nao"]),
        "direction": values[2 * matrix : 4 * matrix].reshape(
            2, fixture["nao"], fixture["nao"]
        ),
        "gradient": values[4 * matrix : 4 * matrix + 2 * coordinates].reshape(
            2, *fixture["centers"].shape
        ),
        "hvp": values[4 * matrix + 2 * coordinates :].reshape(
            2, *fixture["centers"].shape
        ),
    }, receipt


class EcpOracle:
    """Libcint residual channels with independent normalization and projection."""

    def __init__(self, fixture: dict) -> None:
        from pyscf import gto, lib

        lib.num_threads(1)
        labels = [
            f"{'Na' if charge == 11 else 'H'}{index}"
            for index, (charge, _) in enumerate(fixture["atoms"])
        ]
        basis = {label: [] for label in labels}
        for atom, angular, radial in fixture["shells"]:
            basis[labels[atom]].append([angular, *radial])
        self.molecules = []
        for nonlocal_part in (False, True):
            ecp = {}
            for atom, (_, core) in enumerate(fixture["atoms"]):
                if not core:
                    continue
                channels = {}
                for center, channel, power, exponent, coefficient in fixture["terms"]:
                    if center == atom and (channel >= 0) == nonlocal_part:
                        channels.setdefault(channel, [[] for _ in range(5)])[
                            power
                        ].append([exponent, coefficient])
                ecp[labels[atom]] = [
                    core,
                    [[channel, powers] for channel, powers in sorted(channels.items())],
                ]
            self.molecules.append(
                gto.M(
                    atom=list(zip(labels, fixture["centers"], strict=True)),
                    unit="Bohr",
                    basis=basis,
                    ecp=ecp,
                    spin=len(labels) % 2,
                    cart=True,
                    verbose=0,
                )
            )
        molecule = self.molecules[0]
        overlap = molecule.intor("int1e_ovlp_cart")
        transform = (
            molecule.cart2sph_coeff(normalized="sp")
            if fixture["spherical"]
            else np.eye(molecule.nao)
        )
        self.transform = transform / np.sqrt(np.diag(transform.T @ overlap @ transform))
        self.ecp_atoms = [
            atom for atom, (_, core) in enumerate(fixture["atoms"]) if core
        ]

    def evaluate(self, centers: np.ndarray) -> tuple:
        matrices, gradients = [], []
        for template in self.molecules:
            molecule = template.set_geom_(centers, unit="Bohr", inplace=False)
            matrices.append(
                self.transform.T @ molecule.intor("ECPscalar_cart") @ self.transform
            )
            first = -molecule.intor("ECPscalar_ipnuc_cart", comp=3)
            partials = []
            for atom, (_, _, begin, end) in enumerate(molecule.aoslice_by_atom()):
                partial = np.zeros_like(first)
                if atom in self.ecp_atoms:
                    with molecule.with_rinv_at_nucleus(atom):
                        partial += molecule.intor("ECPscalar_iprinv_cart", comp=3)
                partial[:, begin:end] += first[:, begin:end]
                partial += partial.transpose(0, 2, 1).copy()
                partials.append(
                    np.einsum("ia,xij,jb->xab", self.transform, partial, self.transform)
                )
            gradients.append(partials)
        return np.array(matrices), np.array(gradients)


@pytest.mark.parametrize("domain", ("sp", "df", "multi"))
@pytest.mark.parametrize("spherical", (False, True))
def test_native_ecp_second_source_against_independent_libcint(
    domain: str, spherical: bool, tmp_path: Path
) -> None:
    fixture = _fixture(domain, spherical)
    actual, work = _native(fixture, tmp_path, exercise=domain == "sp" and not spherical)
    _assert_ecp_oracle(fixture, actual, work)
    assert work["projection_terms_per_action"] == 160 * work["ao_per_action"]
    assert (
        work["value_actions"]
        == work["contraction_actions"]
        == (2 if domain == "sp" and not spherical else 1)
    )


def _assert_ecp_oracle(fixture: dict, actual: dict, work: dict) -> None:
    """Shared independent molecular gate, not a second native-source oracle."""
    pytest.importorskip("pyscf")
    oracle = EcpOracle(fixture)
    matrix, gradient = oracle.evaluate(fixture["centers"])
    density, density_motion = fixture["density"]
    expected_gradient = np.einsum("ij,caxij->cax", density, gradient)
    np.testing.assert_allclose(actual["matrix"], matrix, atol=3e-9, rtol=3e-9)
    np.testing.assert_allclose(
        actual["direction"],
        np.einsum("ax,caxij->cij", fixture["direction"], gradient),
        atol=3e-8,
        rtol=3e-7,
    )
    np.testing.assert_allclose(
        actual["gradient"], expected_gradient, atol=3e-8, rtol=3e-7
    )
    for step in (3e-4, 1e-4):
        plus, plus_gradient = oracle.evaluate(
            fixture["centers"] + step * fixture["direction"]
        )
        minus, minus_gradient = oracle.evaluate(
            fixture["centers"] - step * fixture["direction"]
        )
        np.testing.assert_allclose(
            np.einsum("ij,cij->c", density, (plus - minus) / (2 * step)),
            np.einsum("ax,cax->c", fixture["direction"], expected_gradient),
            atol=3e-8,
            rtol=3e-6,
        )
        moving_plus = np.einsum(
            "ij,caxij->cax", density + step * density_motion, plus_gradient
        )
        moving_minus = np.einsum(
            "ij,caxij->cax", density - step * density_motion, minus_gradient
        )
        np.testing.assert_allclose(
            actual["hvp"],
            (moving_plus - moving_minus) / (2 * step),
            atol=3e-7,
            rtol=3e-6,
        )
    np.testing.assert_allclose(actual["gradient"].sum(axis=1), 0, atol=2e-10)
    np.testing.assert_allclose(actual["hvp"].sum(axis=1), 0, atol=2e-10)
    count = len(oracle.ecp_atoms)
    assert work["layers_per_action"] == count * (160 + 224)
    assert work["ao_per_action"] == count * fixture["nao"] * (160 * 2048 + 224 * 3872)


@pytest.mark.parametrize("reject", range(1, 10))
def test_native_ecp_source_and_action_admission(reject: int, tmp_path: Path) -> None:
    _native(_fixture(), tmp_path, reject=reject)


def test_native_ecp_nonconverged_quadrature_preserves_all_outputs(
    tmp_path: Path,
) -> None:
    fixture = _fixture()
    fixture["shells"][-1] = (1, 0, [[80.0, 1.0]])
    _native(fixture, tmp_path, reject=10)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("fitted", (False, True))
@pytest.mark.parametrize("spherical", (False, True))
def test_accepted_ks_ecp_source_against_independent_libcint(
    spins: int, fitted: bool, spherical: bool
) -> None:
    executable = os.environ.get("GENERATIVEQC_KS_WB97MV_GRID_PROBE")
    if not executable:
        pytest.skip("requires actual native KS source probe and a Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    mode = (
        ("df-" if fitted else "") + ("spherical-" if spherical else "") + "ecp-source"
    )
    result = json.loads(
        subprocess.run(
            [executable, "rks" if spins == 1 else "uks", mode],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
            env={
                **os.environ,
                "PATH": "/nonexistent",
                "PYTHONHOME": "/nonexistent",
                "PYTHONPATH": "/nonexistent",
            },
        ).stdout
    )
    atoms, nao = result["atoms"], result["nao"]
    radial = [
        [3.425250914, 0.1543289673],
        [0.6239137298, 0.5353281423],
        [0.168855404, 0.4446345422],
    ]
    fixture = {
        "atoms": [(11, 10), *[(1, 0)] * (atoms - 1)],
        "centers": np.array(result["basis"][: 3 * atoms]).reshape(atoms, 3),
        "shells": [
            (0, 0, radial),
            (0, 1, [[0.45, 1.0]]),
            *[(atom, 0, radial) for atom in range(1, atoms)],
        ],
        "terms": _fixture()["terms"],
        "spherical": spherical,
        "nao": nao,
        "density": np.stack(
            (
                np.array(result["density"]).reshape(spins, nao, nao).sum(axis=0),
                np.array(result["direction"]).reshape(spins, nao, nao).sum(axis=0),
            )
        ),
        "direction": np.array(result["ecp_motion"]).reshape(atoms, 3),
    }
    matrices = np.array(result["ecp_matrices"]).reshape(4, nao, nao)
    vectors = np.array(result["ecp_vectors"]).reshape(4, atoms, 3)
    actual = {
        "matrix": matrices[:2],
        "direction": matrices[2:],
        "gradient": vectors[:2],
        "hvp": vectors[2:],
    }
    _assert_ecp_oracle(
        fixture,
        actual,
        {
            "layers_per_action": result["ecp_layers_per_action"],
            "ao_per_action": result["ecp_ao_per_action"],
        },
    )
    assert result["ecp_value_actions"] == result["ecp_contraction_actions"] == 2
    assert result["ecp_spin_sum_terms"] == 4 * spins * nao**2
    assert result["ecp_device_bytes"] > result["ecp_workspace_bytes"] > 0
    assert (
        result["ecp_ordinary_owner_preserved"] and result["revoked_without_publication"]
    )
