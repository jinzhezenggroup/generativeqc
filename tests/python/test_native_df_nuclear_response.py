"""Slurm-only raw DF nuclear AD against independent normalized libcint sources.

Both orbital and auxiliary representations are independently transformed from
libcint Cartesian values/gradients. Finite differences validate the independent
gradient and its moving-weight JVP, never serve as a production implementation.
"""

from __future__ import annotations

import json
import os
import subprocess
from itertools import product
from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
from test_native_wb97mv_df_response import _split
from test_wb97mv_df_response_compiler import df_oracle


def _fixture(
    domain: str = "sp",
    orbital_spherical: bool = False,
    auxiliary_spherical: bool = False,
) -> dict:
    random = np.random.default_rng(2167)
    radial = [[0.8, 0.7], [0.23, -0.15], [0.09, 0.3]]
    if domain == "sp":
        orbital = [(0, 0, radial), (1, 1, [[0.9, 0.6], [0.31, 0.25]])]
        auxiliary = [(0, 0, [[0.6, 0.8], [0.19, -0.2]]), (1, 1, [[0.75, 1.0]])]
    elif domain == "df":
        orbital = [(0, 2, radial), (1, 3, [[0.7, 0.8], [0.22, -0.1]])]
        auxiliary = [(0, 0, [[0.6, 0.8], [0.19, -0.2]])]
    else:
        assert domain in ("g", "fg")
        orbital_order = 0 if domain == "g" else 3
        orbital = [
            (0, orbital_order, radial),
            (1, orbital_order, [[0.7, 0.8], [0.22, -0.1]]),
        ]
        auxiliary = [(0, 4, [[0.6, 0.8], [0.19, -0.2]]), (1, 4, [[0.75, 1.0]])]

    def size(shells: list, spherical: bool) -> int:
        return sum(
            2 * angular + 1 if spherical else (angular + 1) * (angular + 2) // 2
            for _, angular, _ in shells
        )

    nao, naux = size(orbital, orbital_spherical), size(auxiliary, auxiliary_spherical)
    weight = random.normal(scale=0.03, size=(2, nao, nao, naux))
    weight += weight.transpose(0, 2, 1, 3)
    metric_weight = random.normal(scale=0.03, size=(2, naux, naux))
    metric_weight += metric_weight.transpose(0, 2, 1)
    return {
        "centers": np.array([[0.1, -0.2, 0.3], [1.2, 0.7, -0.4]]),
        "orbital": orbital,
        "auxiliary": auxiliary,
        "orbital_spherical": orbital_spherical,
        "auxiliary_spherical": auxiliary_spherical,
        "nao": nao,
        "naux": naux,
        "weight": weight,
        "metric_weight": metric_weight,
        "direction": random.normal(scale=0.15, size=(2, 3)),
    }


def _write_fixture(
    fixture: dict, path: Path, *, tile: int = 64, reject: int = 0
) -> None:
    with path.open("wb") as output:
        np.array(
            [
                len(fixture["centers"]),
                len(fixture["orbital"]),
                len(fixture["auxiliary"]),
                fixture["orbital_spherical"],
                fixture["auxiliary_spherical"],
                tile,
                reject,
            ],
            dtype=np.uint64,
        ).tofile(output)
        fixture["centers"].astype(np.float64).tofile(output)
        for role in ("orbital", "auxiliary"):
            for atom, angular, radial in fixture[role]:
                np.array([atom, angular, len(radial)], dtype=np.uint64).tofile(output)
                np.asarray(radial, dtype=np.float64).tofile(output)
        fixture["direction"].astype(np.float64).tofile(output)
        for index in range(2):
            fixture["weight"][index].astype(np.float64).tofile(output)
            fixture["metric_weight"][index].astype(np.float64).tofile(output)


def _native(fixture: dict, path: Path, *, tile: int = 64, reject: int = 0) -> tuple:
    executable = os.environ.get("GENERATIVEQC_CUDA_DF_NUCLEAR_PROBE")
    if not executable:
        pytest.skip("requires native raw DF probe and an allocated Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    source, result = path / "source.bin", path / "result.bin"
    result.unlink(missing_ok=True)
    _write_fixture(fixture, source, tile=tile, reject=reject)
    completed = subprocess.run(
        [executable, str(source), str(result)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "PATH": "/nonexistent",
            "PYTHONHOME": "/nonexistent",
            "PYTHONPATH": "/nonexistent",
        },
    )
    receipt = json.loads(completed.stdout)
    if reject:
        assert receipt == {"prepare_rejected": True} and not result.exists()
        return (), receipt
    assert receipt["lifecycle"]
    raw, metric = fixture["nao"] ** 2 * fixture["naux"], fixture["naux"] ** 2
    values = np.fromfile(result)
    assert values.size == 2 * (raw + metric) + 6 * len(fixture["centers"])
    result = {}
    offset = 0
    for name, shape in (
        ("three_center", (fixture["nao"], fixture["nao"], fixture["naux"])),
        ("metric", (fixture["naux"], fixture["naux"])),
        ("direction_three_center", (fixture["nao"], fixture["nao"], fixture["naux"])),
        ("direction_metric", (fixture["naux"], fixture["naux"])),
        ("gradient", fixture["centers"].shape),
        ("hvp", fixture["centers"].shape),
    ):
        count = int(np.prod(shape))
        result[name] = values[offset : offset + count].reshape(shape)
        offset += count
    return result, receipt


class DfOracle:
    """Independent libcint physical centers and normalized public projections."""

    def __init__(self, fixture: dict) -> None:
        from pyscf import df, gto, lib

        lib.num_threads(1)
        self.df = df
        self.centers = fixture["centers"]
        labels = tuple(f"H{index}" for index in range(len(self.centers)))
        self.molecules, self.transforms, self.owners = [], [], []
        for role in ("orbital", "auxiliary"):
            basis = {label: [] for label in labels}
            for atom, angular, radial in fixture[role]:
                basis[labels[atom]].append([angular, *radial])
            molecule = gto.M(
                atom=list(zip(labels, self.centers, strict=True)),
                basis={label: shells for label, shells in basis.items() if shells},
                unit="Bohr",
                spin=len(self.centers) % 2,
                cart=True,
                verbose=0,
            )
            overlap = molecule.intor("int1e_ovlp_cart")
            transform = (
                molecule.cart2sph_coeff(normalized="sp")
                if fixture[f"{role}_spherical"]
                else np.eye(molecule.nao)
            )
            transform = transform / np.sqrt(np.diag(transform.T @ overlap @ transform))
            self.molecules.append(molecule)
            self.transforms.append(transform)
            self.owners.append(
                np.repeat(
                    [molecule.bas_atom(shell) for shell in range(molecule.nbas)],
                    np.diff(molecule.ao_loc_nr(cart=True)),
                )
            )

    def evaluate(self, centers: np.ndarray, *, gradients: bool = True) -> dict:
        """Transform independent Cartesian raw values and first center partials."""
        orbital, auxiliary = [
            molecule.set_geom_(centers, unit="Bohr", inplace=False)
            for molecule in self.molecules
        ]
        orbital_transform, auxiliary_transform = self.transforms
        raw = self.df.incore.aux_e2(
            orbital, auxiliary, intor="int3c2e_cart", aosym="s1"
        )
        metric = auxiliary.intor("int2c2e_cart")

        def transform(value: np.ndarray) -> np.ndarray:
            return np.einsum(
                "ia,jb,pc,...ijp->...abc",
                orbital_transform,
                orbital_transform,
                auxiliary_transform,
                value,
                optimize=True,
            )

        result = {
            "three_center": transform(raw),
            "metric": auxiliary_transform.T @ metric @ auxiliary_transform,
        }
        if gradients:
            first = self.df.incore.aux_e2(
                orbital, auxiliary, intor="int3c2e_ip1_cart", aosym="s1", comp=3
            )
            third = self.df.incore.aux_e2(
                orbital, auxiliary, intor="int3c2e_ip2_cart", aosym="s1", comp=3
            )
            metric_first = auxiliary.intor("int2c2e_ip1_cart", comp=3)
            raw_gradient, metric_gradient = [], []
            for atom in range(len(centers)):
                orbital_mask, auxiliary_mask = [
                    owners == atom for owners in self.owners
                ]
                motion = (
                    -first * orbital_mask[None, :, None, None]
                    - first.swapaxes(1, 2) * orbital_mask[None, None, :, None]
                    - third * auxiliary_mask[None, None, None, :]
                )
                metric_motion = (
                    -metric_first * auxiliary_mask[None, :, None]
                    - metric_first.swapaxes(1, 2) * auxiliary_mask[None, None, :]
                )
                raw_gradient.append(transform(motion))
                metric_gradient.append(
                    np.einsum(
                        "pa,xpq,qb->xab",
                        auxiliary_transform,
                        metric_motion,
                        auxiliary_transform,
                    )
                )
            result["gradient_three_center"] = np.array(raw_gradient)
            result["gradient_metric"] = np.array(metric_gradient)
        return result


@pytest.mark.parametrize("domain", ("sp", "df", "g"))
@pytest.mark.parametrize(
    "orbital_spherical,auxiliary_spherical",
    ((False, False), (False, True), (True, False), (True, True)),
)
@pytest.mark.parametrize("tile", (1, 7, 64))
def test_raw_df_two_pass_source_against_independent_oracle(
    domain: str,
    orbital_spherical: bool,
    auxiliary_spherical: bool,
    tile: int,
    tmp_path: Path,
) -> None:
    _qualify_raw_df_source(
        domain, orbital_spherical, auxiliary_spherical, tile, tmp_path
    )


@pytest.mark.parametrize(
    "orbital_spherical,auxiliary_spherical",
    ((False, False), (False, True), (True, False), (True, True)),
)
def test_joint_f_orbital_g_auxiliary_two_pass_source(
    orbital_spherical: bool, auxiliary_spherical: bool, tmp_path: Path
) -> None:
    """Exercise the full high-angular contraction and both independent projections."""
    _qualify_raw_df_source("fg", orbital_spherical, auxiliary_spherical, 64, tmp_path)


def _qualify_raw_df_source(
    domain: str,
    orbital_spherical: bool,
    auxiliary_spherical: bool,
    tile: int,
    tmp_path: Path,
) -> None:
    """Share the independent source, fitted-response and exact semantic-work gates."""
    fixture = _fixture(domain, orbital_spherical, auxiliary_spherical)
    actual, receipt = _native(fixture, tmp_path, tile=tile)
    oracle = DfOracle(fixture)
    reference = oracle.evaluate(fixture["centers"])
    for field in ("three_center", "metric"):
        np.testing.assert_allclose(
            actual[field], reference[field], atol=3e-10, rtol=3e-10
        )
        expected_direction = np.einsum(
            "ax,ax...->...", fixture["direction"], reference[f"gradient_{field}"]
        )
        np.testing.assert_allclose(
            actual[f"direction_{field}"], expected_direction, atol=8e-10, rtol=5e-9
        )

    def weighted_gradient(centers: np.ndarray, displacement: float) -> np.ndarray:
        values = oracle.evaluate(centers)
        return np.einsum(
            "ijp,axijp->ax",
            fixture["weight"][0] + displacement * fixture["weight"][1],
            values["gradient_three_center"],
        ) + np.einsum(
            "pq,axpq->ax",
            fixture["metric_weight"][0] + displacement * fixture["metric_weight"][1],
            values["gradient_metric"],
        )

    np.testing.assert_allclose(
        actual["gradient"],
        weighted_gradient(fixture["centers"], 0),
        atol=2e-9,
        rtol=5e-9,
    )

    def weighted_energy(centers: np.ndarray) -> float:
        values = oracle.evaluate(centers, gradients=False)
        return float(
            np.sum(fixture["weight"][0] * values["three_center"])
            + np.sum(fixture["metric_weight"][0] * values["metric"])
        )

    for step in (3e-4, 1e-4):
        plus_centers = fixture["centers"] + step * fixture["direction"]
        minus_centers = fixture["centers"] - step * fixture["direction"]
        np.testing.assert_allclose(
            (weighted_energy(plus_centers) - weighted_energy(minus_centers))
            / (2 * step),
            np.sum(weighted_gradient(fixture["centers"], 0) * fixture["direction"]),
            atol=2e-8,
            rtol=3e-6,
        )
        plus = weighted_gradient(plus_centers, step)
        minus = weighted_gradient(minus_centers, -step)
        np.testing.assert_allclose(
            actual["hvp"], (plus - minus) / (2 * step), atol=2e-7, rtol=3e-6
        )
    for name in ("gradient", "hvp"):
        np.testing.assert_allclose(actual[name].sum(axis=0), 0, atol=2e-9)

    def primitive_components(shells: list) -> int:
        return sum(
            len(radial) * (angular + 1) * (angular + 2) // 2
            for _, angular, radial in shells
        )

    orbital_work = primitive_components(fixture["orbital"])
    auxiliary_work = primitive_components(fixture["auxiliary"])
    three_center_work = orbital_work**2 * auxiliary_work
    metric_work = auxiliary_work**2
    assert receipt["components_per_pass"] == three_center_work + metric_work
    assert (
        receipt["groups"]
        == len(fixture["orbital"]) ** 2 * len(fixture["auxiliary"])
        + len(fixture["auxiliary"]) ** 2
    )
    assert receipt["reduction_terms_bound"] >= 19 * three_center_work + 13 * metric_work
    groups = [
        *product(fixture["orbital"], fixture["orbital"], fixture["auxiliary"]),
        *product(fixture["auxiliary"], repeat=2),
    ]
    # A linked leaf carries at most 16 public ABI components. More angular
    # components necessarily submit additional radial records, not new physics.
    minimum_records = sum(
        prod(len(radial) for _, _, radial in group)
        * (
            (prod((angular + 1) * (angular + 2) // 2 for _, angular, _ in group) + 15)
            // 16
        )
        for group in groups
    )
    assert receipt["records_per_pass"] >= minimum_records
    assert receipt["records_per_pass"] <= receipt["components_per_pass"]
    assert receipt["projection_terms_bound"] > 0


@pytest.mark.parametrize("reject", (1, 2, 3, 4))
def test_raw_df_reference_and_capacity_fail_closed(reject: int, tmp_path: Path) -> None:
    _native(_fixture(), tmp_path, reject=reject)


@pytest.mark.parametrize("spins", (1, 2))
def test_actual_ks_raw_df_inverse_and_moving_weight_chain(spins: int) -> None:
    """Compose actual fitted-primary owners, not a stationary molecular Hessian."""
    executable = os.environ.get("GENERATIVEQC_KS_WB97MV_GRID_PROBE")
    if not executable:
        pytest.skip("requires actual native KS probe and an allocated Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    pytest.importorskip("pyscf")
    completed = subprocess.run(
        [executable, "rks" if spins == 1 else "uks", "df-nuclear-source"],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
        env={
            **os.environ,
            "PATH": "/nonexistent",
            "PYTHONHOME": "/nonexistent",
            "PYTHONPATH": "/nonexistent",
        },
    )
    result = json.loads(completed.stdout)
    assert result["revoked_without_publication"]
    centers = np.array(result["basis"][: 3 * result["atoms"]]).reshape(-1, 3)
    shells = [
        (
            atom,
            0,
            [
                [3.425250914, 0.1543289673],
                [0.6239137298, 0.5353281423],
                [0.168855404, 0.4446345422],
            ],
        )
        for atom in range(len(centers))
    ]
    oracle = DfOracle(
        {
            "centers": centers,
            "orbital": shells,
            "auxiliary": shells,
            "orbital_spherical": False,
            "auxiliary_spherical": False,
        }
    )
    reference = oracle.evaluate(centers)
    nao, naux = result["nao"], result["df_naux"]
    density = np.array(result["density"]).reshape(spins, nao, nao)
    density_motion = np.array(result["direction"]).reshape(density.shape)
    motion = np.array(result["df_nuclear_motion"]).reshape(centers.shape)
    actual = _split(np.array(result["df_values"]), spins, nao, naux)
    expected = df_oracle(density, reference["three_center"], reference["metric"])
    for field in ("three_center", "metric"):
        np.testing.assert_allclose(
            np.array(
                result["df_raw" if field == "three_center" else "df_metric"]
            ).reshape(reference[field].shape),
            reference[field],
            atol=3e-10,
            rtol=3e-10,
        )
        np.testing.assert_allclose(
            np.array(
                result["df_delta_raw" if field == "three_center" else "df_delta_metric"]
            ).reshape(reference[field].shape),
            np.einsum("ax,ax...->...", motion, reference[f"gradient_{field}"]),
            atol=8e-10,
            rtol=5e-9,
        )
    for field, value in expected.items():
        np.testing.assert_allclose(actual[field], value, atol=3e-10, rtol=3e-9)

    def gradient(values: dict, algebra: dict) -> np.ndarray:
        return np.einsum(
            "ijp,axijp->ax",
            algebra["weight_three_center"],
            values["gradient_three_center"],
        ) + np.einsum(
            "pq,axpq->ax", algebra["weight_metric"], values["gradient_metric"]
        )

    vectors = np.array(result["df_nuclear_vectors"]).reshape(2, *centers.shape)
    np.testing.assert_allclose(
        vectors[0], gradient(reference, expected), atol=2e-9, rtol=5e-9
    )
    for step in (3e-4, 1e-4):
        plus, minus = (
            oracle.evaluate(centers + step * motion),
            oracle.evaluate(centers - step * motion),
        )
        plus_fixed = df_oracle(density, plus["three_center"], plus["metric"])
        minus_fixed = df_oracle(density, minus["three_center"], minus["metric"])
        np.testing.assert_allclose(
            (plus_fixed["energy"] - minus_fixed["energy"]) / (2 * step),
            np.sum(vectors[0] * motion),
            atol=2e-8,
            rtol=3e-6,
        )
        np.testing.assert_allclose(
            actual["fock_direction"],
            (plus_fixed["fock"] - minus_fixed["fock"]) / (2 * step),
            atol=2e-8,
            rtol=3e-6,
        )
        plus_algebra = df_oracle(
            density + step * density_motion, plus["three_center"], plus["metric"]
        )
        minus_algebra = df_oracle(
            density - step * density_motion, minus["three_center"], minus["metric"]
        )
        np.testing.assert_allclose(
            vectors[1],
            (gradient(plus, plus_algebra) - gradient(minus, minus_algebra))
            / (2 * step),
            atol=2e-7,
            rtol=3e-6,
        )
    np.testing.assert_allclose(vectors.sum(axis=1), 0, atol=2e-9)
    assert (
        result["df_nuclear_value_passes"]
        == result["df_nuclear_contraction_passes"]
        == 2
    )
