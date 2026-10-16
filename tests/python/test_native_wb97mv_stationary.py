"""Reconverged independent molecular energies gate the native stationary owner.

Libcint, upstream Libxc and the direct-space VV10 reference supply values only;
no compiler derivatives, CPKS equations or native response enter this oracle.
Python is a test driver/oracle, never loaded by the native numerical process.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_native_grid_geometry import _partition
from test_wb97mv_native_fock import _jk
from test_wb97mv_native_grid import (
    _fields,
    _independent_potential,
    _signed_density_libxc,
)
from test_wb97mv_nuclear_grid import _energy


def _native(
    spins: int,
    fitted: bool,
    spherical: bool,
    ecp: bool,
    full: bool,
    *,
    auxiliary_d: bool = False,
    benchmark_h2: bool = False,
) -> dict:
    executable = os.environ.get("GENERATIVEQC_KS_WB97MV_HESSIAN_PROBE")
    if not executable:
        pytest.skip("requires native stationary probe and Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require finite srun"
    mode = "df" if fitted else "direct"
    mode += "-spherical" if spherical else ""
    mode += "-ecp" if ecp else ""
    mode += "-auxd" if auxiliary_d else ""
    mode += "-benchmark-h2" if benchmark_h2 else ""
    result = subprocess.run(
        [executable, "rks" if spins == 1 else "uks", mode, *(["full"] if full else [])],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        env=dict(
            os.environ,
            PATH="/nonexistent",
            PYTHONHOME="/nonexistent",
            PYTHONPATH="/nonexistent",
        ),
    )
    assert result.returncode == 0, result.stderr
    if directory := os.environ.get("GENERATIVEQC_STATIONARY_RECEIPTS"):
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"stationary-{spins}-{mode}.json").write_text(result.stdout)
    return json.loads(result.stdout)


class MolecularEnergy:
    """Values-only spin-coupled generalized-eigensolve at each moved geometry."""

    def __init__(
        self, result: dict, *, fitted: bool, spherical: bool, ecp: bool, code: str
    ) -> None:
        from pyscf import gto, lib

        lib.num_threads(1)
        self.spins = result["spins"]
        self.nao = result["nao"]
        self.centers = np.array(result["basis"][: 3 * result["atoms"]]).reshape(-1, 3)
        self.points = np.array(result["points"]).reshape(-1, 3)
        self.atomic = np.array(result["atomic"])
        self.owner = np.repeat(
            np.arange(len(self.centers)), len(self.points) // len(self.centers)
        )
        self.seed = np.array(result["density"]).reshape(self.spins, self.nao, self.nao)
        self.fitted, self.code = fitted, code
        labels = [
            f"{'Na' if ecp and atom == 0 else 'H'}{atom}"
            for atom in range(len(self.centers))
        ]
        radial = [
            [3.425250914, 0.1543289673],
            [0.6239137298, 0.5353281423],
            [0.168855404, 0.4446345422],
        ]
        basis = {label: [[0, *radial]] for label in labels}
        auxiliary_d = result.get("auxiliary_d", False)
        if auxiliary_d:
            basis = {
                label: [[0, [0.7 if atom == 0 else 0.9, 1.0]]]
                for atom, label in enumerate(labels)
            }
        potentials = {}
        if ecp:
            from test_native_ecp_nuclear_response import _fixture

            if not auxiliary_d:
                basis[labels[0]].append([1, [0.45, 1.0]])
            terms = (
                ((0, -1, 2, 0.8, -0.4), (0, 0, 2, 0.8, 0.3))
                if auxiliary_d
                else _fixture()["terms"]
            )
            channels = {}
            for _, channel, power, exponent, coefficient in terms:
                channels.setdefault(channel, [[] for _ in range(5)])[power].append(
                    [exponent, coefficient]
                )
            potentials[labels[0]] = [
                10,
                [[channel, powers] for channel, powers in sorted(channels.items())],
            ]
        self.template = gto.M(
            atom=list(zip(labels, self.centers, strict=True)),
            unit="Bohr",
            basis=basis,
            ecp=potentials,
            spin=self.spins - 1,
            cart=not spherical,
            verbose=0,
        )
        self.auxiliary = self.template
        if auxiliary_d:
            auxiliary_basis = {
                label: [[0, [0.8 if atom == 0 else 0.65, 1.0]]]
                for atom, label in enumerate(labels)
            }
            auxiliary_basis[labels[0]].append([2, [0.6, 1.0]])
            self.auxiliary = gto.M(
                atom=list(zip(labels, self.centers, strict=True)),
                unit="Bohr",
                basis=auxiliary_basis,
                spin=self.spins - 1,
                cart=not spherical,
                verbose=0,
            )
        jets = self.template.eval_gto(
            "GTOval_sph_deriv1" if spherical else "GTOval_cart_deriv1", self.points
        )
        self.active = _fields(jets, self.seed).sum(axis=0)[0] >= 1e-8
        self.reference = result

    def __call__(self, displacement: np.ndarray, *, reference: bool = False) -> float:
        from pyscf import df
        from scipy.linalg import eigh
        from scipy.optimize import root

        centers = self.centers + displacement
        points = self.points + displacement[self.owner]
        molecule = self.template.set_geom_(centers, unit="Bohr", inplace=False)
        weights = np.array(
            _partition(
                {
                    "centers": centers,
                    "points": points,
                    "owner": self.owner,
                    "left": np.zeros_like(centers),
                    "right": np.zeros_like(centers),
                    "atomic": self.atomic,
                    "iterations": 3,
                }
            ),
            dtype=np.float64,
        )
        jets = molecule.eval_gto(
            "GTOval_cart_deriv1" if molecule.cart else "GTOval_sph_deriv1", points
        )
        fixture = {
            "jets": jets,
            "points": points,
            "weights": weights,
            "spins": self.spins,
            "libxc_code": self.code,
            "nonlocal_active": self.active,
        }
        overlap = molecule.intor("int1e_ovlp")
        core = molecule.intor("int1e_kin") + molecule.intor("int1e_nuc")
        if molecule.has_ecp():
            core += molecule.intor("ECPscalar")
        eri = molecule.intor("int2e", aosym="s1")
        if self.fitted:
            auxiliary = self.auxiliary.set_geom_(centers, unit="Bohr", inplace=False)
            raw = df.incore.aux_e2(molecule, auxiliary, aosym="s1")
            metric = auxiliary.intor("int2c2e")
            eri = np.einsum("ijp,pq,klq->ijkl", raw, np.linalg.inv(metric), raw)
            if reference and "raw" in self.reference:
                naux = self.reference["naux"]
                native_raw = np.array(self.reference["raw"]).reshape(
                    self.nao, self.nao, naux
                )
                whitening = np.array(self.reference["whitening"]).reshape(naux, naux)
                np.testing.assert_allclose(
                    np.einsum(
                        "ijp,pq,klq->ijkl",
                        native_raw,
                        whitening.T @ whitening,
                        native_raw,
                    ),
                    eri,
                    atol=3e-10,
                )
        with molecule.with_range_coulomb(0.3):
            long_range = molecule.intor("int2e", aosym="s1")
        occupation = 2 if self.spins == 1 else 1
        occupied = (molecule.nelectron // 2,) if self.spins == 1 else molecule.nelec
        rows, columns = np.triu_indices(self.nao)

        def pack(density: np.ndarray) -> np.ndarray:
            return density[:, rows, columns].reshape(-1)

        def unpack(values: np.ndarray) -> np.ndarray:
            values = values.reshape(self.spins, -1)
            density = np.zeros_like(self.seed)
            density[:, rows, columns] = values
            density[:, columns, rows] = values
            return density

        def fock(density: np.ndarray) -> np.ndarray:
            return (
                core
                + _jk(density, eri, long_range)
                + _independent_potential(
                    fixture, density, nonlocal_active=self.active, libxc_code=self.code
                )
            )

        def residual(values: np.ndarray) -> np.ndarray:
            density = unpack(values)
            projected = []
            for matrix, count in zip(fock(density), occupied, strict=True):
                vectors = eigh(matrix, overlap)[1][:, :count]
                projected.append(occupation * vectors @ vectors.T)
            return pack(np.array(projected) - density)

        solution = root(residual, pack(self.seed), tol=1e-10)
        assert np.linalg.norm(residual(solution.x)) < 2e-10, solution.message
        density = unpack(solution.x)
        if reference:
            np.testing.assert_allclose(
                core,
                np.array(self.reference["hcore"]).reshape(self.nao, self.nao),
                atol=4e-9,
            )
            np.testing.assert_allclose(
                overlap,
                np.array(self.reference["overlap"]).reshape(self.nao, self.nao),
                atol=3e-12,
            )
            np.testing.assert_allclose(
                fock(density),
                np.array(self.reference["fock"]).reshape(density.shape),
                atol=4e-8,
            )
            np.testing.assert_allclose(density, self.seed, atol=4e-8)
        return float(
            np.einsum("sij,ji->", density, core)
            + 0.5 * np.einsum("sij,sji", density, _jk(density, eri, long_range))
            + _energy(fixture, density)
            + molecule.energy_nuc()
        )


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("fitted", (False, True))
@pytest.mark.parametrize(
    "spherical,ecp", ((False, False), (True, False), (False, True), (True, True))
)
def test_stationary_hvp_against_reconverged_independent_energies(
    spins: int, fitted: bool, spherical: bool, ecp: bool
) -> None:
    pytest.importorskip("pyscf")
    full = spins == 1 and not spherical and not ecp
    result = _native(spins, fitted, spherical, ecp, full)
    _qualify(result, fitted=fitted, spherical=spherical, ecp=ecp, full=full)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("ecp", (False, True))
def test_independent_spherical_d_auxiliary_stationary_hvp(
    spins: int, ecp: bool
) -> None:
    """A separate d fit must not silently reuse the s-only orbital source."""
    pytest.importorskip("pyscf")
    result = _native(spins, True, True, ecp, False, auxiliary_d=True)
    _qualify(result, fitted=True, spherical=True, ecp=ecp, full=False)


def test_benchmark_h2_full_hessian_against_independent_molecular_energies() -> None:
    """Qualify the actual external-comparison geometry and grid, not a proxy."""
    pytest.importorskip("pyscf")
    result = _native(1, False, True, False, True, benchmark_h2=True)
    actual = np.array(result["hessian"]).reshape(6, 6)
    axes = np.eye(6).reshape(6, 2, 3)
    with _signed_density_libxc() as code:
        energy = MolecularEnergy(
            result, fitted=False, spherical=True, ecp=False, code=code
        )
        np.testing.assert_allclose(
            energy(np.zeros((2, 3)), reference=True), result["energy"], atol=5e-9
        )
        for row in range(6):
            for column in range(row, 6):
                estimates = []
                for step in (0.004, 0.002):
                    estimates.append(
                        sum(
                            first
                            * second
                            * energy(step * (first * axes[row] + second * axes[column]))
                            for first in (1, -1)
                            for second in (1, -1)
                        )
                        / (4 * step**2)
                    )
                expected = (4 * estimates[1] - estimates[0]) / 3
                np.testing.assert_allclose(actual[row, column], expected, atol=8e-7)
                np.testing.assert_allclose(actual[column, row], expected, atol=8e-7)


def _qualify(
    result: dict, *, fitted: bool, spherical: bool, ecp: bool, full: bool
) -> None:
    """Check accepted primal state before independently displaced energy jets."""
    assert result["lifecycle"] and result["directions"] >= 1
    motion = np.array(result["motion"]).reshape(-1, 3)
    actual = np.array(result["hvp"]).reshape(-1, 3)
    np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-8)
    if full:
        hessian = np.array(result["hessian"]).reshape(motion.size, motion.size)
        np.testing.assert_allclose(hessian, hessian.T, atol=2e-7)
        np.testing.assert_allclose(
            hessian @ motion.reshape(-1), actual.reshape(-1), atol=2e-8
        )
    with _signed_density_libxc() as code:
        energy = MolecularEnergy(
            result, fitted=fitted, spherical=spherical, ecp=ecp, code=code
        )
        np.testing.assert_allclose(
            energy(np.zeros_like(motion), reference=True), result["energy"], atol=5e-9
        )
        estimates = []
        for step in (0.004, 0.002):
            estimate = np.zeros_like(motion)
            for coordinate in range(motion.size):
                left = np.eye(motion.size)[coordinate].reshape(motion.shape)
                estimate.reshape(-1)[coordinate] = sum(
                    sign_left
                    * sign_right
                    * energy(step * (sign_left * left + sign_right * motion))
                    for sign_left in (1, -1)
                    for sign_right in (1, -1)
                ) / (4 * step**2)
            estimates.append(estimate)
        np.testing.assert_allclose(
            actual, (4 * estimates[1] - estimates[0]) / 3, atol=8e-7, rtol=3e-5
        )
