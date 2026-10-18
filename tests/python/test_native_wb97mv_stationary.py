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
    auxiliary_g: bool = False,
    benchmark_h2: bool = False,
    benchmark_water: bool = False,
    orbital_angular: int = 0,
    streamed: bool = False,
    unbatched: bool = False,
    batch_chunked: bool = False,
) -> dict:
    executable = os.environ.get("GENERATIVEQC_KS_WB97MV_HESSIAN_PROBE")
    if not executable:
        pytest.skip("requires native stationary probe and Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require finite srun"
    mode = "df" if fitted else "direct"
    mode += "-spherical" if spherical else ""
    mode += "-ecp" if ecp else ""
    mode += "-auxd" if auxiliary_d else ""
    mode += "-auxg" if auxiliary_g else ""
    mode += "-benchmark-h2" if benchmark_h2 else ""
    mode += "-benchmark-water" if benchmark_water else ""
    mode += "-orbd" if orbital_angular == 2 else "-orbf" if orbital_angular == 3 else ""
    mode += "-streamed" if streamed else ""
    mode += "-unbatched" if unbatched else ""
    mode += "-batch-chunked" if batch_chunked else ""
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
        suffix = "-full" if full else ""
        (destination / f"stationary-{spins}-{mode}{suffix}.json").write_text(
            result.stdout
        )
    return json.loads(result.stdout)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("fitted", (False, True))
def test_stationary_resident_and_streamed_hessians_agree(
    spins: int, fitted: bool
) -> None:
    """Optional integral storage must not change the coupled molecular endpoint."""
    resident = _native(spins, fitted, True, fitted, True)
    streamed = _native(spins, fitted, True, fitted, True, streamed=True)
    assert resident["resident_integral_bytes"] > 0
    assert streamed["resident_integral_bytes"] == 0
    for field in ("energy", "hvp", "hessian", "response_residual"):
        np.testing.assert_allclose(
            resident[field], streamed[field], atol=3e-12, rtol=3e-12
        )


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("fitted", (False, True))
def test_stationary_batched_and_small_tile_hessians_agree(
    spins: int, fitted: bool
) -> None:
    """Full coupled endpoints preserve spin, DF, ECP and spherical contracts."""
    batched = _native(spins, fitted, True, fitted, True)
    ordered = _native(spins, fitted, True, fitted, True, unbatched=True)
    assert batched["batched_workspace_bytes"] > 0
    assert ordered["batched_workspace_bytes"] == 0
    for field in ("energy", "hvp", "hessian", "response_residual"):
        np.testing.assert_allclose(
            batched[field], ordered[field], atol=3e-12, rtol=3e-12
        )


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("fitted", (False, True))
def test_stationary_projected_tiles_preserve_full_hessians(
    spins: int, fitted: bool
) -> None:
    """Reusing projected storage preserves full coupled DF/ECP/spin endpoints."""
    batched = _native(spins, fitted, True, fitted, True)
    tiled = _native(spins, fitted, True, fitted, True, batch_chunked=True)
    assert 0 < tiled["batched_workspace_bytes"] < batched["batched_workspace_bytes"]
    for field in ("energy", "hvp", "hessian", "response_residual"):
        np.testing.assert_allclose(batched[field], tiled[field], atol=3e-12, rtol=3e-12)


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
        self.benchmark_water = result.get("benchmark_water", False)
        labels = [
            f"{'Na' if ecp and atom == 0 else 'O' if self.benchmark_water and atom == 0 else 'H'}{atom}"
            for atom in range(len(self.centers))
        ]
        radial = [
            [3.425250914, 0.1543289673],
            [0.6239137298, 0.5353281423],
            [0.168855404, 0.4446345422],
        ]
        basis = {label: [[0, *radial]] for label in labels}
        if self.benchmark_water:
            records = json.loads(
                (
                    Path(__file__).resolve().parents[1]
                    / "data/external_basis/sto-3g-ho.json"
                ).read_text()
            )
            basis[labels[0]] = [
                [
                    angular,
                    *[
                        [float(exponent), float(coefficient)]
                        for exponent, coefficient in zip(
                            shell["exponents"], coefficients, strict=True
                        )
                    ],
                ]
                for shell in records["elements"]["8"]["electron_shells"]
                for angular, coefficients in zip(
                    shell["angular_momentum"], shell["coefficients"], strict=True
                )
            ]
        auxiliary_order = (
            4
            if result.get("auxiliary_g", False)
            else 2
            if result.get("auxiliary_d", False)
            else 0
        )
        self.orbital_angular = result.get("orbital_angular", 0)
        compact_fixture = auxiliary_order or self.orbital_angular
        if compact_fixture:
            basis = {
                label: [[0, [0.7 if atom == 0 else 0.9, 1.0]]]
                for atom, label in enumerate(labels)
            }
        if self.orbital_angular:
            basis[labels[0]].append([self.orbital_angular, [0.48, 1.0]])
        potentials = {}
        if ecp:
            from test_native_ecp_nuclear_response import _fixture

            if not compact_fixture:
                basis[labels[0]].append([1, [0.45, 1.0]])
            terms = (
                ((0, -1, 2, 0.8, -0.4), (0, 0, 2, 0.8, 0.3))
                if compact_fixture
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
        if auxiliary_order:
            auxiliary_basis = {
                label: [[0, [0.8 if atom == 0 else 0.65, 1.0]]]
                for atom, label in enumerate(labels)
            }
            auxiliary_basis[labels[0]].append([auxiliary_order, [0.6, 1.0]])
            self.auxiliary = gto.M(
                atom=list(zip(labels, self.centers, strict=True)),
                unit="Bohr",
                basis=auxiliary_basis,
                spin=self.spins - 1,
                cart=not spherical,
                verbose=0,
            )
        # libcint Cartesian d/f functions carry a shell-radial convention;
        # the public native Cartesian AOs are individually normalized.
        self.orbital_scale = (
            1 / np.sqrt(np.diag(self.template.intor("int1e_ovlp")))
            if not spherical
            else np.ones(self.nao)
        )
        jets = (
            self.template.eval_gto(
                "GTOval_sph_deriv1" if spherical else "GTOval_cart_deriv1", self.points
            )
            * self.orbital_scale
        )
        self.active = _fields(jets, self.seed).sum(axis=0)[0] >= 1e-8
        self.reference = result

    def __call__(self, displacement: np.ndarray, *, reference: bool = False) -> float:
        from pyscf import df
        from scipy.linalg import eigh, expm
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
        jets = (
            molecule.eval_gto(
                "GTOval_cart_deriv1" if molecule.cart else "GTOval_sph_deriv1", points
            )
            * self.orbital_scale
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
        orbital_pair_scale = self.orbital_scale[:, None] * self.orbital_scale[None, :]
        overlap *= orbital_pair_scale
        core *= orbital_pair_scale
        eri = molecule.intor("int2e", aosym="s1")
        eri *= (
            orbital_pair_scale[:, :, None, None] * orbital_pair_scale[None, None, :, :]
        )
        if self.fitted:
            auxiliary = self.auxiliary.set_geom_(centers, unit="Bohr", inplace=False)
            raw = df.incore.aux_e2(molecule, auxiliary, aosym="s1")
            raw *= orbital_pair_scale[:, :, None]
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
        long_range *= (
            orbital_pair_scale[:, :, None, None] * orbital_pair_scale[None, None, :, :]
        )
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

        if self.orbital_angular or self.benchmark_water:
            # Unconstrained density trials can be indefinite and leave VV10's
            # domain. Occupied/virtual rotations preserve positivity and rank
            # without borrowing native response or any derivative equations.
            overlap_values, overlap_vectors = eigh(overlap)
            overlap_root = (
                overlap_vectors * np.sqrt(overlap_values)
            ) @ overlap_vectors.T
            inverse_root = (
                overlap_vectors / np.sqrt(overlap_values)
            ) @ overlap_vectors.T
            frames = [
                inverse_root @ eigh(overlap_root @ seed @ overlap_root)[1][:, ::-1]
                for seed in self.seed
            ]
            offsets = np.cumsum(
                [0, *(count * (self.nao - count) for count in occupied)]
            )

            def orbital_density(parameters: np.ndarray) -> tuple[np.ndarray, list]:
                trial_frames, trial_density = [], []
                for spin, count in enumerate(occupied):
                    rotation = parameters[offsets[spin] : offsets[spin + 1]].reshape(
                        self.nao - count, count
                    )
                    generator = np.zeros((self.nao, self.nao))
                    generator[count:, :count] = rotation
                    generator[:count, count:] = -rotation.T
                    coefficients = frames[spin] @ expm(generator)
                    trial_frames.append(coefficients)
                    trial_density.append(
                        occupation * coefficients[:, :count] @ coefficients[:, :count].T
                    )
                return np.array(trial_density), trial_frames

            def orbital_residual(parameters: np.ndarray) -> np.ndarray:
                trial_density, trial_frames = orbital_density(parameters)
                return np.concatenate(
                    [
                        (
                            coefficients[:, count:].T @ matrix @ coefficients[:, :count]
                        ).ravel()
                        for matrix, coefficients, count in zip(
                            fock(trial_density), trial_frames, occupied, strict=True
                        )
                    ]
                )

            solution = root(
                orbital_residual,
                np.zeros(offsets[-1]),
                tol=1e-10,
                options={"maxfev": 200 * (int(offsets[-1]) + 1)},
            )
            density = orbital_density(solution.x)[0]
            values = pack(density)
        else:
            solution = root(residual, pack(self.seed), tol=1e-10)
            values = solution.x
        # Retain the independent lowest-occupation fixed-point gate, not merely
        # stationarity in the rotation chart, at the original tolerance.
        assert np.linalg.norm(residual(values)) < 2e-10, solution.message
        density = unpack(values)
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


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("ecp", (False, True))
def test_independent_spherical_g_auxiliary_stationary_hvp(
    spins: int, ecp: bool
) -> None:
    """Qualify g/g metric and coupled fitted response, not just primitive leaves."""
    pytest.importorskip("pyscf")
    result = _native(spins, True, True, ecp, False, auxiliary_g=True)
    _qualify(result, fitted=True, spherical=True, ecp=ecp, full=False)


@pytest.mark.parametrize(
    "spins,fitted,spherical,ecp,orbital_angular",
    (
        (1, False, True, False, 2),
        (1, False, True, False, 3),
        (1, False, False, False, 3),
        (2, True, True, True, 3),
        (2, True, False, True, 3),
    ),
    ids=(
        "rks-direct-d",
        "rks-direct-f-spherical",
        "rks-direct-f-cartesian",
        "uks-df-ecp-f-g-spherical",
        "uks-df-ecp-f-g-cartesian",
    ),
)
def test_high_angular_molecular_stationary_hvp(
    spins: int, fitted: bool, spherical: bool, ecp: bool, orbital_angular: int
) -> None:
    """Exercise d/f orbital relaxation and joint f/g response beyond leaf gates."""
    pytest.importorskip("pyscf")
    full = spherical and orbital_angular == 3
    result = _native(
        spins,
        fitted,
        spherical,
        ecp,
        full,
        auxiliary_g=fitted,
        orbital_angular=orbital_angular,
    )
    assert result["orbital_angular"] == orbital_angular
    assert result["nao"] > 6
    _qualify(result, fitted=fitted, spherical=spherical, ecp=ecp, full=full)


def test_benchmark_h2_full_hessian_against_independent_molecular_energies() -> None:
    """Qualify the actual external-comparison geometry and grid, not a proxy."""
    pytest.importorskip("pyscf")
    result = _native(1, False, True, False, True, benchmark_h2=True)
    _qualify_benchmark_matrix(result)


def test_benchmark_water_full_hessian_against_independent_molecular_energies() -> None:
    """Qualify all 81 entries of the actual seven-AO external water workload."""
    pytest.importorskip("pyscf")
    result = _native(1, False, True, False, True, benchmark_water=True)
    assert result["nao"] == 7 and result["atoms"] == 3
    _qualify_benchmark_matrix(result)


def _qualify_benchmark_matrix(result: dict) -> None:
    """Keep each raw entry and both independently reconverged energy stencils."""
    coordinates = 3 * result["atoms"]
    actual = np.array(result["hessian"]).reshape(coordinates, coordinates)
    axes = np.eye(coordinates).reshape(coordinates, result["atoms"], 3)
    expected_matrix = np.zeros_like(actual)
    stencil_matrices = np.zeros((2, coordinates, coordinates))
    with _signed_density_libxc() as code:
        energy = MolecularEnergy(
            result, fitted=False, spherical=True, ecp=False, code=code
        )
        np.testing.assert_allclose(
            energy(np.zeros((result["atoms"], 3)), reference=True),
            result["energy"],
            atol=5e-9,
        )
        for row in range(coordinates):
            for column in range(row, coordinates):
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
                expected_matrix[row, column] = expected_matrix[column, row] = expected
                stencil_matrices[:, row, column] = estimates
                stencil_matrices[:, column, row] = estimates
    if directory := os.environ.get("GENERATIVEQC_STATIONARY_RECEIPTS"):
        from generativeqc_compiler.common.evidence import block_error

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        case = "water" if result.get("benchmark_water", False) else "h2"
        receipt = {
            "case": case,
            "native_energy": result["energy"],
            "steps": [0.004, 0.002],
            "actual": actual.tolist(),
            "energy_stencil_matrices": stencil_matrices.tolist(),
            "expected": expected_matrix.tolist(),
            "error": block_error(actual, expected_matrix, atol=8e-7, rtol=1e-7),
        }
        (destination / f"oracle-full-{case}.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )


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
        reference_energy = energy(np.zeros_like(motion), reference=True)
        np.testing.assert_allclose(reference_energy, result["energy"], atol=5e-9)
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
        expected = (4 * estimates[1] - estimates[0]) / 3
        np.testing.assert_allclose(actual, expected, atol=8e-7, rtol=3e-5)
        if directory := os.environ.get("GENERATIVEQC_STATIONARY_RECEIPTS"):
            from generativeqc_compiler.common.evidence import (
                block_error,
                canonical_hash,
            )

            inputs = {
                "basis": result["basis"],
                "spins": result["spins"],
                "fitted": fitted,
                "spherical": spherical,
                "ecp": ecp,
                "auxiliary_d": result.get("auxiliary_d", False),
                "auxiliary_g": result.get("auxiliary_g", False),
            }
            identity = canonical_hash(inputs)
            receipt = {
                "inputs_hash": identity,
                "spins": result["spins"],
                "fitted": fitted,
                "spherical": spherical,
                "ecp": ecp,
                "orbital_angular": result.get("orbital_angular", 0),
                "auxiliary_g": result.get("auxiliary_g", False),
                "full": full,
                "coordinates_bohr": energy.centers.tolist(),
                "native_energy": result["energy"],
                "reference_energy": reference_energy,
                "direction": motion.tolist(),
                "actual": actual.tolist(),
                "steps": [0.004, 0.002],
                "energy_stencil_hvps": [estimate.tolist() for estimate in estimates],
                "expected": expected.tolist(),
                "error": block_error(actual, expected, atol=8e-7, rtol=3e-5),
                "consumer": result.get("oracle_consumer", "native-stationary-probe"),
            }
            destination = Path(directory)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f"oracle-{receipt['consumer']}-{identity}.json").write_text(
                json.dumps(receipt, indent=2, allow_nan=False) + "\n"
            )
