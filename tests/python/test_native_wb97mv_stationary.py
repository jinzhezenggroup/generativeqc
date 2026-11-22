"""Reconverged independent molecular energies gate the native stationary owner.

Libcint, upstream Libxc and the direct-space VV10 reference supply values only;
no compiler derivatives, CPKS equations or native response enter this oracle.
Python is a test driver/oracle, never loaded by the native numerical process.
"""

from __future__ import annotations

import json
import os
import subprocess
from contextlib import nullcontext
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    KsOptions,
    Primitive,
    ResourceBudget,
    Shell,
)
from generativeqc.basis import BasisShell
from generativeqc_compiler.common.nonlocal_correlation import NonlocalCorrelationSpec
from generativeqc_compiler.dft.nonlocal_reference import (
    assemble_nonlocal_potential_reference,
    nonlocal_energy_reference,
    nonlocal_feature_derivatives_reference,
)
from generativeqc_compiler.method.spec import resolve_method
from r2scan_stationary_oracle import variational_r2scan_grid
from split_hybrid_stationary_oracle import variational_split_hybrid_grid
from test_native_grid_geometry import _partition
from test_wb97mv_native_fock import _jk
from test_wb97mv_native_grid import (
    _fields,
    _independent_potential,
    _molecular_vv10_active,
    _signed_density_libxc,
)
from test_wb97mv_nuclear_grid import _energy, _scaled_libxc_grid

# Independently specified coefficients: Libxc supplies the complementary XC
# energy and potential, not the nonlocal Hartree–Fock contribution.
_SPLIT_HYBRID_EXCHANGE = {"m06-2x": 0.54, "mn15": 0.44}


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
    physical_cpks: bool = False,
    fp32_cpks: bool = False,
    serial_one_electron: bool = False,
    serial_grid: bool = False,
    directional_exact: bool = False,
    fitted_range: bool = False,
    split_grid: bool = False,
    split_treutler: bool = False,
    split_budget: bool = False,
    resource_ledger: bool = False,
    displaced_gradients: bool = False,
    auto_precision: bool = False,
    fitted_xc_warmup: bool = False,
    borrow_force_centers: bool = False,
    metric_branch: str | None = None,
    method_case: str | None = None,
) -> dict:
    executable = os.environ.get("GENERATIVEQC_KS_WB97MV_HESSIAN_PROBE")
    if not executable:
        pytest.skip("requires native stationary probe and Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require finite srun"
    mode = "df" if fitted else "direct"
    mode += "-lrdf" if fitted_range else ""
    mode += "-split-grid" if split_grid else ""
    mode += "-treutler" if split_treutler else ""
    mode += "-budget" if split_budget else ""
    mode += "-resource-ledger" if resource_ledger else ""
    mode += "-gradient-check" if displaced_gradients else ""
    mode += "-auto-precision" if auto_precision else ""
    mode += "-fitted-xc-warmup" if fitted_xc_warmup else ""
    mode += "-borrow-force-centers" if borrow_force_centers else ""
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
    mode += "-physical-cpks" if physical_cpks else ""
    mode += "-fp32-cpks" if fp32_cpks else ""
    mode += "-serial-one-electron" if serial_one_electron else ""
    mode += "-serial-grid" if serial_grid else ""
    mode += "-directional-exact" if directional_exact else ""
    mode += f"-rank-{metric_branch}" if metric_branch else ""
    mode += f"-{method_case}" if method_case else ""
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
            GENERATIVEQC_DF_RANGE_EXCHANGE="df" if fitted_range else "direct",
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


@pytest.mark.parametrize("method_case", ("lda", "pbe"))
def test_direct_semilocal_hessian_borrows_live_force_centers(method_case: str) -> None:
    """A force replay revokes the first lease without changing the HVP."""
    options = {"benchmark_water": True, "method_case": method_case}
    direct = _native(1, False, False, False, False, **options)
    borrowed = _native(
        1, False, False, False, False, borrow_force_centers=True, **options
    )
    np.testing.assert_allclose(borrowed["hvp"], direct["hvp"], atol=2e-12, rtol=2e-11)
    assert borrowed["lifecycle"]


@pytest.mark.parametrize("fitted", [False, True], ids=["direct", "fitted"])
@pytest.mark.parametrize("full", [False, True], ids=["hvp", "matrix"])
def test_hessian_shares_prepared_device_ledger(fitted: bool, full: bool) -> None:
    """C ABI admission counts response concurrently with the retained KS state."""
    result = _native(1, fitted, True, False, full, resource_ledger=True)
    assert 0 < result["response_bytes"] <= result["aggregate_peak_bytes"]
    assert (
        result["response_bytes"] + result["scf_retained_bytes"]
        <= result["aggregate_peak_bytes"]
    )


@pytest.mark.parametrize("method_case", ("lda", "pbe"))
@pytest.mark.parametrize("full", (False, True), ids=("hvp", "matrix"))
def test_cartesian_semilocal_hessian_ledger_replay(
    method_case: str, full: bool
) -> None:
    """Measure live bytes across same-state budget/nonconvergence rejection and replay."""
    result = _native(
        1,
        False,
        False,
        False,
        full,
        benchmark_water=True,
        resource_ledger=True,
        method_case=method_case,
    )
    assert 0 < result["response_bytes"]
    assert (
        result["response_bytes"] + result["scf_retained_bytes"]
        <= result["aggregate_peak_bytes"]
    )


def test_d3_hard_cutoff_preserves_device_output() -> None:
    """Reject undefined curvature without publishing any part of the result."""
    result = _native(1, False, True, False, False, method_case="pbe-d3-hard-cutoff")
    assert result["hard_cutoff_preserves_output"]


@pytest.mark.parametrize("method_case", ("lda", "pbe"))
def test_cartesian_dense_replays_without_resident_cpks_operator(
    method_case: str,
) -> None:
    """The bounded matrix-free route preserves the complete dense result."""
    common = {"benchmark_water": True, "method_case": method_case}
    cached = _native(1, False, False, False, True, **common)
    physical = _native(1, False, False, False, True, physical_cpks=True, **common)
    assert cached["cpks_operator_columns"] > 0
    assert physical["cpks_operator_columns"] == 0
    assert physical["cpks_strict_iterations"] > 0
    np.testing.assert_allclose(
        physical["hessian"], cached["hessian"], atol=2e-10, rtol=0
    )


@pytest.mark.parametrize("method_case", ("lda", "pbe"))
def test_cartesian_direct_hessian_streams_without_resident_integrals(
    method_case: str,
) -> None:
    """Stream both nuclear derivatives and induced Direct J in the same endpoint."""
    common = {"benchmark_water": True, "method_case": method_case}
    cached = _native(1, False, False, False, True, **common)
    streamed = _native(
        1,
        False,
        False,
        False,
        True,
        streamed=True,
        unbatched=True,
        physical_cpks=True,
        **common,
    )
    assert cached["resident_integral_bytes"] > 0
    assert streamed["resident_integral_bytes"] == 0
    assert streamed["cpks_operator_columns"] == 0
    assert streamed["induced_direct_quartets"] > 0
    for field in ("hvp", "hessian"):
        np.testing.assert_allclose(streamed[field], cached[field], atol=2e-10, rtol=0)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("fitted", (False, True))
def test_stationary_response_reuse_and_parallel_integrals_preserve_hessians(
    spins: int, fitted: bool
) -> None:
    """Gate optional schedules on the complete spin/DF/ECP/spherical endpoint."""
    cached = _native(spins, fitted, True, fitted, True)
    physical = _native(
        spins,
        fitted,
        True,
        fitted,
        True,
        physical_cpks=True,
        serial_one_electron=True,
        serial_grid=True,
        directional_exact=True,
    )
    assert cached["cpks_operator_columns"] > 0
    # Symmetric minimal fixtures may have an exactly zero nuclear RHS, so a
    # successful solve need not execute a cached Krylov action.
    assert cached["cpks_physical_checks"] == 3 * cached["atoms"]
    assert cached["parallel_one_electron_actions"] > 0
    assert physical["cpks_operator_columns"] == 0
    assert physical["cpks_physical_checks"] == 0
    assert physical["parallel_one_electron_bytes"] == 0
    assert cached["frozen_exact_bytes"] > 0
    assert cached["frozen_exact_columns"] > 0
    assert physical["frozen_exact_bytes"] == 0
    assert cached["frozen_core_bytes"] > 0
    assert physical["frozen_core_bytes"] == 0
    assert cached["one_electron_components"] < physical["one_electron_components"]
    for field in ("energy", "hvp", "hessian", "response_residual"):
        np.testing.assert_allclose(
            cached[field], physical[field], atol=3e-12, rtol=3e-12
        )


@pytest.mark.parametrize(
    "fitted", (False, True), ids=("wb97mv-direct-rks", "wb97mv-lrdf-uks")
)
def test_fp32_cpks_projection_against_physical_hessian_and_independent_energy(
    fitted: bool,
) -> None:
    """The compressed operator proposes a seed; a physical audit gates every column."""
    pytest.importorskip("pyscf")
    spins = 2 if fitted else 1
    common = {
        "benchmark_water": not fitted,
        "fitted_range": fitted,
        "split_grid": fitted,
    }
    mixed = _native(spins, fitted, True, False, True, fp32_cpks=True, **common)
    resident = _native(spins, fitted, True, False, True, **common)
    uncached = _native(
        spins, fitted, True, False, True, fp32_cpks=True, physical_cpks=True, **common
    )
    dimension = mixed["cpks_operator_columns"]
    coordinates = 3 * mixed["atoms"]
    assert dimension > 1
    assert mixed["cpks_operator_bytes"] == 4 * dimension**2
    assert resident["cpks_operator_bytes"] == 2 * mixed["cpks_operator_bytes"]
    assert mixed["cpks_proposal_iterations"] >= mixed["cpks_proposal_seeds"] > 0
    assert mixed["cpks_strict_iterations"] > 0
    assert mixed["cpks_physical_checks"] == coordinates
    assert uncached["cpks_operator_bytes"] == uncached["cpks_proposal_iterations"] == 0
    matrix = np.asarray(mixed["hessian"]).reshape(coordinates, coordinates)
    for expected in (resident, uncached):
        np.testing.assert_allclose(
            mixed["hessian"], expected["hessian"], atol=3e-9, rtol=0
        )
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix.reshape(coordinates, -1, 3).sum(axis=1), 0, atol=3e-9
    )
    np.testing.assert_allclose(
        matrix @ mixed["motion"], mixed["hvp"], atol=3e-9, rtol=0
    )
    _qualify_benchmark_matrix(mixed, fitted=fitted)


@pytest.mark.parametrize(
    "spins,fitted,fitted_range", [(1, False, False), (2, True, True)]
)
def test_stationary_action_only_matches_dense(
    spins: int, fitted: bool, fitted_range: bool
) -> None:
    """One-shot HVP and column assembly solve the same stationary response."""
    action = _native(spins, fitted, True, False, False, fitted_range=fitted_range)
    dense = _native(spins, fitted, True, False, True, fitted_range=fitted_range)
    matrix = np.reshape(dense["hessian"], (len(action["motion"]), -1))
    np.testing.assert_allclose(action["hvp"], dense["hvp"], atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(
        action["hvp"], matrix @ action["motion"], atol=3e-9, rtol=3e-9
    )


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


def _point_owners(result: dict, prefix: str, points: int) -> np.ndarray:
    """Recover point ownership from the retained per-atom quadrature topology."""
    atoms = result["atoms"]
    if offsets := result.get(f"{prefix}atom_offsets"):
        assert len(offsets) == atoms + 1 and offsets[0] == 0 and offsets[-1] == points
        counts = np.diff(offsets)
        assert (counts > 0).all()
    else:
        assert points % atoms == 0
        counts = points // atoms
    return np.repeat(np.arange(atoms), counts)


class MolecularEnergy:
    """Values-only spin-coupled generalized-eigensolve at each moved geometry."""

    def __init__(
        self,
        result: dict,
        *,
        fitted: bool,
        spherical: bool,
        ecp: bool,
        code: str,
        method_case: str | None = None,
    ) -> None:
        from pyscf import gto, lib

        lib.num_threads(1)
        self.spins = result["spins"]
        self.nao = result["nao"]
        self.centers = np.array(result["basis"][: 3 * result["atoms"]]).reshape(-1, 3)
        self.points = np.array(result["points"]).reshape(-1, 3)
        self.atomic = np.array(result["atomic"])
        self.owner = _point_owners(result, "", len(self.points))
        self.vv10_points = (
            np.array(result["vv10_points"]).reshape(-1, 3)
            if "vv10_points" in result
            else None
        )
        self.seed = np.array(result["density"]).reshape(self.spins, self.nao, self.nao)
        self.fitted, self.code = fitted, code
        self.fitted_range = result.get("fitted_range", False)
        self.metric_cutoff = result.get("metric_relative_threshold", 1e-10)
        self.method_case = method_case
        self.split_hybrid = method_case in _SPLIT_HYBRID_EXCHANGE
        self.revised = method_case == "r2scan-rvv10"
        self.benchmark_water = result.get("benchmark_water", False)
        self.composite = result.get("r2scan3c", False)
        self.pbe_d4 = method_case == "pbe-d4"
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
        if self.composite:
            from dftd4.interface import DampingParam, DispersionModel

            from tools.generativeqc_gcp.reference import evaluate_r2scan3c_gcp

            # Parse the pinned method basis as input data, not from the native
            # normalized AO export. Libcint normalizes its own contractions.
            records = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json"
                ).read_text()
            )
            hydrogen = next(
                record for record in records["elements"] if record["atomic_number"] == 1
            )
            basis = {
                label: [
                    [
                        shell["angular_momentum"],
                        *[
                            [float(exponent), float(coefficient)]
                            for exponent, coefficient in zip(
                                shell["exponents"], coefficients, strict=True
                            )
                        ],
                    ]
                    for shell in hydrogen["shells"]
                    for coefficients in shell["coefficients"]
                ]
                for label in labels
            }
            numbers = np.ones(len(labels), dtype=np.int32)
            damping = DampingParam(s6=1.0, s8=0.0, s9=2.0, a1=0.42, a2=5.65)

            def corrections(centers: np.ndarray) -> tuple[float, float]:
                d4 = DispersionModel(numbers, centers, ga=2.0, gc=1.0).get_dispersion(
                    damping, grad=False
                )
                return float(d4["energy"]), evaluate_r2scan3c_gcp(
                    numbers, centers
                ).energy

            self.corrections = corrections
        if self.pbe_d4:
            from dftd4.interface import DampingParam, DispersionModel

            # The upstream PBE parameter lookup is independent of native tables.
            damping = DampingParam(method="pbe", atm=True)

            def pbe_d4_energy(centers: np.ndarray) -> float:
                model = DispersionModel(np.array([8, 1, 1], dtype=np.int32), centers)
                return float(model.get_dispersion(damping, grad=False)["energy"])

            self.pbe_d4_energy = pbe_d4_energy
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
        compact_fixture = auxiliary_order or self.orbital_angular or self.split_hybrid
        if compact_fixture:
            basis = {
                label: [[0, [0.7 if atom == 0 else 0.9, 1.0]]]
                for atom, label in enumerate(labels)
            }
        if result.get("fitted_xc_warmup", False):
            for shells in basis.values():
                shells.extend(
                    [
                        [0, [0.28, 1.0]],
                        [1, [0.38, 1.0]],
                        [1, [1.3, 1.0]],
                        [2, [0.72, 1.0]],
                    ]
                )
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
        if auxiliary_order or (self.split_hybrid and fitted):
            auxiliary_basis = {
                label: [[0, [0.8 if atom == 0 else 0.65, 1.0]]]
                for atom, label in enumerate(labels)
            }
            if auxiliary_order:
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
        # Hold the SCF active set fixed for the local displaced derivative.
        self.active = _molecular_vv10_active(
            _fields(jets, self.seed).sum(axis=0)[0], np.asarray(result["weights"])
        )
        if self.vv10_points is not None:
            self.active[:] = False
            self.vv10_atomic = np.array(result["vv10_atomic"])
            self.vv10_owner = _point_owners(result, "vv10_", len(self.vv10_points))
            vv10_jets = (
                self.template.eval_gto(
                    "GTOval_sph_deriv1" if spherical else "GTOval_cart_deriv1",
                    self.vv10_points,
                )
                * self.orbital_scale
            )
            self.vv10_active = _molecular_vv10_active(
                _fields(vv10_jets, self.seed).sum(axis=0)[0],
                np.asarray(result["vv10_weights"]),
            )
            self.vv10_spec = resolve_method("WB97M-V").primitives[-1].spec
        if self.revised:
            # Independent paper parameters, not the native model payload.
            self.vv10_spec = NonlocalCorrelationSpec(
                "rvv10", Fraction("11.95"), Fraction("0.0093")
            )
        self.reference = result

    def fitted_eri(self, raw: np.ndarray, metric: np.ndarray, key: str) -> np.ndarray:
        """Recompute the retained subspace, not just its eigenvalues, at every move."""
        eigenvalues, vectors = np.linalg.eigh(metric)
        keep = eigenvalues > self.metric_cutoff * eigenvalues[-1]
        assert np.count_nonzero(keep) == self.reference.get(key, len(keep))
        factors = (raw @ vectors[:, keep]) / np.sqrt(eigenvalues[keep])
        return np.einsum("ijp,klp->ijkl", factors, factors)

    def _partition_weights(
        self,
        centers: np.ndarray,
        points: np.ndarray,
        owner: np.ndarray,
        atomic: np.ndarray,
        prefix: str = "",
    ) -> np.ndarray:
        """Rebuild only partition values, allowing the two atom grids to differ."""
        fixture = {
            "centers": centers,
            "points": points,
            "owner": owner,
            "left": np.zeros_like(centers),
            "right": np.zeros_like(centers),
            "atomic": atomic,
            "iterations": 3,
        }
        if radii := self.reference.get(f"{prefix}atomic_radii"):
            fixture["radii"] = np.asarray(radii)
        return np.asarray(_partition(fixture), dtype=np.float64)

    def __call__(self, displacement: np.ndarray, *, reference: bool = False) -> float:
        from pyscf import df
        from scipy.linalg import eigh, expm
        from scipy.optimize import root

        centers = self.centers + displacement
        points = self.points + displacement[self.owner]
        molecule = self.template.set_geom_(centers, unit="Bohr", inplace=False)
        weights = self._partition_weights(centers, points, self.owner, self.atomic)
        if reference and (self.vv10_points is not None or self.revised):
            # Absolute errors in tiny partition fractions are amplified by
            # large radial weights. Audit the partition before that scaling.
            scale = np.where(self.atomic == 0, 1.0, self.atomic)
            np.testing.assert_allclose(
                weights / scale,
                np.asarray(self.reference["weights"]) / scale,
                atol=2e-13,
                rtol=0,
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
        if self.vv10_points is not None:
            vv10_points = self.vv10_points + displacement[self.vv10_owner]
            vv10_weights = self._partition_weights(
                centers, vv10_points, self.vv10_owner, self.vv10_atomic, "vv10_"
            )
            if reference:
                scale = np.where(self.vv10_atomic == 0, 1.0, self.vv10_atomic)
                np.testing.assert_allclose(
                    vv10_weights / scale,
                    np.asarray(self.reference["vv10_weights"]) / scale,
                    atol=2e-13,
                    rtol=0,
                )
            active = self.vv10_active
            vv10_jets = (
                molecule.eval_gto(
                    "GTOval_cart_deriv1" if molecule.cart else "GTOval_sph_deriv1",
                    vv10_points[active],
                )
                * self.orbital_scale
            )

            def vv10_input(density: np.ndarray) -> tuple:
                fields = _fields(vv10_jets, density).sum(axis=0)
                return (
                    vv10_points[active],
                    vv10_weights[active],
                    fields[0],
                    fields[1:4].T,
                )

        if self.revised:
            if self.vv10_points is None:
                pair_points, pair_weights, pair_jets = points, weights, jets
            else:
                pair_points, pair_weights = vv10_points, vv10_weights
                pair_jets = (
                    molecule.eval_gto(
                        "GTOval_cart_deriv1" if molecule.cart else "GTOval_sph_deriv1",
                        pair_points,
                    )
                    * self.orbital_scale
                )

            def revised_input(density: np.ndarray) -> tuple:
                fields = _fields(pair_jets, density).sum(axis=0)
                # The displaced SCF solves the physical density-cutoff energy,
                # rather than constraining the trials to the center's active set.
                active = _molecular_vv10_active(fields[0], pair_weights)
                return pair_jets[:, active], (
                    pair_points[active],
                    pair_weights[active],
                    fields[0, active],
                    fields[1:4, active].T,
                )

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
            eri = self.fitted_eri(raw, metric, "metric_rank")
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
        if not self.method_case:
            with molecule.with_range_coulomb(0.3):
                if self.fitted_range:
                    with auxiliary.with_range_coulomb(0.3):
                        raw_lr = df.incore.aux_e2(molecule, auxiliary, aosym="s1")
                        metric_lr = auxiliary.intor("int2c2e")
                    raw_lr *= orbital_pair_scale[:, :, None]
                    long_range = self.fitted_eri(raw_lr, metric_lr, "range_metric_rank")
                else:
                    long_range = molecule.intor("int2e", aosym="s1")
                    long_range *= (
                        orbital_pair_scale[:, :, None, None]
                        * orbital_pair_scale[None, None, :, :]
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

        def interaction(density: np.ndarray) -> np.ndarray:
            if self.method_case:
                coulomb = np.einsum("pqrs,rs->pq", eri, density.sum(axis=0))[None]
                fraction = {"pbe0": 0.25, "b3lyp": 0.2, **_SPLIT_HYBRID_EXCHANGE}.get(
                    self.method_case
                )
                if fraction is not None:
                    return coulomb - (fraction / occupation) * np.einsum(
                        "prqt,art->apq", eri, density
                    )
                return coulomb
            return _jk(density, eri, long_range)

        def xc(density: np.ndarray) -> tuple[float, np.ndarray]:
            if self.method_case in ("r2scan", "r2scan-3c", "r2scan-rvv10"):
                return variational_r2scan_grid(fixture, density)
            if self.method_case == "pbe0":
                return _scaled_libxc_grid(fixture, density, "pbe", (0.75, 1.0))
            if self.pbe_d4:
                return _scaled_libxc_grid(fixture, density, "pbe")
            if self.split_hybrid:
                return variational_split_hybrid_grid(fixture, density, self.method_case)
            if self.method_case:
                return _scaled_libxc_grid(fixture, density, self.method_case)
            return wb97mv_energy(density), wb97mv_potential(density)

        def wb97mv_energy(density: np.ndarray) -> float:
            energy = _energy(fixture, density)
            if self.vv10_points is not None:
                energy += nonlocal_energy_reference(
                    *vv10_input(density), self.vv10_spec
                )
            return energy

        def wb97mv_potential(density: np.ndarray) -> np.ndarray:
            potential = _independent_potential(
                fixture, density, nonlocal_active=self.active, libxc_code=self.code
            )
            if self.vv10_points is not None:
                inputs = vv10_input(density)
                derivatives = nonlocal_feature_derivatives_reference(
                    *inputs, self.vv10_spec
                )
                potential += assemble_nonlocal_potential_reference(
                    vv10_jets, inputs[1], inputs[3], *derivatives
                )
            return potential

        def fock(density: np.ndarray) -> np.ndarray:
            potential = (
                xc(density)[1] if self.method_case else wb97mv_potential(density)
            )
            if self.revised:
                active_jets, inputs = revised_input(density)
                derivatives = nonlocal_feature_derivatives_reference(
                    *inputs, self.vv10_spec
                )
                potential += assemble_nonlocal_potential_reference(
                    active_jets, inputs[1], inputs[3], *derivatives
                )
            return core + interaction(density) + potential

        def residual(values: np.ndarray) -> np.ndarray:
            density = unpack(values)
            projected = []
            for matrix, count in zip(fock(density), occupied, strict=True):
                vectors = eigh(matrix, overlap)[1][:, :count]
                projected.append(occupation * vectors @ vectors.T)
            return pack(np.array(projected) - density)

        if (
            self.orbital_angular
            or self.benchmark_water
            or self.composite
            or self.revised
            or self.split_hybrid
            or self.reference.get("fitted_xc_warmup", False)
        ):
            # Rotations preserve positivity and rank, avoiding indefinite VV10
            # trials and the quadratic density chart for the larger PBE fixture.
            # The oracle borrows no native response or derivative equation.
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
        correction = 0.0
        if self.pbe_d4:
            correction = self.pbe_d4_energy(centers)
            if reference:
                assert correction == pytest.approx(
                    self.reference["d4_energy"], abs=2e-12
                )
        elif self.composite:
            dispersion, gcp = self.corrections(centers)
            correction = dispersion + gcp
            if reference:
                assert dispersion == pytest.approx(
                    self.reference["d4_energy"], abs=2e-12
                )
                assert gcp == pytest.approx(self.reference["gcp_energy"], abs=2e-12)
        if self.revised:
            _, inputs = revised_input(density)
            correction += nonlocal_energy_reference(*inputs, self.vv10_spec)
        return float(
            np.einsum("sij,ji->", density, core)
            + 0.5 * np.einsum("sij,sji", density, interaction(density))
            + (xc(density)[0] if self.method_case else wb97mv_energy(density))
            + molecule.energy_nuc()
            + correction
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
def test_fitted_long_range_stationary_hessian(spins: int) -> None:
    """Rebuild both Coulomb and erf metrics at every displaced geometry."""
    pytest.importorskip("pyscf")
    result = _native(
        spins, True, True, False, True, auxiliary_d=True, fitted_range=True
    )
    assert result["fitted_range"]
    assert result["exact_nuclear_components"] == result["induced_direct_quartets"] == 0
    assert result["raw_df_range_value_passes"] == result["raw_df_value_passes"] > 0
    assert (
        result["raw_df_range_contraction_passes"]
        == result["raw_df_contraction_passes"]
        > 0
    )
    assert result["raw_df_range_workspace_bytes"] > 0
    assert result["range_affine_terms"] > 0
    _qualify(result, fitted=True, spherical=True, ecp=False, full=True)


@pytest.mark.parametrize("spins", (1, 2), ids=("rks", "uks"))
def test_fitted_long_range_f_g_ecp_full_hessian(spins: int) -> None:
    """Qualify every element at the joint spherical orbital-f, auxiliary-g boundary."""
    pytest.importorskip("pyscf")
    result = _native(
        spins,
        True,
        True,
        True,
        True,
        orbital_angular=3,
        auxiliary_g=True,
        fitted_range=True,
    )
    assert result["lifecycle"] and result["fitted_range"] and result["auxiliary_g"]
    assert result["orbital_angular"] == 3 and result["nao"] > 6
    assert result["exact_nuclear_components"] == result["induced_direct_quartets"] == 0
    assert result["raw_df_range_value_passes"] == result["raw_df_value_passes"] > 0
    assert (
        result["raw_df_range_contraction_passes"]
        == result["raw_df_contraction_passes"]
        > 0
    )
    coordinates = 3 * result["atoms"]
    matrix = np.asarray(result["hessian"]).reshape(coordinates, coordinates)
    translation = np.tile(np.eye(3), (result["atoms"], 1))
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(matrix @ translation, 0, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=3e-9, rtol=0
    )
    _qualify_benchmark_matrix(result, fitted=True, ecp=True)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("metric_branch", ("lr", "both"))
def test_fixed_rank_fitted_stationary_hessian(spins: int, metric_branch: str) -> None:
    result = _native(
        spins, True, True, False, True, fitted_range=True, metric_branch=metric_branch
    )
    naux = result["naux"]
    assert result["range_metric_rank"] == naux - 1
    assert result["metric_rank"] == naux - (metric_branch == "both")
    assert result["exact_nuclear_components"] == result["induced_direct_quartets"] == 0
    atoms = result["atoms"]
    matrix = np.asarray(result["hessian"]).reshape(3 * atoms, 3 * atoms)
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(matrix @ np.tile(np.eye(3), (atoms, 1)), 0, atol=2e-8)
    np.testing.assert_allclose(matrix @ result["motion"], result["hvp"], atol=2e-8)
    _qualify_benchmark_matrix(result, fitted=True)


@pytest.mark.parametrize("spins", (1, 2))
def test_metric_cutoff_preserves_device_output(spins: int) -> None:
    assert _native(
        spins, True, True, False, True, fitted_range=True, metric_branch="boundary"
    )["metric_cutoff_preserves_output"]


@pytest.mark.parametrize("spins,method_case", ((1, "pbe"), (2, "pbe0")))
def test_fixed_rank_semilocal_stationary_hessian(spins: int, method_case: str) -> None:
    pytest.importorskip("pyscf")
    result = _native(
        spins, True, True, False, True, metric_branch="primary", method_case=method_case
    )
    assert result["metric_rank"] == result["naux"] - 1
    assert result["exact_nuclear_components"] == result["induced_direct_quartets"] == 0
    _qualify_benchmark_matrix(result, method_case=method_case, fitted=True)


@pytest.mark.skipif(
    not os.environ.get("GENERATIVEQC_HESSIAN_TEST_WB97MV_FORCES"),
    reason="requires linked WB97M-V native force profiles in addition to Hessian AOT",
)
@pytest.mark.parametrize("spins,orbital_angular", ((1, 0), (2, 0), (1, 2), (2, 2)))
def test_fitted_long_range_displaced_gradient(spins: int, orbital_angular: int) -> None:
    result = _native(
        spins,
        True,
        True,
        False,
        False,
        auxiliary_d=True,
        orbital_angular=orbital_angular,
        fitted_range=True,
        displaced_gradients=True,
    )
    centered = (
        np.array(result["displaced_gradient_plus"]) - result["displaced_gradient_minus"]
    ) / (2 * result["gradient_step"])
    np.testing.assert_allclose(result["hvp"], centered, atol=8e-7, rtol=3e-5)


def test_fitted_long_range_g_auxiliary_hvp() -> None:
    pytest.importorskip("pyscf")
    result = _native(1, True, True, False, False, auxiliary_g=True, fitted_range=True)
    assert result["exact_nuclear_components"] == result["induced_direct_quartets"] == 0
    _qualify(result, fitted=True, spherical=True, ecp=False, full=False)


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


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("method_case", ("lda", "pbe", "pbe0", "b3lyp"))
def test_direct_semilocal_cuda_full_hessian_against_independent_energies(
    spins: int, method_case: str
) -> None:
    pytest.importorskip("pyscf")
    result = _native(spins, False, True, False, True, method_case=method_case)
    natoms = result["atoms"]
    matrix = np.asarray(result["hessian"]).reshape(3 * natoms, 3 * natoms)
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix.reshape(3 * natoms, natoms, 3).sum(axis=1), 0, atol=3e-9
    )
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=3e-9, rtol=0
    )
    _qualify_benchmark_matrix(result, method_case=method_case)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("method_case", ("pbe0", None), ids=("pbe0", "wb97mv-split"))
def test_auto_scf_complete_hessian_against_independent_energies(
    spins: int, method_case: str | None
) -> None:
    pytest.importorskip("pyscf")
    result = _native(
        spins,
        False,
        True,
        False,
        True,
        auto_precision=True,
        method_case=method_case,
        split_grid=method_case is None,
    )
    coordinates = 3 * result["atoms"]
    matrix = np.asarray(result["hessian"]).reshape(coordinates, coordinates)
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix.reshape(coordinates, -1, 3).sum(axis=1), 0, atol=3e-9
    )
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=3e-9, rtol=0
    )
    _qualify_benchmark_matrix(result, method_case=method_case)


def test_fitted_pbe_xc_warmup_hessian_against_independent_energies() -> None:
    """A mixed XC proposal may close on the same fitted FP64 stationary functional."""
    pytest.importorskip("pyscf")
    common = {"auxiliary_d": True, "fitted_xc_warmup": True, "method_case": "pbe"}
    result = _native(1, True, True, False, True, auto_precision=True, **common)
    strict = _native(1, True, True, False, True, **common)
    assert (
        result["scf_mixed_fock_builds"] > 0 and result["scf_refinement_iterations"] > 0
    )
    assert strict["scf_mixed_fock_builds"] == 0
    assert result["raw_df_value_passes"] and result["raw_df_contraction_passes"]
    np.testing.assert_allclose(result["hessian"], strict["hessian"], atol=3e-9, rtol=0)
    np.testing.assert_allclose(result["density"], strict["density"], atol=2e-10, rtol=0)
    matrix = np.asarray(result["hessian"]).reshape(6, 6)
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(matrix.reshape(6, 2, 3).sum(axis=1), 0, atol=3e-9)
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=3e-9, rtol=0
    )
    _qualify_benchmark_matrix(result, method_case="pbe", fitted=True)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("method_case", ("lda", "pbe", "pbe0", "b3lyp"))
def test_direct_semilocal_cuda_ecp_full_hessian_against_independent_energies(
    spins: int, method_case: str
) -> None:
    pytest.importorskip("pyscf")
    result = _native(spins, False, True, True, True, method_case=method_case)
    natoms = result["atoms"]
    matrix = np.asarray(result["hessian"]).reshape(3 * natoms, 3 * natoms)
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-8, rtol=0)
    np.testing.assert_allclose(
        matrix.reshape(3 * natoms, natoms, 3).sum(axis=1), 0, atol=2e-8
    )
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=2e-8, rtol=0
    )
    _qualify_benchmark_matrix(result, method_case=method_case, ecp=True)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("method_case", ("lda", "pbe", "pbe0", "b3lyp"))
def test_fitted_semilocal_cuda_full_hessian_against_independent_energies(
    spins: int, method_case: str
) -> None:
    pytest.importorskip("pyscf")
    result = _native(spins, True, True, False, True, method_case=method_case)
    assert result["raw_df_value_passes"] and result["raw_df_contraction_passes"]
    assert result["resident_integral_bytes"] == result["frozen_exact_bytes"] == 0
    matrix = np.asarray(result["hessian"]).reshape(3 * result["atoms"], -1)
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=3e-9, rtol=0
    )
    _qualify_benchmark_matrix(result, method_case=method_case, fitted=True)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("method_case", ("lda", "pbe", "pbe0", "b3lyp"))
def test_fitted_semilocal_cuda_ecp_full_hessian_against_independent_energies(
    spins: int, method_case: str
) -> None:
    pytest.importorskip("pyscf")
    result = _native(
        spins, True, True, True, True, auxiliary_d=True, method_case=method_case
    )
    assert result["raw_df_value_passes"] and result["raw_df_contraction_passes"]
    assert result["resident_integral_bytes"] == result["frozen_exact_bytes"] == 0
    matrix = np.asarray(result["hessian"]).reshape(3 * result["atoms"], -1)
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=2e-8, rtol=0
    )
    _qualify_benchmark_matrix(result, method_case=method_case, fitted=True, ecp=True)


@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
@pytest.mark.parametrize("spins", (1, 2), ids=("rks", "uks"))
@pytest.mark.parametrize("fixture", ("valence", "ecp", "orbital-d", "orbital-f"))
def test_r2scan_cuda_molecular_hessian(spins: int, fitted: bool, fixture: str) -> None:
    """Qualify SCF, native gradient and complete molecular response together."""
    pytest.importorskip("pyscf")
    ecp = fixture == "ecp"
    result = _native(
        spins,
        fitted,
        True,
        ecp,
        True,
        method_case="r2scan",
        auxiliary_d=fitted and ecp,
        orbital_angular={"orbital-d": 2, "orbital-f": 3}.get(fixture, 0),
    )
    coordinates = 3 * result["atoms"]
    matrix = np.asarray(result["hessian"]).reshape(coordinates, coordinates)
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix.reshape(coordinates, -1, 3).sum(axis=1), 0, atol=3e-9
    )
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=3e-9, rtol=0
    )
    _qualify_benchmark_matrix(result, method_case="r2scan", fitted=fitted, ecp=ecp)


@pytest.mark.parametrize(
    ("spins", "fitted", "ecp"),
    (
        pytest.param(1, False, False, id="rks-direct-ae"),
        pytest.param(2, True, False, id="uks-df-ae"),
        pytest.param(1, False, True, id="rks-direct-ecp"),
        pytest.param(1, True, True, id="rks-df-ecp"),
        pytest.param(2, False, True, id="uks-direct-ecp"),
        pytest.param(2, True, True, id="uks-df-ecp"),
    ),
)
@pytest.mark.parametrize("split_grid", (False, True), ids=("shared", "separate"))
def test_r2scan_rvv10_complete_cuda_hessian(
    spins: int, fitted: bool, ecp: bool, split_grid: bool
) -> None:
    """Check all ECP spin/provider/grid combinations against the total energy."""
    pytest.importorskip("pyscf")
    result = _native(
        spins,
        fitted,
        True,
        ecp,
        True,
        split_grid=split_grid,
        method_case="r2scan-rvv10",
    )
    coordinates = 3 * result["atoms"]
    matrix = np.asarray(result["hessian"]).reshape(coordinates, coordinates)
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix.reshape(coordinates, -1, 3).sum(axis=1), 0, atol=3e-9
    )
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=3e-9, rtol=0
    )
    _qualify_benchmark_matrix(
        result, method_case="r2scan-rvv10", fitted=fitted, ecp=ecp
    )


def _compact_shells(
    atoms: int,
    first_exponent: float,
    other_exponent: float,
    angular: int,
    exponent: float,
) -> tuple[Shell, ...]:
    """Match the probe's compact fixture, including the first-center shell."""
    return (
        Shell(0, 0, (Primitive(first_exponent, 1.0),)),
        Shell(0, angular, (Primitive(exponent, 1.0),)),
        *(
            Shell(index, 0, (Primitive(other_exponent, 1.0),))
            for index in range(1, atoms)
        ),
    )


@pytest.mark.parametrize(
    "method_case", ("lda", "pbe", "pbe0", "b3lyp", "r2scan", "r2scan-rvv10")
)
@pytest.mark.parametrize(
    "spins,ecp", ((1, False), (2, True)), ids=("rks-ae", "uks-ecp")
)
def test_fitted_semilocal_f_g_hessian_budget(
    method_case: str, spins: int, ecp: bool
) -> None:
    """Qualify bounded Coulomb and hybrid DF at the f-orbital/g-fit boundary."""
    pytest.importorskip("pyscf")
    result = _native(
        spins,
        True,
        True,
        ecp,
        True,
        method_case=method_case,
        orbital_angular=3,
        auxiliary_g=True,
    )
    atom_count = result["atoms"]
    coordinates = 3 * atom_count
    centers = np.asarray(result["basis"][:coordinates]).reshape(-1, 3)
    atoms = [
        ("Na" if ecp and index == 0 else "H", center)
        for index, center in enumerate(centers)
    ]
    basis = _compact_shells(atom_count, 0.7, 0.9, 3, 0.48)
    if ecp:
        from test_wb97mv_ecp_hessian import synthetic_basis

        catalog = synthetic_basis()
        basis = replace(
            catalog,
            elements=tuple(
                replace(
                    element,
                    shells=(*element.shells, BasisShell(3, ("0.48",), (("1",),))),
                )
                if element.atomic_number == 11
                else element
                for element in catalog.elements
            ),
        )
    selector = "r2scan+rvv10" if method_case == "r2scan-rvv10" else method_case
    settings = {
        "method": f"{selector}-{'rks' if spins == 1 else 'uks'}",
        "device": "cuda",
        "basis": basis,
        "basis_representation": "spherical",
        "density_fitting": "cuda",
        "auxiliary_basis": _compact_shells(atom_count, 0.8, 0.65, 4, 0.6),
        "initial_guess": None,
        "max_iterations": 250,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8),
            tile_points=64,
        ),
    }
    scf = (
        Calculator(**settings)
        .estimate_resources([atoms], multiplicities=[spins], properties=("energy",))
        .require_feasible()
    )
    topology = json.loads(scf.requests[0].identity.topology)["items"][0]
    assert topology["orbital"]["maximum_angular"] == 3
    assert topology["auxiliary"]["maximum_angular"] == 4
    assert bool(topology["orbital"].get("ecp_terms")) == ecp
    budget = ResourceBudget(
        host_bytes=scf.peak_bytes["host"] + 16 * coordinates**2,
        device_bytes=scf.peak_bytes["device"] + (128 << 20),
    )
    calculator = Calculator(**settings, resource_budget=budget)
    direction = np.asarray(result["motion"]).reshape(-1, 3)
    dense = calculator.hessian(atoms, multiplicity=spins)
    action = calculator.hessian_vector_product(atoms, direction, multiplicity=spins)
    for output in (dense, action):
        diagnostics = output.diagnostics
        assert diagnostics["df_workspace_bytes"] > 0
        assert diagnostics["numerical_host_fallbacks"] == 0
        resources = diagnostics["resources"]
        ledger = resources["hessian_response"]["device_ledger"]
        assert ledger["peak_bytes"] <= budget.device_bytes
        assert (
            ledger["live_bytes"]
            == resources["observation"]["device_ledger"]["live_bytes"]
        )
    matrix = dense.matrix
    np.testing.assert_allclose(
        matrix, np.asarray(result["hessian"]).reshape(matrix.shape), atol=3e-9, rtol=0
    )
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix.reshape(coordinates, -1, 3).sum(axis=1), 0, atol=3e-9
    )
    np.testing.assert_allclose(
        matrix @ direction.ravel(), action.value.ravel(), atol=3e-9, rtol=0
    )
    # The B3LYP f/g ECP step scan requires a finer pair for the axial curvature.
    # Preserve the independent fixed-point and curvature tolerances.
    steps = (0.00025, 0.000125) if ecp and method_case == "b3lyp" else (0.002, 0.001)
    _qualify_benchmark_matrix(
        {**result, "hessian": matrix, "hvp": action.value.ravel()},
        method_case=method_case,
        fitted=True,
        ecp=ecp,
        steps=steps,
    )


@pytest.mark.parametrize(
    "spins,split_grid", ((1, False), (2, True)), ids=("rks-shared", "uks-separate")
)
@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "fitted-g"))
def test_r2scan_rvv10_f_orbital_hessian_budget(
    spins: int, split_grid: bool, fitted: bool
) -> None:
    """Budget the actual native f/g endpoint, then check independent energies."""
    pytest.importorskip("pyscf")
    result = _native(
        spins,
        fitted,
        True,
        False,
        True,
        method_case="r2scan-rvv10",
        orbital_angular=3,
        auxiliary_g=fitted,
        split_grid=split_grid,
    )
    assert result["orbital_angular"] == 3
    coordinates = 3 * result["atoms"]
    centers = np.asarray(result["basis"][:coordinates]).reshape(-1, 3)
    atoms = [("H", center) for center in centers]
    direction = np.asarray(result["motion"]).reshape(-1, 3)

    settings = {
        "method": "r2scan+rvv10-rks" if spins == 1 else "r2scan+rvv10-uks",
        "device": "cuda",
        "basis": _compact_shells(len(atoms), 0.7, 0.9, 3, 0.48),
        "basis_representation": "spherical",
        "density_fitting": "cuda" if fitted else "none",
        "auxiliary_basis": _compact_shells(len(atoms), 0.8, 0.65, 4, 0.6)
        if fitted
        else None,
        "initial_guess": None,
        "max_iterations": 250,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8),
            nonlocal_grid=GridSpec(radial_points=6, angular_polar=3, angular_azimuth=6)
            if split_grid
            else None,
            tile_points=64,
        ),
    }
    scf = (
        Calculator(**settings)
        .estimate_resources([atoms], multiplicities=[spins], properties=("energy",))
        .require_feasible()
    )
    topology = json.loads(scf.requests[0].identity.topology)["items"][0]
    assert topology["orbital"]["maximum_angular"] == 3
    assert (topology["nonlocal_grid_points"] != topology["grid_points"]) == split_grid
    if fitted:
        assert topology["auxiliary"]["maximum_angular"] == 4
    budget = ResourceBudget(
        host_bytes=scf.peak_bytes["host"] + 16 * coordinates**2,
        device_bytes=scf.peak_bytes["device"] + (128 << 20),
    )
    calculator = Calculator(**settings, resource_budget=budget)
    dense = calculator.hessian(atoms, multiplicity=spins)
    action = calculator.hessian_vector_product(atoms, direction, multiplicity=spins)
    matrix = dense.matrix
    for output in (dense, action):
        diagnostics = output.diagnostics
        assert diagnostics["vv10_reference_pairs"] > 0
        assert diagnostics["numerical_host_fallbacks"] == 0
        assert bool(diagnostics["df_workspace_bytes"]) == fitted
        resources = diagnostics["resources"]
        ledger = resources["hessian_response"]["device_ledger"]
        assert ledger["peak_bytes"] <= budget.device_bytes
        assert (
            ledger["live_bytes"]
            == resources["observation"]["device_ledger"]["live_bytes"]
        )
    np.testing.assert_allclose(
        matrix, np.asarray(result["hessian"]).reshape(matrix.shape), atol=3e-9, rtol=0
    )
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix.reshape(coordinates, -1, 3).sum(axis=1), 0, atol=3e-9
    )
    np.testing.assert_allclose(
        matrix @ direction.ravel(), action.value.ravel(), atol=3e-9, rtol=0
    )
    _qualify_benchmark_matrix(
        {**result, "hessian": matrix, "hvp": action.value.ravel()},
        method_case="r2scan-rvv10",
        fitted=fitted,
        steps=(0.002, 0.001),
    )


@pytest.mark.parametrize("spins", (1, 2), ids=("rks", "uks"))
def test_r2scan_cuda_direct_cartesian_f_orbital(spins: int) -> None:
    """Angular admission must also preserve the distinct Cartesian normalization."""
    pytest.importorskip("pyscf")
    result = _native(
        spins, False, False, False, True, method_case="r2scan", orbital_angular=3
    )
    assert result["orbital_angular"] == 3 and result["nao"] == result["atoms"] + 10
    coordinates = 3 * result["atoms"]
    matrix = np.asarray(result["hessian"]).reshape(coordinates, coordinates)
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix.reshape(coordinates, -1, 3).sum(axis=1), 0, atol=3e-9
    )
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=3e-9, rtol=0
    )
    _qualify_benchmark_matrix(result, method_case="r2scan", spherical=False)


@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
def test_pbe_d4_cuda_complete_hessian(fitted: bool) -> None:
    """Bent water exercises the complete standard-profile D4/ATM/EEQ action."""
    pytest.importorskip("pyscf")
    pytest.importorskip("dftd4")
    result = _native(
        1, fitted, True, False, True, method_case="pbe-d4", benchmark_water=True
    )
    coordinates = 3 * result["atoms"]
    assert result["correction_directions"] == coordinates + 1
    matrix = np.asarray(result["hessian"]).reshape(coordinates, coordinates)
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix @ np.tile(np.eye(3), (result["atoms"], 1)), 0, atol=3e-9
    )
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=3e-9, rtol=0
    )
    _qualify_benchmark_matrix(result, method_case="pbe-d4", fitted=fitted)


@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
@pytest.mark.parametrize("spins", (1, 2), ids=("rks", "uks"))
def test_r2scan3c_cuda_complete_hessian(spins: int, fitted: bool) -> None:
    """Qualify public budgeted dense/action publication against the full surface."""
    pytest.importorskip("pyscf")
    pytest.importorskip("dftd4")
    result = _native(spins, fitted, True, False, True, method_case="r2scan-3c")
    coordinates = 3 * result["atoms"]
    assert result["correction_directions"] == 1 + coordinates
    centers = np.asarray(result["basis"][:coordinates]).reshape(-1, 3)
    atoms = [("H", center) for center in centers]
    direction = np.asarray(result["motion"]).reshape(-1, 3)
    settings = {
        "method": "r2scan-3c-rks" if spins == 1 else "r2scan-3c-uks",
        "device": "cuda",
        "density_fitting": "cuda" if fitted else "none",
        "initial_guess": None,
        "max_iterations": 250,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8),
            tile_points=64,
        ),
    }
    plan = (
        Calculator(**settings)
        .estimate_resources([atoms], multiplicities=[spins], properties=("energy",))
        .require_feasible()
    )
    assert any(
        row.name == "gCP retained CUDA owner" and row.bytes for row in plan.estimates
    )
    budget = ResourceBudget(
        host_bytes=plan.peak_bytes["host"] + 16 * coordinates**2,
        device_bytes=plan.peak_bytes["device"] + (128 << 20),
    )
    owner = Calculator(**settings, resource_budget=budget)
    dense = owner.hessian(atoms, multiplicity=spins)
    action = owner.hessian_vector_product(atoms, direction, multiplicity=spins)
    matrix = dense.matrix
    np.testing.assert_allclose(
        matrix, np.asarray(result["hessian"]).reshape(matrix.shape), atol=3e-9, rtol=0
    )
    for output, count in ((dense, coordinates), (action, 1)):
        diagnostics = output.diagnostics
        assert diagnostics["completed_directions"] == count
        assert diagnostics["numerical_host_fallbacks"] == 0
        resources = diagnostics["resources"]
        ledger = resources["hessian_response"]["device_ledger"]
        assert ledger["peak_bytes"] <= budget.device_bytes
        assert (
            ledger["live_bytes"]
            == resources["observation"]["device_ledger"]["live_bytes"]
        )
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix.reshape(coordinates, -1, 3).sum(axis=1), 0, atol=3e-9
    )
    np.testing.assert_allclose(
        matrix @ direction.ravel(), action.value.ravel(), atol=3e-9, rtol=0
    )
    _qualify_benchmark_matrix(
        {**result, "hessian": matrix, "hvp": action.value.ravel()},
        method_case="r2scan-3c",
        fitted=fitted,
    )


@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
@pytest.mark.parametrize("spins,method_case", ((1, "pbe0"), (2, "b3lyp")))
def test_semilocal_cuda_d_orbital_hessian_against_independent_energies(
    spins: int, method_case: str, fitted: bool
) -> None:
    """Exercise non-s response with the complete linked d-shell inventory."""
    pytest.importorskip("pyscf")
    result = _native(
        spins, fitted, True, False, True, orbital_angular=2, method_case=method_case
    )
    assert result["orbital_angular"] == 2 and result["nao"] > 6
    matrix = np.asarray(result["hessian"]).reshape(3 * result["atoms"], -1)
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=2e-8, rtol=0
    )
    # The d-shell B3LYP energy stencil only reaches its asymptotic regime on
    # the finer pair; keep the same matrix tolerance as the small fixtures.
    _qualify_benchmark_matrix(
        result, method_case=method_case, fitted=fitted, steps=(0.002, 0.001)
    )


@pytest.mark.parametrize("spins", (1, 2), ids=("rks", "uks"))
@pytest.mark.parametrize("provider", ("direct", "df-exact-lr", "df-fitted-lr"))
def test_wb97mv_cuda_d_orbital_hessian_against_independent_energies(
    spins: int, provider: str
) -> None:
    """Independently reconverge the complete physical model with d orbitals."""
    pytest.importorskip("pyscf")
    fitted = provider != "direct"
    result = _native(
        spins,
        fitted,
        True,
        False,
        True,
        auxiliary_d=fitted,
        orbital_angular=2,
        fitted_range=provider == "df-fitted-lr",
    )
    _qualify_wb97mv_d_orbital_receipt(result, fitted=fitted)


def _qualify_wb97mv_d_orbital_receipt(result: dict, *, fitted: bool) -> None:
    """Check a captured GPU result without holding a GPU during CPU reconvergence."""
    assert result["orbital_angular"] == 2 and result["nao"] > 6
    coordinates = 3 * result["atoms"]
    matrix = np.asarray(result["hessian"]).reshape(coordinates, coordinates)
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix @ np.tile(np.eye(3), (result["atoms"], 1)), 0, atol=3e-9, rtol=0
    )
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=3e-9, rtol=0
    )
    _qualify_benchmark_matrix(result, fitted=fitted, steps=(0.002, 0.001))


@pytest.mark.parametrize(
    "spins,fitted,fitted_range,treutler",
    [(1, False, False, False), (2, True, True, False), (1, True, False, True)],
    ids=("direct", "fitted-lr", "exact-lr-treutler"),
)
def test_wb97mv_split_grid_against_reconverged_energies(
    spins: int, fitted: bool, fitted_range: bool, treutler: bool
) -> None:
    """Distinct grid domains must produce the full derivative of the two-grid SCF energy."""
    pytest.importorskip("pyscf")
    # Heteronuclear Na(ECP)H exercises unequal v4 offsets and Bragg radii on
    # both domains, which a same-element tensor fixture cannot distinguish.
    result = _native(
        spins,
        fitted,
        True,
        treutler,
        True,
        fitted_range=fitted_range,
        split_grid=True,
        split_treutler=treutler,
    )
    _qualify_wb97mv_split_grid(result, fitted=fitted, ecp=treutler)


def _qualify_wb97mv_split_grid(result: dict, *, fitted: bool, ecp: bool) -> None:
    """Run the same independent matrix gate on a saved Slurm receipt."""
    if "atom_offsets" in result:
        for prefix in ("", "vv10_"):
            assert len(set(np.diff(result[f"{prefix}atom_offsets"]))) > 1
            assert len(set(result[f"{prefix}atomic_radii"])) > 1
    coordinates = 3 * result["atoms"]
    assert result["vv10_grid_directions"] == 2 * (coordinates + 1)
    matrix = np.asarray(result["hessian"]).reshape(coordinates, coordinates)
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix @ result["motion"], result["hvp"], atol=3e-9, rtol=0
    )
    np.testing.assert_allclose(
        matrix @ np.tile(np.eye(3), (result["atoms"], 1)), 0, atol=3e-9
    )
    _qualify_benchmark_matrix(result, fitted=fitted, ecp=ecp)


def test_wb97mv_split_grid_combined_moving_geometry_budget() -> None:
    """One shared budget forces bounded replay without changing the HVP."""
    replay = _native(1, False, True, False, False, split_grid=True, split_budget=True)
    resident = _native(1, False, True, False, False, split_grid=True)
    assert replay["shared_geometry_budget"]
    np.testing.assert_allclose(replay["hvp"], resident["hvp"], atol=3e-9, rtol=0)


@pytest.mark.parametrize("method", _SPLIT_HYBRID_EXCHANGE)
@pytest.mark.parametrize("spins", (1, 2), ids=("rks", "uks"))
@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
def test_split_hybrid_ecp_stationary_energy(
    method: str, spins: int, fitted: bool
) -> None:
    """Rebuild scalar ECP, J/K, and upstream split XC at each displaced geometry."""
    option = "M062X" if method == "m06-2x" else "MN15"
    if os.environ.get(f"GENERATIVEQC_{option}_HESSIAN_CUDA_TEST") != "1":
        pytest.skip(f"requires the opt-in {method} build")
    result = _native(spins, fitted, True, True, True, method_case=method)
    _qualify_benchmark_matrix(result, method_case=method, fitted=fitted, ecp=True)


@pytest.mark.parametrize("method", _SPLIT_HYBRID_EXCHANGE)
@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
def test_split_hybrid_ecp_f_stationary_energy(method: str, fitted: bool) -> None:
    """Qualify UKS scalar ECP curvature with f orbitals and, for DF, a g fit."""
    option = "M062X" if method == "m06-2x" else "MN15"
    if (
        os.environ.get(f"GENERATIVEQC_{option}_HESSIAN_CUDA_TEST") != "1"
        or os.environ.get("GENERATIVEQC_SPLIT_HYBRID_F_TEST") != "1"
    ):
        pytest.skip(f"requires linked {method} and high-angular second derivatives")
    result = _native(
        2,
        fitted,
        True,
        True,
        True,
        orbital_angular=3,
        auxiliary_g=fitted,
        method_case=method,
    )
    # On this M06-2X grid a coarser stencil crosses the smooth tail join;
    # a finer stencil also keeps the MN15 Direct reference near its SCF seed.
    steps = (0.002, 0.001) if fitted and method == "mn15" else (0.0005, 0.00025)
    _qualify_benchmark_matrix(
        result, method_case=method, fitted=fitted, ecp=True, steps=steps
    )


def _qualify_benchmark_matrix(
    result: dict,
    *,
    method_case: str | None = None,
    ecp: bool = False,
    fitted: bool = False,
    spherical: bool = True,
    steps: tuple[float, float] = (0.004, 0.002),
) -> None:
    """Keep each raw entry and both independently reconverged energy stencils."""
    coordinates = 3 * result["atoms"]
    actual = np.array(result["hessian"]).reshape(coordinates, coordinates)
    axes = np.eye(coordinates).reshape(coordinates, result["atoms"], 3)
    expected_matrix = np.zeros_like(actual)
    stencil_matrices = np.zeros((2, coordinates, coordinates))
    gradient = result.get("gradient")
    expected_gradient = None
    with nullcontext("") if method_case else _signed_density_libxc() as code:
        energy = MolecularEnergy(
            result,
            fitted=fitted,
            spherical=spherical,
            ecp=ecp,
            code=code,
            method_case=method_case,
        )
        np.testing.assert_allclose(
            energy(np.zeros((result["atoms"], 3)), reference=True),
            result["energy"],
            atol=5e-9,
        )
        if gradient is not None:
            # Test the actual stationary force owner against the same independent
            # energy surface used below; a symmetric Hessian cannot prove this.
            gradient_stencils = np.array(
                [
                    [
                        (energy(step * axis) - energy(-step * axis)) / (2 * step)
                        for axis in axes
                    ]
                    for step in steps
                ]
            )
            expected_gradient = (4 * gradient_stencils[1] - gradient_stencils[0]) / 3
            np.testing.assert_allclose(gradient, expected_gradient, atol=3e-8, rtol=0)
        for row in range(coordinates):
            for column in range(row, coordinates):
                estimates = []
                for step in steps:
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
        if (
            result.get("r2scan3c", False)
            or "metric_rank" in result
            or method_case == "r2scan-rvv10"
            or method_case in _SPLIT_HYBRID_EXCHANGE
        ):
            np.testing.assert_allclose(
                result["hvp"], expected_matrix @ result["motion"], atol=8e-7, rtol=0
            )
    if directory := os.environ.get("GENERATIVEQC_STATIONARY_RECEIPTS"):
        from generativeqc_compiler.common.evidence import block_error

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        case = "water" if result.get("benchmark_water", False) else "h2"
        if method_case:
            source = ("-df" if fitted else "") + ("-ecp" if ecp else "")
            if rank := result.get("metric_rank"):
                source += f"-rank{rank}"
            if angular := result.get("orbital_angular", 0):
                source += f"-orbital{angular}"
            if result.get("auxiliary_d", False):
                source += "-auxd"
            if result.get("fitted_xc_warmup", False):
                source += "-xc-warmup"
            if method_case == "r2scan-rvv10":
                source += (
                    "-separate-grid" if "vv10_points" in result else "-shared-grid"
                )
            representation = "" if spherical else "-cartesian"
            case = f"{method_case}{source}{representation}-{result['spins']}-{result['atoms']}"
        elif "range_metric_rank" in result:
            case = f"wb97mv-fixed-rank-{result['spins']}-{result['metric_rank']}-{result['range_metric_rank']}"
        elif "vv10_points" in result:
            topology = "treutler" if "atom_offsets" in result else "product"
            provider = (
                "direct"
                if not fitted
                else "df-fitted-lr"
                if result.get("fitted_range")
                else "df-exact-lr"
            )
            case = f"wb97mv-split-grid-{topology}-{provider}-{result['spins']}-{result['atoms']}"
        elif angular := result.get("orbital_angular", 0):
            provider = "direct"
            if fitted:
                provider = (
                    "df-fitted-lr" if result.get("fitted_range") else "df-exact-lr"
                )
            case = f"wb97mv-{provider}-orbital{angular}-{result['spins']}-{result['atoms']}"
        receipt = {
            "case": case,
            "native_energy": result["energy"],
            "steps": list(steps),
            "actual": actual.tolist(),
            "energy_stencil_matrices": stencil_matrices.tolist(),
            "expected": expected_matrix.tolist(),
            "error": block_error(actual, expected_matrix, atol=8e-7, rtol=1e-7),
        }
        if expected_gradient is not None:
            receipt["gradient_error"] = block_error(
                gradient, expected_gradient, atol=3e-8, rtol=0
            )
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
                "fitted_range": result.get("fitted_range", False),
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
