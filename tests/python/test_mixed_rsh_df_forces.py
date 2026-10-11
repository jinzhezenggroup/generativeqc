"""Graph-based mixed DF admission and complete independently reconverged forces.

Real CUDA endpoints require GENERATIVEQC_DFT_CUDA_TEST=1 inside a finite Slurm
allocation. PySCF is an oracle only: full-range J/K are fitted, while omega-
dependent exchange is exact, matching the native mixed provider contract.
"""

from __future__ import annotations

import ast
import os
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import Atom, Calculator, KsOptions
from generativeqc.ks import (
    cuda_mixed_density_fitted_force_eligible,
    mixed_density_fitted_force_eligible,
)
from generativeqc_compiler.dft.grid import GridSpec, MolecularGrid
from generativeqc_compiler.method import (
    MethodSpec,
    compile_ks_execution_plan,
    original_nonlocal_correlation,
    resolve_method,
)
from generativeqc_compiler.method.stationary_cuda import (
    COMPOSITE_STATIONARY_CUDA_EXTERNAL_SOURCES,
    stationary_external_provider_sources,
)
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.tensor import execute


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_mixed_df_admission_uses_semantics_not_method_name(spin: str) -> None:
    method = resolve_method("WB97M-V", spin=spin)
    assert cuda_mixed_density_fitted_force_eligible(method)
    renamed = replace(method, identifier="unrelated-functional-label")
    assert cuda_mixed_density_fitted_force_eligible(renamed)
    assert mixed_density_fitted_force_eligible(renamed, backend="cpu")
    assert not mixed_density_fitted_force_eligible(renamed, backend="unsupported")
    for identifier in ("PBE", "PBE0", "CAM-B3LYP", "R2SCAN"):
        assert not cuda_mixed_density_fitted_force_eligible(
            resolve_method(identifier, spin=spin)
        )
    assert not cuda_mixed_density_fitted_force_eligible(None)


@pytest.mark.parametrize(
    "capability",
    (
        "cuda_ks",
        "native_range_exchange",
        "native_nonlocal_correlation",
        "cuda_nonlocal_correlation",
        "molecular_nonlocal_domain",
    ),
)
def test_mixed_df_requires_every_generated_provider_capability(
    capability: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import ks

    method = resolve_method("WB97M-V", spin="unpolarized")
    record = dict(ks._native_semilocal_record(method))
    record[capability] = False
    monkeypatch.setattr(ks, "_native_semilocal_record", lambda method: record)
    assert not cuda_mixed_density_fitted_force_eligible(method)


@pytest.mark.parametrize(
    "override,expected",
    (
        ({}, True),
        ({"_device_name": "cpu"}, True),
        ({"_precision_mode": "auto"}, False),
        ({"_basis": "def2-qzvp"}, False),
        ({"_automatic_libxc_name": "unqualified"}, False),
        ({"_dispersion_method_ir": object()}, False),
        ({"ecp": True}, False),
        ({"schedule": "host_unfused"}, False),
        ({"fitted": False}, False),
    ),
)
def test_public_mixed_df_gate_preserves_owner_boundaries(
    override: dict[str, Any], expected: bool
) -> None:
    """Execute the actual public predicate without loading a CUDA library."""
    from generativeqc import _native
    from generativeqc.ks import cuda_nonlocal_force_basis_eligible

    method = resolve_method("WB97M-V", spin="unpolarized")
    owner = SimpleNamespace(
        _device_name="cuda",
        _precision_mode=_native.PRECISION_FP64,
        _basis="sto-3g",
        _automatic_libxc_name=None,
        _dispersion_method_ir=None,
        _ks_options=SimpleNamespace(
            method_ir=method,
            execution_plan=compile_ks_execution_plan(method),
            xc_schedule=override.get("schedule", "device_fused"),
        ),
    )
    for attribute, value in override.items():
        if attribute.startswith("_"):
            setattr(owner, attribute, value)
    source = (
        Path(__file__).resolve().parents[2] / "python/generativeqc/calculator.py"
    ).read_text()
    assignment = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "density_fitted_force"
            for target in node.targets
        )
    )
    namespace = {
        "self": owner,
        "_native": _native,
        "basis_has_ecp": override.get("ecp", False),
        "density_fitting_mode": (
            _native.DENSITY_FITTING_CUDA
            if override.get("fitted", True)
            else _native.DENSITY_FITTING_NONE
        ),
        "cuda_nonlocal_force_basis_eligible": cuda_nonlocal_force_basis_eligible,
        "cuda_mixed_density_fitted_force_eligible": cuda_mixed_density_fitted_force_eligible,
        "mixed_density_fitted_force_eligible": mixed_density_fitted_force_eligible,
    }
    exec(  # noqa: S102 - Only the local repository's public predicate is executed.
        compile(ast.Module(body=[assignment], type_ignores=[]), str(__file__), "exec"),
        namespace,
    )
    assert namespace["density_fitted_force"] is expected


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("long_coefficient", (Fraction(4, 5), Fraction(19, 100)))
def test_other_functional_graphs_share_the_composite_inventory(
    spin: str, long_coefficient: Fraction, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Future provider promotion needs metadata, not a new functional dispatch."""
    from generativeqc import ks

    method = resolve_method(
        MethodSpec(
            "custom-PBE-range-VV10",
            (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
            short_range_exchange=Fraction(19, 100),
            long_range_exchange=long_coefficient,
            range_omega=Fraction(2, 5),
            nonlocal_correlation=original_nonlocal_correlation("vv10"),
        ),
        spin=spin,
    )
    plan = StationaryGradientPlan(method, StationaryMeanField(SCF_POINT_MODEL))
    assert (
        stationary_external_provider_sources(plan)
        == COMPOSITE_STATIONARY_CUDA_EXTERNAL_SOURCES
    )
    assert not cuda_mixed_density_fitted_force_eligible(method)
    record = dict(ks._native_semilocal_record(resolve_method("WB97M-V", spin=spin)))
    record["scf_domain"] = SCF_POINT_MODEL
    monkeypatch.setattr(ks, "_native_semilocal_record", lambda method: record)
    assert cuda_mixed_density_fitted_force_eligible(method) == (
        long_coefficient != Fraction(19, 100)
    )
    assert mixed_density_fitted_force_eligible(method, backend="cpu") == (
        long_coefficient != Fraction(19, 100)
    )


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_mixed_df_lr_weights_use_the_shared_exchange_algebra(spin: str) -> None:
    """The CPU correction is delta K_LR, not a second exact SR/LR Hamiltonian."""
    method = resolve_method("WB97M-V", spin=spin)
    plan = StationaryGradientPlan(method, StationaryMeanField(SCF_POINT_MODEL))
    short, long = (
        plan.range_exchange_primitive("exchange_" + region + "_range")
        for region in ("short", "long")
    )
    density = np.array([[0.7, 0.2], [0.1, 0.4]])[: plan.spin_blocks]
    feeds = {"density_left": density, "density_right": density[:, ::-1].copy()}
    correction = execute(
        plan.integral_block(
            "exchange_long_range", terms=2, mixed_density_fitted_correction=True
        ).weights,
        feeds,
    ).outputs["weights"]
    factor = -float(long.coefficient - short.coefficient) * (
        0.5 if spin == "polarized" else 0.25
    )
    np.testing.assert_allclose(
        correction, factor * np.sum(density * density[:, ::-1], axis=0), atol=1e-15
    )
    with pytest.raises(ValueError, match="matched SR/LR"):
        plan.integral_block(
            "exchange_short_range", terms=2, mixed_density_fitted_correction=True
        )


def _mixed_pyscf_energy(
    calc: Calculator, raw_atoms: list[tuple[str, tuple[float, ...]]], spin: int
) -> float:
    """Rebuild both auxiliary centers and both moving quadratures for each call."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import df, dft, gto, scf

    pyscf.lib.num_threads(1)
    atoms = tuple(Atom.from_value(atom) for atom in raw_atoms)
    labels = [f"{symbol}{index}" for index, (symbol, _) in enumerate(raw_atoms)]

    def basis_dict(selected: Any) -> dict[str, list]:
        result: dict[str, list] = {label: [] for label in labels}
        for shell in calc._shells_for_atoms(atoms, selected):
            result[labels[shell.atom_index]].append(
                [
                    shell.angular_momentum,
                    *[
                        (primitive.exponent, primitive.coefficient)
                        for primitive in shell.primitives
                    ],
                ]
            )
        return result

    mol = gto.M(
        atom=[
            (label, atom.position) for label, atom in zip(labels, atoms, strict=True)
        ],
        basis=basis_dict(calc._basis),
        unit="Bohr",
        cart=calc._representation_name == "cartesian",
        spin=spin,
        verbose=0,
    )
    fitted = df.DF(mol, auxbasis=basis_dict(calc._auxiliary_basis)).build()
    mean_field = dft.UKS(mol) if spin else dft.RKS(mol)
    mean_field.xc = "WB97M_V"

    def get_jk(
        molecule: Any,
        density: np.ndarray,
        hermi: int = 1,
        with_j: bool = True,
        with_k: bool = True,
        omega: float | None = None,
    ) -> Any:
        if omega is not None and omega != 0:
            return scf.hf.get_jk(
                molecule, density, hermi, with_j=with_j, with_k=with_k, omega=omega
            )
        return fitted.get_jk(density, hermi, with_j, with_k)

    mean_field.get_jk = get_jk
    grid = MolecularGrid(atoms, calc.ks_options.grid, multiplicity=spin + 1).explicit()
    for quadrature in (mean_field.grids, mean_field.nlcgrids):
        quadrature.coords = np.asarray(grid.points)
        quadrature.weights = np.asarray(grid.weights)
        quadrature.non0tab = None
    mean_field.small_rho_cutoff = 0
    mean_field.conv_tol = 1e-12
    mean_field.conv_tol_grad = 1e-10
    mean_field.max_cycle = 200
    energy = mean_field.kernel(dm0=mean_field.get_init_guess(key="1e"))
    assert mean_field.converged
    return float(energy)


def test_cpu_mixed_snapshot_is_revoked_by_warm_replay() -> None:
    from generativeqc._ks_snapshot import NativeKsSnapshot

    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.15, 0.13, 1.5))]
    calc = Calculator(
        method="wb97m-v",
        basis="sto-3g",
        device="cpu",
        density_fitting="auto",
        auxiliary_basis="def2-svp",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
    )
    with calc.prepare_batch([atoms], warm_start=True) as batch:
        batch.execute(strict=True, properties=("energy",))
        snapshot = NativeKsSnapshot(batch, 0)
        try:
            snapshot.check_current()
            batch.execute(strict=True, properties=("energy",))
            with pytest.raises(ValueError, match="stale"):
                snapshot.mixed_density_fitted_integral_derivatives(
                    2, 256 << 20, lr_correction=np.zeros((2, 3))
                )
        finally:
            snapshot.close()


def test_cpu_mixed_dense_limit_rejects_before_snapshot_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from generativeqc import _cpu_force_resources
    from generativeqc._dft_gradient import StationaryKsState

    original_inventory = _cpu_force_resources.cpu_force_inventory

    def oversized_inventory(basis: Any, **options: Any) -> Any:
        metadata = SimpleNamespace(
            natom=basis.natom, nao=65, nprimitive=basis.nprimitive, shells=basis.shells
        )
        return original_inventory(metadata, **options)

    monkeypatch.setattr(
        _cpu_force_resources, "cpu_force_inventory", oversized_inventory
    )

    def unexpected_export(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "unsupported orbital domain must reject before snapshot export"
        )

    monkeypatch.setattr(StationaryKsState, "from_native", unexpected_export)
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.1, 1.43, 1.1)),
        ("H", (-0.15, -1.45, 1.12)),
    ]
    calc = Calculator(
        method="wb97m-v",
        basis="def2-svp",
        device="cpu",
        density_fitting="auto",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
    )
    with (
        calc.prepare_batch([atoms]) as batch,
        pytest.raises(ValueError, match="dense-export"),
    ):
        batch._public_dft_cpu_force(0, atoms)


@pytest.mark.parametrize(
    "spin,representation,basis_name",
    (
        (0, "cartesian", "sto-3g"),
        (0, "spherical", "sto-3g"),
        (1, "cartesian", "sto-3g"),
        (1, "spherical", "sto-3g"),
        (0, "spherical", "sto-3g-water"),
        (0, "spherical", "def2-svp"),
        (0, "spherical", "def2-tzvp"),
        (0, "cartesian", "def2-svp"),
        (0, "cartesian", "def2-tzvp"),
        (2, "spherical", "def2-svp"),
        (2, "spherical", "def2-tzvp"),
    ),
)
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_complete_mixed_df_forces_against_independent_energies(
    spin: int, representation: str, basis_name: str, device: str
) -> None:
    if device == "cuda" and os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an allocated native CUDA library/device")
    if device == "cuda":
        assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.15, 0.13, 1.5))]
    if spin:
        atoms.append(("H", (1.8, -0.1, -0.3)))
    if basis_name != "sto-3g":
        atoms = [
            ("O", (0.02, -0.03, 0.04)),
            ("H", (0.1, 1.43, 1.1)),
            ("H", (-0.15, -1.45, 1.12)),
        ]
    calc = Calculator(
        method="wb97m-v-uks" if spin else "wb97m-v-rks",
        basis=basis_name.removesuffix("-water"),
        basis_representation=representation,
        device=device,
        density_fitting="auto",
        auxiliary_basis="def2-svp",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    assert "forces" in calc.capabilities.supported_properties
    with calc.prepare_batch(
        [atoms], multiplicities=[spin + 1], warm_start=True
    ) as batch:
        cpu_work = []
        if device == "cpu":
            original_consumer = batch._public_dft_cpu_force

            def consume_cpu(index: int, current_atoms: Any) -> Any:
                result = original_consumer(index, current_atoms)
                cpu_work.append(result[1])
                return result

            batch._public_dft_cpu_force = consume_cpu
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        cold_reserve = (
            batch._stationary_cuda_execution.last_work[
                "native_integral_additional_device_reserve_bytes"
            ]
            if device == "cuda"
            else None
        )
        warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        work = (
            batch._stationary_cuda_execution.last_work
            if device == "cuda"
            else cpu_work[-1]
        )
        assert cold.converged and warm.converged
        assert (
            cold.executed_backend
            == warm.executed_backend
            == ("cpu_reference" if device == "cpu" else device)
        )
        assert warm.energy == pytest.approx(cold.energy, abs=2e-10)
        np.testing.assert_allclose(warm.forces, cold.forces, atol=2e-8, rtol=0)
        if device == "cuda":
            assert work["prepared_execution_reused"]
            assert cold_reserve > 0
            assert (
                work["native_integral_additional_device_reserve_bytes"] == cold_reserve
            )
            assert work["additional_bounds_exclude_df_response"]
            assert work["native_integral_resources"][
                "additional_device_peak_includes_lazy_lr"
            ]
            assert work["nonlocal_feature_d2h_bytes"] == 0
            assert work["nonlocal_seed_h2d_bytes"] == 0
        else:
            assert (
                work["stationary_integral_derivative_route"]
                == "prepared-mixed-density-fitted"
            )
            assert work["integral_quartet_passes"] == 1
            assert work["range_exchange_primitive_records"] > 0
            assert work["primitive_records"] == work["primitive_record_bound"]
            assert not work["density_fitted_response_resources_included"]
        assert cold.energy == pytest.approx(
            _mixed_pyscf_energy(calc, atoms, spin), abs=2e-8
        )
        step = 2e-4
        oracle_gradient = np.zeros((len(atoms), 3))
        for atom_index in range(len(atoms)):
            for axis in range(3):
                energies = []
                for direction in (1, -1):
                    displaced = list(atoms)
                    position = list(atoms[atom_index][1])
                    position[axis] += direction * step
                    displaced[atom_index] = (atoms[atom_index][0], tuple(position))
                    energies.append(_mixed_pyscf_energy(calc, displaced, spin))
                oracle_gradient[atom_index, axis] = (energies[0] - energies[1]) / (
                    2 * step
                )
        np.testing.assert_allclose(cold.forces, -oracle_gradient, atol=3e-6, rtol=0)
        np.testing.assert_allclose(cold.forces.sum(axis=0), 0, atol=2e-9, rtol=0)
        moved = list(atoms)
        moved[1] = (
            "H",
            tuple(
                position + delta
                for position, delta in zip(
                    atoms[1][1], (0.03, -0.02, 0.04), strict=True
                )
            ),
        )
        rebound = batch.execute(
            coordinates=([position for _, position in moved],),
            strict=True,
            properties=("energy", "forces"),
        ).items[0]
    fresh = calc.singlepoint(
        moved, multiplicity=spin + 1, properties=("energy", "forces")
    )
    assert rebound.energy == pytest.approx(fresh.energy, abs=2e-10)
    np.testing.assert_allclose(rebound.forces, fresh.forces, atol=2e-8, rtol=0)
    assert rebound.energy == pytest.approx(
        _mixed_pyscf_energy(calc, moved, spin), abs=2e-8
    )
