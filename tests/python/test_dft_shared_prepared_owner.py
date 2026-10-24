"""Output selection must not choose a second DFT execution/optimization path."""

from __future__ import annotations

import os
import typing
from types import SimpleNamespace

import pytest
from generativeqc.calculator import Calculator

if typing.TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "method",
    ["lda-rks", "pbe-rks", "pbe0-rks", "pbe0-uks", "r2scan-rks", "b3lyp-uks"],
)
@pytest.mark.parametrize("properties", [("energy",), ("energy", "forces")])
def test_singlepoint_dft_uses_one_prepared_route(
    method: str, properties: tuple[str, ...]
) -> None:
    """Exercise dispatch without allowing a native single-calculation fallback."""
    calls = []
    force = object() if "forces" in properties else None
    item = SimpleNamespace(
        energy=-1.0,
        forces=force,
        converged=True,
        iterations=4,
        energy_change=1e-12,
        density_rms=1e-12,
        executed_backend="cuda",
        basis_metadata={},
        accuracy=None,
        precision=None,
        physical_residual_rms=1e-12,
        ks_diagnostic=None,
        initial_guess=None,
        dispersion=None,
    )

    class Batch:
        resource_diagnostics = None
        ks_transport_diagnostics = (None,)

        def __enter__(self) -> typing.Self:
            return self

        def __exit__(self, *args: object) -> None:
            calls.append("close")

        def execute(
            self, *, strict: bool, properties: frozenset[str]
        ) -> SimpleNamespace:
            calls.append(("execute", strict, properties))
            return SimpleNamespace(items=(item,))

    calculator = Calculator.__new__(Calculator)
    calculator._method_name = method
    calculator._capabilities = SimpleNamespace(
        family="density_functional", supported_properties={"energy", "forces"}
    )
    calculator._resource_budget = None
    calculator._singlepoint_context = None
    calculator._dispersion_method_ir = None
    calculator._preflight_hf_basis = lambda *args, **kwargs: None
    calculator._effective_ks_options = lambda *args, **kwargs: None

    def prepare(systems: list[object], **kwargs: object) -> Batch:
        calls.append(("prepare", kwargs["warm_start"]))
        assert len(systems) == 1
        return Batch()

    calculator.prepare_batch = prepare
    result = calculator.singlepoint([("He", (0.0, 0.0, 0.0))], properties=properties)
    assert result.energy == item.energy
    assert result.forces is force
    assert calls == [
        ("prepare", False),
        ("execute", True, frozenset(properties)),
        "close",
    ]


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires an explicit finite Slurm CUDA allocation",
)
@pytest.mark.parametrize("method", ["pbe-rks", "pbe0-rks", "pbe-uks", "pbe0-uks"])
def test_complete_resident_energy_force_and_independent_gradient(
    method: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Qualify the complete consumer above the actual resident-value threshold."""
    import json
    from time import perf_counter

    import numpy as np
    from generativeqc import GridSpec, KsOptions
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import independent_global_hybrid_gradient
    from test_dft_complete_cuda import no_cpu_derivatives

    from benchmarks._cases import benchmark_cases

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    case = benchmark_cases()["water-tetramer-def2-svp-spherical"]
    calculator = Calculator(
        method=method,
        device="cuda",
        basis=case.generativeqc_basis,
        basis_representation=case.basis_representation,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
        ),
        energy_tolerance=1e-11,
        density_tolerance=1e-9,
        max_iterations=200,
    )
    endpoints = []
    for selection in ("0", "1"):
        monkeypatch.setenv("GENERATIVEQC_KS_RESIDENT_VALUES", selection)
        started = perf_counter()
        energy = calculator.singlepoint(case.atoms, properties=("energy",))
        energy_seconds = perf_counter() - started
        started = perf_counter()
        with no_cpu_derivatives():
            force = calculator.singlepoint(case.atoms, properties=("energy", "forces"))
        force_seconds = perf_counter() - started
        assert energy.converged and force.converged and energy.forces is None
        assert energy.energy == pytest.approx(force.energy, abs=2e-9)
        energy_work = energy.precision["operators"]
        force_work = force.precision["operators"]
        # Independent cold solves may stop on different iterations because CUDA
        # atomic reductions perturb DIIS at the roundoff level. Compare selected
        # operators/precision, and retain both actual work counts as evidence.
        assert [
            {key: value for key, value in row.items() if key != "count"}
            for row in energy_work
        ] == [
            {key: value for key, value in row.items() if key != "count"}
            for row in force_work
        ]
        assert (energy.initial_guess is None) == (force.initial_guess is None)
        for result in (energy, force):
            assert result.precision["operator_inventory_complete"]
            if result.initial_guess is not None:
                assert result.initial_guess["target_attempts"] == 1
                assert result.initial_guess["discarded_target_fock_builds"] == 0
        assert [row for row in energy_work if row["name"] == "coulomb_recurrence"] == [
            row for row in force_work if row["name"] == "coulomb_recurrence"
        ]
        endpoints.append(
            {
                "selection": selection,
                "energy_seconds": energy_seconds,
                "energy_force_seconds": force_seconds,
                "energy": force.energy,
                "forces": force.forces.tolist(),
                "energy_iterations": energy.iterations,
                "force_iterations": force.iterations,
                "energy_operators": energy_work,
                "force_operators": force_work,
            }
        )
    np.testing.assert_allclose(
        endpoints[0]["forces"], endpoints[1]["forces"], atol=2e-8, rtol=0
    )
    assert endpoints[0]["energy"] == pytest.approx(endpoints[1]["energy"], abs=2e-9)
    resident_work = endpoints[1]["energy_operators"]
    assert any(
        row["name"] == "coulomb_recurrence" and row["count"] > 0
        for row in resident_work
    )
    with calculator.prepare_batch([case.atoms], warm_start=False) as batch:
        force_work = []
        consumer = batch._public_dft_cuda_force

        def observed(index: int, atoms: object) -> tuple:
            result, work = consumer(index, atoms)
            force_work.append(work)
            return result, work

        monkeypatch.setattr(batch, "_public_dft_cuda_force", observed)
        with no_cpu_derivatives():
            actual = batch.execute(strict=True, properties=("energy", "forces")).items[
                0
            ]
        with NativeAO(
            case.atoms,
            basis=case.generativeqc_basis,
            representation=case.basis_representation,
        ) as basis:
            assert basis.nao >= 64
            state = StationaryKsState.from_native(batch, basis)
            try:
                reference_energy, reference_gradient = (
                    independent_global_hybrid_gradient(
                        basis,
                        state,
                        method,
                        xc="pbe0" if method.startswith("pbe0-") else "pbe",
                    )
                )
            finally:
                state._source.close()
        assert actual.energy == pytest.approx(reference_energy, abs=2e-8)
        np.testing.assert_allclose(
            actual.forces, -reference_gradient, atol=2e-7, rtol=0
        )
        work = force_work[0]
        assert (
            work["stationary_integral_derivative_route"] == "prepared-native-complete"
        )
        assert work["stationary_state_dw_upload_bytes"] == 0
        assert work["grid_density_source"] == "exact-final-scf-device-binding"
        assert work["native_integral_resources"]["final_state_export_reads"] == 0
        assert work["native_integral_resources"]["one_electron_h2d_bytes"] == 0
    (tmp_path / f"{method}-shared-owner.json").write_text(
        json.dumps(
            {
                "slurm_job": os.environ["SLURM_JOB_ID"],
                "endpoints": endpoints,
                "force_work": work,
                "independent_energy_error_hartree": abs(
                    actual.energy - reference_energy
                ),
                "independent_force_max_error_hartree_per_bohr": float(
                    np.max(np.abs(actual.forces + reference_gradient))
                ),
            },
            indent=2,
        )
        + "\n"
    )
