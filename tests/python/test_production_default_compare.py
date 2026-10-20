"""Host regression for the three unmodified production-performance policies."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks import production_default_compare as runner


def receipt(arm: str, *, grid_response: bool | None = None) -> dict[str, Any]:
    if grid_response is None:
        grid_response = arm != "gpu4pyscf"
    observations = []
    for i in (0, 1):
        settings = {
            "initial_guess": "minao" if arm == "gpu4pyscf" else "huckel",
            "xc_grid": {
                "rule": arm,
                "points": 200 + i,
                **(
                    {"version": 4, "level": 3, "pruning": "nwchem"}
                    if arm in runner.NATIVE_ARMS
                    else {}
                ),
            },
            **(
                {"grid_source": "implicit-native-default"}
                if arm in runner.NATIVE_ARMS
                else {}
            ),
            "selected_candidate_precision": "fp64" if arm == "standard" else "auto",
        }
        observation: dict[str, Any] = {
            "repeat": i,
            "seconds": float({"gpu4pyscf": 6, "standard": 4, "fast": 2}[arm] + i),
            "energy": -1.0,
            "forces": [[0.0, 0.0, 0.0]],
            "converged": True,
            "iterations": 12 + i,
            "grid_response": grid_response,
            "settings": settings,
        }
        if arm == "fast":
            observation["precision_validation"] = {
                "profile": "fast",
                "status": "qualified_mixed_gradient",
                "solve_count": 1,
                "scope": "qualification_not_per_call_observable_audit",
                "gradient_admission": "fixture-mixed",
                "fallback_reasons": [],
                "errors": [],
                "rejected_errors": [],
                "budget": {
                    "energy_abs": 1e-5,
                    "force_max_abs": 1e-4,
                    "force_rms": 3e-5,
                },
                "solves": [
                    {
                        "role": "candidate",
                        "requested_precision": "auto",
                        "seconds": 1.0,
                        "iterations": 12,
                        "force_evaluations": 1,
                        "fock_builds": 12,
                        "executed_backend": "cuda",
                        "energy_tolerance": 1e-6,
                        "density_tolerance": 1e-4,
                        "screening_tolerance": 1e-10,
                        "error": None,
                    }
                ],
            }
        observations.append(observation)
    return {
        "schema": runner.SCHEMA,
        "engine": arm,
        "status": "complete",
        "problem": runner.problem_identity("pbe", [("H", (0, 0, 0))]),
        "phase": "fresh-owner",
        "endpoint": "constructor-scf-energy-analytic-force-host-export",
        "repeats": 2,
        "samples": observations,
        "controls": {
            "scf_overrides": False,
            "initial_guess_overrides": False,
            "reference_grid_overrides": False,
            "native_explicit_grid_overrides": False,
            "fast_default_policy": "fast" if arm == "fast" else None,
            "force_grid_response_override": False,
        },
        "accelerator": {
            "name": "host-test-synthetic-GPU",
            "compute_capability": [12, 0],
            "total_global_memory_bytes": 32000000000,
            "multiprocessor_count": 170,
        },
        "environment": {"git": {"commit": "a" * 40, "dirty": False}},
        "source_sha256": "b" * 64,
        **({"native_library_sha256": "c" * 64} if arm in runner.NATIVE_ARMS else {}),
    }


def receipts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return receipt("gpu4pyscf"), receipt("standard"), receipt("fast")


def test_three_way_default_main_table_and_speedups() -> None:
    gpu, standard, fast = receipts()
    summary = runner.compare(gpu, standard, fast)
    row = summary["main_table"][0]
    assert row["gpu4pyscf_default_seconds"] == pytest.approx(6.5)
    assert row["generativeqc_standard_seconds"] == pytest.approx(4.5)
    assert row["generativeqc_fast_seconds"] == pytest.approx(2.5)
    assert row["standard_speedup_vs_gpu4pyscf"] == pytest.approx(6.5 / 4.5)
    assert row["fast_speedup_vs_gpu4pyscf"] == pytest.approx(6.5 / 2.5)
    assert row["fast_speedup_vs_standard"] == pytest.approx(4.5 / 2.5)
    assert summary["fast_statuses"] == ["qualified_mixed_gradient"] * 2
    assert summary["fast_internal_solve_counts"] == [1, 1]
    assert not summary["same_moving_grid_force_definition"]
    assert summary["force_definition_warning"]
    assert summary["science_equivalence_proved"] is False
    assert summary["accuracy"].startswith("not_qualified")
    table = runner.main_table_markdown([summary])
    assert table.count("| Method |") == 1
    assert "PBE0" not in table
    assert "6.500" in table and "4.500" in table and "2.500" in table


def test_fast_mode_never_masquerades_as_fp32_when_it_audits() -> None:
    gpu, standard, fast = receipts()
    report = fast["samples"][0]["precision_validation"]
    report.update(status="observed_met", solve_count=2, errors=audit_errors())
    report["solves"].append(
        {
            **report["solves"][0],
            "role": "fp64_audit",
            "requested_precision": "fp64",
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "screening_tolerance": 1e-14,
        }
    )
    summary = runner.compare(gpu, standard, fast)
    assert summary["fast_internal_solve_counts"] == [2, 1]
    assert summary["fast_statuses"] == ["observed_met", "qualified_mixed_gradient"]


@pytest.mark.parametrize(
    "fault",
    (
        "wrong_geometry",
        "wrong_method",
        "missing_repeat",
        "failed_run",
        "not_converged",
        "nan_time",
        "nan_energy",
        "nan_force",
        "wrong_role",
        "missing_fast_report",
        "missing_fast_solve_count",
        "different_force_within_fast",
        "standard_fast_force_mismatch",
        "partial_hardware",
        "different_gpu",
    ),
)
def test_fail_closed_on_invalid_three_arm_receipts(fault: str) -> None:
    gpu, standard, fast = receipts()
    if fault == "wrong_geometry":
        fast["problem"]["geometry_sha256"] = "different"
    elif fault == "wrong_method":
        fast["problem"]["method"] = "pbe0"
    elif fault == "missing_repeat":
        standard["samples"].pop()
    elif fault == "failed_run":
        gpu["status"] = "failed"
    elif fault == "not_converged":
        fast["samples"][0]["converged"] = False
    elif fault == "nan_time":
        fast["samples"][0]["seconds"] = float("nan")
    elif fault == "nan_energy":
        standard["samples"][0]["energy"] = float("nan")
    elif fault == "nan_force":
        gpu["samples"][0]["forces"] = [[float("nan"), 0.0, 0.0]]
    elif fault == "wrong_role":
        standard["engine"] = "fast"
    elif fault == "missing_fast_report":
        del fast["samples"][0]["precision_validation"]
    elif fault == "missing_fast_solve_count":
        del fast["samples"][0]["precision_validation"]["solve_count"]
    elif fault == "different_force_within_fast":
        fast["samples"][0]["grid_response"] = False
    elif fault == "standard_fast_force_mismatch":
        for row in fast["samples"]:
            row["grid_response"] = False
    elif fault in ("partial_hardware", "different_gpu"):
        for arm, record in (("gpu4pyscf", gpu), ("standard", standard), ("fast", fast)):
            record["accelerator"] = {
                "name": "RTX 5090",
                "compute_capability": [12, 0],
                "total_global_memory_bytes": 32000000000,
                "multiprocessor_count": 170,
            }
        if fault == "partial_hardware":
            del fast["accelerator"]
        else:
            fast["accelerator"]["name"] = "RTX 6000 Ada"
    with pytest.raises(ValueError):
        runner.compare(gpu, standard, fast)


def test_native_constructor_is_default_and_fast_uses_real_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runner.native_constructor_kwargs("pbe") == {
        "method": "pbe-rks",
        "basis": "def2-svp",
        "basis_representation": "spherical",
        "device": "cuda",
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    def standard(**kwargs: Any) -> Any:
        calls.append(("standard", kwargs))
        return SimpleNamespace()

    def fast(*, policy: str, **kwargs: Any) -> Any:
        calls.append(("fast", {"policy": policy, **kwargs}))
        return SimpleNamespace()

    fake = SimpleNamespace(Calculator=standard, PrecisionCalculator=fast)
    monkeypatch.setitem(sys.modules, "generativeqc", fake)
    runner.make_native_calculator("pbe", "standard")
    runner.make_native_calculator("pbe", "fast")
    assert calls[0] == ("standard", runner.native_constructor_kwargs("pbe"))
    assert calls[1] == (
        "fast",
        {"policy": "fast", **runner.native_constructor_kwargs("pbe")},
    )
    assert "energy_tolerance" not in calls[0][1]
    assert "screening_tolerance" not in calls[1][1]
    with pytest.raises(ValueError, match="standard or fast"):
        runner.make_native_calculator("pbe", "invented")


@pytest.mark.parametrize("method", ("pbe0", "wb97m-v", "pbe", "r2scan"))
def test_native_default_grid_is_implicit_after_2626(method: str) -> None:
    args = runner.native_constructor_kwargs(method)
    assert "ks_options" not in args
    assert args["basis"] == "def2-svp"
    assert not {
        "max_iterations",
        "energy_tolerance",
        "density_tolerance",
        "screening_tolerance",
        "initial_guess",
        "precision",
    }.intersection(args)


@pytest.mark.parametrize("method", ("pbe0-rks", "wb97m-v-rks", "r2scan-rks"))
def test_production_ks_default_is_treutler_level3(method: str) -> None:
    from generativeqc import GridSpec, KsOptions
    from generativeqc.ks import resolve_ks_options

    assert GridSpec().version == 1  # The explicit legacy tensor prescription.
    resolved = resolve_ks_options(method, KsOptions())
    assert resolved.grid == GridSpec.treutler(level=3)


def test_r2scan3c_has_canonical_basis_and_full_corrections() -> None:
    model = runner.problem_identity("r2scan-3c", [("H", (0, 0, 0))])
    assert model["method"] == "r2scan-3c"
    assert model["basis"] == "def2-mTZVPP"
    assert model["composite_corrections"] == ["D4", "gCP"]
    assert runner.METHODS["r2scan-3c"] == ("r2scan-3c-rks", "R2SCAN")
    options = runner.native_constructor_kwargs("r2scan-3c")
    assert options["method"] == "r2scan-3c-rks"
    assert "basis" not in options  # Native canonical method supplies def2-mTZVPP.
    assert "ks_options" not in options
    assert runner.problem_identity("r2scan", [("H", (0, 0, 0))]) != model


def test_offline_compare_and_render_without_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu, standard, fast = receipts()
    paths = {
        name: tmp_path / f"{name}.json" for name in ("gpu4pyscf", "standard", "fast")
    }
    for name, record in zip(paths, (gpu, standard, fast), strict=True):
        paths[name].write_text(json.dumps(record))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    report = tmp_path / "summary.json"
    runner.main(
        [
            "compare",
            "--gpu4pyscf",
            str(paths["gpu4pyscf"]),
            "--standard",
            str(paths["standard"]),
            "--fast",
            str(paths["fast"]),
            "--output",
            str(report),
        ]
    )
    result = json.loads(report.read_text())
    assert len(result["main_table"]) == 1
    table = tmp_path / "table.md"
    runner.main(["render", "--comparisons", str(report), "--output", str(table)])
    assert table.read_text().count("| Method |") == 1
    assert "2.500" in table.read_text()


def test_measurement_requires_finite_slurm_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    for arm in (*runner.ARMS, "all"):
        with pytest.raises(SystemExit) as result:
            runner.main(
                [arm, "--method", "pbe", "--output", str(tmp_path / "notrun.json")]
            )
        assert result.value.code == 2


def test_render_rejects_old_and_incomplete_records(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "old.json"
    invalid.write_text(json.dumps({"schema": "old", "main_table": []}))
    with pytest.raises(ValueError, match="older comparison"):
        runner.main(
            [
                "render",
                "--comparisons",
                str(invalid),
                "--output",
                str(tmp_path / "table.md"),
            ]
        )


def test_all_mode_runs_exactly_three_isolated_public_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user-facing all command must never substitute AUTO for Fast."""
    monkeypatch.setenv("SLURM_JOB_ID", "host-fixture")
    launched: list[str] = []

    def fake_run(arguments: list[str], *, check: bool) -> SimpleNamespace:
        assert check
        assert arguments[:3] == [
            sys.executable,
            "-m",
            "benchmarks.production_default_compare",
        ]
        arm = arguments[3]
        assert arm in runner.ARMS
        destination = Path(arguments[arguments.index("--output") + 1])
        destination.write_text(json.dumps(receipt(arm)))
        launched.append(arm)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    combined = tmp_path / "all.json"
    runner.main(
        [
            "all",
            "--method",
            "pbe",
            "--atoms",
            "3",
            "--repeats",
            "2",
            "--output",
            str(combined),
        ]
    )
    assert launched == list(runner.ARMS)
    result = json.loads(combined.read_text())
    assert len(result["main_table"]) == 1
    assert result["main_table"][0]["generativeqc_fast_seconds"] == pytest.approx(2.5)
    for arm in runner.ARMS:
        assert (tmp_path / f"all-{arm}.json").is_file()


def test_gpu4_scientific_defaults_and_grid_response_remain_unmodified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An independent reference default is not a grid/tolerance override."""
    import numpy as np

    measured: list[tuple[str, Any]] = []
    gradient = SimpleNamespace(grid_response=False)

    def gradient_kernel() -> np.ndarray:
        measured.append(("grid_response", gradient.grid_response))
        return np.zeros((1, 3))

    gradient.kernel = gradient_kernel
    grids = SimpleNamespace(
        level=3,
        coords=np.zeros((2, 3)),
        radi_method=lambda: None,
        prune=lambda: None,
        radii_adjust=None,
        becke_scheme=lambda: None,
    )
    engine = SimpleNamespace(
        xc=None,
        converged=True,
        cycles=12,
        grids=grids,
        max_cycle=50,
        conv_tol=1e-9,
        conv_tol_grad=None,
        diis_space=8,
        direct_scf=True,
        direct_scf_tol=1e-13,
        init_guess="minao",
        nuc_grad_method=lambda: gradient,
    )

    def kernel() -> float:
        measured.append(("scf", (engine.conv_tol, engine.max_cycle, engine.xc)))
        return -1.0

    engine.kernel = kernel
    cp = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None)),
        asnumpy=np.asarray,
    )
    gto = SimpleNamespace(M=lambda **kwargs: SimpleNamespace(**kwargs))
    dft = SimpleNamespace(rks=SimpleNamespace(RKS=lambda molecule: engine))
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "pyscf", SimpleNamespace(gto=gto))
    monkeypatch.setitem(sys.modules, "gpu4pyscf", SimpleNamespace(dft=dft))
    monkeypatch.setitem(sys.modules, "gpu4pyscf.dft", dft)
    sample = runner.gpu4_sample("pbe", [("H", (0.0, 0.0, 0.0))])
    assert measured == [
        ("scf", (1e-9, 50, "PBE")),
        ("grid_response", False),
    ]
    assert not sample["grid_response"]
    assert sample["settings"]["xc_grid"]["level"] == 3
    assert sample["settings"]["initial_guess"] == "minao"


def test_fast_reports_actual_strict_fallback_not_automatic_fp32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default Fast fallback and all its solves belong to the timed endpoint."""
    import numpy as np

    grid = SimpleNamespace(
        version=1,
        radial_points=48,
        angular_polar=16,
        angular_azimuth=32,
        pruning="none",
    )
    candidate = SimpleNamespace(
        ks_options=SimpleNamespace(grid=grid),
        initial_guess=SimpleNamespace(kind="huckel"),
        _max_iterations=100,
        _energy_tolerance=1e-10,
        _density_tolerance=1e-8,
        _diis_history=8,
        _screening_tolerance=1e-12,
        _library=SimpleNamespace(_name="/tmp/fake-native-library.so"),
    )
    solve = SimpleNamespace(
        role="fp64_audit",
        requested_precision="fp64",
        iterations=11,
        fock_builds=12,
        seconds=0.1,
    )
    validation = SimpleNamespace(
        profile="fast",
        status="observed_met",
        scope="relaxed_target",
        solve_count=2,
        gradient_admission="fp64-after-budget-rejection",
        fallback_reasons=("candidate_exceeded_observable_budget",),
        budget=None,
        solves=(solve, solve),
    )
    validation.to_dict = lambda: {
        "profile": validation.profile,
        "status": validation.status,
        "solve_count": validation.solve_count,
        "solves": [
            {
                **vars(solve),
                "executed_backend": "cuda",
                "energy_tolerance": 1e-12,
                "density_tolerance": 1e-10,
                "screening_tolerance": 1e-14,
                "force_evaluations": 1,
                "error": None,
            }
        ]
        * 2,
    }
    result = SimpleNamespace(
        converged=True,
        executed_backend="cuda",
        energy=-1.5,
        forces=np.zeros((1, 3)),
        iterations=11,
        initial_guess={"kind": "huckel", "outcome": "used"},
        precision_validation=validation,
        precision={"mixed_stage_fock_builds": 0, "gradient": {"returned_mode": "fp64"}},
        ks_diagnostic=SimpleNamespace(grid_points=24576),
    )
    precision_calculator = SimpleNamespace(
        _candidate=candidate,
        _candidate_precision="fp64",
        singlepoint=lambda atoms, properties: result,
    )
    cp = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None))
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(
        runner, "make_native_calculator", lambda method, arm: precision_calculator
    )
    sample = runner.native_sample("pbe", [("H", (0.0, 0.0, 0.0))], "fast")
    assert sample["settings"]["selected_candidate_precision"] == "fp64"
    assert sample["precision_validation"]["solve_count"] == 2
    assert sample["precision_validation"]["status"] == "observed_met"
    assert sample["gradient_arithmetic"] == "fp64"
    assert sample["mixed_fock_builds"] == 0
    assert sample["precision_validation"]["solves"][0]["energy_tolerance"] == 1e-12
    assert sample["precision_validation"]["solves"][0]["executed_backend"] == "cuda"


def test_gpu4_composite_correction_is_timed_and_completes_energy_forces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent CPU D4/gCP must be *inside* the full GPU4 E+F call."""
    import numpy as np

    executed: list[str] = []
    canonical_basis = {"H": [[0, [1.0, 1.0]]]}
    monkeypatch.setattr(runner, "reference_r2scan3c_basis", lambda: canonical_basis)

    def reference_correction(atoms: Any) -> Any:
        assert len(atoms) == 1
        executed.append("prepare-D4-gCP")

        def evaluate() -> tuple[float, np.ndarray]:
            executed.append("evaluate-D4-gCP")
            return 0.125, np.array([[0.25, 0.5, 0.75]])

        return evaluate

    monkeypatch.setattr(runner, "reference_r2scan3c_correction", reference_correction)
    grad = SimpleNamespace(
        grid_response=False,
        kernel=lambda: np.array([[1.0, 2.0, 3.0]]),
    )
    grids = SimpleNamespace(
        level=3,
        coords=np.zeros((2, 3)),
        radi_method=lambda: None,
        prune=lambda: None,
        radii_adjust=None,
        becke_scheme=lambda: None,
    )
    model = SimpleNamespace(
        xc=None,
        converged=True,
        cycles=10,
        grids=grids,
        max_cycle=50,
        conv_tol=1e-9,
        conv_tol_grad=None,
        diis_space=8,
        direct_scf=True,
        direct_scf_tol=1e-13,
        init_guess="minao",
        nuc_grad_method=lambda: grad,
    )

    def kernel() -> float:
        executed.append("electronic-R2SCAN")
        assert model.xc == "R2SCAN"
        return -2.0

    model.kernel = kernel

    def molecule(**kwargs: Any) -> Any:
        assert kwargs["basis"] == canonical_basis
        return SimpleNamespace(**kwargs)

    cp = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None)),
        asnumpy=np.asarray,
    )
    dft = SimpleNamespace(rks=SimpleNamespace(RKS=lambda mol: model))
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(
        sys.modules, "pyscf", SimpleNamespace(gto=SimpleNamespace(M=molecule))
    )
    monkeypatch.setitem(sys.modules, "gpu4pyscf", SimpleNamespace(dft=dft))
    monkeypatch.setitem(sys.modules, "gpu4pyscf.dft", dft)

    result = runner.gpu4_sample("r2scan-3c", [("H", (0.0, 0.0, 0.0))])
    assert executed == [
        "electronic-R2SCAN",
        "prepare-D4-gCP",
        "evaluate-D4-gCP",
    ]
    assert result["energy"] == pytest.approx(-1.875)
    np.testing.assert_allclose(result["forces"], [[-1.25, -2.5, -3.75]], atol=0, rtol=0)
    assert result["settings"]["composite_corrections"]["included_in_timing"]
    assert result["settings"]["composite_corrections"]["energy_hartree"] == 0.125
    assert result["grid_response"] is False


def audit_errors() -> list[dict[str, Any]]:
    return [
        {
            "observable": observable,
            "norm": norm,
            "unit": unit,
            "difference": 0.0,
            "reference_norm": 1.0,
            "allowance": {"absolute": 1e-5, "max_abs": 1e-4, "rms": 3e-5}[norm],
        }
        for observable, norm, unit in (
            ("energy", "absolute", "Eh"),
            ("forces", "max_abs", "Eh/bohr"),
            ("forces", "rms", "Eh/bohr"),
        )
    ]


@pytest.mark.parametrize(
    "fault",
    (
        "warm",
        "energy_only",
        "missing_science",
        "missing_hardware",
        "empty_hardware",
        "override",
        "duplicate_repeat",
        "missing_repeat_id",
        "unequal_repeats",
        "different_binary",
        "different_source",
        "missing_source",
        "failed_fast_audit",
        "missing_solves",
        "wrong_solve_count",
        "missing_budget",
        "missing_solve_controls",
        "failed_last_solve",
        "missing_candidate_precision",
        "boolean_time",
        "boolean_converged",
    ),
)
def test_reject_incomplete_protocol_and_provenance(fault: str) -> None:
    gpu, standard, fast = records = receipts()
    report = fast["samples"][0]["precision_validation"]
    if fault == "warm":
        for record in records:
            record["phase"] = "warm-reused-density"
    elif fault == "energy_only":
        for record in records:
            record["endpoint"] = "energy-only"
    elif fault == "missing_science":
        for record in records:
            record["problem"] = {"method": "pbe", "atoms": 1}
    elif fault == "missing_hardware":
        for record in records:
            del record["accelerator"]
    elif fault == "empty_hardware":
        for record in records:
            record["accelerator"] = {}
    elif fault == "override":
        for record in records:
            record["controls"]["scf_overrides"] = True
    elif fault == "duplicate_repeat":
        fast["samples"][1]["repeat"] = 0
    elif fault == "missing_repeat_id":
        del fast["samples"][1]["repeat"]
    elif fault == "unequal_repeats":
        fast["samples"].pop()
        fast["repeats"] = 1
    elif fault == "different_binary":
        fast["native_library_sha256"] = "d" * 64
    elif fault == "different_source":
        fast["source_sha256"] = "d" * 64
    elif fault == "missing_source":
        del fast["environment"]
    elif fault == "failed_fast_audit":
        report["status"] = "observed_unmet"
    elif fault == "missing_solves":
        del report["solves"]
    elif fault == "wrong_solve_count":
        report["solve_count"] = 2
    elif fault == "missing_budget":
        del report["budget"]
    elif fault == "missing_solve_controls":
        del report["solves"][0]["screening_tolerance"]
    elif fault == "failed_last_solve":
        report["solves"][-1]["error"] = "SCF failed"
    elif fault == "missing_candidate_precision":
        del fast["samples"][0]["settings"]["selected_candidate_precision"]
    elif fault == "boolean_time":
        fast["samples"][0]["seconds"] = True
    elif fault == "boolean_converged":
        fast["samples"][0]["converged"] = "true"
    with pytest.raises(ValueError):
        runner.compare(gpu, standard, fast)


def test_render_recomputes_rows_from_retained_receipts() -> None:
    summary = runner.compare(*receipts())
    summary["main_table"][0].update(
        generativeqc_fast_seconds=float("nan"), fast_speedup_vs_gpu4pyscf=999
    )
    summary["science_equivalence_proved"] = True
    summary["accuracy"] = "qualified"
    table = runner.main_table_markdown([summary])
    assert "2.500" in table and "2.60x" in table
    assert "nan" not in table and "999" not in table
    assert "Accuracy unverified" in table and "Stock force definitions differ" in table
    summary["receipts"]["fast"]["samples"][0]["energy"] = float("nan")
    with pytest.raises(ValueError):
        runner.main_table_markdown([summary])


def test_render_rejects_forged_schema_only_table(tmp_path: Path) -> None:
    fake = {
        "schema": runner.COMPARISON_SCHEMA,
        "main_table": [
            {
                "method": "pbe",
                "atoms": 96,
                "gpu4pyscf_default_seconds": 1.0,
                "generativeqc_standard_seconds": -1.0,
                "generativeqc_fast_seconds": float("nan"),
                "standard_speedup_vs_gpu4pyscf": 999.0,
                "fast_speedup_vs_gpu4pyscf": float("inf"),
            }
        ],
    }
    source, output = tmp_path / "forged.json", tmp_path / "table.md"
    source.write_text(json.dumps(fake))
    with pytest.raises(ValueError):
        runner.main(["render", "--comparisons", str(source), "--output", str(output)])
    assert not output.exists()


def test_legitimate_failed_candidate_and_strict_audit_remain_reported() -> None:
    gpu, standard, fast = receipts()
    for sample in fast["samples"]:
        report = sample["precision_validation"]
        successful = report["solves"][0]
        report["solves"] = [
            {
                **successful,
                "iterations": None,
                "force_evaluations": None,
                "error": "candidate failed",
            },
            {
                **successful,
                "role": "fp64_fallback",
                "requested_precision": "fp64",
                "energy_tolerance": 1e-12,
            },
            {
                **successful,
                "role": "fp64_audit",
                "requested_precision": "fp64",
                "energy_tolerance": 1e-12,
            },
        ]
        report.update(
            solve_count=3,
            status="observed_met",
            fallback_reasons=["budget_fock_failed_fp64_fallback"],
            errors=audit_errors(),
        )
    summary = runner.compare(gpu, standard, fast)
    assert summary["fast_internal_solve_counts"] == [3, 3]
    assert summary["receipts"]["fast"]["samples"][0]["precision_validation"]["solves"][
        0
    ]["error"]
    assert "Accuracy unverified" in runner.main_table_markdown([summary])


@pytest.mark.parametrize("fault", ("missing", "incomplete", "nonfinite", "unmet"))
def test_fast_observed_met_requires_successful_complete_audit(fault: str) -> None:
    gpu, standard, fast = receipts()
    report = fast["samples"][0]["precision_validation"]
    report["status"] = "observed_met"
    report["solves"].append(
        {**report["solves"][0], "role": "fp64_audit", "requested_precision": "fp64"}
    )
    report["solve_count"] = 2
    if fault != "missing":
        report["errors"] = audit_errors()
        if fault == "incomplete":
            report["errors"].pop()
        elif fault == "nonfinite":
            report["errors"][0]["difference"] = float("nan")
        else:
            report["errors"][0]["difference"] = 1.0
    with pytest.raises(ValueError):
        runner.compare(gpu, standard, fast)


def test_receipt_budget_tracks_public_fast_default() -> None:
    from dataclasses import asdict

    from generativeqc import PrecisionPolicy

    assert runner.DEFAULT_FAST_BUDGET == asdict(PrecisionPolicy("fast").budget)


@pytest.mark.parametrize(
    "fault",
    (
        "nondefault_budget",
        "candidate_only_audit",
        "wrong_audit_precision",
        "missing_audit_role",
        "nondefault_allowance",
        "unrecorded_verification",
        "false_rejected_errors",
    ),
)
def test_fast_default_and_audit_census_fail_closed(fault: str) -> None:
    gpu, standard, fast = receipts()
    report = fast["samples"][0]["precision_validation"]
    if fault == "nondefault_budget":
        report["budget"]["energy_abs"] = 1.0
    else:
        report.update(status="observed_met", errors=audit_errors())
        if fault != "candidate_only_audit":
            report["solves"].append(
                {
                    **report["solves"][0],
                    "role": "fp64_audit",
                    "requested_precision": "fp64",
                }
            )
            report["solve_count"] = 2
        if fault == "wrong_audit_precision":
            report["solves"][-1]["requested_precision"] = "auto"
        elif fault == "missing_audit_role":
            report["solves"][-1]["role"] = "candidate"
        elif fault == "nondefault_allowance":
            report["errors"][0]["allowance"] = 1.0
        elif fault in ("unrecorded_verification", "false_rejected_errors"):
            report["rejected_errors"] = audit_errors()
            report["fallback_reasons"].append("candidate_exceeded_observable_budget")
            if fault == "unrecorded_verification":
                report["rejected_errors"][0]["difference"] = 1.0
            else:
                report["solves"].append(
                    {**report["solves"][-1], "role": "fp64_fallback_verification"}
                )
                report["solve_count"] = 3
    with pytest.raises(ValueError):
        runner.compare(gpu, standard, fast)


@pytest.mark.parametrize(
    "initial", ("candidate", "failed_candidate", "temporary_failure")
)
@pytest.mark.parametrize("rejected", (False, True))
def test_fast_audit_preserves_public_fallback_shapes(
    initial: str, rejected: bool
) -> None:
    gpu, standard, fast = receipts()
    for sample in fast["samples"]:
        report = sample["precision_validation"]
        candidate = report["solves"][0]
        strict = {
            **candidate,
            "requested_precision": "fp64",
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "screening_tolerance": 1e-14,
        }
        if initial == "candidate":
            solves = [candidate]
        else:
            report["fallback_reasons"].append("budget_fock_failed_fp64_fallback")
            solves = [{**strict, "role": "fp64_fallback"}]
            if initial == "failed_candidate":
                solves.insert(
                    0,
                    {
                        **candidate,
                        "error": "candidate failed",
                        "iterations": None,
                        "force_evaluations": None,
                        "fock_builds": None,
                    },
                )
        solves.append({**strict, "role": "fp64_audit"})
        if rejected:
            report["rejected_errors"] = audit_errors()
            report["rejected_errors"][0]["difference"] = 1.0
            report["fallback_reasons"].append("candidate_exceeded_observable_budget")
            solves.append({**strict, "role": "fp64_fallback_verification"})
        report.update(
            status="observed_met",
            errors=audit_errors(),
            solves=solves,
            solve_count=len(solves),
        )
    summary = runner.compare(gpu, standard, fast)
    assert summary["fast_internal_solve_counts"] == [len(solves)] * 2
    assert summary["science_equivalence_proved"] is False


def composite_receipts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    records = receipts()
    for arm, record in zip(runner.ARMS, records, strict=True):
        record["problem"] = runner.problem_identity("r2scan-3c", [("H", (0, 0, 0))])
        for sample in record["samples"]:
            correction = {"included_in_timing": True, "energy_hartree": -0.125}
            if arm in runner.NATIVE_ARMS:
                sample["composite_corrections"] = {
                    **correction,
                    "provider": "native-canonical-D4+gCP",
                    "complete": True,
                }
            else:
                sample["settings"]["composite_corrections"] = {
                    **correction,
                    "provider": "independent-CPU-dftd4+gCP",
                    "d4": {
                        "s6": 1.0,
                        "s8": 0.0,
                        "s9": 2.0,
                        "a1": 0.42,
                        "a2": 5.65,
                        "ga": 2.0,
                        "gc": 1.0,
                    },
                }
    return records


def test_complete_r2scan3c_receipts_remain_accuracy_unverified() -> None:
    summary = runner.compare(*composite_receipts())
    assert summary["problem"]["basis"] == "def2-mTZVPP"
    assert summary["problem"]["composite_corrections"] == ["D4", "gCP"]
    table = runner.main_table_markdown([summary])
    assert "r2scan-3c" in table and "Accuracy unverified" in table
    assert "GPU4PySCF r2SCAN + independent CPU D4/gCP" in table
    assert table.count("| Method |") == 1
    assert "independent CPU D4/gCP" not in runner.main_table_markdown(
        [runner.compare(*receipts())]
    )


@pytest.mark.parametrize(
    "fault",
    (
        "plain_basis",
        "plain_science",
        "missing_native",
        "incomplete_native",
        "missing_reference",
        "missing_gcp",
        "untimed_reference",
        "nonfinite_energy",
        "missing_d4_parameter",
        "changed_d4_parameter",
    ),
)
def test_composite_receipts_require_complete_canonical_provenance(fault: str) -> None:
    gpu, standard, fast = records = composite_receipts()
    if fault == "plain_basis":
        for record in records:
            record["problem"]["basis"] = "def2-svp"
    elif fault == "plain_science":
        for record in records:
            record["problem"]["composite_corrections"] = []
    elif fault == "missing_native":
        del fast["samples"][0]["composite_corrections"]
    elif fault == "incomplete_native":
        standard["samples"][0]["composite_corrections"]["complete"] = False
    elif fault == "missing_reference":
        del gpu["samples"][0]["settings"]["composite_corrections"]
    else:
        correction = gpu["samples"][0]["settings"]["composite_corrections"]
        if fault == "missing_gcp":
            correction["provider"] = "independent-CPU-dftd4"
        elif fault == "untimed_reference":
            correction["included_in_timing"] = False
        elif fault == "nonfinite_energy":
            correction["energy_hartree"] = float("nan")
        elif fault == "missing_d4_parameter":
            del correction["d4"]["s9"]
        else:
            correction["d4"]["s9"] = 1.0
    with pytest.raises(ValueError):
        runner.compare(gpu, standard, fast)


@pytest.mark.parametrize("fault", ("legacy_version", "wrong_level", "no_pruning"))
def test_rejects_legacy_native_grids_in_default_comparison(fault: str) -> None:
    gpu, standard, fast = receipts()
    grid = standard["samples"][0]["settings"]["xc_grid"]
    if fault == "legacy_version":
        grid["version"] = 1
    elif fault == "wrong_level":
        grid["level"] = 4
    else:
        grid["pruning"] = "none"
    with pytest.raises(ValueError, match="missing executed settings"):
        runner.compare(gpu, standard, fast)
