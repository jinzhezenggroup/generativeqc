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
            "xc_grid": {"rule": arm, "points": 200 + i},
        }
        observation: dict[str, Any] = {
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
    fast["samples"][0]["precision_validation"] = {
        "profile": "fast",
        "status": "observed_met",
        "solve_count": 2,
    }
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


def test_hybrid_grid_is_mandatory_api_input_not_reference_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = SimpleNamespace(
        GridSpec=lambda: "default-native-grid",
        KsOptions=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "generativeqc", fake)
    args = runner.native_constructor_kwargs("pbe0")
    assert args["ks_options"].grid == "default-native-grid"
    assert not {
        "max_iterations",
        "energy_tolerance",
        "density_tolerance",
        "screening_tolerance",
        "initial_guess",
        "precision",
    }.intersection(args)


def test_unsupported_composite_is_not_labeled_r2scan() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        runner.problem_identity("r2scan-3c", ())


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
