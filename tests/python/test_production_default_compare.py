"""Host-only safety checks for independent production-default timings."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import production_default_compare as runner


def receipt(engine: str, *, response: bool = True) -> dict:
    data = {
        "schema": runner.SCHEMA,
        "engine": engine,
        "status": "complete",
        "problem": runner.problem_identity("pbe", [("H", (0, 0, 0))]),
        "force_policy": "engine-default",
        "phase": "fresh-owner",
        "endpoint": "constructor-scf-energy-analytic-force-host-export",
        "repeats": 2,
        "samples": [],
    }
    for i in (0, 1):
        data["samples"].append(
            {
                "seconds": 1.0 + i if engine == "generativeqc" else 3.0 + i,
                "energy": -1.0,
                "forces": [[0.0, 0.0, 0.0]],
                "converged": True,
                "iterations": 10 + i,
                "grid_response": response,
                "settings": {
                    "initial_guess": "huckel" if engine == "generativeqc" else "minao",
                    "xc_grid": {"rule": engine, "points": 100 * (i + 1)},
                },
            }
        )
    return data


def test_distinct_defaults_report_runtime_but_never_claim_accuracy() -> None:
    left = receipt("generativeqc")
    right = receipt("gpu4pyscf", response=False)
    output = runner.compare(left, right)
    assert output["observed_default_runtime_ratio_gpu4_over_native"] == pytest.approx(
        3.5 / 1.5
    )
    assert output["native_initial_guess"] == "huckel"
    assert output["gpu4pyscf_initial_guess"] == "minao"
    assert output["native_grid"] != output["gpu4pyscf_grid"]
    assert not output["same_moving_grid_response"]
    assert "Different analytic-force" in output["force_definition_warning"]
    assert "unverified" in output["accuracy"]
    assert output["science_equivalence_proved"] is False


def test_full_response_is_separate_mode() -> None:
    left = receipt("generativeqc")
    right = receipt("gpu4pyscf", response=True)
    left["force_policy"] = right["force_policy"] = "full-moving-grid"
    result = runner.compare(left, right)
    assert result["same_moving_grid_response"]
    assert result["force_definition_warning"] is None
    assert result["science_equivalence_proved"] is False


@pytest.mark.parametrize(
    "change",
    [
        "geometry",
        "method",
        "force_policy",
        "missing_sample",
        "failed",
        "nonconverged",
        "nonfinite",
        "mismatched_backend",
        "bad_energy",
        "bad_forces",
    ],
)
def test_comparator_rejects_unmatched_or_incomplete_receipts(change: str) -> None:
    left, right = receipt("generativeqc"), receipt("gpu4pyscf")
    if change == "geometry":
        right["problem"]["geometry_sha256"] = "different"
    elif change == "method":
        right["problem"]["method"] = "pbe0"
    elif change == "force_policy":
        right["force_policy"] = "full-moving-grid"
    elif change == "missing_sample":
        right["samples"].pop()
    elif change == "failed":
        right["status"] = "failed"
    elif change == "nonconverged":
        right["samples"][0]["converged"] = False
    elif change == "nonfinite":
        right["samples"][0]["seconds"] = float("nan")
    elif change == "bad_energy":
        right["samples"][0]["energy"] = float("nan")
    elif change == "bad_forces":
        right["samples"][0]["forces"] = [[float("nan"), 0.0, 0.0]]
    else:
        right["engine"] = "generativeqc"
    with pytest.raises(ValueError):
        runner.compare(left, right)


def test_native_builder_does_not_force_scf_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = runner.native_constructor_kwargs("pbe")
    assert options == {
        "method": "pbe-rks",
        "basis": "def2-svp",
        "basis_representation": "spherical",
        "device": "cuda",
    }
    fake = SimpleNamespace(
        GridSpec=lambda: "native-default-grid",
        KsOptions=lambda **kw: SimpleNamespace(**kw),
    )
    monkeypatch.setitem(sys.modules, "generativeqc", fake)
    hybrid = runner.native_constructor_kwargs("pbe0")
    assert hybrid["ks_options"].grid == "native-default-grid"
    assert not {
        "energy_tolerance",
        "density_tolerance",
        "screening_tolerance",
        "max_iterations",
        "initial_guess",
        "precision",
    }.intersection(hybrid)


def test_unsupported_composite_is_not_relabelled_r2scan() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        runner.problem_identity("r2scan-3c", ())


def test_combine_receipts_needs_no_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native, gpu = tmp_path / "native.json", tmp_path / "gpu.json"
    native.write_text(json.dumps(receipt("generativeqc")))
    gpu.write_text(json.dumps(receipt("gpu4pyscf", response=False)))
    out = tmp_path / "report.json"
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    runner.main(
        [
            "compare",
            "--native",
            str(native),
            "--reference",
            str(gpu),
            "--output",
            str(out),
        ]
    )
    result = json.loads(out.read_text())
    assert "unverified" in result["accuracy"]


def test_gpu_measurement_refuses_unscheduled_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit) as error:
        runner.main(
            ["gpu4pyscf", "--method", "pbe", "--output", str(tmp_path / "x.json")]
        )
    assert error.value.code == 2


def test_missing_settings_or_invalid_default_policy_fails() -> None:
    left, right = receipt("generativeqc"), receipt("gpu4pyscf")
    incomplete = copy.deepcopy(left)
    incomplete["samples"][0]["settings"].pop("initial_guess")
    with pytest.raises(KeyError):
        runner.compare(incomplete, right)
    right["force_policy"] = left["force_policy"] = "invented"
    with pytest.raises(ValueError, match="unknown"):
        runner.compare(left, right)
