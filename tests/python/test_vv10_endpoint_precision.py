"""A single joint reference allowance must cover all endpoint error sources."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest
from generativeqc import GridSpec

from benchmarks.readme_dft import SCHEMA
from tools.analyze_vv10_grid_study import PROFILES
from tools.qualify_vv10_endpoint_precision import CONTROLS, main, qualify_endpoint

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def cohort(profile: str = "fp64") -> tuple[dict, dict, dict]:
    anchor = {
        "schema": "generativeqc.reference-vv10-grid-accuracy.v1",
        "engine": "GPU4PySCF",
        "stage": "complete",
        "atoms": 3,
        "basis": "spherical def2-SVP",
        "density_fitting": "none",
        "geometries_bohr": [
            [["O", [0, 0, 0]], ["H", [1, 0, 0]], ["H", [0, 1, 0]]],
            [["O", [0, 0, 0]], ["H", [1, 0, 0.001]], ["H", [0, 1, 0]]],
        ],
        "xc_grid": asdict(GridSpec(240, 80, 160)),
        "nlc_grid": asdict(GridSpec(240, 40, 80)),
        "source_sha256": {"reference.py": "f" * 64},
        "accuracy_protocol": {
            "vv10_density_threshold": 1e-12,
            "full_grid_response": True,
            "fock_policy": "full-density-rebuild",
            "scf_tolerances": {
                "energy": 1e-12,
                "gradient": 1e-10,
                "direct_scf": 1e-14,
                "max_cycle": 100,
            },
        },
        "records": [
            {
                "geometry": geometry,
                "phase": phase,
                "repeat": 0,
                "energy": -76.0,
                "forces": [[0.0] * 3 for _ in range(3)],
                "converged": True,
            }
            for geometry, phase in ((0, "accuracy-original"), (1, "accuracy-moved"))
        ],
    }
    controls = {name: copy.deepcopy(anchor) for name in CONTROLS}
    for name in CONTROLS[:-1]:
        axis, direction = name.split("_")
        grid = controls[name][f"{axis}_grid"]
        if direction == "radial":
            grid["radial_points"] -= 48
        else:
            grid["angular_polar"] -= 8
            grid["angular_azimuth"] -= 16
    controls["screening"]["accuracy_protocol"]["vv10_density_threshold"] = 1e-14
    native = {
        "schema": SCHEMA + (".fast.v1" if profile == "fast" else ""),
        "engine": "native",
        "stage": "complete",
        "native_experiment": {
            "precision_policy": profile,
            "solver_control_policy": "application-policy-selected",
            "requested_solver_controls": {"max_iterations": 100},
            "public_initialization_policy": True,
            "preliminary_provider": "none",
        },
        "application_budget": asdict(PROFILES[profile]),
        "protocol": {
            "method": "WB97M-V/RKS",
            "basis": "def2-SVP",
            "representation": "spherical",
            "atoms": 3,
            "density_fitting": False,
            "geometries_bohr": copy.deepcopy(anchor["geometries_bohr"]),
            "repeats": 5,
            "force_return": "host_array_in_timed_endpoint",
        },
        "records": [
            {
                "phase": phase,
                "geometry": geometry,
                "repeat": repeat,
                "energy": -76.0,
                "forces": [[0.0] * 3 for _ in range(3)],
                "converged": True,
                "status": 0,
                "complete_seconds": 1.0,
                "precision": {"effective_bits": 64},
                "gate": None,
            }
            for phase, geometry, count in (
                ("cold", 0, 1),
                ("warm", 0, 5),
                ("moved", 1, 1),
                ("moved-warm", 1, 5),
            )
            for repeat in range(count)
        ],
    }
    return native, anchor, controls


def test_joint_gate_checks_every_endpoint_without_changing_measurements() -> None:
    native, anchor, controls = cohort()
    original = copy.deepcopy((native, anchor, controls))
    result = qualify_endpoint(native, anchor, controls, profile="fp64")
    assert result["total_endpoint_gate"] is True
    assert len(result["records"]) == 12
    assert result["timing_qualification"] == "not_evaluated"
    assert (native, anchor, controls) == original
    native["records"][-1]["energy"] += 2e-8
    result = qualify_endpoint(native, anchor, controls, profile="fp64")
    assert result["total_endpoint_gate"] is False
    assert result["records"][-1]["gate"] is False


def test_reference_controls_share_one_reserved_allowance() -> None:
    native, anchor, controls = cohort()
    for control in controls.values():
        control["records"][0]["energy"] += 3e-10
    result = qualify_endpoint(native, anchor, controls, profile="fp64")
    assert all(
        control["maximum_errors"]["energy_abs"] < 1e-9
        for control in result["reference_controls"].values()
    )
    assert result["reference_uncertainty"]["energy_abs"] > 1e-9
    assert result["reference_stability_gate"] is False
    assert result["total_endpoint_gate"] is False


def test_reference_uncertainty_is_added_to_native_error() -> None:
    native, anchor, controls = cohort()
    controls["screening"]["records"][1]["energy"] += 8e-10
    native["records"][-1]["energy"] += 9.5e-9
    result = qualify_endpoint(native, anchor, controls, profile="fp64")
    assert result["reference_stability_gate"] is True
    assert result["maximum_errors"]["energy_abs"] < 1e-8
    assert result["maximum_errors_with_reference_allowance"]["energy_abs"] > 1e-8
    assert result["total_endpoint_gate"] is False


@pytest.mark.parametrize("name", ("energy", "gradient", "direct_scf"))
def test_self_consistent_but_loose_reference_controls_do_not_certify_accuracy(
    name: str,
) -> None:
    native, anchor, controls = cohort()
    for population in (anchor, *controls.values()):
        population["accuracy_protocol"]["scf_tolerances"][name] = 1e-6
    with pytest.raises(ValueError, match="too loose"):
        qualify_endpoint(native, anchor, controls, profile="fp64")


def test_fast_checks_rms_and_reports_actual_fp64_fallback() -> None:
    native, anchor, controls = cohort("fast")
    result = qualify_endpoint(native, anchor, controls, profile="fast")
    assert result["total_endpoint_gate"] is True
    assert result["actual_fast_precision_counts"] == {"64": 12}
    native["records"][-1]["forces"] = [[5e-5] * 3 for _ in range(3)]
    result = qualify_endpoint(native, anchor, controls, profile="fast")
    assert result["maximum_errors"]["force_max_abs"] < 1e-4
    assert result["maximum_errors"]["force_rms"] > 3e-5
    assert result["total_endpoint_gate"] is False


@pytest.mark.parametrize("profile", ("fp64", "fast"))
def test_gpu4pyscf_timing_endpoints_must_meet_the_same_total_budget(
    profile: str,
) -> None:
    candidate, anchor, controls = cohort(profile)
    candidate["schema"] = SCHEMA
    candidate["engine"] = "reference"
    candidate.pop("native_experiment")
    for row in candidate["records"]:
        row.pop("precision")
    result = qualify_endpoint(candidate, anchor, controls, profile=profile)
    assert result["total_endpoint_gate"] is True
    assert result["engine"] == "reference"
    assert result["actual_fast_precision_counts"] == {}
    candidate["records"][-1]["energy"] += PROFILES[profile].energy_abs * 2
    assert (
        qualify_endpoint(candidate, anchor, controls, profile=profile)[
            "total_endpoint_gate"
        ]
        is False
    )


@pytest.mark.parametrize(
    "key,value",
    (
        ("solver_control_policy", "pinned-tight-pilot"),
        (
            "requested_solver_controls",
            {"energy_tolerance": 1e-12, "max_iterations": 100},
        ),
        ("public_initialization_policy", False),
        ("preliminary_provider", "reference"),
        ("force_max_device_bytes", 1024),
        ("force_active_ao_cutoff", 1e-12),
    ),
)
def test_private_or_pinned_fast_controls_cannot_claim_default_fast(
    key: str, value: object
) -> None:
    native, anchor, controls = cohort("fast")
    native["native_experiment"][key] = value
    with pytest.raises(ValueError, match="public controls differ"):
        qualify_endpoint(native, anchor, controls, profile="fast")


@pytest.mark.parametrize("name", CONTROLS)
def test_all_independent_controls_are_mandatory(name: str) -> None:
    native, anchor, controls = cohort()
    controls.pop(name)
    with pytest.raises(ValueError, match="controls are required"):
        qualify_endpoint(native, anchor, controls, profile="fp64")


@pytest.mark.parametrize(
    "change",
    (
        lambda controls: controls["xc_radial"]["nlc_grid"].update(radial_points=120),
        lambda controls: controls["screening"]["xc_grid"].update(radial_points=120),
        lambda controls: controls["xc_angular"]["accuracy_protocol"].update(
            vv10_density_threshold=1e-14
        ),
        lambda controls: controls["nlc_radial"]["accuracy_protocol"].update(
            fock_policy="incremental"
        ),
        lambda controls: controls["screening"]["accuracy_protocol"].update(
            vv10_density_threshold=1e-8
        ),
        lambda controls: controls["xc_radial"]["xc_grid"].update(radial_points=240),
        lambda controls: controls["nlc_angular"]["nlc_grid"].update(
            angular_polar=40, angular_azimuth=80
        ),
        lambda controls: controls["xc_angular"]["records"].pop(),
    ),
)
def test_controls_cannot_hide_changed_or_missing_scientific_work(
    change: Callable[[dict], None],
) -> None:
    native, anchor, controls = cohort()
    change(controls)
    with pytest.raises(ValueError):
        qualify_endpoint(native, anchor, controls, profile="fp64")


@pytest.mark.parametrize("failure", ("missing", "duplicate", "unconverged", "nan"))
def test_incomplete_or_invalid_native_calls_cannot_pass(failure: str) -> None:
    native, anchor, controls = cohort()
    if failure == "missing":
        native["records"].pop()
    elif failure == "duplicate":
        native["records"][-1] = copy.deepcopy(native["records"][0])
    elif failure == "unconverged":
        native["records"][-1]["converged"] = False
    else:
        native["records"][-1]["forces"][0][0] = float("nan")
    with pytest.raises(ValueError):
        qualify_endpoint(native, anchor, controls, profile="fp64")


def test_method_and_profile_mismatch_are_not_quadrature_error() -> None:
    native, anchor, controls = cohort()
    native["protocol"]["density_fitting"] = True
    with pytest.raises(ValueError, match="contract differs"):
        qualify_endpoint(native, anchor, controls, profile="fp64")
    native, anchor, controls = cohort("fast")
    with pytest.raises(ValueError, match="contract differs"):
        qualify_endpoint(native, anchor, controls, profile="fp64")


def test_missing_process_outcome_is_incomplete_not_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native, _, _ = cohort()
    path = tmp_path / "native.json"
    path.write_text(json.dumps(native))
    output = tmp_path / "qualification.json"
    args = ["qualify", "--native", str(path), "--anchor", "absent", "--profile", "fp64"]
    for name in CONTROLS:
        args.extend(["--" + name.replace("_", "-"), "absent"])
    args.extend(["--output", str(output)])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    result = json.loads(output.read_text())
    assert result["total_endpoint_gate"] is None
    assert "native.outcome" in result["incomplete_or_invalid"]


@pytest.mark.parametrize("engine", ("native", "reference"))
def test_cli_binds_both_candidate_engines_to_successful_processes_and_reference_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    candidate, anchor, controls = cohort()
    candidate["engine"] = engine
    candidate.update(
        scheduler={"SLURM_JOB_ID": "candidate-job"},
        source_file_sha256={"benchmark.py": "d" * 64},
        native_build={"library_sha256": "a" * 64},
    )
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    candidate_path.with_suffix(".outcome").write_text('{"exit_code": 0}')
    paths = {}
    for name, population in {"anchor": anchor, **controls}.items():
        population.update(
            purpose="accuracy-only-not-a-latency-population",
            scheduler={"SLURM_JOB_ID": "reference-job"},
            accelerator={"name": "independent GPU"},
        )
        population["accuracy_protocol"].update(
            density_sources="independent-GPU4PySCF-only", cold_timing_qualified=False
        )
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(population))
        path.with_suffix(".outcome").write_text('{"exit_code": 0}')
        paths[name] = path
    output = tmp_path / "qualification.json"
    args = ["qualify", "--candidate", str(candidate_path), "--profile", "fp64"]
    for name, path in paths.items():
        args.extend(["--" + name.replace("_", "-"), str(path)])
    args.extend(["--output", str(output)])
    monkeypatch.setattr(sys, "argv", args)
    main()
    result = json.loads(output.read_text())
    assert result["total_endpoint_gate"] is True
    assert result["candidate_receipt"]["sha256"]
    assert result["reference_controls"]["anchor_replay"]["receipt"]["outcome_sha256"]
    candidate_path.with_suffix(".outcome").write_text('{"exit_code": 1}')
    with pytest.raises(SystemExit):
        main()
    assert json.loads(output.read_text())["total_endpoint_gate"] is None
