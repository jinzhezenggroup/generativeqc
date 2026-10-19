"""A single joint reference allowance must cover all endpoint error sources."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc import GridSpec

from benchmarks.readme_dft import SCHEMA
from tools.analyze_vv10_grid_study import PROFILES, load_population
from tools.qualify_vv10_endpoint_precision import (
    CONTROLS,
    load_candidate,
    main,
    qualify_endpoint,
    replay_qualification,
)
from tools.render_qualified_vv10_grids import collect_qualified_sweep, validate_summary
from tools.render_readme_grids import compact_summary, render

if TYPE_CHECKING:
    from collections.abc import Callable


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
        "status": "completed_unchecked",
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
            "grid": asdict(GridSpec(48, 16, 32)),
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
                "iterations": 1,
                "fock_builds": 2,
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


def test_a_later_dense_reference_never_erases_a_known_measurement_failure() -> None:
    native, anchor, controls = cohort()
    native["records"][-1]["gate"] = False
    result = qualify_endpoint(native, anchor, controls, profile="fp64")
    assert result["maximum_errors"]["energy_abs"] == 0
    assert result["total_endpoint_gate"] is False
    assert result["records"][-1]["measurement_gate"] is False


def test_failed_run_status_is_not_requalified_as_a_successful_endpoint() -> None:
    native, anchor, controls = cohort()
    native["status"] = "failed"
    with pytest.raises(ValueError, match="contract differs"):
        qualify_endpoint(native, anchor, controls, profile="fp64")


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
    replayed_candidate, replayed = replay_qualification(output)
    assert replayed_candidate["engine"] == engine
    assert replayed == result
    modified = copy.deepcopy(result)
    modified["records"][0]["energy_abs"] = 1e-12
    output.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="differs from"):
        replay_qualification(output)
    output.write_text(json.dumps(result))
    modified_candidate = copy.deepcopy(candidate)
    modified_candidate["records"][0]["energy"] += 1e-6
    candidate_path.write_text(json.dumps(modified_candidate))
    with pytest.raises(ValueError, match="differs from"):
        replay_qualification(output)
    candidate_path.write_text(json.dumps(candidate))
    candidate_path.with_suffix(".outcome").write_text('{"exit_code": 1}')
    with pytest.raises(SystemExit):
        main()
    assert json.loads(output.read_text())["total_endpoint_gate"] is None


def publication_manifest(tmp_path: Path, profile: str) -> Path:
    """Write real input/outcome/report files for publication replay, without a GPU."""
    native, anchor, controls = cohort(profile)
    oracles = {}
    for name, population in {"anchor_replay": anchor, **controls}.items():
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
        oracles[name] = load_population(path)
    reports = {}
    for engine in ("native", "reference"):
        candidate = copy.deepcopy(native)
        candidate.update(
            engine=engine,
            scheduler={"SLURM_JOB_ID": f"{engine}-job"},
            source_file_sha256={"benchmark.py": "d" * 64},
        )
        if engine == "native":
            candidate["native_build"] = {"library_sha256": "a" * 64}
        else:
            candidate["schema"] = SCHEMA
            candidate["status"] = "measured"
            candidate.pop("native_experiment")
            for row in candidate["records"]:
                row.pop("precision")
                row["gate"] = True
        path = tmp_path / f"{engine}.json"
        path.write_text(json.dumps(candidate))
        path.with_suffix(".outcome").write_text('{"exit_code": 0}')
        report = qualify_endpoint(
            load_candidate(path),
            oracles["anchor_replay"],
            {name: oracles[name] for name in CONTROLS},
            profile=profile,
        )
        report_path = tmp_path / f"{engine}-precision.json"
        report_path.write_text(json.dumps(report))
        reports[engine] = report_path.name
    manifest_path = tmp_path / "publication.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "generativeqc.readme-qualified-grid-inputs.v1",
                "profile": profile,
                "points": [
                    {
                        "mode": "none",
                        "grid": [48, 16, 32],
                        "atoms": 3,
                        "reports": reports,
                    }
                ],
            }
        )
    )
    return manifest_path


@pytest.mark.parametrize("profile", ("fp64", "fast"))
def test_qualified_grid_archive_preserves_both_engines_and_every_force_sample(
    tmp_path: Path, profile: str
) -> None:
    manifest = publication_manifest(tmp_path, profile)
    summary = collect_qualified_sweep(manifest, [(48, 16, 32)], [3], ["none"])
    point = summary["modes"]["none"][0]["points"][0]
    assert set(point["engines"]) == {"native", "reference"}
    for entry in point["engines"].values():
        assert len(entry["records"]) == 12
        assert all(len(row["forces"]) == 3 for row in entry["records"])
        assert entry["endpoint_precision_qualification"]["total_endpoint_gate"] is True
    assert set(point["independent_accuracy_oracles"]) == {*CONTROLS, "anchor_replay"}
    retained = compact_summary(summary)
    validate_summary(retained)
    for path in tmp_path.glob("*.json"):
        path.unlink()
    render(retained, tmp_path)
    table = (tmp_path / "grid-timings.md").read_text()
    assert f"{profile.upper()} total-error budget" in table
    assert "| native |" in table and "| reference |" in table
    assert "| Warm work | Effective precision |" in table
    if profile == "fast":
        assert "64-bit: 12" in table
        assert (
            "Fast request (FP64 execution)"
            in (tmp_path / "wb97mv-grids.svg").read_text()
        )


@pytest.mark.parametrize("missing", ("reference", "size", "mode", "grid"))
def test_qualified_grid_publication_never_omits_a_missing_requested_population(
    tmp_path: Path, missing: str
) -> None:
    manifest_path = publication_manifest(tmp_path, "fast")
    grids, sizes, modes = [(48, 16, 32)], [3], ["none"]
    if missing == "reference":
        manifest = json.loads(manifest_path.read_text())
        manifest["points"][0]["reports"].pop("reference")
        manifest_path.write_text(json.dumps(manifest))
    elif missing == "size":
        sizes.append(12)
    elif missing == "mode":
        modes.append("cuda")
    else:
        grids.append((64, 24, 48))
    with pytest.raises(ValueError):
        collect_qualified_sweep(manifest_path, grids, sizes, modes)


@pytest.mark.parametrize(
    "alter", ("force", "oracle", "source-proof", "profile", "precision", "gate")
)
def test_retained_grid_archive_cannot_promote_changed_or_inaccurate_evidence(
    tmp_path: Path, alter: str
) -> None:
    manifest = publication_manifest(tmp_path, "fast")
    summary = collect_qualified_sweep(manifest, [(48, 16, 32)], [3], ["none"])
    point = summary["modes"]["none"][0]["points"][0]
    native = point["engines"]["native"]
    if alter == "force":
        native["records"][-1]["forces"][0][0] = 1e-3
    elif alter == "oracle":
        point["independent_accuracy_oracles"]["screening"]["records"][0]["energy"] += (
            1e-3
        )
    elif alter == "source-proof":
        for oracle in point["independent_accuracy_oracles"].values():
            oracle["source_sha256"] = {"reference.py": "b" * 64}
    elif alter == "profile":
        summary["profile"] = "fp64"
    elif alter == "precision":
        native["records"][0]["precision"]["effective_bits"] = 32
    else:
        native["records"][0]["measurement_gate"] = False
    with pytest.raises(ValueError):
        validate_summary(summary)
    with pytest.raises(ValueError):
        render(summary, tmp_path)
    assert not (tmp_path / "wb97mv-grids.svg").exists()


def test_retained_grid_archive_preserves_declared_modes_and_build_receipts(
    tmp_path: Path,
) -> None:
    manifest = publication_manifest(tmp_path, "fast")
    summary = collect_qualified_sweep(manifest, [(48, 16, 32)], [3], ["none"])
    missing = copy.deepcopy(summary)
    missing["requested_modes"].append("cuda")
    with pytest.raises(ValueError, match="invalid qualified"):
        validate_summary(missing)
    entry = summary["modes"]["none"][0]["points"][0]["engines"]["native"]
    entry["native_build"]["library_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="metadata differs"):
        validate_summary(summary)


@pytest.mark.parametrize("difference", ("controls", "library"))
def test_grid_comparisons_do_not_mix_source_cohorts_or_numerical_controls(
    tmp_path: Path, difference: str
) -> None:
    manifest = publication_manifest(tmp_path, "fast")
    summary = collect_qualified_sweep(manifest, [(48, 16, 32)], [3], ["none"])
    second = copy.deepcopy(summary["modes"]["none"][0])
    second["dimensions"] = [64, 24, 48]
    point = second["points"][0]
    for protocol in (
        point["protocol"],
        *(entry["protocol"] for entry in point["engines"].values()),
    ):
        protocol["grid"] = asdict(GridSpec(64, 24, 48))
        if difference == "controls":
            protocol["energy_tolerance"] = 1e-10
    if difference == "library":
        native = point["engines"]["native"]
        native["native_build"]["library_sha256"] = "b" * 64
        native["endpoint_precision_qualification"]["candidate_receipt"]["native_build"][
            "library_sha256"
        ] = "b" * 64
    summary["grids"].append([64, 24, 48])
    summary["modes"]["none"].append(second)
    with pytest.raises(ValueError, match="controls|cohorts"):
        validate_summary(summary)


def test_a_reproducible_failed_qualification_cannot_be_published(
    tmp_path: Path,
) -> None:
    manifest = publication_manifest(tmp_path, "fast")
    candidate_path = tmp_path / "native.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["records"][-1]["gate"] = False
    candidate_path.write_text(json.dumps(candidate))
    saved = json.loads((tmp_path / "native-precision.json").read_text())
    oracles = {
        name: load_population(Path(control["receipt"]["file"]))
        for name, control in saved["reference_controls"].items()
    }
    failed = qualify_endpoint(
        load_candidate(candidate_path),
        oracles["anchor_replay"],
        {name: oracles[name] for name in CONTROLS},
        profile="fast",
    )
    assert failed["total_endpoint_gate"] is False
    (tmp_path / "native-precision.json").write_text(json.dumps(failed))
    with pytest.raises(ValueError, match="did not pass"):
        collect_qualified_sweep(manifest, [(48, 16, 32)], [3], ["none"])
