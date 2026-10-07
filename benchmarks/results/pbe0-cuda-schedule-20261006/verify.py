"""Recompute independent gates and paired whole-endpoint schedule qualification."""

import argparse
import hashlib
import json
import lzma
import math
import statistics
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--candidate", default="adaptive512")
parser.add_argument("--integration", action="store_true")
parser.add_argument("--output", type=Path, required=True)
arguments = parser.parse_args()
base_commit = (
    "25675d88d64bfec13a8b7100eebcabef2a83d534"
    if arguments.integration
    else "21f6314f5a812e9e83a41b3c9921549d6e944778"
)
prefix = "integration-" if arguments.integration else ""
bundle_path = arguments.directory / f"{prefix}bundle.json.xz"
if bundle_path.exists():
    manifest = json.loads((arguments.directory / f"{prefix}manifest.json").read_text())
    compressed = bundle_path.read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == manifest["bundle_sha256"]
    decoded = lzma.decompress(compressed)
    assert len(decoded) == manifest["decoded_bytes"]
    payload = json.loads(decoded)
    assert payload["schema"] == manifest["schema"]
    assert payload["base_commit"] == manifest["base_commit"]
    assert payload["base_commit"] == base_commit
    assert payload["artifacts"].keys() == manifest["artifacts"].keys()
    temporary = tempfile.TemporaryDirectory(prefix="cuda-schedule-verify-")
    expanded = Path(temporary.name)
    for name, item in payload["artifacts"].items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts
        assert "\\" not in name and ":" not in name
        data = item["text"].encode()
        expected = manifest["artifacts"][name]
        assert len(data) == item["bytes"] == expected["bytes"], name
        assert hashlib.sha256(data).hexdigest() == item["sha256"] == expected["sha256"], name
        target = expanded / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    arguments.directory = expanded / "campaign"
    assert (expanded / "support/exit").read_text().strip() == "0"
    for mode in ("base", arguments.candidate):
        for kind in ("source", "harness"):
            for stage in ("before", "after"):
                receipt = expanded / f"support/{mode}-{kind}-{stage}.txt"
                assert receipt.exists()
                lines = receipt.read_text().splitlines()
                assert lines and all(line.endswith(": OK") for line in lines), receipt
        lines = (expanded / f"support/{mode}-binaries-after.txt").read_text().splitlines()
        assert lines and all(line.endswith(": OK") for line in lines), mode
phases = ("cold", "warm", "moved", "moved-warm")
identities = {}
runtime_identities = {}
max_energy_error = max_force_error = 0.0
result = {
    "schema": "generativeqc.cuda-schedule-qualification.v1",
    "base_commit": base_commit,
    "cases": {},
}

def load_run(atoms, round_number, mode, reference, reference_sha):
    """Recompute whole-vector gates and authenticate actual, unforced work."""
    global max_energy_error, max_force_error
    path = arguments.directory / f"{atoms}-{round_number}-{mode}.json"
    record = json.loads(path.read_text())
    assert record["status"] == "measured" and record["stage"] == "complete", path
    assert record["cuda_schedule_intrusive"] is False, path
    assert record["reference_sha256"] == reference_sha, path
    assert record["protocol"] == reference["protocol"], path
    assert tuple(row["phase"] for row in record["records"]) == phases, path
    assert len(record["cuda_schedule_force_work"]) == len(phases), path
    identity = (record["source_file_sha256"], record["native_build"]["library_sha256"],
                record["native_build"]["probe"]["source_identity"])
    if mode in identities:
        assert identities[mode] == identity, (path, "changing source/binary population")
    identities[mode] = identity
    for row, work in zip(record["records"], record["cuda_schedule_force_work"], strict=True):
        runtime_identity = [(item["key"], item["binary_sha256"]) for item in work["artifacts"]]
        runtime_scope = f"{mode}/{atoms}/{row['phase']}"
        if runtime_scope in runtime_identities:
            assert runtime_identities[runtime_scope] == runtime_identity, (path, "changing runtime artifact")
        runtime_identities[runtime_scope] = runtime_identity
        oracle = next(item for item in reference["records"] if item["geometry"] == row["geometry"]
                      and item["phase"] in ("cold", "moved"))
        actual, expected = np.asarray(row["forces"]), np.asarray(oracle["forces"])
        assert row["gate"] and row["converged"] and row["status"] == 0, path
        assert actual.shape == expected.shape and np.isfinite(actual).all(), path
        energy_error = abs(row["energy"] - oracle["energy"])
        force_error = float(np.max(np.abs(actual - expected)))
        assert energy_error <= 1e-8 and force_error <= 1e-7, path
        max_energy_error = max(max_energy_error, energy_error)
        max_force_error = max(max_force_error, force_error)
        assert math.isfinite(row["complete_seconds"]) and row["complete_seconds"] > 0, path
        assert row["fock_builds"] == len(row["native_ks_diagnostic"]["history"]), path
        assert work["additional_device_peak_bound"] <= work["additional_device_budget"], path
        assert work["additional_host_numeric_bound"] <= work["additional_host_budget"], path
        assert work["device_profile_enabled"] is False, path
        assert work["becke_phase_profile_enabled"] is False, path
        assert work["xc_points"] == row["native_ks_diagnostic"]["grid_points"], path
        policy = work["force_active_ao_policy"]
        assert policy["decision"] == "selected" and policy["producer"] == "sampled-jets", path
        assert policy["cutoff"] == 1e-16 and policy["cache_bytes"] == 16 << 20, path
    return record

for atoms in (48, 96):
    reference_path = arguments.directory / f"reference-{atoms}.json"
    reference = json.loads(reference_path.read_text())
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    populations = {mode: [] for mode in ("base", arguments.candidate)}
    for round_number in range(5):
        control = load_run(atoms, round_number, "base", reference, reference_sha)
        candidate = load_run(atoms, round_number, arguments.candidate, reference, reference_sha)
        for control_row, candidate_row, control_work, candidate_work in zip(
            control["records"], candidate["records"], control["cuda_schedule_force_work"],
            candidate["cuda_schedule_force_work"], strict=True
        ):
            if control_row["phase"] in ("warm", "moved-warm"):
                for field in ("iterations", "fock_builds", "warm_start_used", "warm_start_fallback"):
                    assert control_row[field] == candidate_row[field], (atoms, round_number, field)
                assert all(
                    control_row["native_scf_ao_work"][field] == candidate_row["native_scf_ao_work"][field]
                    for field in control_row["native_scf_ao_work"] if field != "discovery_seconds"
                ), (atoms, round_number, "changed warm SCF work")
            for field in ("xc_points", "ordered_pairs", "ordered_quartets", "exchange_ordered_quartets", "grid_pair_visits",
                          "becke_pair_primal_visits", "becke_log_incident_visits",
                          "becke_normalization_atom_entries", "becke_reverse_pair_visits",
                          "becke_gather_incident_visits", "becke_motion_atom_entries"):
                assert control_work[field] == candidate_work[field], (atoms, round_number, field)
        populations["base"].append(control)
        populations[arguments.candidate].append(candidate)
    summaries = {}
    for index, phase in enumerate(phases):
        controls = [record["records"][index]["complete_seconds"] for record in populations["base"]]
        candidates = [record["records"][index]["complete_seconds"] for record in populations[arguments.candidate]]
        gains = [(control - candidate) / control for control, candidate in zip(controls, candidates, strict=True)]
        summaries[phase] = {
            "control_seconds": controls, "candidate_seconds": candidates,
            "control_median": statistics.median(controls),
            "candidate_median": statistics.median(candidates),
            "median_fractional_gain": statistics.median(gains),
            "positive_pairs": sum(value > 0 for value in gains),
            "paired_fractional_gains": gains,
            "scope": "matched-work-qualification" if phase in ("warm", "moved-warm") else "diagnostic-only-SCF-work-may-vary",
            "control_iterations": [record["records"][index]["iterations"] for record in populations["base"]],
            "candidate_iterations": [record["records"][index]["iterations"] for record in populations[arguments.candidate]],
        }
    assert all(summaries[phase]["median_fractional_gain"] > 0.02 and summaries[phase]["positive_pairs"] >= 4
               for phase in ("warm", "moved-warm")), atoms
    result["cases"][str(atoms)] = summaries

result.update(
    decision="qualified", independent_energy_gate=1e-8, independent_force_gate=1e-7,
    minimum_warm_gain=0.02, minimum_positive_pairs=4,
    qualification_scope="warm-and-moved-warm-matched-work",
    cold_and_moved_scope="diagnostic-only; natural SCF iteration variability; no speed claim",
    max_energy_error=max_energy_error, max_force_error=max_force_error,
    source_and_binary_identities=identities,
    runtime_artifact_identities=runtime_identities,
    deployment_compile_envelope="not-run", global_allocator_peak="not-run",
)
arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({"decision": result["decision"], "max_energy_error": max_energy_error,
                  "max_force_error": max_force_error}))
