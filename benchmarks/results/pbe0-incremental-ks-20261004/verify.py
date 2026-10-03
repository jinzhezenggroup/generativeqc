"""Check every retained repeat and actual work census without a GPU."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def read(name: str) -> dict:
    raw = (ROOT / name).read_bytes()
    return json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)


def verify() -> dict:
    manifest = read("publication.json")
    for entry in manifest["files"]:
        raw = (ROOT / entry["path"]).read_bytes()
        assert len(raw) == entry["bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    phases = ["cold", *(["warm"] * 5), "moved", *(["moved-warm"] * 5)]
    comparisons = 0
    maximum_energy = maximum_force = 0.0
    for label, cases in [("initial", (48, 96)), ("audit", (48, 96, 4))]:
        for atoms in cases:
            entries = read(f"{label}-{atoms}.json.gz")
            reference = entries["reference"]["result"]
            assert entries["reference"]["receipt"]["reference_xc_objects"]
            assert all(
                item["on_gpu"]
                for item in entries["reference"]["receipt"]["reference_xc_objects"]
            )
            for variant in ("off", "on"):
                entry = entries[variant]
                assert entry["outcome"]["exit_code"] == 0
                assert entry["result"]["protocol"] == reference["protocol"]
                assert [
                    point["phase"] for point in entry["result"]["records"]
                ] == phases
                observations = entry["receipt"]["native_build_observations"]
                assert len(observations) == 12
                for point, observation in zip(
                    entry["result"]["records"], observations, strict=True
                ):
                    assert point["gate"] and point["converged"] and point["status"] == 0
                    assert (
                        observation["iterations"]
                        == point["iterations"]
                        == observation["fock_builds"]
                    )
                    if variant == "on":
                        counts = observation["incremental"]
                        assert (
                            sum(
                                counts[key]
                                for key in (
                                    "anchor_full_builds",
                                    "delta_builds",
                                    "post_scf_full_builds",
                                )
                            )
                            == observation["fock_builds"]
                        )
                    matches = [
                        row
                        for row in reference["records"]
                        if row["geometry"] == point["geometry"]
                    ]
                    assert len(matches) == 6
                    for oracle in matches:
                        energy_error = abs(point["energy"] - oracle["energy"])
                        force_error = float(
                            np.max(
                                np.abs(np.asarray(point["forces"]) - oracle["forces"])
                            )
                        )
                        assert energy_error <= 1e-8 and force_error <= 1e-7
                        maximum_energy = max(maximum_energy, energy_error)
                        maximum_force = max(maximum_force, force_error)
                        comparisons += 1
    assert comparisons == 720
    initial = read("initial-48.json.gz")["on"]["result"]["records"]
    assert sum(point.get("warm_start_fallback", False) for point in initial) == 2
    work = read("observer-work.json.gz")
    reference = read("initial-96.json.gz")["reference"]["result"]
    tokens = (ROOT / "input-96-basis.txt").read_text().split()
    assert list(map(int, tokens[:2])) == [96, 384]
    for index, atom in enumerate(reference["protocol"]["geometries_bohr"][0]):
        assert int(tokens[2 + 4 * index]) == {"H": 1, "O": 8}[atom[0]]
        assert list(map(float, tokens[3 + 4 * index : 6 + 4 * index])) == atom[1]
    for mode, run in work["runs"].items():
        assert run["outcome"]["exit_code"] == 0
        identity = run["rows"][0]
        assert identity["present_mask"] == identity["covered_mask"] == (1 << 21) - 1
        for replay in range(3):
            iterations = [
                row
                for row in run["rows"]
                if row["kind"] == "iteration" and row["replay"] == replay
            ]
            results = [
                row
                for row in run["rows"]
                if row["kind"] == "result" and row["replay"] == replay
            ]
            assert len(results) == 1 and results[0]["fock_builds"] == len(iterations)
            assert results[0]["converged"] and results[0]["final_residual_audits"] == 1
            for oracle in reference["records"]:
                if oracle["geometry"] == 0:
                    assert abs(results[0]["energy"] - oracle["energy"]) <= 1e-8
            for row in iterations:
                assert not row["failed"] and sum(row["j_admitted"]) == 142757104
                for channel in ("j_admitted", "k_admitted"):
                    assert len(row[channel]) == 55 and all(
                        value == 0 for value in row[channel][21:]
                    )
            groups = work["summary"]["variants"][mode]["replays"][replay]["by_build"]
            for build, group in groups.items():
                selected = [row for row in iterations if row["build"] == build]
                assert group["builds"] == len(selected)
                for channel in ("j_admitted", "k_admitted"):
                    assert group[channel] == sum(sum(row[channel]) for row in selected)
    return {
        "endpoint_comparisons": comparisons,
        "maximum_energy_error": maximum_energy,
        "maximum_force_error": maximum_force,
        "work_energy_comparisons": 36,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
