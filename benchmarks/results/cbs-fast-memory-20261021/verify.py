"""Replay the retained workspace campaign's observable and provenance gates."""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path

import numpy as np

from tools.generativeqc_validation.publication import validate_publication
from tools.render_readme_cbs import attach_fast, constituents, qualification, table


def errors(actual: dict, expected: dict, budget: dict) -> None:
    """Check each energy and full force vector, including the RMS allowance."""
    delta = np.asarray(actual["forces"]) - np.asarray(expected["forces"])
    measured = {
        "energy_abs": abs(actual["energy"] - expected["energy"]),
        "force_max_abs": float(np.abs(delta).max()),
        "force_rms": float(np.sqrt(np.mean(delta**2))),
    }
    assert all(math.isfinite(v) and v <= budget[k] for k, v in measured.items()), (
        measured
    )


def main() -> None:
    """Verify immutable measurements without requiring the present source to match history."""
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "publication.json").read_text())
    validate_publication(
        manifest,
        {
            entry["path"]: (root / entry["path"]).read_bytes()
            for entry in manifest["files"]
        },
    )
    summary = json.loads(lzma.decompress((root / "summary.json.xz").read_bytes()))
    detail = summary["qualification"]
    source = detail["source"]
    patch = lzma.decompress((root / "qualified-source.patch.xz").read_bytes())
    assert hashlib.sha256(patch).hexdigest() == source["candidate"]["patch_sha256"]
    assert source["candidate"]["base_revision"] == source["baseline"]["base_revision"]
    screen = detail["interrupted_screen"]
    assert (
        hashlib.sha256(screen["reconstruction_patch"].encode()).hexdigest()
        == screen["source"]["patch_sha256"]
    )
    for series in (summary, summary["fast"]):
        assert [p["carbon"] for p in series["points"]] == list(range(1, 8))
        for point in series["points"]:
            record = point["native"]
            assert (
                record["status"] == "completed"
                and point["native_outcome"]["exit_code"] == 0
            )
            assert record["library_sha256"] == source["candidate"]["library_sha256"]
            assert (
                record["driver_sha256"]
                == source["candidate"]["file_sha256"]["benchmarks/readme_cbs.py"]
            )
            assert len(record["samples"]) == record["requested_samples"] == 1
            for sample in record["samples"]:
                check = qualification(sample, record["protocol"], point["reference"])
                assert check["execution_gate"] and check["accuracy_gate"] is not False
        assert [
            p["carbon"] for p in series["points"] if p["reference"] is not None
        ] == [2]
    replay = copy.deepcopy(summary)
    attach_fast(replay, replay["fast"])
    assert all(p["standard_comparison"][0]["passed"] for p in replay["fast"]["points"])
    assert table(replay) == (root / "timings.md").read_text()

    for carbon, cohort in detail["pairs"].items():
        control = summary["points"][int(carbon) - 1]["native"]
        expected = constituents(control["samples"][0], control["protocol"])
        expected_count = {"4": 3, "6": 1, "7": 2}[carbon]
        assert len(cohort["baseline"]) == len(cohort["candidate"]) == expected_count
        for before, after in zip(cohort["baseline"], cohort["candidate"], strict=True):
            assert before["gpu"] == after["gpu"]
            assert before["protocol"] == after["protocol"] == control["protocol"]
        for variant in ("baseline", "candidate"):
            times = []
            for record in cohort[variant]:
                assert record["status"] == "completed"
                assert record["library_sha256"] == source[variant]["library_sha256"]
                assert (
                    record["driver_sha256"]
                    == source[variant]["file_sha256"]["benchmarks/readme_cbs.py"]
                )
                assert len(record["samples"]) == record["requested_samples"] == 1
                sample = record["samples"][0]
                assert qualification(sample, record["protocol"], None)["execution_gate"]
                for key, actual in constituents(sample, record["protocol"]).items():
                    limits = sample["convergence"][
                        "budget" if key == "composite" else "component_budget"
                    ]
                    errors(actual, expected[key], limits)
                times.append(sample["complete_seconds"])
            assert times == cohort["statistics"][variant]["times"]
            if carbon != "7":
                assert (
                    statistics.median(times) == cohort["statistics"][variant]["median"]
                )
        ratios = [
            b["samples"][0]["complete_seconds"] / c["samples"][0]["complete_seconds"]
            for b, c in zip(cohort["baseline"], cohort["candidate"], strict=True)
        ]
        assert ratios == cohort["paired_speedups"]
        print("C" + carbon, "paired old/new Fast speedups:", ratios)

    # The host-vector controller emits action deltas once, not a second total
    # at the parent GMRES scope. Keep the complete closed Lambda subtree and
    # reject overlap if a future trace format changes this ownership.
    for variant, work in detail["diagnostic_work"].items():
        record = work["receipt"]
        assert record["status"] == "completed"
        assert record["library_sha256"] == source[variant]["library_sha256"]
        control = summary["points"][5]["native"]
        assert record["protocol"] == control["protocol"]
        sample = record["samples"][0]
        assert qualification(sample, record["protocol"], None)["execution_gate"]
        expected = constituents(control["samples"][0], control["protocol"])
        for key, actual in constituents(sample, record["protocol"]).items():
            errors(
                actual,
                expected[key],
                sample["convergence"][
                    "budget" if key == "composite" else "component_budget"
                ],
            )
        rows = work["selected_lambda_trace"]
        starts = {r["id"]: r for r in rows if r["event"] == "BEGIN"}
        ends = {r["id"]: r for r in rows if r["event"] == "END"}
        assert starts.keys() == ends.keys()
        assert all(r["status"] == "returned" for r in ends.values())
        roots = [i for i, r in starts.items() if r["name"] == "cc_lambda_complete"]
        assert len(roots) == 1
        assert all(r["parent"] in starts for i, r in starts.items() if i != roots[0])
        counts = collections.Counter()
        for r in rows:
            if r["event"] == "VALUE" and r["key"].startswith("lambda_"):
                assert r["name"] != "cc_lambda_gmres"
                counts[r["key"]] += r["value"]
        assert dict(counts) == work["nonoverlapping_work"]
        assert (
            sum(
                r["name"] == "df_lambda_core_reuse_preparation" for r in starts.values()
            )
            == work["core_reuse_preparations"]
        )

    water = detail["water"]
    assert water["source"] == source["candidate"]
    assert water["library_sha256"] == source["candidate"]["library_sha256"]
    assert {(c["reference"], c["geometry_id"]) for c in water["cases"]} == {
        (r, g) for r in ("exact", "df") for g in (0, 1)
    }
    for case in water["cases"]:
        oracle = case["oracle"]
        expected_e = np.asarray(oracle["component_energies"])
        force = np.empty((5, 3, 3))
        assert {(r["atom"], r["axis"]) for r in oracle["displaced_energies"]} == {
            (i, j) for i in range(3) for j in range(3)
        }
        assert len(oracle["displaced_energies"]) == 9
        for row in oracle["displaced_energies"]:
            minus, plus = np.asarray(row["energies_minus_plus"])
            force[:, row["atom"], row["axis"]] = -(plus - minus) / (
                2 * oracle["step_bohr"]
            )
        np.testing.assert_array_equal(force, oracle["component_forces"])
        assert {
            (s["policy"], s["budget_mib"], s["repeat"]) for s in case["samples"]
        } == {
            ("standard", 2048, 0),
            ("standard", 2048, 1),
            ("fast", 2048, 0),
            ("fast", 2048, 1),
            ("fast", 256, 0),
        }
        assert len(case["samples"]) == 5
        for sample in case["samples"]:
            assert (
                sample["executed_backend"] == "cuda" and sample["complete_seconds"] > 0
            )
            budget = sample["convergence"]["budget"] or {
                "energy_abs": 1e-8,
                "force_max_abs": 3e-6,
                "force_rms": 3e-6,
            }
            component_budget = sample["convergence"]["component_budget"] or budget
            weights = np.array([c["coefficient"] for c in sample["components"]])
            errors(
                sample,
                {
                    "energy": weights @ expected_e,
                    "forces": np.einsum("c,cij->ij", weights, force),
                },
                budget,
            )
            for i, component in enumerate(sample["components"]):
                assert (
                    component["converged"] and component["executed_backend"] == "cuda"
                )
                errors(
                    component,
                    {"energy": expected_e[i], "forces": force[i]},
                    component_budget,
                )
    assert "ERROR SUMMARY: 0 errors" in detail["logs"]["memcheck.log"]
    print(
        "Verified 14 curve receipts and 12 paired ablation receipts (25 unique calls; the first C7 candidate overlaps), plus 20 independent water endpoints, two separate diagnostic calls and C2 directional gates."
    )


if __name__ == "__main__":
    main()
