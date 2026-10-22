"""Replay the limited numerical and profiling conclusions; no default promotion."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.df_lambda_core_reuse_ablation import trace_categories
from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_json
from tools.render_readme_cbs import constituents, qualification


def same_diagnostics(actual: object, expected: object) -> bool:
    """Allow only reduction roundoff; raw receipt bytes and gates stay exact."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            same_diagnostics(value, expected[key]) for key, value in actual.items()
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            same_diagnostics(a, b) for a, b in zip(actual, expected, strict=True)
        )
    if isinstance(actual, float):
        return (
            math.isfinite(actual)
            and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
        )
    return actual == expected


def main() -> None:
    """Check complete endpoints, independent C2 gates and disjoint Lambda phases."""
    root = Path(__file__).resolve().parent
    manifest = load_json(root / "publication.json")
    validate_publication(
        manifest,
        {row["path"]: (root / row["path"]).read_bytes() for row in manifest["files"]},
    )
    data = load_json(root / "samples.json.xz")
    source = load_json(root / "source.json.xz")
    patches = load_json(root / "candidate-patches.json.xz")
    profiles = load_json(root / "profiles.json.xz")
    for variant, patch in patches.items():
        assert (
            hashlib.sha256(patch.encode()).hexdigest()
            == source["patch_sha256"][variant]
        )
    records = data["records"]
    assert len(records) == 9
    for name, receipt in records.items():
        assert receipt["status"] == "completed"
        assert len(receipt["samples"]) == receipt["requested_samples"] == 1
        assert receipt["gpu"] == records["c4-fast"]["gpu"]
        variant = next((v for v in reversed(tuple(patches)) if v in name), None)
        expected = (
            source["native_candidate_library_sha256"][variant]
            if variant
            else source["baseline_library_sha256"]
        )
        assert receipt["library_sha256"] == expected
        sample = receipt["samples"][0]
        oracle = (
            data["independent_c2_reference"]
            if receipt["protocol"]["carbon"] == 2
            else None
        )
        check = qualification(sample, receipt["protocol"], oracle)
        assert same_diagnostics(check, data["checks"][name])
        assert check["execution_gate"] and check["accuracy_gate"] is not False
        if receipt["policy"] != "fast":
            continue
        baseline = records[
            "c4-baseline-clean-standard"
            if receipt["protocol"]["carbon"] == 4
            else "c2-baseline-clean"
        ]
        assert baseline["protocol"] == receipt["protocol"]
        expected_rows = constituents(baseline["samples"][0], baseline["protocol"])
        for key, row in constituents(sample, receipt["protocol"]).items():
            delta = np.asarray(row["forces"]) - np.asarray(expected_rows[key]["forces"])
            errors = {
                "energy_abs": abs(row["energy"] - expected_rows[key]["energy"]),
                "force_max_abs": float(np.max(np.abs(delta))),
                "force_rms": float(np.sqrt(np.mean(delta**2))),
            }
            recorded = data["comparisons"][name]["components"][key]
            budget = sample["convergence"][
                "budget" if key == "composite" else "component_budget"
            ]
            assert same_diagnostics(recorded, dict(errors, budget=budget))
            assert all(value <= budget[k] for k, value in errors.items())
    with tempfile.TemporaryDirectory() as directory:
        for name, profile in profiles.items():
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in profile["excerpt"]) + "\n"
            )
            assert trace_categories(path) == profile["categories"], name
    for name in (
        "c4-baseline-clean-standard",
        "c4-baseline-clean",
        "c4-lambda-mixed-q32-clean",
    ):
        assert "GENERATIVEQC_DF_PROGRESS_TRACE" not in records[name]["controls"]
        print(name, records[name]["samples"][0]["complete_seconds"])
    print(
        "Verified nine endpoints, C2 independent gates and four Lambda trace excerpts; performance remains inconclusive."
    )


if __name__ == "__main__":
    main()
