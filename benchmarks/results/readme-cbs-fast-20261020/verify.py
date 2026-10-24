"""Recheck frozen README measurements and their publication hashes without CUDA."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record
from tools.render_readme_cbs import attach_fast, completed_times, qualification, table


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
    """Recompute every scientific/control gate from the retained raw results."""
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "publication.json").read_text())
    validate_publication(
        manifest,
        {row["path"]: (root / row["path"]).read_bytes() for row in manifest["files"]},
    )
    original = load_publication_record(root, role="samples", name="summary.json")
    summary = copy.deepcopy(original)
    for series in (summary, summary["fast"]):
        assert [point["carbon"] for point in series["points"]] == list(range(1, 8))
        for point in series["points"]:
            native = point["native"]
            assert native["status"] == "completed"
            assert len(native["samples"]) == native["requested_samples"] == 1
            assert point["native_outcome"]["exit_code"] == 0
            checks = [
                qualification(sample, native["protocol"], point["reference"])
                for sample in native["samples"]
            ]
            assert same_diagnostics(checks, point["checks"])
            assert all(c["execution_gate"] for c in checks)
        checked = [
            p["carbon"] for p in series["points"] if p["checks"][0]["accuracy_gate"]
        ]
        assert checked == [2], checked
    attach_fast(summary, summary["fast"])
    assert same_diagnostics(summary, original), (
        "stored matched-control diagnostics differ"
    )
    assert all(completed_times(point) for point in summary["fast"]["points"])
    for point in summary["fast"]["points"]:
        assert point["native"]["samples"][0]["convergence"]["arithmetic_request"] == (
            "df_ccsd_fp32_preconvergence"
        )
        assert point["standard_comparison"][0]["arithmetic"] is not None
    assert table(summary) == (root / "timings.md").read_text()
    df = load_publication_record(root, role="samples", name="df-water-paired.json")
    source = json.loads((root / "source-manifest.json").read_text())
    assert df["library_sha256"] == source["library_sha256"]
    assert df["source"]["file_sha256"] == source["file_sha256"]
    assert {(row["reference"], row["policy"]) for row in df["samples"]} == {
        (reference, policy)
        for reference in ("exact", "df")
        for policy in ("standard", "fast")
    }
    for row in df["samples"]:
        assert len(row["samples"]) == 4
        for sample in row["samples"]:
            assert sample["executed_backend"] == "cuda"
            assert sample["endpoint_seconds"] > 0
            assert len(sample["components"]) == 5
            assert all(component["converged"] for component in sample["components"])
            if row["policy"] == "fast":
                assert sample["convergence"]["arithmetic"] == "mixed"
                cc = next(
                    c for c in sample["components"] if c["method"] == "df-ccsd(t)"
                )
                counts = cc["cc_performance"]
                assert counts["ccsd_fp32_requested"] and counts["ccsd_fp32_executed"]
                assert not counts["ccsd_fp32_refused"]
                assert counts["ccsd_fp32_gemm_calls"] > 0
                assert counts["ccsd_fp32_evaluations"] > 0
        convergence = row["samples"][-1]["convergence"]
        budget = convergence["budget"] or {
            "energy_abs": 1e-8,
            "force_max_abs": 3e-6,
            "force_rms": 3e-6,
        }
        component_budget = convergence["component_budget"] or budget
        for errors, limits in [
            (row["oracle"], budget),
            *[(errors, component_budget) for errors in row["component_errors"]],
        ]:
            assert all(
                math.isfinite(value) and value <= limits[key]
                for key, value in errors.items()
            )
    print(
        "Verified 14 QM7 endpoints, FP32 provenance, all Fast vector controls, "
        "C2 directional checks and exact/full-DF water independent error gates."
    )


if __name__ == "__main__":
    main()
