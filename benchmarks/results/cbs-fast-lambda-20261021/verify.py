"""Replay retained CBS Lambda numerical gates and complete endpoint observations."""

from __future__ import annotations

import copy
import hashlib
import lzma
import math
import statistics
from pathlib import Path

import numpy as np

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_json
from tools.render_readme_cbs import (
    attach_fast,
    completed_times,
    constituents,
    qualification,
    table,
)


def equal(actual: object, expected: object) -> bool:
    """Allow derived reduction roundoff only; raw receipt bytes remain hash-bound."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            equal(v, expected[k]) for k, v in actual.items()
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            equal(a, b) for a, b in zip(actual, expected, strict=True)
        )
    if isinstance(actual, float):
        return math.isfinite(actual) and math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=1e-15
        )
    return actual == expected


def errors(
    energy: float,
    forces: list | np.ndarray,
    expected_energy: float,
    expected_forces: list | np.ndarray,
    limits: dict,
) -> dict:
    """Gate every observable without cancellation between constituent errors."""
    delta = np.asarray(forces) - np.asarray(expected_forces)
    values = {
        "energy_abs": float(abs(energy - expected_energy)),
        "force_max_abs": float(abs(delta).max()),
        "force_rms": float(np.sqrt(np.mean(delta**2))),
    }
    assert all(math.isfinite(v) and v <= limits[k] for k, v in values.items()), (
        values,
        limits,
    )
    return values


def same_composite_errors(actual: dict, recorded: dict, terms: np.ndarray) -> bool:
    """Allow only dot-product roundoff in the derived composite energy error.

    Two BLAS reductions can differ by one ULP of the total energy even though
    the underlying component energies are byte-identical. Bound both reductions
    by gamma_n times their absolute term sum. Scientific gates are recomputed
    separately, and force diagnostics retain the tighter ordinary comparison.
    """
    if actual.keys() != recorded.keys():
        return False
    if not equal(
        {k: v for k, v in actual.items() if k != "energy_abs"},
        {k: v for k, v in recorded.items() if k != "energy_abs"},
    ):
        return False
    values = actual["energy_abs"], recorded["energy_abs"]
    if any(type(v) is not float or not math.isfinite(v) or v < 0 for v in values):
        return False
    neps = len(terms) * np.finfo(float).eps
    roundoff = 2 * neps / (1 - neps) * float(np.abs(terms).sum())
    return abs(values[0] - values[1]) <= roundoff


def main() -> None:
    """Verify all receipts, both independent scopes, repeated controls and source identity."""
    root = Path(__file__).resolve().parent
    publication = load_json(root / "publication.json")
    validate_publication(
        publication,
        {r["path"]: (root / r["path"]).read_bytes() for r in publication["files"]},
    )
    original = load_json(root / "summary.json.xz")
    detail = load_json(root / "qualification.json.xz")
    source = detail["source"]
    patch = lzma.decompress((root / "qualified-source.patch.xz").read_bytes())
    assert hashlib.sha256(patch).hexdigest() == source["patch_sha256"]
    summary = copy.deepcopy(original)
    for series in (summary, summary["fast"]):
        assert [p["carbon"] for p in series["points"]] == list(range(1, 8))
        for point in series["points"]:
            receipt = point["native"]
            assert (
                receipt["status"] == "completed"
                and point["native_outcome"]["exit_code"] == 0
            )
            assert receipt["library_sha256"] == source["library_sha256"]
            assert (
                receipt["driver_sha256"]
                == source["file_sha256"]["benchmarks/readme_cbs.py"]
            )
            assert len(receipt["samples"]) == receipt["requested_samples"] == 1
            checks = [
                qualification(s, receipt["protocol"], point["reference"])
                for s in receipt["samples"]
            ]
            assert equal(checks, point["checks"])
            assert all(c["execution_gate"] for c in checks)
        assert [
            p["carbon"] for p in series["points"] if p["checks"][0]["accuracy_gate"]
        ] == [2]
    attach_fast(summary, summary["fast"])
    assert equal(summary, original)
    assert all(completed_times(p) for p in summary["fast"]["points"])
    assert table(summary) == (root / "timings.md").read_text()

    control = summary["points"][3]["native"]
    expected = constituents(control["samples"][0], control["protocol"])
    times = {}
    for variant, receipts in detail["c4_repeats"].items():
        assert len(receipts) == 3
        times[variant] = []
        for receipt in receipts:
            assert receipt["status"] == "completed"
            assert (
                receipt["protocol"] == control["protocol"]
                and receipt["gpu"] == control["gpu"]
            )
            assert (
                receipt["library_sha256"]
                == source[
                    "baseline_library_sha256"
                    if variant == "baseline"
                    else "library_sha256"
                ]
            )
            sample = receipt["samples"][0]
            assert qualification(sample, receipt["protocol"], None)["execution_gate"]
            for key, row in constituents(sample, receipt["protocol"]).items():
                budget = sample["convergence"][
                    "budget" if key == "composite" else "component_budget"
                ]
                errors(
                    row["energy"],
                    row["forces"],
                    expected[key]["energy"],
                    expected[key]["forces"],
                    budget,
                )
            times[variant].append(sample["complete_seconds"])
        measured = {
            "median": statistics.median(times[variant]),
            "min": min(times[variant]),
            "max": max(times[variant]),
        }
        assert equal(measured, detail["c4_statistics"][variant])
    assert max(times["candidate"]) < min(times["baseline"])

    larger = summary["points"][5]["native"]
    expected = constituents(larger["samples"][0], larger["protocol"])
    for variant, receipt in detail["c6_ablation"].items():
        assert receipt["status"] == "completed"
        assert (
            receipt["protocol"] == larger["protocol"]
            and receipt["gpu"] == larger["gpu"]
        )
        assert (
            receipt["library_sha256"]
            == source[
                "baseline_library_sha256" if variant == "baseline" else "library_sha256"
            ]
        )
        sample = receipt["samples"][0]
        assert qualification(sample, receipt["protocol"], None)["execution_gate"]
        for key, row in constituents(sample, receipt["protocol"]).items():
            budget = sample["convergence"][
                "budget" if key == "composite" else "component_budget"
            ]
            errors(
                row["energy"],
                row["forces"],
                expected[key]["energy"],
                expected[key]["forces"],
                budget,
            )
    water = detail["water"]
    assert water["library_sha256"] == source["library_sha256"]
    assert water["source"]["file_sha256"] == source["file_sha256"]
    assert {(c["reference"], c["geometry_id"]) for c in water["cases"]} == {
        (r, g) for r in ("exact", "df") for g in (0, 1)
    }
    for case in water["cases"]:
        oracle = case["oracle"]
        expected_energy = np.array(oracle["component_energies"])
        expected_force = np.zeros((5, 3, 3))
        assert len(oracle["displaced_energies"]) == 9
        for row in oracle["displaced_energies"]:
            minus, plus = np.asarray(row["energies_minus_plus"])
            expected_force[:, row["atom"], row["axis"]] = -(plus - minus) / (
                2 * oracle["step_bohr"]
            )
        np.testing.assert_array_equal(expected_force, oracle["component_forces"])
        assert len(case["samples"]) == 5
        assert {
            (s["policy"], s["budget_mib"], s["repeat"]) for s in case["samples"]
        } == {
            ("standard", 2048, 0),
            ("standard", 2048, 1),
            ("fast", 2048, 0),
            ("fast", 2048, 1),
            ("fast", 256, 0),
        }
        for sample in case["samples"]:
            assert (
                sample["executed_backend"] == "cuda" and sample["complete_seconds"] > 0
            )
            budget = sample["convergence"]["budget"] or {
                "energy_abs": 1e-8,
                "force_max_abs": 3e-6,
                "force_rms": 3e-6,
            }
            cb = sample["convergence"]["component_budget"] or budget
            weights = np.array([c["coefficient"] for c in sample["components"]])
            actual = errors(
                sample["energy"],
                sample["forces"],
                weights @ expected_energy,
                np.einsum("c,cij->ij", weights, expected_force),
                budget,
            )
            assert same_composite_errors(
                actual, sample["oracle_errors"], weights * expected_energy
            )
            for i, component in enumerate(sample["components"]):
                assert (
                    component["converged"] and component["executed_backend"] == "cuda"
                )
                actual = errors(
                    component["energy"],
                    component["forces"],
                    expected_energy[i],
                    expected_force[i],
                    cb,
                )
                assert equal(actual, sample["component_oracle_errors"][i])
            cc = next(
                c["cc_performance"]
                for c in sample["components"]
                if c["method"] == "df-ccsd(t)"
            )
            assert cc["lambda_fp32_requested"] is (sample["policy"] == "fast")
            assert cc["lambda_fp32_executed"] is (cc["lambda_fp32_gemm_calls"] > 0)
            if cc["lambda_fp32_executed"]:
                assert not cc["lambda_fp32_refused"] and cc["lambda_fp64_actions"] > 0
    assert (
        "joint DF triples response exceeds complete numeric budget"
        in detail["memory_refusal"]["log"]
    )
    assert len(detail["memory_refusal"]["partial"]["cases"][0]["samples"]) == 4
    print(
        "Verified 14 homepage endpoints, six matched C4 repeats, two C6 ablations, 20 independent water endpoints, C2 directional gates and explicit memory refusal."
    )
    print("C4 median seconds:", {k: statistics.median(v) for k, v in times.items()})


if __name__ == "__main__":
    main()
