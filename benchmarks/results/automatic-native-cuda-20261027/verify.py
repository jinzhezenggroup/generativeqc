"""Recheck retained numerical, binary/resource and complete endpoint evidence.

Run explicitly from the repository root with PYTHONPATH=python:.; this is an
offline campaign audit, not a required CI fingerprint of production source.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from tools.generativeqc_validation.record import load_json

ROOT = Path(__file__).resolve().parent


def performance_rows(records: dict[str, Any]) -> list[dict[str, Any]]:
    """Use every ABBA sample, grouping equivalent complete endpoint workloads."""
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for filename, record in sorted(records.items()):
        arm = "candidate" if "candidate" in filename else "baseline"
        for row in record["rows"]:
            grouped.setdefault((row["method"], row["phase"]), {}).setdefault(
                arm, []
            ).append(row)
    result = []
    for (method, phase), arms in sorted(grouped.items()):
        expected = 12 if phase == "warm" else 4
        assert set(arms) == {"baseline", "candidate"}
        assert all(len(rows) == expected for rows in arms.values())
        anchor = arms["candidate"][0]
        statistics_by_arm = {}
        for arm, rows in arms.items():
            for row in rows:
                np.testing.assert_allclose(
                    row["energy"], anchor["energy"], atol=3e-9, rtol=0
                )
                np.testing.assert_allclose(
                    row["forces"], anchor["forces"], atol=3e-8, rtol=0
                )
                assert row["seconds"] > 0 and row["fock_builds"] > 0
                assert row["coulomb_exchange_work"]
                assert row["executed_backend"] == "cuda"
            values = [row["seconds"] for row in rows]
            median = statistics.median(values)
            statistics_by_arm[arm] = {
                "count": len(values),
                "median_seconds": median,
                "mad_seconds": statistics.median(
                    abs(value - median) for value in values
                ),
                "minimum_seconds": min(values),
                "maximum_seconds": max(values),
                "iterations": [row["iterations"] for row in rows],
                "fock_builds": [row["fock_builds"] for row in rows],
                "warm_fallbacks": [row["warm_fallback"] for row in rows],
            }
        ratio = (
            statistics_by_arm["candidate"]["median_seconds"]
            / statistics_by_arm["baseline"]["median_seconds"]
        )
        result.append(
            {
                "method": method,
                "phase": phase,
                "arms": statistics_by_arm,
                "ratio": ratio,
                "passed": ratio <= 1.05,
            }
        )
    return result


def main() -> None:
    """Authenticate stored bytes before re-evaluating the declared gates."""
    if not __debug__:
        raise SystemExit("Do not run the evidence verifier with Python -O")
    manifest = load_json(ROOT / "manifest.json")
    for filename, identity in manifest["files"].items():
        data = (ROOT / filename).read_bytes()
        assert len(data) == identity["bytes"]
        assert hashlib.sha256(data).hexdigest() == identity["sha256"], filename
    references = load_json(ROOT / "references.json.gz")
    executions = load_json(ROOT / "executions.json.gz")
    audit = load_json(ROOT / "aot-audit.json.gz")
    review_path = ROOT / "reference-reuse-source-review.json"
    review = load_json(review_path)
    assert len(executions) == len(references) == 34
    endpoints = full_vectors = fd_directions = 0
    for filename, execution in executions.items():
        reference = references[filename]
        assert (
            execution["source_reuse_review_sha256"]
            == hashlib.sha256(review_path.read_bytes()).hexdigest()
        )
        # Original producer uses atomic_json's exact sorted representation.
        original = (json.dumps(reference, indent=2, sort_keys=True) + "\n").encode()
        assert (
            hashlib.sha256(original).hexdigest()
            == execution["reference_receipt_sha256"]
        )
        assert execution["name"] == reference["name"]
        assert execution["spins"] == reference["spins"]
        assert len(execution["rows"]) == len(reference["rows"])
        for actual, expected in zip(execution["rows"], reference["rows"], strict=True):
            assert actual["phase"] == expected["phase"]
            receipt = actual["artifact"]
            assert receipt["force_completed"]
            assert receipt["point_artifact"]["sha256"] == audit["library"]["sha256"]
            assert receipt["force_artifact"]["sha256"] == audit["library"]["sha256"]
            for key in (
                "point_expression_identity",
                "point_source_identity",
                "force_plan_identity",
            ):
                assert receipt[key] == expected["artifact"][key]
            profile = receipt["force_profile"]
            transition = review["profiles"][profile]
            assert (
                expected["artifact"]["force_source_identity"]
                == transition["previous_identity"]
            )
            assert receipt["force_source_identity"] == transition["current_identity"]
            assert (
                audit["profiles"][profile]["source"]["sha256"]
                == transition["current_source_sha256"]
            )
            np.testing.assert_allclose(
                actual["energy"], actual["reference_energy"], atol=3e-9, rtol=0
            )
            np.testing.assert_allclose(
                actual["forces"], expected["forces"], atol=3e-8, rtol=0
            )
            np.testing.assert_allclose(
                np.sum(actual["forces"], axis=0), 0, atol=2e-8, rtol=0
            )
            forces = np.asarray(actual["forces"])
            if len(expected["independent_fd"]) == forces.size:
                projected = forces.ravel()
                full_vectors += 1
            else:
                projected = [np.dot(forces[-1], [0.2, -0.3, 0.4])]
            for field in ("native_fd", "independent_fd"):
                np.testing.assert_allclose(
                    projected, expected[field], atol=3e-5, rtol=0
                )
            fd_directions += len(projected)
            endpoints += 1
    assert (endpoints, full_vectors, fd_directions) == (58, 22, 300)
    assert len(audit["profiles"]) == 22 and len(audit["executions"]) == 58
    for profile in audit["profiles"].values():
        for rows in profile["scopes"].values():
            assert rows and any(row["REG"] > 0 for row in rows.values())
            for row in rows.values():
                for field, maximum in (
                    ("REG", 255),
                    ("STACK", 4096),
                    ("LOCAL", 4096),
                    ("SHARED", 49152),
                ):
                    assert 0 <= row[field] <= maximum
    for row in audit["executions"]:
        assert 0 <= row["dynamic_becke_shared_bytes"] <= 49152
    performance = load_json(ROOT / "performance.json.gz")
    for filename, record in performance.items():
        arm = "candidate" if "candidate" in filename else "baseline"
        assert record["library_sha256"] == manifest["native_libraries"][arm]
    timings = performance_rows(performance)
    assert len(timings) == 24 and all(row["passed"] for row in timings)
    assert timings == manifest["performance"]
    print(
        f"Verified {endpoints} force endpoints, {fd_directions} FD directions, "
        f"22 executed AOT profiles and {len(timings)} performance gates"
    )


if __name__ == "__main__":
    main()
