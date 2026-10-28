"""Recheck retained DIIS endpoint observations without a GPU or native library."""

from __future__ import annotations

import json
import lzma
import math
from pathlib import Path


def verify() -> dict[str, float | int]:
    """Apply fixed scientific gates independently of receipt success labels."""
    data = json.loads(
        lzma.decompress(Path(__file__).with_name("qualification.json.xz").read_bytes())
    )
    maxima = {
        "energy_error": 0.0,
        "force_error": 0.0,
        "paired_energy": 0.0,
        "paired_force": 0.0,
    }
    endpoints = 0
    assert len(data["cases"]) == 3
    for case in data["cases"]:
        rows, receipts = case["endpoints"], case["receipts"]
        assert len(rows) == 20 and len(receipts) == 10
        for row in rows:
            error = abs(row["energy"] - row["reference_energy"])
            assert math.isfinite(error) and error <= 1e-8
            maxima["energy_error"] = max(maxima["energy_error"], error)
            for actual, reference in zip(
                row["forces"], row["reference_forces"], strict=True
            ):
                for a, r in zip(actual, reference, strict=True):
                    error = abs(a - r)
                    assert math.isfinite(error) and error <= 1e-6
                    maxima["force_error"] = max(maxima["force_error"], error)
            endpoints += 1
        for before, after in zip(rows[:10], rows[10:], strict=True):
            assert not before["diagnostics"] and after["diagnostics"]
            for key in ("phase", "geometry", "iterations"):
                assert before[key] == after[key]
            error = abs(before["energy"] - after["energy"])
            assert error <= 1e-12
            maxima["paired_energy"] = max(maxima["paired_energy"], error)
            for left, right in zip(before["forces"], after["forces"], strict=True):
                for a, b in zip(left, right, strict=True):
                    error = abs(a - b)
                    assert error <= 1e-10
                    maxima["paired_force"] = max(maxima["paired_force"], error)
        for row in receipts:
            assert row["work_coverage"] == "observed"
            assert (
                row["residual_vector_elements"]
                == row["gram_dot_reductions"] * row["nbf"] ** 2 * row["spins"]
            )
            assert row["live_history"] <= row["history_capacity"]
            assert row["history_head"] < row["history_capacity"]
        assert [r["gram_dot_reductions"] for r in receipts[:2]] == [
            r["gram_dot_reductions"] for r in receipts[-2:]
        ]
        if case["method"] == "rhf" or case["chunk"] == "2":
            assert any(r["graph_launches_lifetime"] for r in receipts)
    assert len(data["terminal_controls"]) == 4
    for name, rows in data["terminal_controls"].items():
        if "nonconvergence" in name:
            assert [r["converged"] for r in rows] == [0, 0, 1]
            assert rows[0]["gram_dot_reductions"] == rows[1]["gram_dot_reductions"]
        else:
            assert len(rows) == 1 and rows[0]["work_coverage"] == "unavailable"
            assert "gram_dot_reductions" not in rows[0]
    return {"complete_endpoints": endpoints, **maxima}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
