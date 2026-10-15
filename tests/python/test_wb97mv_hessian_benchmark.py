"""Host-only acceptance checks for the opt-in GPU Hessian benchmark."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from benchmarks.compare_wb97mv_hessian_gpu4pyscf import _run_child, _summarize


def _sample(engine: str, seconds: float, matrix_value: float = 1.0) -> dict:
    return {
        "engine": engine,
        "wall_seconds": seconds,
        "result": {"matrix": [[matrix_value]]},
    }


def test_primed_medians_exclude_first_observed_sample() -> None:
    rows = [
        _sample("generativeqc", 100.0),
        _sample("gpu4pyscf", 200.0),
        _sample("gpu4pyscf", 2.0),
        _sample("generativeqc", 6.0),
        _sample("generativeqc", 4.0),
        _sample("gpu4pyscf", 4.0),
    ]
    summary = _summarize(rows, 1e-6)
    assert summary["accepted"]
    assert summary["artifact_primed_fresh_process_median_seconds"] == {
        "generativeqc": 5.0,
        "gpu4pyscf": 3.0,
    }
    assert summary["generativeqc_over_gpu4pyscf_wall_ratio"] == pytest.approx(5 / 3)


def test_inaccurate_repeat_prevents_a_qualified_speed_ratio() -> None:
    rows = [
        _sample("generativeqc", 3.0, 1.001),
        _sample("gpu4pyscf", 2.0),
        _sample("generativeqc", 1.0),
        _sample("gpu4pyscf", 1.0),
    ]
    summary = _summarize(rows, 1e-6)
    assert not summary["accepted"]
    assert summary["maximum_hessian_error_hartree_per_bohr2"] == pytest.approx(0.001)
    assert summary["generativeqc_over_gpu4pyscf_wall_ratio"] is None


def test_single_engine_preflight_is_not_a_speed_comparison() -> None:
    summary = _summarize([_sample("gpu4pyscf", 2.0)], 1e-6)
    assert not summary["accepted"]
    assert summary["maximum_hessian_error_hartree_per_bohr2"] is None
    assert summary["generativeqc_over_gpu4pyscf_wall_ratio"] is None


def test_nonfinite_repeat_cannot_be_hidden_by_a_finite_maximum_error() -> None:
    rows = [
        _sample("generativeqc", 2.0),
        _sample("gpu4pyscf", 1.0),
        _sample("generativeqc", 1.0, float("nan")),
    ]
    with pytest.raises(ValueError, match="finite real"):
        _summarize(rows, 1e-6)


def test_accuracy_comparison_does_not_broadcast_different_matrix_shapes() -> None:
    reference = _sample("gpu4pyscf", 1.0)
    reference["result"]["matrix"] = [[1.0, 1.0], [1.0, 1.0]]
    with pytest.raises(ValueError, match="broadcasting"):
        _summarize([_sample("generativeqc", 2.0), reference], 1e-6)


def test_retained_h2_baseline_reproduces_all_repeat_accuracy_and_timing_gates() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/wb97mv-hessian-bridge-h2-20261015.json"
    )
    case = json.loads(path.read_text())["cases"]["h2"]
    samples = [
        {
            **row,
            "result": {
                **row["result"],
                "matrix": case["matrices"][row["result"]["matrix_index"]],
            },
        }
        for row in case["samples"]
    ]
    assert len(samples) == 6
    summary = _summarize(samples, case["summary"]["hessian_absolute_tolerance"])
    assert summary == case["summary"]
    assert not summary["accepted"]
    assert summary["generativeqc_over_gpu4pyscf_wall_ratio"] is None


def test_retained_water_timeout_does_not_invent_native_latency() -> None:
    root = Path(__file__).resolve().parents[2] / "benchmarks/results"
    receipt = json.loads(
        (root / "wb97mv-hessian-bridge-water-timeout-20261015.json").read_text()
    )
    assert (root / receipt["workload_reference"]).is_file()
    assert not receipt["accepted"]
    assert receipt["qualified_wall_ratio"] is None
    assert "timed out" in receipt["failure"]
    assert {row["engine"] for row in receipt["samples"]} == {"gpu4pyscf"}


def test_timeout_cleans_a_separately_grouped_endpoint_worker(tmp_path: Path) -> None:
    prefix = tmp_path / "nested-worker"
    command = [
        sys.executable,
        "-c",
        (
            "import subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],process_group=0); "
            "print(child.pid,flush=True); child.wait()"
        ),
    ]
    with pytest.raises(RuntimeError, match="timed out"):
        _run_child(command, prefix, 1)
    child_id = int(prefix.with_suffix(".stdout").read_text())
    status = Path(f"/proc/{child_id}/status")
    for _attempt in range(100):
        try:
            state = status.read_text()
        except FileNotFoundError:
            return
        if "\nState:\tZ" in state:
            return
        time.sleep(0.01)
    pytest.fail("timed-out endpoint worker is still running")
