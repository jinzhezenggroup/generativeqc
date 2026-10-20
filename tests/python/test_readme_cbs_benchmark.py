"""Complete CBS timing boundaries and honest independent-oracle publication."""

import copy
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks import readme_cbs
from tools.render_readme_cbs import (
    attach_fast,
    collect,
    completed_times,
    outcome,
    qualification,
    reference_label,
    render,
    table,
)


def records() -> tuple[dict, dict, dict]:
    """A five-constituent DF composite with a normalized independent direction."""
    protocol = {
        "carbon": 1,
        "atoms_bohr": [[6, [0, 0, 0]]],
        "orbital_identities": ["dz", "tz"],
    }
    result = {
        "energy": -40.0,
        "forces": [[0.0, 0.0, 0.0]],
        "converged": True,
        "executed_backend": "cuda",
    }
    sample = dict(
        result,
        complete_seconds=12.0,
        components=[
            {"basis_identity": basis, "method": method, "result": copy.deepcopy(result)}
            for basis, method in (
                ("dz", "rhf"),
                ("tz", "rhf"),
                ("dz", "mp2"),
                ("tz", "mp2"),
                ("dz", "df-ccsd(t)"),
            )
        ],
    )
    keys = ("composite", "low/rhf", "high/rhf", "low/mp2", "high/mp2", "low/df-ccsd(t)")
    reference = {
        "schema": readme_cbs.SCHEMA,
        "engine": "reference",
        "protocol": copy.deepcopy(protocol),
        "status": "completed",
        "direction": [[1.0, 0.0, 0.0]],
        "energies": dict.fromkeys(keys, -40.0),
        "derivatives": [
            {"step_bohr": step, "values": dict.fromkeys(keys, 0.0)}
            for step in readme_cbs.STEPS
        ],
    }
    return protocol, sample, reference


@pytest.mark.parametrize(
    "payload", [b"", b"\x00x" * 1048577], ids=["empty", "multiple-chunks"]
)
def test_receipt_digest_matches_standard_sha256(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    assert readme_cbs.digest(path) == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("policy", ("standard", "fast"))
@pytest.mark.parametrize("hf_reference", ("exact", "df"))
def test_cold_timer_includes_construction_and_returned_forces(
    monkeypatch: pytest.MonkeyPatch,
    hf_reference: str,
    policy: str,
) -> None:
    import generativeqc

    events = []

    @dataclass
    class Returned:
        forces: np.ndarray

    class Calculator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            events.append(("construct", kwargs))

        def singlepoint(self, atoms: list, *, properties: tuple) -> Returned:
            events.append(("host result", properties))
            return Returned(np.zeros((len(atoms), 3)))

    def timer() -> float:
        events.append(("timer", None))
        return 7.0 * len(events)

    monkeypatch.setattr(generativeqc, "CCSDTCBSCalculator", Calculator)
    monkeypatch.setattr(
        generativeqc, "CBSBasis", lambda basis, cardinal: (basis, cardinal)
    )
    monkeypatch.setattr(readme_cbs.time, "perf_counter", timer)
    result = readme_cbs.native_call(
        [[1, [0, 0, 0]]], ("dz", "tz"), ("dz-ri", "tz-ri"), 4096, hf_reference, policy
    )
    assert [event[0] for event in events] == [
        "timer",
        "construct",
        "host result",
        "timer",
    ]
    assert events[1][1]["auxiliary_bases"] == ("dz-ri", "tz-ri")
    assert events[1][1].get("hf_reference", "exact") == hf_reference
    assert events[1][1]["policy"] == policy
    assert events[2][1] == ("energy", "forces")
    assert result["complete_seconds"] == 21
    assert result["forces"].shape == (1, 3)


def test_distinct_reference_models_cannot_be_pooled_or_checked_together() -> None:
    protocol, sample, reference = records()
    protocol["hf_reference"] = "df"
    with pytest.raises(ValueError, match="different scientific input"):
        qualification(sample, protocol, reference)
    summary = {"points": [{"native": {"protocol": protocol}}]}
    assert "Full DF: RI-JK" in reference_label(summary)
    summary["points"].append({"native": reference})
    with pytest.raises(ValueError, match="one explicit HF reference"):
        reference_label(summary)


@pytest.mark.parametrize("hf_reference", ("exact", "df"))
def test_reference_selector_reaches_independent_oracle(
    monkeypatch: pytest.MonkeyPatch, hf_reference: str
) -> None:
    from tools.generativeqc_validation import correlation_df_oracle

    calls = []

    def independent(atoms: list, orbital: str, auxiliary: str, **options: Any) -> dict:
        calls.append((orbital, auxiliary, options))
        return {"rhf": -1.0, "mp2": -1.1, "df-ccsd(t)": -1.12}

    monkeypatch.setattr(correlation_df_oracle, "energies", independent)
    result = readme_cbs.oracle_energies([], hf_reference)
    assert result["composite"] == pytest.approx(-1.12)
    assert all(
        call[2].get("df_reference", False) == (hf_reference == "df") for call in calls
    )
    assert all(
        call[1].endswith("-jkfit" if hf_reference == "df" else "-ri") for call in calls
    )


def test_missing_or_incomplete_oracle_never_passes_accuracy() -> None:
    protocol, sample, reference = records()
    for oracle in (
        None,
        dict(reference, status="running"),
        dict(reference, derivatives=reference["derivatives"][:1]),
    ):
        check = qualification(sample, protocol, oracle)
        assert check["execution_gate"] is True
        assert check["accuracy_gate"] is None
        assert check["maximum_energy_error_hartree"] is None
        assert check["maximum_directional_error_hartree_per_bohr"] is None


def test_component_error_cannot_hide_in_composite_cancellation() -> None:
    protocol, sample, reference = records()
    assert qualification(sample, protocol, reference)["accuracy_gate"] is True
    sample["components"][2]["result"]["energy"] += 1e-5
    check = qualification(sample, protocol, reference)
    assert check["execution_gate"] is True
    assert check["accuracy_gate"] is False
    assert check["component_errors"]["composite"]["energy"] == 0
    assert check["maximum_energy_error_hartree"] > 1e-6


@pytest.mark.parametrize(
    "mutation",
    [
        "energy_only",
        "unconverged",
        "wrong_backend",
        "nan_force",
        "translation",
        "negative_time",
    ],
)
def test_partial_or_invalid_native_result_has_no_endpoint_gate(mutation: str) -> None:
    protocol, sample, _ = records()
    row = sample["components"][0]["result"]
    if mutation == "energy_only":
        row["forces"] = None
    elif mutation == "unconverged":
        row["converged"] = False
    elif mutation == "wrong_backend":
        row["executed_backend"] = "cpu"
    elif mutation == "nan_force":
        row["forces"][0][0] = float("nan")
    elif mutation == "translation":
        row["forces"][0][0] = 0.1
    else:
        sample["complete_seconds"] = -1
    assert qualification(sample, protocol, None)["execution_gate"] is False


def test_mismatched_geometry_and_incomplete_constituents_are_rejected() -> None:
    protocol, sample, reference = records()
    reference["protocol"]["atoms_bohr"][0][1][0] = 0.1
    with pytest.raises(ValueError, match="different scientific input"):
        qualification(sample, protocol, reference)
    sample["components"].pop()
    with pytest.raises(ValueError, match="incomplete DF-CBS"):
        qualification(sample, protocol, None)


def test_two_step_force_check_cannot_ignore_second_step() -> None:
    protocol, sample, reference = records()
    reference["derivatives"][1]["values"]["high/mp2"] = 1e-4
    check = qualification(sample, protocol, reference)
    assert check["accuracy_gate"] is False
    assert check["maximum_directional_error_hartree_per_bohr"] == 1e-4


def test_later_timeout_preserves_cold_but_not_partial_replays(tmp_path: Path) -> None:
    protocol, sample, _ = records()
    directory = tmp_path / "1"
    directory.mkdir()
    readme_cbs.checkpoint(
        directory / "native.json",
        {
            "schema": readme_cbs.SCHEMA,
            "engine": "native",
            "protocol": protocol,
            "status": "running",
            "requested_samples": 4,
            "samples": [sample, sample],
        },
    )
    (directory / "native.outcome").write_text(
        json.dumps({"exit_code": 124, "point_time_limit_seconds": 7200})
    )
    point = collect(tmp_path, 1)
    assert completed_times(point) == [12]
    assert completed_times(point, "replay") == []
    assert outcome(point) == "unchecked ‡"
    assert point["native_outcome"]["exit_code"] == 124
    point["native"]["samples"] = []
    point["checks"] = []
    assert completed_times(point) == []
    assert outcome(point) == "point time limit"


def test_failed_numerics_never_supply_plot_time() -> None:
    protocol, sample, reference = records()
    reference["energies"]["composite"] += 1
    point = {
        "native": {"samples": [sample], "requested_samples": 1},
        "checks": [qualification(sample, protocol, reference)],
    }
    assert completed_times(point) == []
    assert outcome(point) == "numerical gate failed"


def test_primary_cc_counter_is_not_reported_as_companion_scf_work() -> None:
    protocol, sample, _ = records()
    for component in sample["components"]:
        component["result"]["iterations"] = (
            22 if component["basis_identity"] == "dz" else 14
        )
        component["result"]["correlation"] = {
            "response_iterations": 3,
            "ccsd_iterations": 22,
        }
    sample["execution_groups"] = [
        {"basis_identity": "dz", "elapsed_seconds": 3.0},
        {"basis_identity": "tz", "elapsed_seconds": 9.0},
    ]
    point = {
        "carbon": 1,
        "native": {"protocol": protocol, "samples": [sample], "requested_samples": 1},
        "checks": [qualification(sample, protocol, None)],
    }
    result = table({"points": [point]})
    assert "| C1 | —/14 | 22 |" in result
    assert "22/14" not in result


def test_readme_chart_hides_accuracy_markers_without_changing_gates(
    tmp_path: Path,
) -> None:
    """README highlights complete times; scientific acceptance stays in the archive."""
    pytest.importorskip("matplotlib")
    protocol, sample, _ = records()
    check = qualification(sample, protocol, None)
    assert check["execution_gate"] is True
    assert check["accuracy_gate"] is None
    point = {
        "carbon": 1,
        "native": {"protocol": protocol, "samples": [sample], "requested_samples": 1},
        "checks": [check],
    }
    chart = tmp_path / "readme.svg"
    render({"points": [point]}, chart, readme_style=True)
    content = chart.read_text()
    assert "12.0 s" in content
    assert "RHF uses exact J/K" in content
    assert "directional-force checks" not in content
    assert "accuracy unchecked" not in content
    assert "‡" not in content
    assert "No full-vector or CBS basis-limit accuracy claim" not in content


def test_plot_retains_all_seven_positions_and_timeout_gaps(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    points = [
        {
            "carbon": carbon,
            "native": None,
            "checks": [],
            "native_outcome": {"exit_code": 124},
        }
        for carbon in range(1, 8)
    ]
    destination = tmp_path / "figure.svg"
    render({"points": points}, destination)
    content = destination.read_text()
    assert "point time limit" in content
    assert "Complete E+F" in content
    assert "RHF uses exact J/K" in content


def test_raw_checkpoints_cannot_write_reviewed_results() -> None:
    root = Path(readme_cbs.__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="raw benchmark output"):
        readme_cbs.checkpoint(root / "benchmarks/results/disallowed.json", {})


def test_wrong_dataset_is_rejected_before_loading_coordinates(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    dataset = tmp_path / "qm7.mat"
    dataset.write_bytes(b"not the pinned QM7 dataset")
    with pytest.raises(ValueError, match="dataset checksum"):
        readme_cbs.geometry(dataset, 7)


def test_native_cli_rejects_unscheduled_gpu_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "readme_cbs",
            "native",
            "--qm7",
            str(tmp_path / "absent.mat"),
            "--carbon",
            "1",
            "--source-revision",
            "test-revision",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        readme_cbs.main()
    assert error.value.code == 2
    assert not (tmp_path / "result.json").exists()


def test_runner_continues_after_failed_point_without_reference(tmp_path: Path) -> None:
    launcher = tmp_path / "fake-python"
    launcher.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == '-c': sys.exit(0)\n"
        "assert sys.argv[3] == 'native'\n"
        "carbon = sys.argv[sys.argv.index('--carbon') + 1]\n"
        "output = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "output.write_text(carbon)\n"
        "sys.exit(1 if carbon == '2' else 0)\n"
    )
    launcher.chmod(0o755)
    root = Path(readme_cbs.__file__).resolve().parents[1]
    environment = dict(
        os.environ,
        SLURM_JOB_ID="host-test-no-gpu-command",
        QM7_DATASET="not-loaded",
        GENERATIVEQC_LIBRARY="not-loaded",
        README_CBS_SOURCE_REVISION="host-test",
        README_BENCHMARK_PYTHON=str(launcher),
        README_BENCHMARK_OUTPUT=str(tmp_path / "runs"),
        README_BENCHMARK_SIZES="1 2 3",
        README_BENCHMARK_POINT_TIMEOUT="5",
    )
    result = subprocess.run(
        ["bash", "benchmarks/run_readme_cbs_benchmarks.sh", "native"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 1
    for carbon, expected in ((1, 0), (2, 1), (3, 0)):
        directory = tmp_path / "runs" / str(carbon)
        assert (directory / "native.json").read_text() == str(carbon)
        assert (
            json.loads((directory / "native.outcome").read_text())["exit_code"]
            == expected
        )
        assert not (directory / "reference.json").exists()


def paired_records() -> tuple[dict, dict]:
    """Matched measurements with explicit Fast composite/constituent budgets."""
    protocol, sample, _ = records()
    point = {
        "carbon": 1,
        "native": {
            "protocol": protocol,
            "samples": [sample],
            "requested_samples": 1,
            "policy": "standard",
            "source_revision": "same",
            "driver_sha256": "same",
            "library_sha256": "same",
            "gpu": "same",
        },
        "checks": [qualification(sample, protocol, None)],
    }
    control = {"points": [point]}
    fast = copy.deepcopy(control)
    native = fast["points"][0]["native"]
    native["policy"] = "fast"
    native["samples"][0]["complete_seconds"] = 9.0
    native["samples"][0]["convergence"] = {
        "requested_policy": "fast",
        "budget": {"energy_abs": 1e-5, "force_max_abs": 1e-4, "force_rms": 3e-5},
        "component_budget": {
            "energy_abs": 2e-6,
            "force_max_abs": 2e-5,
            "force_rms": 6e-6,
        },
    }
    return control, fast


def test_fast_plot_uses_measured_values_and_matched_vector_gates(
    tmp_path: Path,
) -> None:
    control, fast = paired_records()
    attach_fast(control, fast)
    assert completed_times(fast["points"][0]) == [9.0]
    assert fast["points"][0]["checks"][0]["accuracy_gate"] is None
    assert fast["points"][0]["standard_comparison"][0]["passed"]
    pytest.importorskip("matplotlib")
    path = tmp_path / "paired.svg"
    render(control, path, readme_style=True)
    svg = path.read_text()
    assert all(value in svg for value in ("Standard", "Fast", "12.0 s", "9.0 s"))


@pytest.mark.parametrize(
    "field", ("protocol", "source_revision", "driver_sha256", "library_sha256", "gpu")
)
def test_fast_comparison_rejects_unmatched_measurements(field: str) -> None:
    control, fast = paired_records()
    fast["points"][0]["native"][field] = "different"
    with pytest.raises(ValueError, match="differs"):
        attach_fast(control, fast)


@pytest.mark.parametrize(
    "field", ("source_revision", "driver_sha256", "library_sha256", "gpu")
)
def test_fast_comparison_rejects_missing_provenance(field: str) -> None:
    """Two missing identities cannot count as a matched measurement pair."""
    control, fast = paired_records()
    for series in (control, fast):
        del series["points"][0]["native"][field]
    with pytest.raises(ValueError, match="provenance is missing"):
        attach_fast(control, fast)


@pytest.mark.parametrize("observable", ("energy", "maximum", "rms"))
def test_fast_constituent_error_cannot_hide_in_composite(observable: str) -> None:
    control, fast = paired_records()
    row = fast["points"][0]["native"]["samples"][0]["components"][0]["result"]
    if observable == "energy":
        row["energy"] += 3e-6
    elif observable == "maximum":
        row["forces"] = [[3e-5, 0.0, 0.0]]
    else:
        row["forces"] = [[1e-5, 1e-5, 1e-5]]
    attach_fast(control, fast)
    assert completed_times(fast["points"][0]) == []
    assert outcome(fast["points"][0]) == "Fast control missing/failed"


def test_fast_missing_control_cannot_supply_plot_time() -> None:
    control, fast = paired_records()
    control["points"][0]["native"]["samples"] = []
    attach_fast(control, fast)
    assert completed_times(fast["points"][0]) == []


def test_fast_independent_oracle_uses_explicit_component_allowances() -> None:
    _, fast = paired_records()
    protocol, _, reference = records()
    sample = fast["points"][0]["native"]["samples"][0]
    sample["energy"] += 5e-6
    assert qualification(sample, protocol, reference)["accuracy_gate"]
    sample["components"][0]["result"]["energy"] += 5e-6
    assert not qualification(sample, protocol, reference)["accuracy_gate"]


@pytest.mark.parametrize("executed", (True, False))
def test_fast_plot_precision_provenance_includes_bounded_fp64_fallback(
    executed: bool,
) -> None:
    control, fast = paired_records()
    sample = fast["points"][0]["native"]["samples"][0]
    sample["convergence"].update(
        arithmetic_request="df_ccsd_fp32_preconvergence",
        arithmetic="mixed" if executed else "fp64",
    )
    cc = next(c["result"] for c in sample["components"] if c["method"] == "df-ccsd(t)")
    cc["cc_performance"] = {
        "ccsd_fp32_requested": True,
        "ccsd_fp32_executed": executed,
        "ccsd_fp32_refused": not executed,
        "ccsd_fp32_evaluations": 2 if executed else 0,
        "ccsd_fp32_gemm_calls": 10 if executed else 0,
        "ccsd_fp32_cast_elements": 100 if executed else 0,
    }
    attach_fast(control, fast)
    assert completed_times(fast["points"][0]) == [9.0]
    check = fast["points"][0]["standard_comparison"][0]["arithmetic"]
    assert check["arithmetic"] == ("mixed" if executed else "fp64")
    sample["convergence"]["arithmetic"] = "fp64" if executed else "mixed"
    with pytest.raises(ValueError, match="disagree"):
        attach_fast(control, fast)
    cc["cc_performance"].pop("ccsd_fp32_evaluations")
    with pytest.raises(ValueError, match="lacks native"):
        attach_fast(control, fast)


def test_frozen_verifier_allows_only_reduction_roundoff() -> None:
    """Portable derived diagnostics must retain exact gates and raw integrity."""
    import runpy

    module = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "benchmarks/results/readme-cbs-fast-20261020/verify.py"
        )
    )
    same = module["same_diagnostics"]
    expected = {"accuracy_gate": True, "errors": [5e-9], "count": 1}
    assert same({**expected, "errors": [5e-9 + 1e-18]}, expected)
    assert not same({**expected, "accuracy_gate": False}, expected)
    assert not same({**expected, "count": 1.0}, expected)
    assert not same({**expected, "errors": [5e-9 + 1e-12]}, expected)
    assert not same({**expected, "errors": [float("nan")]}, expected)
    assert not same({**expected, "extra": None}, expected)
