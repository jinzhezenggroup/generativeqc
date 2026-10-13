"""Scientific identities and complete timing boundaries for README comparisons."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import GridSpec, load_r2scan3c_basis

from benchmarks.readme_dft import SCHEMA, specification
from benchmarks.readme_omol25 import (
    protocol,
    reference_baselines,
    reference_energy_forces,
    retain_endpoint_record,
)
from tools.render_omol25_benchmarks import independent_phase_status
from tools.render_readme_dft import collect_point, phase_statistics, timing_cell


def test_hf_and_df_protocols_have_distinct_scientific_identities() -> None:
    spec = specification("hf")
    assert spec.native_method == "rhf"
    basis = load_r2scan3c_basis()
    direct = protocol(3, basis, GridSpec(), 5, benchmark=spec)
    auxiliary = SimpleNamespace(name="explicit-jkfit", identity="auxiliary-identity")
    fitted = protocol(
        3,
        basis,
        GridSpec(),
        5,
        benchmark=replace(spec, density_fitting=True),
        auxiliary_basis=auxiliary,
    )
    assert direct["method"] == "RHF"
    assert direct["grid"] is None
    assert not direct["density_fitting"]
    assert fitted["density_fitting"]
    assert fitted["auxiliary_basis_identity"] == auxiliary.identity
    assert fitted["reference_auxbasis_response"]
    with pytest.raises(ValueError, match="auxiliary basis identity"):
        protocol(3, basis, GridSpec(), 5, benchmark=replace(spec, density_fitting=True))


def test_varying_work_keeps_endpoint_median_but_failed_points_remain_gaps() -> None:
    entry = {
        "status": "measured",
        "records": [
            {
                "phase": "warm",
                "iterations": iterations,
                "scf_jk_builds": iterations + 1,
                "complete_seconds": seconds,
            }
            for iterations, seconds in [
                (1, 1.0),
                (2, 3.0),
                (1, 2.0),
                (3, 9.0),
                (1, 1.5),
            ]
        ],
    }
    assert phase_statistics(entry, "warm") == (2.0, 1.0, 7.0)
    assert timing_cell(entry, "warm") == "2.000 [1.000–9.000]†"
    assert np.isnan(phase_statistics(entry, "cold")[0])
    entry["status"] = "timeout"
    assert np.isnan(phase_statistics(entry, "warm")[0])


@pytest.mark.parametrize(
    "reason", ["operator_cancelled", "stopped_after_12_atom_energy_gate_failure"]
)
def test_operator_stop_retains_completed_cold_but_not_incomplete_warm(
    tmp_path: Path, reason: str
) -> None:
    directory = tmp_path / "96"
    directory.mkdir()
    row = dict(
        completed_sample(),
        complete_seconds=1220.0,
        forces=[[0.0, 0.0, 0.0]] * 96,
        gate=None,
        execution_gate=True,
    )
    (directory / "native.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "running",
                "stage": "warm/0",
                "protocol": {
                    "atoms": 96,
                    "repeats": 5,
                    "force_return": "host_array_in_timed_endpoint",
                },
                "records": [row],
            }
        )
    )
    outcome = {"exit_code": 143, "reason": reason}
    (directory / "native.outcome").write_text(json.dumps(outcome))
    entry = collect_point(tmp_path, 96)["engines"]["native"]
    assert entry["status"] == "interrupted"
    assert entry["outcome"] == outcome
    assert entry["records"] == [
        {key: value for key, value in row.items() if key != "forces"}
    ]
    assert phase_statistics(entry, "cold")[0] == 1220.0
    assert timing_cell(entry, "cold") == "1220.000‡"
    assert np.isnan(phase_statistics(entry, "warm")[0])


def completed_sample(phase: str = "cold", geometry: int = 0) -> dict:
    """A converged host E+F result, independent of any native implementation."""
    return {
        "phase": phase,
        "geometry": geometry,
        "repeat": 0,
        "energy": -76.0,
        "forces": [[0.0, 0.0, 0.0]] * 3,
        "converged": True,
        "status": 0,
        "iterations": 1,
        "complete_seconds": 1.0,
        "gate": True,
    }


def test_partial_reference_checks_only_its_completed_geometry() -> None:
    scientific = {"atoms": 3}
    oracle = {
        "protocol": scientific,
        "status": "running",
        "records": [completed_sample()],
    }
    baselines = reference_baselines(oracle, scientific, allow_missing=True)
    assert set(baselines) == {0}
    record = {"protocol": scientific, "stage": "cold/0", "records": []}
    checked = completed_sample()
    retain_endpoint_record(record, checked, baselines[0], lambda _: None)
    assert checked["gate"] is True
    unchecked = completed_sample("moved", 1)
    retain_endpoint_record(record, unchecked, baselines.get(1), lambda _: None)
    assert unchecked["execution_gate"] is True
    assert unchecked["gate"] is None
    assert unchecked["energy_error"] is None
    assert unchecked["force_error"] is None
    with pytest.raises(ValueError, match="incomplete"):
        reference_baselines(oracle, scientific)
    with pytest.raises(ValueError, match="protocol differs"):
        reference_baselines(oracle, {"atoms": 12}, allow_missing=True)
    assert reference_baselines(None, scientific, allow_missing=True) == {}


@pytest.mark.parametrize(
    "failure", ["forces", "convergence", "status", "timing", "energy"]
)
def test_missing_reference_never_accepts_an_incomplete_endpoint(failure: str) -> None:
    row = completed_sample()
    if failure == "forces":
        row["forces"] = [[0.0, 0.0, 0.0]]
    elif failure == "convergence":
        row["converged"] = False
    elif failure == "status":
        row["status"] = 5
    elif failure == "timing":
        row["complete_seconds"] = float("inf")
    else:
        row["energy"] = float("nan")
    record = {"protocol": {"atoms": 3}, "stage": "cold/0", "records": []}
    with pytest.raises(RuntimeError, match="gate failed"):
        retain_endpoint_record(record, row, None, lambda _: None)
    assert record["records"][0]["gate"] is False


def test_phase_reduction_checks_partial_oracles_and_every_requested_repeat() -> None:
    scientific = {
        "atoms": 3,
        "repeats": 2,
        "force_return": "host_array_in_timed_endpoint",
    }
    reference = {
        "schema": SCHEMA,
        "protocol": scientific,
        "status": "running",
        "records": [completed_sample()],
    }
    rows = [
        dict(completed_sample(phase, geometry), repeat=repeat)
        for phase, geometry, count in (
            ("cold", 0, 1),
            ("warm", 0, 2),
            ("moved", 1, 1),
            ("moved-warm", 1, 2),
        )
        for repeat in range(count)
    ]
    for row in rows:
        if row["geometry"] == 1:
            row.update(gate=None, execution_gate=True)
    native = {"schema": SCHEMA, "protocol": scientific, "records": rows}
    expected = {
        "cold": "measured",
        "warm": "measured",
        "moved": "completed_unchecked",
        "moved-warm": "completed_unchecked",
    }
    assert (
        independent_phase_status(native, reference, atoms=3, schema=SCHEMA) == expected
    )
    rows.pop()
    expected["moved-warm"] = "incomplete"
    assert (
        independent_phase_status(native, reference, atoms=3, schema=SCHEMA) == expected
    )
    rows[1]["energy"] += 0.01
    expected["warm"] = "failed"
    assert (
        independent_phase_status(native, reference, atoms=3, schema=SCHEMA) == expected
    )
    rows.append(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        independent_phase_status(native, reference, atoms=3, schema=SCHEMA)


@pytest.mark.parametrize("partial_file", [False, True])
def test_reference_timeout_never_skips_native_runner(
    tmp_path: Path, partial_file: bool
) -> None:
    """Exercise the real shell orchestration with a CPU-only process fixture."""
    helper = tmp_path / "endpoint-probe"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "if sys.argv[1] == '-':\n"
        "    pathlib.Path(sys.argv[2]).write_text('{}'); sys.exit(0)\n"
        "engine = sys.argv[sys.argv.index('--method') + 2]\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "if engine == 'reference':\n"
        + (
            "    output.write_text(json.dumps({'status': 'running', 'records': []}))\n"
            if partial_file
            else ""
        )
        + "    sys.exit(124)\n"
        "assert '--allow-missing-reference' in sys.argv\n"
        "output.write_text(json.dumps({'native_ran': True}))\n"
    )
    helper.chmod(0o700)
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", str(root / "benchmarks/run_readme_dft_benchmarks.sh"), "both"],
        cwd=root,
        env={
            **os.environ,
            "SLURM_JOB_ID": "host-only-control-flow",
            "GENERATIVEQC_LIBRARY": "not-loaded",
            "README_BENCHMARK_PYTHON": str(helper),
            "README_BENCHMARK_OUTPUT": str(output),
            "README_BENCHMARK_METHODS": "wb97m-v",
            "README_BENCHMARK_SIZES": "48",
            "README_BENCHMARK_POINT_TIMEOUT": "1",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    point = output / "wb97m-v/48"
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads((point / "reference.outcome").read_text())["exit_code"] == 124
    assert json.loads((point / "native.outcome").read_text())["exit_code"] == 0
    assert json.loads((point / "native.json").read_text())["native_ran"]


def test_composite_protocol_retains_basis_and_corrections() -> None:
    spec = specification("r2scan-3c")
    result = protocol(3, load_r2scan3c_basis(), GridSpec(), 5, benchmark=spec)
    assert spec.native_method == "r2scan-3c-rks"
    assert spec.reference_xc == "R2SCAN"
    assert result["basis"] == "def2-mTZVPP"
    assert result["aos"] == 34
    assert result["d4_parameters"]["s9"] == 2.0
    assert "included in every endpoint" in result["reference_corrections"]
    assert not specification("r2scan").is_r2scan3c


@pytest.mark.parametrize("name,xc", [("b3lyp", "B3LYP"), ("m06-2x", "M06_2X")])
def test_hybrid_identity(name: str, xc: str) -> None:
    spec = specification(name)
    assert spec.reference_xc == xc
    assert spec.native_method == name + "-rks"
    assert spec.basis == "def2-SVP"


def test_composite_correction_and_host_export_are_inside_endpoint_timer() -> None:
    events = []
    gradient = SimpleNamespace(kernel=lambda: np.ones((1, 3)))
    engine = SimpleNamespace(
        kernel=lambda **kwargs: -10.0,
        converged=True,
        nuc_grad_method=lambda: gradient,
    )

    def clock() -> float:
        events.append("clock")
        return float(len(events))

    def export(value: np.ndarray) -> np.ndarray:
        events.append("export")
        return value

    def correction() -> tuple[float, np.ndarray]:
        events.append("correction")
        return -0.5, np.full((1, 3), 0.25)

    energy, forces, seconds = reference_energy_forces(
        engine,
        None,
        export=export,
        synchronize=lambda: None,
        clock=clock,
        correction=correction,
    )
    assert energy == -10.5
    np.testing.assert_array_equal(forces, np.full((1, 3), -1.25))
    assert events == ["clock", "export", "correction", "clock"]
    assert seconds == 3.0
