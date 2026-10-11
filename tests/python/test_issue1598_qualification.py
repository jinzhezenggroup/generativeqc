"""Host regressions for the historical #1598 qualification runner/reducer."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import reduce_issue1598_incremental_qualification as reducer

ROOT = Path(__file__).resolve().parents[2]
CAPSULE = ROOT / "benchmarks/results/issue1598-incremental-direct-jk-20261011"


def _record(round: int) -> dict:
    record = {
        "benchmark": "compare_gpu4pyscf_batch",
        "workload": {"case": "water-12", "ao_count": 96, "batch_size": 1},
        "gate": {"passed": True},
        "native_build": {
            "library_sha256": "a" * 64,
            "probe": {"source_identity": "b" * 64},
        },
        "environment": {
            "git": {"commit": "c" * 40, "dirty": False},
            "host": {"node": "host"},
            "accelerator": {
                "name": "gpu",
                "device_id": 0,
                "driver_version": 1,
                "runtime_version": 1,
            },
        },
        "settings": {
            "gates": {
                "maximum_energy_error_hartree": 1e-8,
                "maximum_force_error_hartree_per_bohr": 1e-7,
            }
        },
        "generativeqc": {
            "cold_seconds": 1.0,
            "warm_median_seconds": 0.1,
            "energies_hartree": [float(round)],
            "forces_hartree_per_bohr": [[[0.0, 0.0, 0.0]]],
            "warm_samples": [
                {
                    "convergence": [
                        {
                            "iterations": 1,
                            "incremental_direct_jk": {
                                "active": True,
                                "anchor_full_builds": 1,
                                "delta_builds": 0,
                                "full_admitted_shell_quartets": 0,
                                "delta_admitted_shell_quartets": 0,
                                "full_admitted_quartet_tiles": 0,
                                "delta_admitted_quartet_tiles": 0,
                                "quartet_work_counters_valid": False,
                            },
                        }
                    ]
                }
            ],
        },
    }

    sample = record["generativeqc"]["warm_samples"][0]
    sample["convergence"][0]["converged"] = True
    sample["seconds"] = 0.1
    record["generativeqc"]["warm_seconds"] = [0.1] * 3
    for key in ("energies_hartree", "forces_hartree_per_bohr"):
        sample[key] = copy.deepcopy(record["generativeqc"][key])
    record["generativeqc"]["warm_samples"] = [copy.deepcopy(sample) for _ in range(3)]
    record["gpu4pyscf"] = copy.deepcopy(record["generativeqc"])
    return record


def _point(root: Path, mode: str, round: int, record: dict | None) -> None:
    path = root / "rhf" / f"direct-12-{mode}-{round}.json"
    path.parent.mkdir(exist_ok=True)
    path.with_suffix(".outcome").write_text(
        json.dumps(
            {
                "exit_code": 0 if record is not None else 42,
                "attempt": 1 if record is not None else 2,
            }
        )
    )
    if record is not None:
        path.write_text(json.dumps(record))


@pytest.fixture
def matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(reducer, "POPULATIONS", ("rhf",))
    monkeypatch.setattr(reducer, "SIZES", ((12, 96),))
    for mode in ("baseline", "incremental"):
        for round in (1, 2):
            _point(tmp_path, mode, round, _record(round))
    return tmp_path


def _rows(root: Path) -> tuple:
    return reducer._hf_rows(
        argparse.Namespace(
            maximum_energy_difference=1e-8,
            maximum_force_difference=1e-7,
        ),
        root,
    )


def test_runner_outcomes_and_all_rounds_are_reduced(matrix: Path) -> None:
    rows, sha, failures = _rows(matrix)
    assert sha == "a" * 64
    assert not failures
    assert rows[0]["matched_rounds"] == [1, 2]
    assert rows[0]["incremental_work"]["sample_count"] == 6


def test_partial_success_pairs_by_round_without_shifting(matrix: Path) -> None:
    _point(matrix, "incremental", 1, None)
    rows, _, failures = _rows(matrix)
    assert rows[0]["matched_rounds"] == [2]
    assert rows[0]["maximum_energy_difference_hartree"] == 0
    assert len(failures) == 1


def test_disjoint_successes_are_not_paired(matrix: Path) -> None:
    _point(matrix, "incremental", 1, None)
    _point(matrix, "baseline", 2, None)
    rows, _, failures = _rows(matrix)
    assert not rows
    assert len(failures[0]["failed_points"]) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True])
def test_unpaired_success_still_validates_timing(matrix: Path, value: object) -> None:
    _point(matrix, "incremental", 1, None)
    _point(matrix, "incremental", 2, None)
    record = _record(1)
    record["generativeqc"]["cold_seconds"] = value
    _point(matrix, "baseline", 1, record)
    with pytest.raises(ValueError):
        _rows(matrix)


@pytest.mark.parametrize("field,value", [("passed", "true"), ("passed", False)])
def test_unpaired_success_cannot_skip_reference_gate(
    matrix: Path, field: str, value: object
) -> None:
    _point(matrix, "incremental", 1, None)
    _point(matrix, "incremental", 2, None)
    record = _record(1)
    record["gate"][field] = value
    _point(matrix, "baseline", 1, record)
    with pytest.raises(ValueError, match="independent reference"):
        _rows(matrix)


def test_round_two_numerical_failure_is_retained(matrix: Path) -> None:
    record = _record(2)
    for engine in ("generativeqc", "gpu4pyscf"):
        for sample in (record[engine], *record[engine]["warm_samples"]):
            sample["energies_hartree"][0] += 1
    _point(matrix, "incremental", 2, record)
    rows, _, _ = _rows(matrix)
    assert not rows[0]["numerical_gate_passed"]


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_invalid_work_counts_fail(value: object) -> None:
    record = _record(1)
    record["generativeqc"]["warm_samples"][0]["convergence"][0][
        "incremental_direct_jk"
    ]["delta_builds"] = value
    with pytest.raises(ValueError, match="nonnegative integers"):
        reducer._incremental_work(record)


def test_all_failed_matrix_writes_negative_summary(
    matrix: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for mode in ("baseline", "incremental"):
        for round in (1, 2):
            _point(matrix, mode, round, None)
    monkeypatch.setattr(reducer, "KS_SAMPLES", ())
    output = matrix / "summary.json"
    monkeypatch.setattr(
        sys, "argv", ["reduce", "--input", str(matrix), "--output", str(output)]
    )
    reducer.main()
    result = json.loads(output.read_text())
    assert result["any_point_failed"]
    assert result["all_numerical_gates_passed"] is False
    assert result["best_warm_speedup"] is None


@pytest.mark.parametrize(
    "function,call,prefix",
    [
        (
            "run_point",
            "run_point rhf incremental water-12 12 1",
            "rhf/direct-12-incremental-1",
        ),
        ("ks_point", "ks_point on 48 1", "ks/pbe0-48-on-1"),
    ],
)
def test_failed_runner_keeps_real_exit_code(
    tmp_path: Path, function: str, call: str, prefix: str
) -> None:
    source = (
        ROOT / "benchmarks/run_issue1598_incremental_qualification.sh"
    ).read_text()
    start = source.index(f"{function}() {{")
    end = source.index("\n}\n", start) + 3
    (tmp_path / "ks").mkdir()
    script = f"""set -eu
output={str(tmp_path)!r}
point_timeout=1
rebuild_interval=8
python=unused
ledger() {{ :; }}
sleep() {{ :; }}
timeout() {{ return 42; }}
{source[start:end]}
{call}
"""
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    assert "exit=42" in result.stderr
    for attempt in (1, 2):
        assert json.loads(
            (tmp_path / f"{prefix}-attempt{attempt}.outcome").read_text()
        ) == {
            "exit_code": 42,
            "attempt": attempt,
        }
    assert json.loads((tmp_path / f"{prefix}.outcome").read_text()) == {
        "exit_code": 42,
        "attempt": 2,
        "retried": True,
    }


def test_ks_oracle_selection_matches_both_supported_sizes() -> None:
    tree = ast.parse((ROOT / "benchmarks/issue1598_ks_cold_control.py").read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assignment = next(
        node
        for node in main.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "reference_path"
            for target in node.targets
        )
    )
    expression = ast.Expression(assignment.value)
    for atoms in (48, 96):
        path = eval(  # noqa: S307 - evaluate one reviewed local Path expression
            compile(expression, "reference selector", "eval"),
            {"Path": Path, "atoms": atoms},
        )
        assert path.name == f"reference-{atoms}.json.gz"
        assert (ROOT / path).is_file()


@pytest.mark.parametrize("mutation", ["source", "device", "oracle", "nan", "gate"])
def test_measured_identity_and_reference_values_fail_closed(
    matrix: Path, mutation: str
) -> None:
    record = _record(2)
    if mutation == "source":
        record["environment"]["git"]["commit"] = "d" * 40
    elif mutation == "device":
        record["environment"]["accelerator"]["name"] = "another GPU"
    elif mutation == "gate":
        record["settings"]["gates"]["maximum_energy_error_hartree"] = 1.0
    else:
        record["gpu4pyscf"]["warm_samples"][2]["forces_hartree_per_bohr"][0][0][1] = (
            float("nan") if mutation == "nan" else 1.0
        )
    _point(matrix, "incremental", 2, record)
    with pytest.raises(ValueError):
        _rows(matrix)


@pytest.mark.parametrize("accepted", [True, False, "true", None])
def test_ks_reducer_consumes_check_endpoint_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accepted: object
) -> None:
    monkeypatch.setattr(reducer, "KS_SAMPLES", (("off", 48, 1),))
    root = tmp_path / "ks"
    root.mkdir()
    (root / "pbe0-48-off-1.outcome").write_text('{"exit_code":0,"attempt":1}')
    (root / "pbe0-48-off-1.json").write_text(
        json.dumps(
            {
                "schema": "issue1598.ks-cold-control.v1",
                "status": "measured",
                "complete_seconds": 1.0,
                "iterations": 2,
                "fock_builds": 2,
                "incremental_direct_jk": {"active": False},
                "acceptance": {
                    "energy_error": 0.0,
                    "force_error": 0.0,
                    "gate": accepted,
                },
            }
        )
    )
    if accepted is True:
        rows, failures = reducer._ks_rows(tmp_path)
        assert len(rows) == 2 and not failures
    else:
        with pytest.raises(ValueError, match="independent acceptance failed"):
            reducer._ks_rows(tmp_path)


def test_ks_runner_consumes_real_gate_key() -> None:
    source = (ROOT / "benchmarks/issue1598_ks_cold_control.py").read_text()
    tree = ast.parse(source)
    conditions = [node.test for node in ast.walk(tree) if isinstance(node, ast.If)]
    gate_condition = next(
        node for node in conditions if "acceptance.get" in ast.unparse(node)
    )
    for value in (True, False, "true", None):
        rejected = eval(  # noqa: S307 - reviewed local comparison expression only
            compile(ast.Expression(gate_condition), "KS acceptance", "eval"),
            {"acceptance": {"gate": value}},
        )
        assert rejected is (value is not True)


@pytest.mark.parametrize("mutation", ["median", "seconds", "missing_success"])
def test_timing_summary_and_success_inventory_are_strict(
    matrix: Path, mutation: str
) -> None:
    record = _record(2)
    if mutation == "median":
        record["generativeqc"]["warm_median_seconds"] = 0.01
    elif mutation == "seconds":
        record["generativeqc"]["warm_samples"][0]["seconds"] = -1
    _point(matrix, "incremental", 2, record)
    if mutation == "missing_success":
        (matrix / "rhf/direct-12-incremental-2.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        _rows(matrix)


def test_paired_gate_checks_nonfinal_samples(matrix: Path) -> None:
    record = _record(2)
    for engine in ("generativeqc", "gpu4pyscf"):
        record[engine]["warm_samples"][0]["energies_hartree"][0] += 1
    _point(matrix, "incremental", 2, record)
    rows, _, _ = _rows(matrix)
    assert rows[0]["maximum_energy_difference_hartree"] == 1.0
    assert not rows[0]["numerical_gate_passed"]
