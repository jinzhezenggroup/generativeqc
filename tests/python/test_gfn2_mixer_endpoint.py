"""Behavioral report checks; synthetic rows are never device qualification."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "mixer_report", ROOT / "benchmarks/gfn2_mixer_endpoint.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def experiment() -> tuple[dict, list[dict]]:
    oracle = {
        "oracle": "tblite 0.6.0",
        "cases": [
            {
                "name": name,
                "energy": -1.0,
                "forces": [0.0] * 6,
                "vector_elements": 20,
                "numbers": [1, 1],
                "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            }
            for name in ("h2", "h2-moved")
        ],
    }
    rows = []
    for case in range(2):
        for rep in range(4):
            for order in range(2):
                enabled = bool((order + rep + case) % 2)
                row = {
                    "schema": 1,
                    "case_index": case,
                    "repetition": rep,
                    "order": order,
                    "phase": ("moved_replay" if rep else "moved")
                    if case
                    else ("warm" if rep else "cold"),
                    "enabled": enabled,
                    "maximum_iterations": 300,
                    "numbers": [1, 1],
                    "positions": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "construction_seconds": 0.1 if not case and not rep else 0.0,
                    "endpoint_seconds": 0.1,
                    "readback_seconds": 0.01 if enabled else 0.0,
                    "status": 0,
                    "system_status": 0,
                    "converged": 1,
                    "iterations": 11,
                    "flags": 0,
                    "energy": -1.0,
                    "forces": [0.0] * 6,
                    "snapshot_available": enabled,
                    "receipt_bytes": 112,
                    "call_id": case * 4 + rep + 1 if enabled else 0,
                    "plan_token": 100 if enabled else 0,
                    "device_id": 0 if enabled else -1,
                    "graph_family": 1 if enabled else 0,
                    "graph_submitted": enabled,
                    "endpoint_completed": enabled,
                    "attempted_receipts": 11 if enabled else 0,
                    "receipts": [],
                }
                if enabled:
                    for index in range(11):
                        # Independent hand-written boundary expectations cover
                        # history fill, wrap and the first empty invocation.
                        live = min(index, 8)
                        row["receipts"].append(
                            {
                                "invocation": index,
                                "system": 0,
                                "iteration_before": index,
                                "restart_before": 0,
                                "vector_elements": 20,
                                "live_history": live,
                                "new_slot": (index - 1) % 8 if index else -1,
                                "completed_stages": 31 if index else 17,
                                "status": 0,
                                "coefficient_dot_elements": live * 20,
                                "gram_dot_elements": live * live * 20,
                                "combination_elements": live * 20,
                                **{
                                    f"{stage}_elapsed_cycles": 10
                                    if index or stage == "residual"
                                    else 0
                                    for stage in MODULE.STAGES
                                },
                            }
                        )
                rows.append(row)
    return oracle, rows


def write_report(tmp_path: Path, rows: list[dict], oracle: dict) -> dict:
    journal = tmp_path / "journal.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return MODULE.report(journal, oracle, 3)


def test_complete_report_preserves_timing_domain(tmp_path: Path) -> None:
    oracle, rows = experiment()
    result = write_report(tmp_path, rows, oracle)
    assert result["status"] == "PASS"
    assert result["mixer_scc_fraction"] == "UNAVAILABLE"
    assert result["retained_768_ao_gate"].startswith("historical independent FAIL")
    assert result["endpoint_pairs"][0]["on_seconds"] == pytest.approx(0.21)
    assert (
        result["rows"][1]["work"]["per_invocation_measured_stage_shares"][1]["history"]
        == 0.25
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "dimension",
        "visits",
        "slot",
        "call",
        "plan",
        "order",
        "duplicate",
        "forces",
        "nan_time",
        "invalid_bool",
    ],
)
def test_invalid_evidence_is_rejected(tmp_path: Path, mutation: str) -> None:
    oracle, rows = experiment()
    on = next(row for row in rows if row["enabled"])
    if mutation == "dimension":
        on["receipts"][1]["vector_elements"] += 1
    if mutation == "visits":
        on["receipts"][1]["gram_dot_elements"] -= 1
    if mutation == "slot":
        on["receipts"][9]["new_slot"] = 7
    if mutation == "call":
        on["call_id"] += 1
    if mutation == "plan":
        next(row for row in rows[2:] if row["enabled"])["plan_token"] += 1
    if mutation == "order":
        rows.reverse()
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    if mutation == "forces":
        rows[0]["forces"].pop()
    if mutation == "nan_time":
        rows[0]["endpoint_seconds"] = float("nan")
    if mutation == "invalid_bool":
        rows[0]["enabled"] = 1
    with pytest.raises(ValueError):
        write_report(tmp_path, rows, oracle)


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "truncated",
        "partial",
        "restart",
        "clock",
        "science",
        "parity",
        "missing",
    ],
)
def test_negative_or_incomplete_evidence_remains_visible(
    tmp_path: Path, mutation: str
) -> None:
    oracle, rows = experiment()
    on = next(row for row in rows if row["enabled"])
    if mutation == "empty":
        on["receipts"] = []
        on["attempted_receipts"] = 0
    if mutation == "truncated":
        on["receipts"].pop()
    if mutation == "partial":
        on["receipts"][1]["completed_stages"] = 1
    if mutation == "restart":
        on["receipts"][1]["restart_before"] = 1
    if mutation == "clock":
        for stage in MODULE.STAGES:
            on["receipts"][0][f"{stage}_elapsed_cycles"] = 0
    if mutation == "science":
        for row in rows:
            row["energy"] += 1e-6
    if mutation == "parity":
        on["energy"] += 1e-10
    if mutation == "missing":
        rows.pop()
    result = write_report(tmp_path, rows, oracle)
    assert result["status"] == "INCOMPLETE"
    if mutation == "science":
        assert result["rows"][0]["numerical"]["status"] == "FAIL"


def test_journal_bound(tmp_path: Path) -> None:
    journal = tmp_path / "large.jsonl"
    with journal.open("wb") as stream:
        stream.truncate(MODULE.LIMIT + 1)
    oracle, _ = experiment()
    with pytest.raises(ValueError, match="capacity"):
        MODULE.report(journal, oracle, 3)


@pytest.mark.parametrize(
    "field",
    [
        "device_id",
        "graph_family",
        "plan_token",
        "schema",
        "converged",
        "flags",
        "order",
    ],
)
def test_boolean_identity_is_not_an_integer(tmp_path: Path, field: str) -> None:
    oracle, rows = experiment()
    on = next(row for row in rows if row["enabled"])
    on[field] = bool(on[field])
    with pytest.raises(ValueError):
        write_report(tmp_path, rows, oracle)


@pytest.mark.parametrize(
    "field", ["system", "vector_elements", "live_history", "new_slot", "status"]
)
def test_float_receipt_identity_is_rejected(tmp_path: Path, field: str) -> None:
    oracle, rows = experiment()
    on = next(row for row in rows if row["enabled"])
    on["receipts"][1][field] = float(on["receipts"][1][field])
    with pytest.raises(ValueError):
        write_report(tmp_path, rows, oracle)


def lifecycle() -> tuple[dict, list[dict]]:
    oracle, samples = experiment()
    healthy = next(row for row in samples if row["enabled"])
    rows = []
    for index, phase in enumerate(
        (
            "first_failed",
            "recovered",
            "replacement_failed",
            "replay",
            "rejected",
            "after_rejection",
        )
    ):
        row = copy.deepcopy(healthy)
        row.update(
            phase=phase,
            call_id=index + 1,
            case_index=0,
            plan_token=100,
            maximum_iterations=301 if index == 2 else 0 if index == 4 else 300,
        )
        if index in (0, 2, 4):
            row.update(
                status=1,
                system_status=-43,
                converged=165,
                iterations=-42,
                energy=123.0,
                forces=[-321.0] * 6,
                flags=0,
                receipts=[],
                attempted_receipts=0,
                endpoint_completed=False,
                graph_family=0,
                graph_submitted=False,
            )
        if index in (0, 2):
            row["positions"][0] = None
            row["plan_token"] = 77 if index == 0 else 777
        if index == 4:
            row.update(snapshot_available=False, call_id=0, plan_token=0, device_id=-1)
        rows.append(row)
    return oracle, rows


def test_lifecycle_readout_preserves_failure_and_recovery(tmp_path: Path) -> None:
    oracle, rows = lifecycle()
    path = tmp_path / "lifecycle.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = MODULE.report_lifecycle(path, oracle)
    assert result["status"] == "PASS"
    assert result["rows"][0]["failure_observed"]
    assert result["rows"][1]["numerical"]["status"] == "PASS"


@pytest.mark.parametrize(
    "mutation", ["sentinel", "stale", "replacement", "rollback", "geometry", "call"]
)
def test_lifecycle_corruption_is_rejected(tmp_path: Path, mutation: str) -> None:
    oracle, rows = lifecycle()
    if mutation == "sentinel":
        rows[0]["energy"] = -1.0
    if mutation == "stale":
        rows[4]["snapshot_available"] = True
    if mutation == "replacement":
        rows[2]["plan_token"] = rows[1]["plan_token"]
    if mutation == "rollback":
        rows[3]["plan_token"] += 1
    if mutation == "geometry":
        rows[1]["positions"][0] += 0.1
    if mutation == "call":
        rows[5]["call_id"] -= 1
    path = tmp_path / "lifecycle.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        MODULE.report_lifecycle(path, oracle)


def test_lifecycle_partial_execution_stays_incomplete(tmp_path: Path) -> None:
    oracle, rows = lifecycle()
    path = tmp_path / "lifecycle.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    assert MODULE.report_lifecycle(path, oracle)["status"] == "INCOMPLETE"


@pytest.mark.parametrize(
    "mutation",
    ["call_float", "status_bool", "snapshot_int", "plan_float", "schema_bool"],
)
def test_lifecycle_metadata_types_fail_closed(tmp_path: Path, mutation: str) -> None:
    oracle, rows = lifecycle()
    if mutation == "call_float":
        for row in rows:
            row["call_id"] = float(row["call_id"])
    if mutation == "status_bool":
        for index in (1, 3, 5):
            rows[index].update(status=False, system_status=False, converged=True)
    if mutation == "snapshot_int":
        for row in rows:
            row["snapshot_available"] = int(row["snapshot_available"])
    if mutation == "plan_float":
        rows[0]["plan_token"] = float(rows[0]["plan_token"])
    if mutation == "schema_bool":
        rows[0]["schema"] = True
    path = tmp_path / "lifecycle.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        MODULE.report_lifecycle(path, oracle)


def test_consistently_missing_tail_is_incomplete(tmp_path: Path) -> None:
    oracle, rows = experiment()
    on = next(row for row in rows if row["enabled"])
    on["receipts"].pop()
    on["attempted_receipts"] -= 1
    result = write_report(tmp_path, rows, oracle)
    assert result["status"] == "INCOMPLETE"


@pytest.mark.parametrize("drift", ["none", "energy", "force", "nonfinite"])
def test_pinned_reference_gate_survives_optimized_python(drift: str) -> None:
    record = {"energy": -1.0, "forces": [0.0, 0.0, 0.0]}
    if drift == "energy":
        record["energy"] += 1e-6
    if drift == "force":
        record["forces"][0] = 1e-6
    if drift == "nonfinite":
        record["energy"] = None
    script = (
        "import importlib.util,json; "
        f"s=importlib.util.spec_from_file_location('probe',{str(ROOT / 'benchmarks/gfn2_mixer_endpoint.py')!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"m.check_pinned_reference(json.loads({json.dumps(record)!r}),"
        "{'energy':-1.0,'forces':[[0.0,0.0,0.0]]})"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == (0 if drift == "none" else 1)
    if drift != "none":
        assert "original pinned energy/force gate" in result.stderr
