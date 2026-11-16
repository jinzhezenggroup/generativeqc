"""Exercise endpoint acceptance without compiling or timing a GPU workload."""

import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPARE = ROOT / "benchmarks/results/issue597-direct-jk-policy-20261114/compare.py"


@pytest.fixture
def campaign(tmp_path: Path) -> Path:
    rows = [
        {
            "spin": spin,
            "nbf": nbf,
            "forces": forces,
            "batch": batch,
            "phase": phase,
            "repetition": repetition,
            "endpoint_ms": 1.0,
            "energy": 0.0,
            "force_values": [0.0, 0.0, 0.0] if forces else [],
            "iterations": repetition + 1,
            "fock_builds": repetition + 2,
            "tracked_numeric_peak_bytes": 100,
            "matrix_provider": "native",
            "persistent_eri": 1,
        }
        for spin, nbf, forces, batch, phase, repetition in itertools.product(
            ("RHF", "UHF"),
            (2, 18),
            (0, 1),
            (1, 2),
            ("cold", "warm", "moved-warm"),
            range(3),
        )
    ]
    for variant, process in itertools.product(("baseline", "candidate"), range(2)):
        write_rows(tmp_path / f"{variant}-{process}.jsonl", rows)
    (tmp_path / "oracle.json").write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "unrestricted": spin,
                        "nbf": nbf,
                        "moved": moved,
                        "energy": 0.0,
                        "forces": [[0.0, 0.0, 0.0]],
                    }
                    for spin, nbf, moved in itertools.product(
                        (False, True), (2, 18), (False, True)
                    )
                ]
            }
        )
    )
    return tmp_path


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def run_comparison(campaign: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            "--baseline",
            str(campaign / "baseline-*.jsonl"),
            "--candidate",
            str(campaign / "candidate-*.jsonl"),
            "--reference",
            str(campaign / "oracle.json"),
            "--output",
            str(campaign / "report.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads((campaign / "report.json").read_text())
    assert result.returncode == int(not report["passed"]), result.stderr
    return report


def test_complete_equal_work_campaign_passes(campaign: Path) -> None:
    report = run_comparison(campaign)
    assert report["passed"]
    assert len(report["groups"]) == 48


@pytest.mark.parametrize("field", ["iterations", "fock_builds"])
def test_changed_semantic_work_fails(campaign: Path, field: str) -> None:
    path = campaign / "candidate-0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row[field] += 100
    write_rows(path, rows)
    report = run_comparison(campaign)
    assert not report["passed"]
    assert any("semantic work changed" in error for error in report["errors"])


def test_work_is_matched_per_owner_not_just_as_a_set(campaign: Path) -> None:
    path = campaign / "candidate-0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["iterations"] = 3 - row["repetition"]
        row["fock_builds"] = 4 - row["repetition"]
    write_rows(path, rows)
    assert not run_comparison(campaign)["passed"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-process",
        "missing-owner",
        "duplicate-owner",
        "unexpected-owner",
        "duplicate-process",
    ],
)
def test_incomplete_or_duplicate_populations_fail_even_when_matched(
    campaign: Path, mutation: str
) -> None:
    for variant in ("baseline", "candidate"):
        path = campaign / f"{variant}-0.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "missing-process":
            path.unlink()
            continue
        if mutation == "duplicate-process":
            path.rename(campaign / f"{variant}-00.jsonl")
            write_rows(campaign / f"{variant}-extra-0.jsonl", rows)
            continue
        if mutation == "missing-owner":
            rows = [row for row in rows if row["repetition"] != 2]
        else:
            for row in rows:
                if row["repetition"] == 2:
                    row["repetition"] = 1 if mutation == "duplicate-owner" else 3
        write_rows(path, rows)
    report = run_comparison(campaign)
    assert not report["passed"]
    assert any("population" in error for error in report["errors"])
