"""Offline source-bound PR ratchet controls; no GPU, native build or pytest plugins."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.audit_producer_work import ReceiptError
from tools.ratchet_producer_schedule import CASES, DEPENDENCIES, SCHEDULE, audit

SCHEDULE_FIXTURE = '''from dataclasses import dataclass
from .df_occupied_gram_cuda import emit_occupied_gram

@dataclass(frozen=True)
class ProjectedExchangeSchedule:
    rows: int = 0
    blocks: int = 0
    generated_rows: int = 0

def projected_exchange_schedule(n, auxiliaries, rank, capacity, dense_row_blocks,
                                dense_output_blocks, triangular):
    if min(n, auxiliaries, rank, dense_row_blocks, dense_output_blocks) <= 0:
        return ProjectedExchangeSchedule()
    maximum_rows = min(n, capacity // (auxiliaries * rank), capacity // n)
    if maximum_rows <= 0:
        return ProjectedExchangeSchedule()
    blocks = (n + maximum_rows - 1) // maximum_rows
    rows = (n + blocks - 1) // blocks
    generated = n + rows * (blocks - 1) * max(0, blocks - 2) // 2 if triangular else n * blocks
    if generated >= n * dense_row_blocks * dense_output_blocks:
        return ProjectedExchangeSchedule()
    return ProjectedExchangeSchedule(rows, blocks, generated)
'''


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "checkout"
    root.mkdir()
    for path in (*DEPENDENCIES, SCHEDULE):
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            SCHEDULE_FIXTURE if path == SCHEDULE
            else "def emit_occupied_gram():\n    return ''\n" if path.endswith("df_occupied_gram_cuda.py")
            else "",
            encoding="utf-8",
        )
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "cpu-qa@example.invalid")
    _git(root, "config", "user.name", "CPU Fixture")
    _git(root, "add", "python")
    _git(root, "commit", "-qm", "fixture source baseline")
    return root, _git(root, "rev-parse", "HEAD")


def test_equal_source_has_complete_pass_receipts(checkout: tuple[Path, str]) -> None:
    root, base = checkout
    result = audit(root, base)
    assert result["status"] == "PASS"
    assert len(result["cases"]) == len(CASES)
    assert all(row["status"] == "PASS" for row in result["cases"])
    assert any(row["baseline_executed"] > 12 for row in result["cases"])
    assert "not runtime" in result["scope"]


def test_nested_producer_replay_is_flagged_not_declared_a_bug(
    checkout: tuple[Path, str],
) -> None:
    root, base = checkout
    path = root / SCHEDULE
    path.write_text(
        path.read_text().replace(
            "maximum_rows = min(n, capacity // (auxiliaries * rank), capacity // n)",
            "maximum_rows = min(n, 2)",
        ),
        encoding="utf-8",
    )
    result = audit(root, base)
    assert result["status"] == "FAIL"
    repeat = next(row for row in result["cases"] if row["case"] == "triangular-repeat")
    assert repeat["candidate_executed"] > repeat["baseline_executed"]
    assert repeat["candidate_callbacks"] > repeat["baseline_callbacks"]
    assert repeat["classification"] == "work change; reuse unproven"
    assert repeat["runtime_acceptance"] == "INCOMPLETE"


def test_transitive_import_mutation_fails_closed(checkout: tuple[Path, str]) -> None:
    root, base = checkout
    (root / DEPENDENCIES[-1]).write_text("def emit_occupied_gram():\n    return 'changed'\n")
    result = audit(root, base)
    assert result["status"] == "INCOMPLETE"
    assert "import dependency changed" in result["reason"]
    assert result["cases"] == []


def test_production_schedule_rejection_is_not_a_pass(checkout: tuple[Path, str]) -> None:
    root, base = checkout
    path = root / SCHEDULE
    path.write_text(
        path.read_text().replace(
            "maximum_rows = min(n, capacity // (auxiliaries * rank), capacity // n)",
            "maximum_rows = 0",
        )
    )
    result = audit(root, base)
    assert result["status"] == "INCOMPLETE"
    assert all(row["status"] == "INCOMPLETE" for row in result["cases"])


def test_requires_full_base_sha(checkout: tuple[Path, str]) -> None:
    root, _ = checkout
    with pytest.raises(ReceiptError, match="full 40-character"):
        audit(root, "HEAD")
