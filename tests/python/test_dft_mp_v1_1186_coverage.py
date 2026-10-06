"""The #1186 inventory only credits rows accepted by the shared validator."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tools.dft_mp_v1 import issue1186_coverage as coverage
from tools.dft_mp_v1.freeze_contract import REPO
from tools.dft_mp_v1.validate import InvalidEvidence

if TYPE_CHECKING:
    from pathlib import Path


def test_frozen_semilocal_fp64_inventory_starts_missing() -> None:
    result = coverage.build_coverage()
    assert result["contract_sha256"] == (
        "de6c847b1ed93e537c1422679ae3df53a4cca3cd13e7268483e419afbf2faa00"
    )
    assert result["required_fp64_rows"] == 44
    assert result["validated_pass_rows"] == 0
    assert {row["method"] for row in result["rows"]} == {"lda", "pbe", "r2scan"}
    assert {row["status"] for row in result["rows"]} == {"missing"}
    assert all(row["input_sha256"] and row["grid_identity"] for row in result["rows"])


def test_retained_missing_map_reproduces_exactly() -> None:
    report = (
        REPO
        / "benchmarks/results/dft-mp-1186-frozen-row-coverage-20261006/coverage.json"
    )
    assert json.loads(report.read_text(encoding="utf-8")) == coverage.build_coverage()


def test_rejected_pass_remains_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    contract = coverage.manifest()
    entries = [
        {"id": row["id"], "status": "not-run", "reason": "pending"}
        for row in contract["rows"]
    ]
    target = "pbe/rks/water/fp64_energy_forces"
    next(entry for entry in entries if entry["id"] == target)["status"] = "pass"
    monkeypatch.setattr(
        coverage,
        "audit",
        lambda _path: {
            "passed_rows": 0,
            "source_commit": "a" * 40,
            "failures": [f"{target}: invalid pass: wrong grid identity"],
        },
    )
    receipt_path.write_text(
        json.dumps({"rows": entries, "campaign": {"hardware": {"sm": 120}}}),
        encoding="utf-8",
    )

    result = coverage.build_coverage(receipt_path)
    rejected = next(row for row in result["rows"] if row["id"] == target)
    assert rejected["status"] == "invalid"
    assert result["validated_pass_rows"] == 0
    assert result["missing_or_blocked_rows"] == 44


def test_accepted_row_is_counted_without_crediting_other_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    contract = coverage.manifest()
    entries = [
        {"id": row["id"], "status": "not-run", "reason": "pending"}
        for row in contract["rows"]
    ]
    target = "r2scan/uks/oh/fp64_energy_forces"
    next(entry for entry in entries if entry["id"] == target)["status"] = "pass"
    monkeypatch.setattr(
        coverage,
        "audit",
        lambda _path: {"passed_rows": 1, "source_commit": "a" * 40, "failures": []},
    )
    receipt_path.write_text(
        json.dumps({"rows": entries, "campaign": {"hardware": {"sm": 120}}}),
        encoding="utf-8",
    )

    result = coverage.build_coverage(receipt_path)
    accepted = next(row for row in result["rows"] if row["id"] == target)
    assert accepted["status"] == "pass"
    assert result["validated_pass_rows"] == 1
    assert result["missing_or_blocked_rows"] == 43


def test_inconsistent_validator_pass_inventory_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        coverage,
        "audit",
        lambda _path: {"passed_rows": 1, "source_commit": "a" * 40, "failures": []},
    )
    receipt_path.write_text(
        json.dumps({"rows": [], "campaign": {"hardware": {"sm": 120}}}),
        encoding="utf-8",
    )
    with pytest.raises(InvalidEvidence, match="could not be reconciled"):
        coverage.build_coverage(receipt_path)


def test_legacy_receipt_is_rejected_by_real_validator(tmp_path: Path) -> None:
    receipt_path = tmp_path / "legacy.json"
    receipt_path.write_text(
        json.dumps({"schema_version": 1, "contract_sha256": "a" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(InvalidEvidence, match="receipt contract mismatch"):
        coverage.build_coverage(receipt_path)


def test_live_receipt_change_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")

    def update_during_audit(path: Path) -> dict:
        path.write_text('{"rows": []}', encoding="utf-8")
        return {}

    monkeypatch.setattr(coverage, "audit", update_during_audit)
    with pytest.raises(InvalidEvidence, match="receipt changed during validation"):
        coverage.build_coverage(receipt_path)


@pytest.mark.parametrize("alias", ["same", "relative", "symlink", "hardlink"])
def test_cli_cannot_overwrite_receipt_alias(tmp_path: Path, alias: str) -> None:
    receipt = tmp_path / "receipt.json"
    original = b'{"retained_raw_receipt": true}\n'
    receipt.write_bytes(original)
    output = receipt
    if alias == "relative":
        nested = tmp_path / "nested"
        nested.mkdir()
        output = nested / ".." / receipt.name
    elif alias == "symlink":
        output = tmp_path / "symlink.json"
        output.symlink_to(receipt)
    elif alias == "hardlink":
        output = tmp_path / "hardlink.json"
        output.hardlink_to(receipt)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.dft_mp_v1.issue1186_coverage",
            "--receipt",
            str(receipt),
            "--output",
            str(output),
        ],
        cwd=REPO,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 2
    assert b"must not overwrite the input --receipt" in result.stderr
    assert receipt.read_bytes() == original


def test_cli_writes_separate_coverage_output(tmp_path: Path) -> None:
    output = tmp_path / "coverage.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.dft_mp_v1.issue1186_coverage",
            "--output",
            str(output),
        ],
        cwd=REPO,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(output.read_text(encoding="utf-8")) == coverage.build_coverage()
