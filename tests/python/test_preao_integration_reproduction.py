"""Offline integration reproduction works before any local campaign has run."""

import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = ROOT / "benchmarks/results/preao-force-integration-20261006"


@pytest.mark.parametrize("existing_parent", [False, True])
def test_reproduction_creates_output_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_parent: bool
) -> None:
    output = tmp_path / ".artifacts/preao/reproduced-integration.json"
    if existing_parent:
        output.parent.mkdir(parents=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PUBLICATION / "reproduce.py"),
            str(PUBLICATION / "bundle.json.xz"),
            "--output",
            str(output),
        ],
    )
    runpy.run_path(str(PUBLICATION / "reproduce.py"), run_name="__main__")
    result = json.loads(output.read_text())
    assert result["warm_qualification"]["default_promotion_eligible"]
    assert set(result["smoke"]) == {
        "3/current-default",
        "3/auto",
        "48/current-default",
        "48/auto",
    }
