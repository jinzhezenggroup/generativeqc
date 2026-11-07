"""Historical replay rejects corruption without importing a GPU runtime."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
READER = ROOT / "benchmarks/results/vv10-history-cold-20261107/replay.py"


@pytest.fixture(scope="module")
def reader() -> Any:
    specification = importlib.util.spec_from_file_location(
        "vv10_history_cold_evidence", READER
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def replace_text(evidence: dict[str, Any], device: str, name: str, text: str) -> None:
    evidence["devices"][device]["files"][name] = {
        "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def damaged(evidence: dict[str, Any], case: str) -> dict[str, Any]:
    evidence = copy.deepcopy(evidence)
    files = evidence["devices"]["RTX5090"]["files"]
    if case == "missing_device":
        del evidence["devices"]["A100"]
    elif case == "promotion":
        evidence["decision"] = "promote"
    elif case == "relaxed_gate":
        evidence["gates"]["paired"][1] = 1e-4
    elif case == "zero_exit":
        replace_text(evidence, "RTX5090", "exit-code.txt", "0\n")
    elif case == "lost_fixture":
        replace_text(evidence, "RTX5090", "linked-fixture.log", "no device proof\n")
    elif case == "lost_sanitizer":
        replace_text(evidence, "RTX5090", "memcheck.log", "not executed\n")
    elif case == "raw_checksum":
        files["forward-candidate.json"]["text"] += "\n"
    elif case == "lost_receipt":
        text = files["forward-candidate.log"]["text"]
        text = "\n".join(
            line for line in text.splitlines() if "REBASE_HISTORY" not in line
        )
        replace_text(evidence, "RTX5090", "forward-candidate.log", text)
    else:
        record = json.loads(files["forward-candidate.json"]["text"])
        sample = record["samples"][0]
        if case == "wrong_image":
            record["native_build"]["library_sha256"] = "0" * 64
        elif case == "lost_cold":
            del record["samples"][0]
        elif case == "nonfinite_time":
            sample["seconds"] = float("inf")
        elif case == "nonfinite_force":
            sample["forces"][0][0] = float("nan")
        elif case == "wrong_force":
            sample["forces"][0][0] += 1
        elif case == "wrong_geometry":
            record["geometry_bohr"][0][1][0] += 1
        elif case == "wrong_selector":
            record["history_transport_experiment"]["request"] = "0"
        elif case == "wrong_tolerance":
            sample["settings"]["density_tolerance"] = 1e-3
        elif case == "incomplete_precision":
            sample["precision"]["operator_inventory_complete"] = False
        elif case == "lost_work":
            sample["ks_diagnostic"]["nonlocal_history_scopes"].pop()
        else:
            raise ValueError(case)
        replace_text(evidence, "RTX5090", "forward-candidate.json", json.dumps(record))
    return evidence


def test_saved_rejection_is_reproducible(reader: Any) -> None:
    expected = json.loads((READER.parent / "summary.json").read_text())
    assert reader.replay() == expected
    assert expected["decision"] == "reject-default"
    assert expected["independent_comparisons"] == 32
    assert all(
        not row["cold_protection_passed"] for row in expected["devices"].values()
    )


@pytest.mark.parametrize("optimized", (False, True))
@pytest.mark.parametrize(
    "case",
    (
        "missing_device",
        "promotion",
        "relaxed_gate",
        "zero_exit",
        "lost_fixture",
        "lost_sanitizer",
        "raw_checksum",
        "lost_receipt",
        "wrong_image",
        "lost_cold",
        "nonfinite_time",
        "nonfinite_force",
        "wrong_force",
        "wrong_geometry",
        "wrong_selector",
        "wrong_tolerance",
        "incomplete_precision",
        "lost_work",
    ),
)
def test_corruption_refuses_without_assertions(
    reader: Any, tmp_path: Path, optimized: bool, case: str
) -> None:
    payload = tmp_path / "corrupt.json"
    payload.write_text(json.dumps(damaged(reader.load(), case)))
    program = (
        "import importlib.util,json,sys;from pathlib import Path;"
        f"spec=importlib.util.spec_from_file_location('replay',{str(READER)!r});"
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);"
        "module.replay(json.loads(Path(sys.argv[1]).read_text()))"
    )
    command = [
        sys.executable,
        *(["-O"] if optimized else []),
        "-c",
        program,
        str(payload),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode != 0
    assert "ValueError" in result.stderr


def test_optimized_cli_preserves_rejected_decision() -> None:
    result = subprocess.run(
        [sys.executable, "-O", str(READER)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout)["decision"] == "reject-default"
