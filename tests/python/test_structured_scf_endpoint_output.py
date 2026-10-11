"""Endpoint runners reject retained destinations before native or oracle work."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import _retention
from benchmarks import structured_scf_endpoint_qualification as qualification

ROOT = Path(__file__).resolve().parents[2]


def command(runner: str, module: bool) -> list[str]:
    name = f"structured_scf_endpoint_{runner}"
    return (
        [sys.executable, "-m", f"benchmarks.{name}"]
        if module
        else [
            sys.executable,
            str(ROOT / "benchmarks" / f"{name}.py"),
        ]
    )


@pytest.mark.parametrize("runner", ["qualification", "report"])
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("alias", ["direct", "symlink", "traversal"])
def test_cli_rejects_retained_output_before_scientific_work(
    tmp_path: Path, runner: str, module: bool, alias: str
) -> None:
    retained = ROOT / "benchmarks/results"
    name = f"blocked-{tmp_path.name}.json"
    destination = retained / name
    if alias == "symlink":
        link = tmp_path / "retained-link"
        link.symlink_to(retained, target_is_directory=True)
        destination = link / name
    elif alias == "traversal":
        destination = ROOT / ".artifacts/../benchmarks/results" / name
    arguments = (
        ["oracle", "--case", "small", "--method", "rks"]
        if runner == "qualification"
        else [str(tmp_path / "missing-receipt.json")]
    )
    result = subprocess.run(
        [*command(runner, module), *arguments, "--output", str(destination)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "raw_output_path" in result.stderr
    assert not (retained / name).exists()
    assert "Traceback" not in result.stderr


def test_direct_writer_preserves_retained_bytes_and_accepts_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    existing = retained / "existing.json"
    existing.write_text("original evidence")
    with pytest.raises(ValueError, match="raw benchmark output"):
        qualification.write_json(existing, {"changed": True})
    assert existing.read_text() == "original evidence"
    scratch = tmp_path / "scratch.json"
    qualification.write_json(scratch, {"forces": [[0.0, None, 1.0]]})
    assert json.loads(scratch.read_text()) == {"forces": [[0.0, None, 1.0]]}


@pytest.mark.parametrize("module", [False, True])
def test_report_preserves_scratch_output(tmp_path: Path, module: bool) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"mask": 0}))
    output = tmp_path / "report.json"
    result = subprocess.run(
        [*command("report", module), str(receipt), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["default_promoted"] is False
