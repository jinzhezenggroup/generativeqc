"""The one-shot provider diagnostic must never become a scientific fallback."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/diagnose_cumetal_ptx.py"
WORKFLOW = ROOT / ".github/workflows/cumetal-cuda.yml"


def _diagnostic() -> dict[str, object]:
    return runpy.run_path(str(SCRIPT), run_name="cumetal_diagnostic_test")


@pytest.mark.parametrize("failure", (None, "missing", "ambiguous", "timeout", "pin"))
def test_diagnostic_captures_exact_input_without_retrying_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    namespace = _diagnostic()
    diagnose = namespace["diagnose"]
    output = tmp_path / "evidence"
    prefix = tmp_path / "provider"
    compiler = prefix / "bin/cumetalc"
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"fixture compiler")
    env = {
        "CUMETAL_PREFIX": str(prefix),
        "CUMETAL_SOURCE": str(tmp_path / "source"),
        "CUMETAL_COMMIT": "pinned-source",
    }
    calls = []
    ptx_bytes = f".version 8.0\n.visible .entry {namespace['KERNEL']}() {{}}\n".encode()

    def run(
        command: list[str],
        name: str,
        directory: Path,
        command_env: dict[str, str],
        timeout: float,
        deadline: float,
    ) -> dict[str, object]:
        calls.append((name, command, command_env.copy(), timeout, deadline))
        stdout = ""
        if name == "provider-head":
            stdout = "different" if failure == "pin" else env["CUMETAL_COMMIT"]
        if name == "legacy-capture" and failure != "missing":
            path = Path(command_env["CUMETAL_DEBUG_DUMP_PTX_DIR"])
            path.mkdir()
            (path / f"{namespace['KERNEL']}.ptx").write_bytes(
                ptx_bytes * (2 if failure == "ambiguous" else 1)
            )
        (directory / f"{name}.stdout.txt").write_text(stdout)
        (directory / f"{name}.stderr.txt").write_text(
            "cumetalc failed: fixture typed error" if name == "typed-compiler" else ""
        )
        return {
            "command": command,
            "returncode": 1 if name == "typed-compiler" else 0,
            "timed_out": failure == "timeout" and name == "legacy-capture",
        }

    monkeypatch.setitem(diagnose.__globals__, "run_command", run)
    assert diagnose(output, env) == (0 if failure is None else 1)
    report = json.loads((output / "diagnostic.json").read_text())
    assert "not QC acceptance" in report["purpose"]
    assert report["compiler_sha256"] == hashlib.sha256(b"fixture compiler").hexdigest()
    assert len({call[4] for call in calls}) == 1
    legacy = [call for call in calls if call[0] == "legacy-capture"]
    typed = [call for call in calls if call[0] == "typed-compiler"]
    assert len(legacy) == (0 if failure == "pin" else 1)
    assert len(typed) == (1 if failure is None else 0)
    if legacy:
        _, command, child_env, timeout, _ = legacy[0]
        assert command.count(namespace["ENDPOINT"]) == 1
        assert command[-1] == f"--junitxml={output / 'legacy-capture.xml'}"
        assert child_env["CUMETAL_PTX_BACKEND"] == "legacy"
        assert child_env["CUMETAL_FP64_MODE"] == "fast48"
        assert timeout == 90
        assert "CUMETAL_PTX_BACKEND" not in env
    if typed:
        _, command, child_env, timeout, _ = typed[0]
        assert command[0] == str(compiler)
        assert command[command.index("--entry") + 1] == namespace["KERNEL"]
        assert "--backend=cumetal-ir" in command
        assert "--fp64=fast48" in command
        assert "--emit=msl" in command
        assert "--no-link" in command
        assert "--ptx-strict" in command
        assert child_env["CUMETAL_PTX_BACKEND"] == "cumetal-ir"
        assert timeout == 60
        assert report["input"]["sha256"] == hashlib.sha256(ptx_bytes).hexdigest()
        assert report["input"]["bytes"] == len(ptx_bytes)
        assert report["commands"]["typed-compiler"]["returncode"] == 1
        assert (
            "fixture typed error" in (output / "typed-compiler.stderr.txt").read_text()
        )


def test_diagnostic_refuses_stale_evidence(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        _diagnostic()["diagnose"](tmp_path, {})


def test_diagnostic_subprocess_preserves_streams_and_bounds_time(
    tmp_path: Path,
) -> None:
    namespace = _diagnostic()
    run = namespace["run_command"]
    monotonic = run.__globals__["time"].monotonic
    result = run(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        ],
        "streams",
        tmp_path,
        dict(os.environ),
        5,
        monotonic() + 10,
    )
    assert result["returncode"] == 0
    assert (tmp_path / "streams.stdout.txt").read_text().strip() == "out"
    assert (tmp_path / "streams.stderr.txt").read_text().strip() == "err"
    result = run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        "timeout",
        tmp_path,
        dict(os.environ),
        0.05,
        monotonic() + 10,
    )
    assert result["timed_out"]
    assert result["returncode"] < 0
    result = run(["must-not-run"], "exhausted", tmp_path, {}, 5, monotonic() - 1)
    assert result["error"] == "diagnostic suite budget exhausted"
    assert not (tmp_path / "exhausted.stdout.txt").exists()


def test_workflow_diagnostic_is_one_shot_and_cannot_replace_failed_gate() -> None:
    workflow = WORKFLOW.read_text()
    gate = workflow.split("- name: Run bounded CuMetal QC endpoint gate", 1)[1]
    gate = gate.split("\n      - name:", 1)[0]
    assert "id: qc_gate" in gate
    assert "continue-on-error" not in gate
    assert "CUMETAL_PTX_BACKEND: cumetal-ir" in gate
    diagnostic = workflow.split(
        "- name: Capture one CuMetal PTX compiler diagnostic", 1
    )[1]
    diagnostic = diagnostic.split("\n      - name:", 1)[0]
    for condition in (
        "failure()",
        "github.event_name == 'pull_request'",
        "github.event.pull_request.number == 1997",
        "github.event.action == 'synchronize'",
        "github.event.before == '0f67b413da60fa3d146eead32360901adeda4698'",
        "github.run_attempt == 1",
        "steps.qc_gate.outcome == 'failure'",
    ):
        assert condition in diagnostic
    assert "continue-on-error: true" in diagnostic
    assert "timeout-minutes: 4" in diagnostic
    artifact = workflow.split(
        "- name: Preserve one-shot CuMetal compiler diagnostic", 1
    )[1]
    artifact = artifact.split("\n      - name:", 1)[0]
    assert "always()" in artifact
    assert "name: cumetal-ptx-diagnostic-only" in artifact
    assert "retention-days: 1" in artifact
    assert "ptx/*.ptx" in artifact
    assert "legacy-capture.xml" in artifact
    assert "diagnostic-jit" not in artifact
