"""Current-provider comparison must use the old exact input, not regenerated PTX."""

from __future__ import annotations

import base64
import hashlib
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts"


def _comparison(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return runpy.run_path(str(SCRIPTS / "compare_cumetal_ptx.py"))


def test_retained_ptx_matches_exact_captured_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _comparison(monkeypatch)
    fixture = json.loads(namespace["FIXTURE"].read_text())
    source = base64.b64decode(fixture["input_base64"], validate=True)
    assert len(source) == 7051 == fixture["input_bytes"]
    assert hashlib.sha256(source).hexdigest() == namespace["INPUT_SHA256"]
    assert fixture["input_sha256"] == namespace["INPUT_SHA256"]
    assert fixture["entry"] == namespace["KERNEL"]
    assert (
        "aggregate PTX parameter load is not an aligned field"
        in fixture["source_typed_error"]
    )
    assert source.splitlines()[25].endswith(b"_param_0+32];")
    assert not source.endswith(b"\n")  # Preserve the original absent final newline.


@pytest.mark.parametrize(
    "outcome", ("pass", "fail", "timeout", "dirty", "wrong-pin", "corrupt")
)
def test_comparison_never_recaptures_or_changes_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    namespace = _comparison(monkeypatch)
    compare = namespace["compare"]
    env = {
        "CUMETAL_PREFIX": str(tmp_path / "prefix"),
        "CUMETAL_SOURCE": "source",
        "CUMETAL_COMMIT": "current-pin",
    }
    compiler = Path(env["CUMETAL_PREFIX"]) / "bin/cumetalc"
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"fixture compiler")
    fixture = namespace["FIXTURE"]
    if outcome == "corrupt":
        data = json.loads(fixture.read_text())
        data["input_base64"] = base64.b64encode(b"not the original PTX").decode()
        fixture = tmp_path / "corrupt.json"
        fixture.write_text(json.dumps(data))
    calls = []

    def run(
        command: list[str],
        name: str,
        output: Path,
        child_env: dict[str, str],
        timeout: float,
        deadline: float,
    ) -> dict[str, object]:
        calls.append((name, command, timeout, deadline))
        stdout = ""
        if name == "provider-head":
            stdout = "old-pin" if outcome == "wrong-pin" else env["CUMETAL_COMMIT"]
        if name == "provider-status" and outcome == "dirty":
            stdout = " M runtime/registration/registration.cpp"
        (output / f"{name}.stdout.txt").write_text(stdout)
        (output / f"{name}.stderr.txt").write_text("fixture compiler result")
        return {
            "command": command,
            "returncode": 1 if name == "typed-compiler" and outcome == "fail" else 0,
            "timed_out": name == "typed-compiler" and outcome == "timeout",
        }

    monkeypatch.setitem(compare.__globals__, "run_command", run)
    output = tmp_path / "comparison"
    if outcome == "corrupt":
        with pytest.raises(ValueError, match="captured input"):
            compare(output, env, fixture)
        assert calls == []
        return
    assert compare(output, env, fixture) == (0 if outcome == "pass" else 1)
    report = json.loads((output / "comparison.json").read_text())
    assert "not QC acceptance" in report["purpose"]
    assert report["input_sha256"] == namespace["INPUT_SHA256"]
    assert (
        hashlib.sha256((output / "retained-input.ptx").read_bytes()).hexdigest()
        == namespace["INPUT_SHA256"]
    )
    assert len({call[3] for call in calls}) == 1
    assert not any(
        "pytest" in command or "legacy" in command for _, command, _, _ in calls
    )
    typed = [call for call in calls if call[0] == "typed-compiler"]
    assert len(typed) == (0 if outcome in {"dirty", "wrong-pin"} else 1)
    if typed:
        _, command, timeout, _ = typed[0]
        assert "--backend=cumetal-ir" in command
        assert "--fp64=fast48" in command
        assert "--emit=msl" in command
        assert "--ptx-strict" in command
        assert "--no-link" in command
        assert timeout == 60


def test_latest_pin_comparison_is_separate_from_qc_acceptance() -> None:
    workflow = (ROOT / ".github/workflows/cumetal-cuda.yml").read_text()
    runtime = workflow.split("\n  cuda-tests:", 1)[1].split(
        "\n  cumetal-benchmark:", 1
    )[0]
    assert "CUMETAL_COMMIT: e87f368060cf09a45b148b4f8892470c74093ebd" in runtime
    comparison = runtime.split("- name: Compare exact retained CuMetal PTX", 1)[
        1
    ].split("\n      - name:", 1)[0]
    assert "continue-on-error: true" in comparison
    assert "timeout-minutes: 2" in comparison
    assert "compare_cumetal_ptx.py" in comparison
    for condition in (
        "github.event_name == 'pull_request'",
        "github.event.pull_request.number == 1997",
        "github.event.action == 'synchronize'",
        "github.event.before == '48e7afd1cb8e52f0abb85af08727e7684b969631'",
        "github.run_attempt == 1",
    ):
        assert condition in comparison
    assert runtime.index("Compare exact retained CuMetal PTX") < runtime.index(
        "Run bounded CuMetal QC endpoint gate"
    )
    gate = runtime.split("- name: Run bounded CuMetal QC endpoint gate", 1)[1].split(
        "\n      - name:", 1
    )[0]
    assert "continue-on-error" not in gate
    assert "run_cumetal_cuda_pytests.py" in gate
    # The completed one-shot legacy capture stays ineligible for this update.
    capture = runtime.split("- name: Capture one CuMetal PTX compiler diagnostic", 1)[
        1
    ].split("\n      - name:", 1)[0]
    assert (
        "github.event.before == '0f67b413da60fa3d146eead32360901adeda4698'" in capture
    )
    assert (
        "github.event.before == '48e7afd1cb8e52f0abb85af08727e7684b969631'"
        not in capture
    )
