"""The compiler tuner has a separate entrypoint and no runtime CLI fallback."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from generativeqc import tune_cli


def test_tuner_requires_xyz_and_preserves_quick_full_limits() -> None:
    parser = tune_cli.parser()
    default = parser.parse_args(["water.xyz"])
    assert default.input == Path("water.xyz")
    assert default.method == "rhf"
    assert default.compile_jobs == 2
    assert default.device_id == 0
    assert default.budget_seconds == 7200
    assert parser.parse_args(["--full", "water.xyz"]).full
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["profile", "show"])
    assert error.value.code == 2


def test_tuner_never_exposes_runtime_or_profile_commands() -> None:
    with pytest.raises(SystemExit):
        tune_cli.parser().parse_args(["--show-profile", "water.xyz"])
    with pytest.raises(SystemExit):
        tune_cli.parser().parse_args(["--clear-profile", "water.xyz"])
    help_text = tune_cli.parser().format_help()
    assert "generativeqc-tune" in help_text
    assert "profiles" not in help_text.lower()


def test_tuner_validates_limits_without_importing_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_run(args: object) -> dict[str, object]:
        calls.append(args)
        return {"installed": None, "directory": "/tmp/synthetic", "reason": "no work"}

    from generativeqc import autotune

    monkeypatch.setattr(autotune, "run", fake_run)
    assert tune_cli.main(["--full", "--batch", "4", "water.xyz"]) == 0
    assert len(calls) == 1
    assert calls[0].coverage == 1.0
    assert calls[0].max_classes == 55
    assert calls[0].max_candidates == 16
    assert tune_cli.main(["water.xyz"]) == 0
    assert calls[1].coverage == 0.97
    with pytest.raises(SystemExit) as error:
        tune_cli.main(["--batch", "0", "water.xyz"])
    assert error.value.code == 2
    assert len(calls) == 2


def test_module_help_has_no_native_or_gpu_dependency() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "generativeqc.tune_cli", "--help"],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.stdout.startswith("usage: generativeqc-tune")
