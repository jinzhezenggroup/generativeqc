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
    assert parser.parse_args(["water.xyz", "--device-id", "2"]).device_id == 2
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
    assert (
        tune_cli.main(["--full", "--batch", "4", "--device-id", "2", "water.xyz"]) == 0
    )
    assert len(calls) == 1
    assert calls[0].coverage == 1.0
    assert calls[0].device_id == 2
    assert calls[0].max_classes == 55
    assert calls[0].max_candidates == 16
    assert tune_cli.main(["water.xyz"]) == 0
    assert calls[1].coverage == 0.97
    with pytest.raises(SystemExit) as error:
        tune_cli.main(["--batch", "0", "water.xyz"])
    assert error.value.code == 2
    assert len(calls) == 2
    for invalid in ("-1", str(2**31)):
        with pytest.raises(SystemExit) as error:
            tune_cli.main(["--device-id", invalid, "water.xyz"])
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


@pytest.mark.parametrize(
    "method", ("lda-rks", "pbe-rks", "pbe-uks", "pbe0-rks", "b3lyp-uks", "r2scan-rks")
)
def test_dft_tuner_preserves_scientific_and_execution_controls(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc import autotune

    calls = []

    def fake_run(args: object) -> dict:
        calls.append(args)
        return {"installed": None, "directory": "/tmp/dft-fixture"}

    monkeypatch.setattr(autotune, "run", fake_run)
    assert (
        tune_cli.main(
            [
                "water.xyz",
                "--method",
                method,
                "--grid-level",
                "2",
                "--batch",
                "4",
                "--xc-tiles",
                "128",
                "512",
                "--device-id",
                "2",
                "--repeats",
                "5",
            ]
        )
        == 0
    )
    assert calls[0].method == method
    assert calls[0].grid_level == 2
    assert calls[0].xc_tiles == [128, 512]
    assert calls[0].batch == 4 and calls[0].device_id == 2
    assert calls[0].repeats == 5


@pytest.mark.parametrize(
    "arguments",
    (
        ["--method", "pbe-rks", "--repeats", "4"],
        ["--method", "pbe-rks", "--grid-level", "10"],
        ["--method", "pbe-rks", "--xc-tiles", "0"],
        ["--method", "pbe-rks", "--xc-tiles", "4097"],
        ["--method", "rhf", "--xc-tiles", "128"],
        ["--method", "uhf", "--grid-level", "2"],
    ),
)
def test_invalid_dft_controls_fail_before_tuning(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        tune_cli.main(["water.xyz", *arguments])
    assert error.value.code == 2
