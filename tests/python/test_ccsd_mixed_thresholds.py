"""The threshold runner preserves endpoint controls and absent force results."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from benchmarks.ccsd_mixed_thresholds import controls, maximum_force_error


def test_threshold_scan_changes_only_the_warmup_policy() -> None:
    strict = controls(True, None)
    mixed = controls(True, 1e-4)
    assert strict[:27] == mixed[:27]
    assert strict[27:] == ["0", "0", "1e-05"]
    assert mixed[27:] == ["1", "0", "0.0001"]
    assert controls(False, 1e-4)[2] == "0"
    assert controls(True, None, auto=True)[27:] == ["auto", "0"]


def test_force_error_uses_flat_public_components() -> None:
    actual = {"forces_requested": 1, "forces": [0.1, -0.2, 0.3]}
    expected = {"forces_requested": 1, "forces": [0.1, -0.21, 0.3]}
    assert maximum_force_error(actual, expected) == pytest.approx(0.01)
    assert maximum_force_error({"forces_requested": 0}, {}) is None
    with pytest.raises(ValueError):
        maximum_force_error(actual, {"forces": [0.1]})


@pytest.mark.parametrize("destination_kind", ["retained", "alias", "traversal"])
def test_threshold_runner_rejects_retained_output_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    destination_kind: str,
) -> None:
    """Reject raw output and preserve existing receipts before any GPU work."""
    from benchmarks import _retention, ccsd_mixed_thresholds

    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    existing = retained / "existing.json"
    existing.write_text("original evidence", encoding="utf-8")
    destination = retained
    if destination_kind == "alias":
        destination = tmp_path / "alias"
        destination.symlink_to(retained, target_is_directory=True)
    elif destination_kind == "traversal":
        destination = tmp_path / ".artifacts/../benchmarks/results"
    monkeypatch.setattr(
        ccsd_mixed_thresholds,
        "digest",
        lambda *_args: pytest.fail("benchmark work started before output admission"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ccsd_mixed_thresholds",
            "--binary",
            "not-executed",
            "--library",
            "not-read",
            "--input",
            "not-read",
            "--output",
            str(destination),
        ],
    )
    with pytest.raises(SystemExit) as error:
        ccsd_mixed_thresholds.main()
    assert error.value.code == 2
    assert "argument --output:" in capsys.readouterr().err
    assert existing.read_text(encoding="utf-8") == "original evidence"
    assert not (tmp_path / ".artifacts").exists()
