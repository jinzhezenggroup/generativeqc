"""Exercise the finite benchmark wrapper without loading a CUDA runtime."""

import subprocess
from pathlib import Path
from typing import Any

import pytest

from benchmarks import direct_force_reuse


def test_scheduler_visibility_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never create an output or start a GPU process outside Slurm."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SystemExit, match="2"):
        direct_force_reuse.main([str(tmp_path / "not-started")])
    assert not (tmp_path / "not-started").exists()


@pytest.mark.parametrize("timeout", [False, True])
def test_population_budget_and_owned_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout: bool
) -> None:
    """Preserve device identity, five-call worker protocol and finite cleanup."""
    for name, value in (
        ("SLURM_JOB_ID", "fixture"),
        ("CUDA_VISIBLE_DEVICES", "GPU-scheduler-assigned"),
        ("GENERATIVEQC_LIBRARY", "/fixture/libgenerativeqc.so"),
    ):
        monkeypatch.setenv(name, value)
    launches, signals = [], []

    class Child:
        pid = 123456

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            launches.append((command, kwargs))
            self.waits = 0

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            self.waits += 1
            if self.waits == 1:
                assert timeout == 1800
            if self.waits == 2:
                assert timeout == 10
            if timeout is not None and should_timeout:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return 0

    should_timeout = timeout
    monkeypatch.setattr(direct_force_reuse.subprocess, "Popen", Child)
    monkeypatch.setattr(
        direct_force_reuse.os, "killpg", lambda *args: signals.append(args)
    )
    output = tmp_path / "population"
    if timeout:
        with pytest.raises(SystemExit, match="124"):
            direct_force_reuse.main([str(output)])
        assert (output / "baseline/exit-status.txt").read_text() == "124\n"
        assert len(signals) == 2 and all(row[0] == Child.pid for row in signals)
        assert not (output / "candidate").exists()
    else:
        direct_force_reuse.main([str(output)])
        assert len(launches) == 2 and not signals
        assert (output / "candidate/exit-status.txt").read_text() == "0\n"
    for command, options in launches:
        assert "--worker" in command and options["start_new_session"]
        assert options["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-scheduler-assigned"
