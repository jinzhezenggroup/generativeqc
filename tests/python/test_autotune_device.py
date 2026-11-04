"""Discovery subprocesses preserve scheduling and fail before endpoint execution."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import _autotune_device


def test_device_discovery_does_not_load_cuda_in_parent(monkeypatch: Any) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-scheduler-allocated")
    monkeypatch.setenv("GENERATIVEQC_PROFILE", "off")
    before = dict(os.environ)
    observation = {"probe": {"source_identity": "synthetic"}, "library": "/selected.so"}

    def forbidden_parent_load(*args: Any) -> None:
        pytest.fail("the coordinator must not retain a CUDA context")

    def child(command: list[str], **kwargs: Any) -> Any:
        assert command[-1] == "--select" and command[-2] == "2"
        assert kwargs["timeout"] == 7
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-scheduler-allocated"
        assert kwargs["env"]["GENERATIVEQC_PROFILE"] == "off"
        assert kwargs["env"]["GENERATIVEQC_LIBRARY"] == "/baseline.so"
        return SimpleNamespace(returncode=0, stdout=json.dumps(observation))

    monkeypatch.setattr(_autotune_device.ctypes, "CDLL", forbidden_parent_load)
    monkeypatch.setattr(_autotune_device.subprocess, "run", child)
    assert (
        _autotune_device.inspect_library(
            Path("/baseline.so"), 2, select=True, timeout=7
        )
        == observation
    )
    assert dict(os.environ) == before


@pytest.mark.parametrize("failure", ("exit", "timeout", "invalid-output"))
def test_device_discovery_never_reuses_failed_probe(
    monkeypatch: Any, failure: str
) -> None:
    def child(*args: Any, **kwargs: Any) -> Any:
        if failure == "timeout":
            raise subprocess.TimeoutExpired("synthetic discovery", 1)
        return SimpleNamespace(
            returncode=int(failure == "exit"),
            stderr="synthetic failure",
            stdout="bad-json",
        )

    monkeypatch.setattr(_autotune_device.subprocess, "run", child)
    with pytest.raises((RuntimeError, ValueError, subprocess.TimeoutExpired)):
        _autotune_device.inspect_library(Path("/baseline.so"), 0, timeout=1)
