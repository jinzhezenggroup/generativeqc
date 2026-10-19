"""Reject retained-tree output before qz authentication or observation."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks import qualify_residency_qz as runner


def test_retained_output_is_rejected_before_loading_inspire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = Path(runner.__file__).resolve().parent / "results/qz-parse-guard.json"
    previous = destination.read_bytes() if destination.exists() else None
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] == "inspire":
            pytest.fail("retained output reached qz SDK import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify_residency_qz",
            "--no-env-file",
            "--account",
            "qz",
            "--workspace",
            "unused",
            "--job",
            "unused",
            "--python",
            "/usr/bin/python",
            "--output",
            str(destination),
        ],
    )
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert (destination.read_bytes() if destination.exists() else None) == previous
