"""Roundtrip publication argv without loading a reference oracle or using CUDA."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def validator(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Only replace unrelated PySCF-dependent system-reporting utilities."""
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(
        sys.modules,
        "validate_range_eri",
        SimpleNamespace(command=None, cpu_model=None),
    )
    path = Path(__file__).resolve().parents[2] / "tools/validate_low_rank.py"
    spec = importlib.util.spec_from_file_location("low_rank_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "find_nvcc", lambda: None)
    return module


@pytest.mark.parametrize("samples", [1, 7])
def test_cpu_reproduction_keeps_existing_arguments(
    validator: ModuleType, samples: int, tmp_path: Path
) -> None:
    original = validator.parse_args(
        ["--samples", str(samples), "--output", str(tmp_path / "original")]
    )
    command = validator.reproduction_command(original)
    assert command == [
        "python",
        "tools/validate_low_rank.py",
        "--backend",
        "cpu",
        "--samples",
        str(samples),
        "--output",
        ".artifacts/low-rank-cpu-reproduction",
    ]
    replay = validator.parse_args(command[2:])
    assert replay.backend == "cpu"
    assert replay.samples == samples
    assert replay.output == Path(".artifacts/low-rank-cpu-reproduction")
    assert replay.cuda_target is None
    assert replay.nvcc is None
    assert replay.publish is None


@pytest.mark.parametrize("target", ["sm_86", "sm_90", "sm_120"])
@pytest.mark.parametrize("explicit_compiler", [False, True])
def test_cuda_reproduction_keeps_selected_target_and_compiler(
    validator: ModuleType,
    target: str,
    explicit_compiler: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    discovered_compiler = tmp_path / "selected toolkit/bin/nvcc"
    compiler = discovered_compiler
    monkeypatch.setattr(validator, "find_nvcc", lambda: discovered_compiler)
    argv = [
        "--backend",
        "cuda",
        "--cuda-target",
        target,
        "--samples",
        "5",
        "--output",
        str(tmp_path / "original"),
        "--publish",
        str(tmp_path / "published"),
    ]
    if explicit_compiler:
        compiler = tmp_path / "explicit toolkit/bin/nvcc"
        argv.extend(["--nvcc", str(compiler)])
    original = validator.parse_args(argv)
    command = validator.reproduction_command(original)
    # Replay on a host where discovery would choose a different toolkit.
    monkeypatch.setattr(validator, "find_nvcc", lambda: tmp_path / "other/nvcc")
    replay = validator.parse_args(command[2:])
    assert command[:2] == ["python", "tools/validate_low_rank.py"]
    assert replay.backend == "cuda"
    assert replay.samples == 5
    assert replay.cuda_target == target
    assert replay.nvcc == compiler
    assert replay.output == Path(".artifacts/low-rank-cuda-reproduction")
    assert replay.publish is None
