"""Fail closed before scientific qualification consumes incomplete inputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.experiments.mp2_materialization import oracle, qualify

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("script", ["oracle", "qualify", "endpoint", "memory"])
@pytest.mark.parametrize("alias", [False, True])
def test_retained_output_rejected_before_scientific_imports(
    tmp_path: Path, script: str, alias: bool
) -> None:
    destination = ROOT / "benchmarks/results/must-not-overwrite.json"
    if alias:
        link = tmp_path / "retained"
        link.symlink_to(destination.parent, target_is_directory=True)
        destination = link / destination.name
    args = [str(destination)]
    if script == "oracle":
        args = ["--observations", str(tmp_path), "--output", str(destination)]
    elif script == "qualify":
        args = ["--build", str(tmp_path), "--output", str(destination)]
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/experiments/mp2_materialization" / (script + ".py")),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 2, result.stderr
    assert "invalid raw_output_path value" in result.stderr
    assert not destination.exists()


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("case", ["h2", "lih", "water"])
def test_incomplete_endpoint_history_rejected(
    tmp_path: Path, count: int, case: str
) -> None:
    cases = {name: {"samples": [{}, {}]} for name in ("h2", "lih", "water")}
    cases[case]["samples"] = [{}] * count
    (tmp_path / "endpoint-00.json").write_text(json.dumps({"cases": cases}))
    with pytest.raises(ValueError, match=f"expected two samples for {case}"):
        oracle.observation_histories(tmp_path)


def test_complete_timing_and_memory_histories_admitted(tmp_path: Path) -> None:
    cases = {
        name: {"samples": [{"energy": 0}, {"energy": 0}]}
        for name in ("h2", "lih", "water")
    }
    for name in ("endpoint-00.json", "memory-00.endpoints.json"):
        (tmp_path / name).write_text(json.dumps({"cases": cases}))
    histories = oracle.observation_histories(tmp_path)
    assert len(histories) == 2
    assert all(value == cases for value in histories.values())


@pytest.mark.parametrize("other_checkout", [False, True])
def test_retained_build_must_match_checkout_and_be_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, other_checkout: bool
) -> None:
    home = tmp_path / "other" if other_checkout else ROOT
    (tmp_path / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={home}\n")
    calls = []

    def planned(command: list[str], *, cwd: Path) -> str:
        calls.append((command, cwd))
        return "[1/2] compiler -c stale.cpp\n[2/2] linker\n"

    monkeypatch.setattr(qualify, "run", planned)
    with pytest.raises(
        ValueError, match="source checkout" if other_checkout else "stale"
    ):
        qualify.verify_build_freshness(tmp_path)
    assert bool(calls) is not other_checkout


def test_fresh_retained_build_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={ROOT}\n")
    calls = []

    def planned(command: list[str], *, cwd: Path) -> str:
        calls.append((command, cwd))
        return "ninja: no work to do.\n"

    monkeypatch.setattr(qualify, "run", planned)
    qualify.verify_build_freshness(tmp_path)
    assert calls == [(["ninja", "-n", "-v", "libgenerativeqc.so.0.1.0"], tmp_path)]
