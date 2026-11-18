"""Compiled exact native AO prefix: content identity, capacity and lifecycle."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = (
    "capacity",
    "tails",
    "streaming",
    "allocation",
    "basis",
    "points",
    "weights",
    "environment",
    "failure",
    "nonfinite",
    "shape",
    "empty",
    "unaligned",
)


@pytest.fixture(scope="module")
def prefix_probe(
    tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any
) -> Path:
    """Compile the production header against the actual native AO evaluator."""
    directory = tmp_path_factory.mktemp("host-ao-prefix")
    core = Path(policy_library._name).resolve()
    return required_native_cxx.build_executable(
        [ROOT / "tests/native/host_ao_prefix_probe.cpp"],
        directory / "probe",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ],
        link_args=[str(core), "-Wl,-rpath," + str(core.parent)],
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_exact_native_ao_prefix(prefix_probe: Path, scenario: str) -> None:
    """Misses/failures stream; hits preserve native AO bytes and floating flags."""
    result = subprocess.run(
        [str(prefix_probe), scenario],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
