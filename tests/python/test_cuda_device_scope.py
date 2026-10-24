"""Compile real device owners against a host-only CUDA failure-injection double."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def device_scope_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    """Link the unmodified production adapter without loading CUDA or a GPU."""
    executable = tmp_path_factory.mktemp("cuda-device-scope") / "probe"
    return native_cxx.build_executable(
        [
            ROOT / "tests/native/test_cuda_device_scope.cpp",
            ROOT / "src/xtb/native/src/runtime/cuda_descriptor_validation.cu",
        ],
        executable,
        compile_args=[
            "-std=c++20",
            "-O0",
            "-x",
            "c++",
            "-I" + str(ROOT / "tests/native/fixtures/cuda_device_scope"),
            "-I" + str(ROOT / "src/xtb/native/src"),
            "-I" + str(ROOT / "src"),
        ],
    )


@pytest.mark.parametrize(
    "consumer,scenario",
    [
        ("shared", scenario)
        for scenario in (
            "nested",
            "mixed",
            "same-device",
            "unwind",
            "query-failure",
            "selection-failure",
            "allocation-failure",
            "restore-retry",
            "destructor-retry",
            "cleanup-failure",
        )
    ]
    + [
        ("xtb", scenario)
        for scenario in (
            "negative",
            "same-device",
            "query-failure",
            "query-invalid-device",
            "selection-failure",
            "invalid-device",
            "restore-retry",
            "destructor-retry",
            "cleanup-failure",
        )
    ],
)
def test_device_scope_contract(
    device_scope_probe: Path, consumer: str, scenario: str
) -> None:
    """Check observable device state, cleanup, diagnostics and status mapping."""
    result = subprocess.run(
        [str(device_scope_probe), consumer, scenario],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
