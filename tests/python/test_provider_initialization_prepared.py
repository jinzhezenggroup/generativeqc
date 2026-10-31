"""Actual publisher/resource-ledger integration with metadata-only method factories."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def prepared_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    directory = tmp_path_factory.mktemp("provider-prepared-resource")
    sources = [
        "tests/native/test_provider_initialization_prepared.cpp",
        "src/api/c_api_calculation.cpp",
        "src/api/c_api_provider_initialization_lifetime.cpp",
        "src/api/c_api_device_resources.cpp",
        "src/api/error.cpp",
        "src/methods/prepared_execution.cpp",
        "src/runtime/resource_ledger_plan.cpp",
        "src/runtime/resource_planning.cpp",
    ]
    return required_native_cxx.build_executable(
        [ROOT / path for path in sources],
        directory / "prepared-resource",
        compile_args=[
            "-std=c++20",
            "-DGENERATIVEQC_HAS_CUDA=0",
            "-ffunction-sections",
            "-fdata-sections",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ],
        link_args=["-pthread", "-Wl,--gc-sections"],
    )


@pytest.mark.parametrize(
    "mode", ["resource-retention", "pending-unknown-ledger", "resource-device-refusal"]
)
def test_actual_prepared_publisher_and_ledger(prepared_probe: Path, mode: str) -> None:
    result = subprocess.run(
        [str(prepared_probe), mode],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
