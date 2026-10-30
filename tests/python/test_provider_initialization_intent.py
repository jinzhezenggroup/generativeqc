"""Compiled C/C++ pending-intent transport contracts; no numerical execution."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def intent_abi_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    directory = tmp_path_factory.mktemp("provider-intent-abi")
    c_object = directory / "client.o"
    required_native_cxx.compile_object(
        ROOT / "tests/native/provider_initialization_intent_client.c",
        c_object,
        args=["-x", "c", "-std=c11", "-I" + str(ROOT / "include")],
    )
    cpp_sources = [ROOT / "tests/native/test_provider_initialization_intent_abi.cpp"]
    implementation = ROOT / "src/api/c_api_provider_initialization.cpp"
    if implementation.exists():
        cpp_sources.append(implementation)
    objects = [c_object]
    for index, source in enumerate(cpp_sources):
        obj = directory / f"implementation-{index}.o"
        required_native_cxx.compile_object(
            source,
            obj,
            args=["-std=c++20", "-I" + str(ROOT / "include"), "-I" + str(ROOT / "src")],
        )
        objects.append(obj)
    executable = directory / "intent-abi"
    required_native_cxx.link(objects, executable)
    return executable


@pytest.mark.parametrize(
    "mode", ["capture", "overlap", "legacy", "negative", "lifetime"]
)
def test_pending_intent_c_abi(intent_abi_probe: Path, mode: str) -> None:
    result = subprocess.run(
        [str(intent_abi_probe), mode],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def generation_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    directory = tmp_path_factory.mktemp("provider-intent-generation")
    return required_native_cxx.build_executable(
        [ROOT / "tests/native/test_provider_initialization_generation.cpp"],
        directory / "generation",
        compile_args=[
            "-std=c++20",
            "-DGENERATIVEQC_HAS_CUDA=0",
            "-I" + str(ROOT / "include"),
            "-I" + str(ROOT / "src"),
        ],
    )


@pytest.mark.parametrize(
    "mode",
    [
        "domain",
        "bind",
        "legacy",
        "context",
        "loaded-owner",
        "ledger",
        "query-error",
        "retire-query-error",
        "retirement",
    ],
)
def test_pending_generation_contract(generation_probe: Path, mode: str) -> None:
    result = subprocess.run(
        [str(generation_probe), mode],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
