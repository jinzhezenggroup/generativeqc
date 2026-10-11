"""Build-graph regressions for optional CUDA XC, without compiling CUDA."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _target_block(source: str, target: str) -> str:
    start = source.index(f"add_executable({target}")
    end = source.index(f"set_target_properties({target}", start)
    return source[start:end]


def _assert_grid_registry_closure(source: str) -> None:
    consumers = []
    for target in re.findall(r"add_executable\((\w+)", source):
        start = source.index(f"add_executable({target}")
        declaration = source[start : source.index(")", start) + 1]
        if '"${GENERATIVEQC_GRID_SOURCE}"' not in declaration:
            continue
        consumers.append(target)
        block = _target_block(source, target)
        assert "$<TARGET_OBJECTS:generativeqc_libxc_semilocal_cuda>" in block
        assert '"${GENERATIVEQC_LIBXC_SEMILOCAL_CUDA_DIRECTORY}"' in block
        dependencies = re.findall(r"add_dependencies\([^)]*\)", block)
        assert any(
            "generativeqc_libxc_semilocal_cuda_codegen" in value
            for value in dependencies
        )
    assert consumers, "the generated grid must have standalone native consumers"


def test_every_standalone_grid_consumer_has_registry_compile_and_link_closure() -> None:
    source = (ROOT / "cmake/GenerativeQCTests.cmake").read_text()
    _assert_grid_registry_closure(source)


@pytest.mark.parametrize(
    "removed",
    (
        "$<TARGET_OBJECTS:generativeqc_libxc_semilocal_cuda>",
        '"${GENERATIVEQC_LIBXC_SEMILOCAL_CUDA_DIRECTORY}"',
        "generativeqc_libxc_semilocal_cuda_codegen",
    ),
)
def test_grid_closure_guard_rejects_each_missing_cosx_dependency(removed: str) -> None:
    source = (ROOT / "cmake/GenerativeQCTests.cmake").read_text()
    block = _target_block(source, "generativeqc_cosx_cuda_tests")
    with pytest.raises(AssertionError):
        _assert_grid_registry_closure(source.replace(block, block.replace(removed, "")))


_SCIENCE_INPUTS = (
    "python/generativeqc_compiler/xc/libxc_bulk_catalog.json",
    "python/generativeqc_compiler/common/identity_only.py",
    "python/generativeqc_compiler/integral/production_df_values.json",
    "upstream/libxc-fulltree/7.0.0/maple/input.mpl",
    "upstream/libxc-fulltree/7.0.0/src/owner.c",
    "upstream/libxc/7.0.0/functionals.c",
    "upstream/libxc/7.0.0/work_mgga_inc.c",
)


@pytest.mark.parametrize("enabled", (False, True), ids=("off", "on"))
def test_registry_incremental_build_tracks_science_only_when_enabled(
    tmp_path: Path, enabled: bool
) -> None:
    """Execute the real declaration with a counter generator, not CUDA/NVCC."""
    cmake, ninja = shutil.which("cmake"), shutil.which("ninja")
    if cmake is None or ninja is None:
        pytest.skip("incremental graph probe requires CMake and Ninja")
    source = (ROOT / "cmake/GenerativeQCGeneratedSources.cmake").read_text()
    start = source.index("  set(GENERATIVEQC_LIBXC_SEMILOCAL_CUDA_DIRECTORY")
    end = source.index("  # Reuse the exact AOT objects", start)
    declaration = source[start:end]
    inputs = set(re.findall(r'"\$\{CMAKE_CURRENT_SOURCE_DIR\}/([^"*$]+)"', declaration))
    for relative in inputs | set(_SCIENCE_INPUTS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source input\n")
    generator = tmp_path / "tools/generate_libxc_semilocal_cuda_registry.py"
    generator.parent.mkdir(parents=True, exist_ok=True)
    generator.write_text(
        "import argparse, pathlib\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--output-directory', type=pathlib.Path)\n"
        "p.add_argument('--disabled', action='store_true')\n"
        "a = p.parse_args()\n"
        "a.output_directory.mkdir(parents=True, exist_ok=True)\n"
        "counter = a.output_directory / 'runs.txt'\n"
        "counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))\n"
        "names = ['registry.hpp', 'registry.cpp']\n"
        "if not a.disabled: names += [str(i) + '.cu' for i in range(32)]\n"
        "for name in names:\n"
        "    (a.output_directory / ('generated_libxc_semilocal_cuda_' + name)).write_text('// inert probe\\n')\n"
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(AutomaticCudaGraph LANGUAGES NONE)\n"
        f'include("{ROOT / "cmake/GenerativeQCGenerated.cmake"}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        f"set(GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT {'ON' if enabled else 'OFF'})\n"
        + declaration
    )
    build = tmp_path / "build"
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(build), "-G", "Ninja"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    command = [
        cmake,
        "--build",
        str(build),
        "--target",
        "generativeqc_libxc_semilocal_cuda_codegen",
    ]

    def run() -> int:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        return int((build / "generated/libxc_semilocal_cuda/runs.txt").read_text())

    assert run() == 1
    assert run() == 1
    expected = 1
    for relative in (*_SCIENCE_INPUTS, "python/generativeqc_compiler/common/added.py"):
        path = tmp_path / relative
        path.write_text("# changed scientific or identity input\n")
        expected += int(enabled)
        assert run() == expected, relative
    shards = list((build / "generated/libxc_semilocal_cuda").glob("*.cu"))
    assert len(shards) == (32 if enabled else 0)
