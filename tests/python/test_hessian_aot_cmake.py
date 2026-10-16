"""Exercise build-time package selection without a GPU or a CUDA toolchain."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.complete_hessian_aot_inventory import complete_inventory
from tools.generate_hessian_aot import parse_inventory

ROOT = Path(__file__).resolve().parents[2]


def _configure(
    directory: Path, inventory: str = "", *, python: str = sys.executable
) -> subprocess.CompletedProcess[str]:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the package-selection gate")
    (directory / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(hessian_package_selection NONE)\n"
        f'set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")\n'
        f'set(Python3_EXECUTABLE "{python}")\n'
        f'include("{ROOT.as_posix()}/cmake/GenerativeQCHessianAot.cmake")\n'
        f'generativeqc_resolve_hessian_aot_inventory(selected "{inventory}")\n'
        'file(WRITE "${CMAKE_BINARY_DIR}/selection.txt" "${selected}")\n',
        encoding="utf-8",
    )
    return subprocess.run(
        [cmake, "-S", str(directory), "-B", str(directory / "build")],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_default_package_selects_complete_factorized_orbital_f_auxiliary_g(
    tmp_path: Path,
) -> None:
    """The enabled default must not inherit the partial development closure."""
    result = _configure(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    inventory_path = Path((tmp_path / "build/selection.txt").read_text())
    actual = parse_inventory(json.loads(inventory_path.read_text()))
    seed = parse_inventory(
        json.loads((ROOT / "manifests/hessian_aot_development.json").read_text())
    )
    expected = complete_inventory(
        seed,
        (
            "overlap",
            "kinetic",
            "nuclear_attraction",
            "eri",
            "three_center_eri",
            "coulomb_metric",
        ),
        3,
        max_auxiliary_angular=4,
    )
    assert actual.integrals == expected.integrals
    assert actual.arrays == expected.arrays
    assert actual.ao == expected.ao
    assert actual.integral_schedule == "factorized"
    original_mtime = inventory_path.stat().st_mtime_ns
    repeated = _configure(tmp_path)
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert inventory_path.stat().st_mtime_ns == original_mtime


def test_explicit_partial_inventory_is_preserved_without_completion(
    tmp_path: Path,
) -> None:
    """A deliberate small development package remains a bounded opt-in."""
    inventory_path = tmp_path / "partial inventory.json"
    original = (ROOT / "manifests/hessian_aot_development.json").read_bytes()
    inventory_path.write_bytes(original)
    result = _configure(tmp_path, inventory_path.name)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path((tmp_path / "build/selection.txt").read_text()) == inventory_path
    assert inventory_path.read_bytes() == original
    assert not (tmp_path / "build/generated/hessian_aot/inventory.json").exists()


def test_explicit_missing_inventory_fails_closed(tmp_path: Path) -> None:
    result = _configure(tmp_path, "missing.json")
    assert result.returncode != 0
    assert "Explicit Hessian AOT inventory is missing" in result.stderr


def test_package_selection_requires_build_time_python(tmp_path: Path) -> None:
    result = _configure(tmp_path, python="")
    assert result.returncode != 0
    assert "Hessian AOT generation requires build-time Python" in result.stderr
