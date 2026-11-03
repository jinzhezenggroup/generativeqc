"""Exercise native automatic tiling without requiring a CUDA device."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_automatic_tile_preserves_overrides_and_resource_fallbacks(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """CUDA XC defaults to 1024 without a profile, shape or FP64-device gate."""
    unit = tmp_path / "tiles.cpp"
    unit.write_text(
        r"""
#include <iostream>
#include "dft/ks_tile_policy.hpp"
int main() {
  std::size_t requested, nao, points;
  bool fitted, unconstrained, nonlocal;
  int major, ratio;
  while (std::cin >> requested >> fitted >> unconstrained >> nao >> points
                 >> nonlocal >> major >> ratio)
    std::cout << generativeqc::dft::automatic_ks_scf_tile(
        requested, fitted, unconstrained, nao, points, nonlocal, major, ratio) << '\n';
}
"""
    )
    exe = required_native_cxx.build_executable(
        [unit],
        tmp_path / "tiles",
        compile_args=(
            "-std=c++20",
            "-I" + str(Path(__file__).resolve().parents[2] / "src"),
        ),
    )
    eligible = (0, 1, 1, 768, 1 << 20, 0, 8, 2)
    cases = [(eligible, 1024)]
    # Explicit requests have priority even in an otherwise ineligible domain.
    for tile in (1, 31, 256, 512, 4096):
        cases.append(((tile, 0, 0, 1, 1, 0, 0, 0), tile))
    for index, value, expected in (
        (1, 0, 1024),
        (2, 0, 256),
        (3, 511, 1024),
        (4, (1 << 19) - 1, 1024),
        (5, 1, 512),
        (6, 7, 1024),
        (6, 0, 1024),
        (7, 0, 1024),
        (7, 64, 1024),
    ):
        row = list(eligible)
        row[index] = value
        cases.append((tuple(row), expected))
    cases.extend(
        [
            ((0, 1, 1, 512, 1 << 19, 0, 9, 2), 1024),
            ((0, 0, 1, 1, 1, 0, 12, 64), 1024),
            ((0, 0, 0, 768, 1 << 20, 0, 12, 64), 256),
            ((0, 1, 1, 768, 1 << 20, 1, 12, 64), 256),
            ((0, 0, 1, 768, 1 << 20, 1, 8, 2), 256),
            ((0, 1, 1, 511, 1 << 20, 1, 8, 2), 256),
        ]
    )
    result = subprocess.run(
        [str(exe)],
        input="".join(" ".join(map(str, row)) + "\n" for row, _ in cases),
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    assert list(map(int, result.stdout.split())) == [expected for _, expected in cases]


def test_force_tile_has_independent_capability_and_geometry_limits(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Force batching admits low-FP64 devices but protects geometry fallback cliffs."""
    unit = tmp_path / "force_tiles.cpp"
    unit.write_text(
        r"""
#include <iostream>
#include "dft/ks_tile_policy.hpp"
int main() {
  bool fitted, unconstrained, all_electron;
  std::size_t angular, atoms, nao, points;
  int major;
  while (std::cin >> fitted >> unconstrained >> all_electron >> angular >> atoms
                 >> nao >> points >> major)
    std::cout << generativeqc::dft::automatic_ks_force_tile(
        fitted, unconstrained, all_electron, angular, atoms, nao, points, major) << '\n';
}
"""
    )
    exe = required_native_cxx.build_executable(
        [unit],
        tmp_path / "force_tiles",
        compile_args=(
            "-std=c++20",
            "-I" + str(Path(__file__).resolve().parents[2] / "src"),
        ),
    )
    eligible = (1, 1, 1, 2, 96, 768, 1 << 20, 8)
    cases = [(eligible, 1024)]
    for index, value in (
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 3),
        (4, 47),
        (4, 129),
        (5, 511),
        (6, (1 << 19) - 1),
        (7, 7),
        (7, 0),
    ):
        row = list(eligible)
        row[index] = value
        cases.append((tuple(row), 256))
    for major in (8, 9, 10, 12):
        for atoms in (48, 128):
            cases.append(((1, 1, 1, 2, atoms, 512, 1 << 19, major), 1024))
    result = subprocess.run(
        [str(exe)],
        input="".join(" ".join(map(str, row)) + "\n" for row, _ in cases),
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    assert list(map(int, result.stdout.split())) == [expected for _, expected in cases]


def test_direct_scf_tiles_are_independent_of_gpu_generation(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Direct semilocal defaults need no device profile, size or FP64 capability."""
    unit = tmp_path / "direct_tiles.cpp"
    unit.write_text(
        r"""
#include <iostream>
#include "dft/ks_tile_policy.hpp"
int main() {
  bool unconstrained;
  std::size_t requested, nao, points;
  int major, ratio;
  while (std::cin >> requested >> unconstrained >> nao >> points >> major >> ratio)
    std::cout << generativeqc::dft::automatic_ks_scf_tile(
        requested, false, unconstrained, nao, points, false, major, ratio) << '\n';
}
"""
    )
    executable = required_native_cxx.build_executable(
        [unit],
        tmp_path / "direct_tiles",
        compile_args=(
            "-std=c++20",
            "-I" + str(Path(__file__).resolve().parents[2] / "src"),
        ),
    )
    cases: list[tuple[tuple[int, ...], int]] = []
    for major, ratio in ((0, 0), (6, 64), (7, 2), (8, 2), (8, 64), (9, 2), (12, 64)):
        for nao, points in ((1, 1), (511, (1 << 19) - 1), (768, 1 << 20)):
            cases.append(((0, 1, nao, points, major, ratio), 1024))
            cases.append(((0, 0, nao, points, major, ratio), 256))
        for requested in (1, 31, 256, 512, 1024, 4096):
            cases.append(((requested, 0, 1, 1, major, ratio), requested))
    inputs = "".join(" ".join(map(str, row)) + "\n" for row, _ in cases)
    result = subprocess.run(
        [str(executable)],
        input=inputs,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    assert list(map(int, result.stdout.split())) == [expected for _, expected in cases]
