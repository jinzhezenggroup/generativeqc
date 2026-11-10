"""Execute automatic spatial admission at its scientific and resource boundaries."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def spatial_policy_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    """Compile the production policy directly without CUDA or source extraction."""
    folder = tmp_path_factory.mktemp("ks-spatial-policy")
    source, executable = folder / "spatial.cpp", folder / "spatial"
    source.write_text(
        r"""
#include <cstdlib>
#include <iostream>
#include "dft/ks_tile_policy.hpp"

int main(int count, char** arguments) {
  if (count != 9) return 2;
  std::cout << generativeqc::dft::automatic_ks_spatial_bisection(
      std::atoi(arguments[1]) != 0, std::atoi(arguments[2]) != 0,
      std::atoi(arguments[3]) != 0, std::strtoull(arguments[4], nullptr, 10),
      std::strtoull(arguments[5], nullptr, 10), std::strtoull(arguments[6], nullptr, 10),
      std::atoi(arguments[7]), std::atoi(arguments[8]));
}
"""
    )
    required_native_cxx.build_executable(
        [source], executable, compile_args=["-std=c++20", f"-I{ROOT / 'src'}"]
    )
    return executable


@pytest.mark.parametrize(
    (
        "restricted_fp64",
        "unconstrained",
        "range_separated",
        "ao_jets",
        "aos",
        "points",
        "major",
        "ratio",
        "expected",
    ),
    [
        (True, True, False, 4, 1088, 1078336, 12, 64, True),
        (True, True, False, 4, 544, 539168, 12, 64, True),
        (True, True, False, 4, 768, 1078336, 8, 64, True),
        (True, True, True, 4, 768, 1078336, 12, 64, False),
        (True, True, False, 1, 768, 1078336, 12, 64, False),
        (True, True, False, 3, 768, 1078336, 12, 64, False),
        (True, True, False, 4, 384, 539168, 12, 64, False),
        (True, True, False, 4, 96, 134792, 12, 64, False),
        (True, True, False, 4, 512, 524288, 12, 64, True),
        (True, True, False, 4, 511, 524288, 12, 64, False),
        (True, True, False, 4, 512, 524287, 12, 64, False),
        (False, True, False, 4, 1088, 1078336, 12, 64, False),
        (True, False, False, 4, 1088, 1078336, 12, 64, False),
        (True, True, False, 4, 1088, 1078336, 8, 2, False),
        (True, True, False, 4, 1088, 1078336, 9, 2, False),
        (True, True, False, 4, 1088, 1078336, 7, 64, False),
        (True, True, False, 4, 1088, 1078336, 12, 0, False),
        (True, True, False, 4, 1088, 1078336, 0, 0, False),
    ],
)
def test_automatic_spatial_domain(
    spatial_policy_probe: Path,
    restricted_fp64: bool,
    unconstrained: bool,
    range_separated: bool,
    ao_jets: int,
    aos: int,
    points: int,
    major: int,
    ratio: int,
    expected: bool,
) -> None:
    """Protect threshold, precision/spin, operator, device and budget fallbacks."""
    result = subprocess.run(
        [
            str(spatial_policy_probe),
            str(int(restricted_fp64)),
            str(int(unconstrained)),
            str(int(range_separated)),
            str(ao_jets),
            str(aos),
            str(points),
            str(major),
            str(ratio),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout == str(int(expected))
