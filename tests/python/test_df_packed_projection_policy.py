"""Exercise portable packed-projection admission without a CUDA installation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.fixture(scope="module")
def projection_policy(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    """Compile the production policy with the repository's cached host toolchain."""
    directory = tmp_path_factory.mktemp("packed-projection-policy")
    source = directory / "query.cpp"
    source.write_text(
        r"""
#include <iostream>
#include "scf/df_exchange_policy.hpp"
int main() {
  using namespace generativeqc::scf;
  std::size_t n, a, rank, capacity;
  int ratio;
  while (std::cin >> n >> a >> rank >> capacity >> ratio)
    std::cout << df_packed_projection_preferred(n, a, rank, capacity, ratio) << " "
              << df_packed_projection_policy_valid() << " "
              << df_packed_projection_auto_requested() << " "
              << df_packed_projection_blas_requested() << "\n";
}
"""
    )
    return required_native_cxx.build_executable(
        [source],
        directory / "query",
        compile_args=(
            "-std=c++20",
            "-I" + str(Path(__file__).resolve().parents[2] / "src"),
        ),
    )


@pytest.mark.parametrize("policy", [None, "auto", "generated", "blas", "unknown"])
def test_capability_work_capacity_and_explicit_controls(
    projection_policy: Path, policy: str | None
) -> None:
    """Large FP64 work can select BLAS; unknown/slow capabilities and tight scratch cannot."""
    shapes = [
        (384, 1808, 80, 32 * 384 * 1808, 2),
        (768, 3616, 160, 8 * 768 * 3616, 2),
        (391, 1813, 83, 32 * 391 * 1813, 2),
        (768, 3616, 160, 8 * 768 * 3616 - 1, 2),
        (768, 3616, 160, 32 * 768 * 3616, 64),
        (768, 3616, 160, 32 * 768 * 3616, 0),
        (768, 3616, 32, 32 * 768 * 3616, 2),
        (37, 67, 11, 128 * 37 * 67, 2),
        (0, 0, 0, 0, 2),
        ((1 << 64) - 1, (1 << 64) - 1, 64, (1 << 64) - 1, 2),
    ]
    env = dict(os.environ)
    env.pop("GENERATIVEQC_DF_PACKED_PROJECTION", None)
    if policy is not None:
        env["GENERATIVEQC_DF_PACKED_PROJECTION"] = policy
    result = subprocess.run(
        [str(projection_policy)],
        input="".join(" ".join(map(str, s)) + "\n" for s in shapes),
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows = [list(map(int, line.split())) for line in result.stdout.splitlines()]
    assert [row[0] for row in rows] == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert all(
        row[1:] == [policy != "unknown", policy in (None, "auto"), policy == "blas"]
        for row in rows
    )
