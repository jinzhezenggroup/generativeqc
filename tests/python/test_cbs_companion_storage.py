"""Native companion storage must construct and move on supported host compilers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("compiler_name", ("c++", "clang++"))
def test_companion_storage_constructs_and_moves(
    tmp_path: Path, native_cxx: NativeCxx, compiler_name: str
) -> None:
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"{compiler_name} is not installed")
    source = tmp_path / "companions.cpp"
    executable = tmp_path / "companions"
    source.write_text(
        r"""
#include <type_traits>
#include <utility>
#include "methods/method.hpp"

int main() {
  using generativeqc::methods::Result;
  static_assert(std::is_default_constructible_v<Result::Companion>);
  static_assert(std::is_nothrow_move_constructible_v<Result>);
  Result result;
  if (result.hf_component || result.mp2_component) return 1;
  auto& hf = result.hf_component.emplace();
  auto& mp2 = result.mp2_component.emplace();
  if (hf.energy != 0 || mp2.energy != 0 || !hf.forces.empty() ||
      !mp2.forces.empty() || hf.correlation || mp2.correlation) return 2;
  hf.energy = -1;
  hf.forces = {1, 2, 3};
  mp2.energy = -2;
  mp2.forces = {4, 5, 6};
  mp2.correlation.emplace().reference_energy = hf.energy;
  const auto* hf_storage = hf.forces.data();
  const auto* mp2_storage = mp2.forces.data();
  auto moved = std::move(result);
  if (!moved.hf_component || !moved.mp2_component ||
      moved.hf_component->energy != -1 || moved.mp2_component->energy != -2 ||
      moved.hf_component->forces.data() != hf_storage ||
      moved.mp2_component->forces.data() != mp2_storage ||
      moved.hf_component->forces.at(2) != 3 || moved.mp2_component->forces.at(2) != 6 ||
      !moved.mp2_component->correlation ||
      moved.mp2_component->correlation->reference_energy != -1) return 3;
  moved.hf_component.reset();
  moved.mp2_component.reset();
  if (moved.hf_component || moved.mp2_component) return 4;
}
""",
        encoding="utf-8",
    )
    replace(native_cxx, compiler=compiler).build_executable(
        [source],
        executable,
        compile_args=(
            "-std=c++20",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ),
        compile_timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
