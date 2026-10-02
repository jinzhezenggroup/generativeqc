"""Exercise the production checkpoint export and its density allocation contract."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def checkpoint_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the checkpoint export contract")
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/methods/mp2_method.cpp").read_text()
    begin = source.index("      if (retained_warm_state) {")
    end = source.index("      const auto& ref = *hf.reference;", begin)
    directory = tmp_path_factory.mktemp("mp2_checkpoint_export")
    cpp = directory / "export.cpp"
    cpp.write_text(
        r"""
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <utility>
#include "core/types.hpp"
#include "scf/types.hpp"
#include "scf/warm_state.hpp"
#include "methods/correlated_warm_reference.hpp"

// Count the four-double H2 density allocations only while publishing the seed.
// The six-double coordinate allocation is deliberately a different size.
bool recording = false;
std::size_t density_allocations = 0;
void* operator new(std::size_t size) {
  if (recording && size == 4 * sizeof(double)) ++density_allocations;
  if (void* p = std::malloc(size)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

using namespace generativeqc;
namespace warm_reference = generativeqc::methods::warm_reference;
scf::HfWarmState capture(scf::ScfResult& hf, const core::System& system_) {
  scf::HfWarmState output;
  auto* retained_warm_state = &output;
"""
        + source[begin:end]
        + r"""
  return output;
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const bool iterative = std::atoi(argv[1]) != 0;
  core::System system;
  system.atoms = {{1, {0., 0., -.7}}, {1, {0., 0., .7}}};
  const std::vector<double> density{.6, .6, .6, .6};
  auto reference = std::make_shared<scf::PhysicalReference>();
  reference->density = density;
  scf::ScfResult hf;
  hf.reference = reference;
  hf.converged = true;
  hf.energy = -1.1;
  hf.iterations = 3;
  if (iterative) hf.density = density;
  const auto* old_data = hf.density.data();
  recording = true;
  const auto state = capture(hf, system);
  recording = false;
  if (state.density != density || state.coordinates !=
      std::vector<double>{0., 0., -.7, 0., 0., .7}) {
    std::cerr << "incomplete checkpoint\n";
    return 1;
  }
  if (reference->density != density || state.energy != hf.energy ||
      state.iterations != 3 || !std::isfinite(state.energy)) return 3;
  for (double x : state.density) if (!std::isfinite(x)) return 4;
  if (density_allocations != (iterative ? 0 : 1)) {
    std::cerr << "unexpected density allocations: " << density_allocations << '\n';
    return 5;
  }
  if (iterative &&
      (state.density.data() != old_data || !hf.density.empty())) return 6;
  if (!iterative && state.density.data() == reference->density.data()) return 7;
}
"""
    )
    obj = directory / "export.o"
    binary = directory / "export"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
            "-c",
            str(cpp),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        [compiler, str(obj), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize("iterative", [False, True])
def test_mp2_checkpoint_density_owns_only_one_complete_candidate(
    checkpoint_export: Path, iterative: bool
) -> None:
    result = subprocess.run(
        [str(checkpoint_export), str(int(iterative))],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
