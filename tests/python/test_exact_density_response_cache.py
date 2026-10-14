"""Execute exact key, finite-result and bounded-storage cache contracts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_exact_density_response_cache(native_cxx: NativeCxx, tmp_path: Path) -> None:
    source = tmp_path / "probe.cpp"
    source.write_text(r"""
#include "methods/exact_density_response_cache.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
using generativeqc::methods::detail::ExactDensityResponseCache;
// Inject failure only into the retained array allocation, not the input vectors.
bool reject_array_allocation = false;
void* operator new[](std::size_t size) {
  if (!reject_array_allocation)
    if (auto* memory = std::malloc(size)) return memory;
  throw std::bad_alloc();
}
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }
int main() {
  ExactDensityResponseCache cache;
  ExactDensityResponseCache::Blocks density{{1, 0, 0, 1}, {2, 0, 0, 2}};
  std::vector<double> force{.1, -.2, .3};
  const auto bytes = (8 + 3) * sizeof(double);
  assert(!cache.matches(density, 3, bytes));
  assert(cache.store(density, force, bytes));
  assert(cache.bytes() == bytes && cache.matches(density, 3, bytes));
  assert(!cache.matches(density, 3, bytes - 1));
  assert(!cache.matches(density, 2, bytes));
  auto different = density;
  different[1][0] = std::nextafter(different[1][0], 3.);
  assert(!cache.matches(different, 3, bytes));
  different = density; different[1][1] = -0.;
  assert(!cache.matches(different, 3, bytes));
  different = density; different.pop_back();
  assert(!cache.matches(different, 3, bytes));
  different = density; different[0].pop_back();
  assert(!cache.matches(different, 3, bytes));
  force[0] = 8.;
  assert(cache.response()[0] == .1); // Retention owns an immutable copy.
  assert(!cache.store(density, force, bytes - 1));
  assert(cache.bytes() == 0 && cache.response().empty());
  assert(cache.store(density, force, bytes));
  different = density; different[0][0] = std::numeric_limits<double>::quiet_NaN();
  assert(!cache.store(different, force, bytes));
  assert(!cache.matches(density, 3, bytes));
  force[0] = std::numeric_limits<double>::infinity();
  assert(!cache.store(density, force, bytes));
  assert(cache.bytes() == 0);
  force[0] = .1;
  assert(cache.store(density, force, bytes));
  cache.clear();
  assert(cache.bytes() == 0 && !cache.matches(density, 3, bytes));
  assert(cache.store(density, force, bytes));
  reject_array_allocation = true;
  assert(!cache.store(density, force, bytes));
  assert(cache.bytes() == 0 && cache.response().empty());
  reject_array_allocation = false;
  assert(cache.store(density, force, bytes)); // Refusal does not poison recovery.
}
""")
    binary = tmp_path / "probe"
    native_cxx.build_executable(
        [source],
        binary,
        compile_args=[
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "src"),
        ],
    )
    subprocess.run([str(binary)], check=True, timeout=10)
