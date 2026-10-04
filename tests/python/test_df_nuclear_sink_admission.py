"""Host-only allocation-peak checks of the production DF nuclear sink setup.

Extract the real packer, Arena and constructor, link real basis.cpp, and replace
only CUDA runtime allocation/copy calls. This verifies metadata admission and
CPU packing, not CUDA execution, kernels or synchronization.
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = r"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#include "molecule/basis.hpp"
#include "scf/cuda_df_nuclear_sink.hpp"

namespace allocation_probe {
bool enabled = false;
std::size_t live = 0, peak = 0, device_live = 0, combined_peak = 0;
struct alignas(std::max_align_t) Header { std::size_t bytes; bool tracked; };
void observe() {
  peak = std::max(peak, live);
  combined_peak = std::max(combined_peak, live + device_live);
}
void start() {
  if (live || device_live) std::abort();
  peak = combined_peak = 0;
  enabled = true;
}
void* allocate(std::size_t n) {
  auto* p = static_cast<Header*>(std::malloc(sizeof(Header) + n));
  if (!p) throw std::bad_alloc();
  *p = {n, enabled};
  if (enabled) { live += n; observe(); }
  return p + 1;
}
void release(void* value) noexcept {
  if (!value) return;
  auto* p = static_cast<Header*>(value) - 1;
  if (p->tracked) live -= p->bytes;
  std::free(p);
}
}
void* operator new(std::size_t n) { return allocation_probe::allocate(n); }
void* operator new[](std::size_t n) { return allocation_probe::allocate(n); }
void operator delete(void* p) noexcept { allocation_probe::release(p); }
void operator delete[](void* p) noexcept { allocation_probe::release(p); }
void operator delete(void* p, std::size_t) noexcept { allocation_probe::release(p); }
void operator delete[](void* p, std::size_t) noexcept { allocation_probe::release(p); }

using cudaError_t = int;
using cudaStream_t = void*;
constexpr int cudaSuccess = 0, cudaStreamNonBlocking = 1, cudaMemcpyHostToDevice = 1;
namespace fake_cuda { unsigned allocations = 0, streams = 0; }
int cudaGetDevice(int* device) { *device = 0; return cudaSuccess; }
int cudaSetDevice(int) { return cudaSuccess; }
int cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned) {
  ++fake_cuda::streams;
  *stream = reinterpret_cast<void*>(1);
  return cudaSuccess;
}
int cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
int cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
int cudaMemcpy(void* dst, const void* src, std::size_t n, int) {
  std::memcpy(dst, src, n); return cudaSuccess;
}
int cudaMemsetAsync(void* dst, int value, std::size_t n, cudaStream_t) {
  std::memset(dst, value, n); return cudaSuccess;
}
const char* cudaGetErrorString(int) { return "mock CUDA failure"; }
namespace generativeqc::runtime {
int resource_cuda_malloc(void** out, std::size_t bytes) {
  auto* p = static_cast<allocation_probe::Header*>(
      std::malloc(sizeof(allocation_probe::Header) + bytes));
  if (!p) return 1;
  *p = {bytes, false};
  *out = p + 1;
  ++fake_cuda::allocations;
  allocation_probe::device_live += bytes;
  allocation_probe::observe();
  return cudaSuccess;
}
int resource_cuda_free(void* out) {
  auto* p = static_cast<allocation_probe::Header*>(out) - 1;
  allocation_probe::device_live -= p->bytes;
  std::free(p);
  return cudaSuccess;
}
}
namespace generativeqc::scf {
"""

# Frozen pre-reservation loop from 0aa34782; all eight packed arrays must retain
# their exact bytes after the resource-only repair. The basis expansion oracle
# and numerical derivative gates remain in the existing auxiliary-g tests.
LEGACY_PACK = r"""
HostBasis legacy_pack(const core::System& system, unsigned maximum_angular = 3,
               std::size_t expansion_terms = molecule::kMaximumAoExpansionTerms) {
  HostBasis h;
  h.primitive_offsets.push_back(0);
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > maximum_angular || shell.atom_index >= system.atoms.size())
      throw std::invalid_argument("generated DF gradient shell exceeds its admitted basis role");
    const auto si = static_cast<std::int32_t>(h.shell_atoms.size());
    h.shell_atoms.push_back(shell.atom_index);
    for (const auto& p : shell.primitives) {
      h.exponents.push_back(p.exponent);
      h.coefficients.push_back(p.coefficient);
    }
    h.primitive_offsets.push_back(h.exponents.size());
    for (const auto& expansion :
         molecule::ao_expansions(shell.angular_momentum, system.basis_representation)) {
      h.ao_shells.push_back(si);
      if (expansion.size() > expansion_terms)
        throw std::invalid_argument("DF gradient AO expansion exceeds its metadata stride");
      h.term_counts.push_back(expansion.size());
      for (std::size_t term = 0; term < expansion_terms; ++term) {
        if (term < expansion.size()) {
          const auto& item = expansion[term];
          for (auto power : item.component) h.term_angular.push_back(power);
          h.term_coefficients.push_back(
              item.coefficient * molecule::cartesian_component_normalization(item.component));
        } else {
          h.term_angular.insert(h.term_angular.end(), 3, 0);
          h.term_coefficients.push_back(0.0);
        }
      }
    }
  }
  return h;
}

"""

DRIVER = r"""
}  // namespace generativeqc::scf
using namespace generativeqc;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
core::System system(unsigned angular, unsigned representation,
                    unsigned primitives, unsigned shells) {
  core::System result;
  result.basis_representation = static_cast<generativeqc_basis_representation>(representation);
  result.atoms.push_back({1, {0.13, -0.27, 0.41}, 0});
  for (unsigned s = 0; s < shells; ++s) {
    core::Shell shell{0, angular, {}};
    for (unsigned p = 0; p < primitives; ++p)
      shell.primitives.push_back({0.5 + .01 * (p + s), (p % 2 ? -.7 : 1.0) / primitives});
    result.shells.push_back(std::move(shell));
  }
  return result;
}
template<class T> void equal(const std::vector<T>& a, const std::vector<T>& b) {
  require(a.size() == b.size(), "packed array size changed");
  require(a.empty() || !std::memcmp(a.data(), b.data(), a.size() * sizeof(T)),
          "packed scientific bytes changed");
}
void check_pack(const core::System& system, unsigned role, std::size_t terms) {
  const auto a = scf::pack(system, role, terms), b = scf::legacy_pack(system, role, terms);
  equal(a.shell_atoms, b.shell_atoms); equal(a.ao_shells, b.ao_shells);
  equal(a.primitive_offsets, b.primitive_offsets); equal(a.term_counts, b.term_counts);
  equal(a.term_angular, b.term_angular); equal(a.term_coefficients, b.term_coefficients);
  equal(a.exponents, b.exponents); equal(a.coefficients, b.coefficients);
}
int main(int argc, char** argv) {
  if (argc != 7) return 99;
  const auto ol = std::stoul(argv[1]), xl = std::stoul(argv[2]);
  const auto ro = std::stoul(argv[3]), rx = std::stoul(argv[4]);
  const auto primitives = std::stoul(argv[5]), shells = std::stoul(argv[6]);
  try {
    const auto orbital = system(ol, ro, primitives, shells);
    const auto auxiliary = system(xl, rx, 1, 1);
    const auto terms = xl == 4 ? molecule::kMaximumAuxiliaryAoExpansionTerms
                              : molecule::kMaximumAoExpansionTerms;
    check_pack(orbital, 3, terms); check_pack(auxiliary, 4, terms);
    std::size_t numeric = 0, host_bound = 0, host_peak = 0, combined_peak = 0;
    {
      scf::CudaDfNuclearSink sink(0, orbital, auxiliary, 1ULL << 30);
      numeric = sink.numeric_capacity_bytes();
      host_bound = numeric - sink.resources().device_bytes;
      host_peak = allocation_probe::peak;
      combined_peak = allocation_probe::combined_peak;
      std::cout << "host_bound=" << host_bound << " host_peak=" << host_peak
                << " combined_peak=" << combined_peak << " numeric=" << numeric << '\n';
      require(host_peak <= host_bound, "host construction exceeded admitted bound");
      require(combined_peak <= numeric, "host/device overlap exceeded admitted bound");
      require(fake_cuda::allocations == 18, "sink allocation topology changed");
    }
    require(!allocation_probe::live && !allocation_probe::device_live,
            "successful construction leaked storage");
    {
      scf::CudaDfNuclearSink sink(0, orbital, auxiliary, numeric);
      require(sink.numeric_capacity_bytes() == numeric, "exact budget changed admission");
      require(allocation_probe::combined_peak <= numeric, "exact budget exceeded in setup");
    }
    bool rejected = false;
    try { scf::CudaDfNuclearSink sink(0, orbital, auxiliary, numeric - 1); }
    catch (const std::bad_alloc&) { rejected = true; }
    allocation_probe::enabled = false;
    require(rejected, "one-byte-short complete budget was admitted");
    require(allocation_probe::combined_peak <= numeric - 1,
            "rejected construction exceeded its complete budget");
    require(!allocation_probe::live && !allocation_probe::device_live,
            "failed construction leaked storage");
    for (auto budget : {host_bound - 1, host_bound}) {
      fake_cuda::allocations = fake_cuda::streams = 0;
      rejected = false;
      try { scf::CudaDfNuclearSink sink(0, orbital, auxiliary, budget); }
      catch (const std::length_error&) { rejected = true; }
      require(rejected && !fake_cuda::allocations && !fake_cuda::streams,
              "host-only insufficient budget reached CUDA setup");
    }
    for (auto invalid : {std::size_t{0}, std::numeric_limits<std::size_t>::max()}) {
      bool rejected = false;
      try { (void)scf::pack(orbital, 3, invalid); }
      catch (const std::length_error&) { rejected = true; }
      require(rejected, "overflowed/zero metadata stride was not rejected");
    }
  } catch (const std::exception& error) {
    allocation_probe::enabled = false;
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""


def extract(source: str, begin: str, end: str) -> str:
    first = source.index(begin)
    return source[first : source.index(end, first)]


@pytest.fixture(scope="module")
def sink_admission_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires a C++20 compiler and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True, timeout=10)
    source = (ROOT / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    basis = (ROOT / "src/runtime/cuda_gaussian_products.cuh").read_text()
    view = extract(basis, "struct BasisView {", "/** One real factor")
    view += "using DfDerivativeBasisView = BasisView;\n"
    failure = extract(source, "struct CudaFailure {", "/** Prepare shell envelopes")
    guard = extract(
        source, "void check(cudaError_t status)", "/** Diagnostic staging only"
    )
    packing = extract(
        source, "struct HostBasis {", "/** Reuse only drained diagnostic storage"
    )
    constructor = extract(
        source, "struct CudaDfNuclearSink::Impl {", "void CudaDfNuclearSink::consume"
    )
    # Exclude the fixed C++ owner objects, just as the numeric-capacity API does;
    # trace every heap-backed metadata buffer from the first Arena reservation.
    marker = "auto& arena = *state->arena;"
    assert constructor.count(marker) == 1
    constructor = constructor.replace(marker, marker + "\nallocation_probe::start();")
    constructor = constructor.replace(
        "implementation_ = std::move(state);",
        "allocation_probe::enabled = false;\nimplementation_ = std::move(state);",
    )
    accessors = extract(
        source,
        "std::size_t CudaDfNuclearSink::numeric_capacity_bytes()",
        "generativeqc_status execute_cuda_df_gradient(",
    )
    directory = tmp_path_factory.mktemp("df-nuclear-sink-admission")
    unit, executable = directory / "probe.cpp", directory / "probe"
    unit.write_text(
        PREFIX
        + view
        + failure
        + guard
        + packing
        + LEGACY_PACK
        + constructor
        + accessors
        + DRIVER
    )
    objects = []
    for source_file in (unit, ROOT / "src/molecule/basis.cpp"):
        obj = directory / (source_file.stem + ".o")
        objects.append(str(obj))
        subprocess.run(
            [
                cache,
                compiler,
                "-std=c++20",
                "-O2",
                "-fsanitize=undefined",
                "-I" + str(ROOT / "src"),
                "-I" + str(ROOT / "include"),
                "-c",
                str(source_file),
                "-o",
                str(obj),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    subprocess.run(
        [compiler, "-fsanitize=undefined", *objects, "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


def run_probe(probe: Path, args: tuple[int, ...]) -> None:
    result = subprocess.run(
        [str(probe), *map(str, args)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "ol,xl,ro,rx", list(itertools.product(range(4), range(5), range(2), range(2)))
)
def test_all_supported_shell_workspace_peaks(
    sink_admission_probe: Path, ol: int, xl: int, ro: int, rx: int
) -> None:
    # Minimal spherical d/f/g cases expose expansion temporaries which the old
    # 2x packed-size bound missed even without long primitive contractions.
    run_probe(sink_admission_probe, (ol, xl, ro, rx, 1, 1))


@pytest.mark.parametrize("primitives,shells", [(257, 1), (513, 1), (17, 17), (1, 33)])
@pytest.mark.parametrize("ro,rx,xl", [(0, 0, 0), (1, 1, 3), (0, 1, 4), (1, 0, 4)])
def test_primitive_and_metadata_growth_boundaries(
    sink_admission_probe: Path, primitives: int, shells: int, ro: int, rx: int, xl: int
) -> None:
    run_probe(sink_admission_probe, (0, xl, ro, rx, primitives, shells))
