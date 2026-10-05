"""Exercise actual RHF admission with a live, inactive caller-owned cache.

Only AO counting and the direct-device resource query are host doubles. The
generated map queries, identity/cache owners, and response admission are real;
this bounded resource test does not execute physical integrals or CUDA work.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner

from tools.generate_rhf_frame_response import cpu_header

ROOT = Path(__file__).resolve().parents[2]


def test_inactive_recycle_is_charged_or_released(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    native = (ROOT / "src/hf/rhf_frame_response.cu").read_text()
    start = native.index("RHFFrameResponseResult rhf_frame_response_cuda(")
    end = native.index("\n  result.matrix_blas = options.matrix_blas;", start)
    admission = native[start:end]
    assert "result.numeric_capacity_bytes = total;" in admission
    (tmp_path / "generated_rhf_frame_response_cpu.hpp").write_text(cpu_header())
    source = tmp_path / "inactive_recycle.cpp"
    source.write_text(PREFIX + admission + "\n  return result;\n}\n}\n" + MAIN)
    executable = tmp_path / "inactive_recycle"
    compile_owner(
        compiler,
        tmp_path,
        [
            source,
            ROOT / "src/response/native_gmres.cpp",
            ROOT / "src/tensor/cpu_linalg.cpp",
        ],
        executable,
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=15, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "generated_rhf_frame_response_cpu.hpp"
#include "hf/rhf_frame_response.hpp"
#include "response/low_rank_preconditioner.hpp"
#include "posthf/capacity.hpp"

namespace generativeqc::molecule {
std::size_t ao_count(const core::System& system) noexcept {
  // Every fixture shell is s-type, so this agrees with its actual AO count.
  return system.shells.size();
}
}
namespace generativeqc::scf {
std::size_t cuda_direct_coulomb_device_bytes(std::size_t, std::size_t, std::size_t,
                                            std::size_t, std::size_t, std::size_t) {
  return 1024;
}
}
namespace generativeqc::hf {
namespace maps = scf::generated::rhf_frame;
using posthf::checked_add;
using posthf::checked_mul;
constexpr std::size_t kProviderAllowance = 96ULL << 20;
std::size_t bytes(std::size_t n) { return checked_mul(n, sizeof(double)); }
bool finite(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}
void require(bool condition, const char* reason) {
  if (!condition) throw std::invalid_argument(reason);
}
"""

MAIN = r"""
int main() {
  try {
    using namespace generativeqc;
    auto require = [](bool condition, const char* reason) {
      if (!condition) throw std::runtime_error(reason);
    };
    core::System system;
    system.electron_count = 2;
    system.atoms = {{2, {0., 0., 0.}, 0}};
    system.shells = {{0, 0, {{1., 1.}}}, {0, 0, {{2., 1.}}}};
    hf::PhysicalReference ref;
    ref.nbf = 2;
    ref.nocc = 1;
    ref.energy = -1.;
    ref.coefficients = ref.overlap = {1., 0., 0., 1.};
    ref.hcore = ref.fock = {-1., 0., 0., 1.};
    ref.density = {2., 0., 0., 0.};
    ref.weighted_density = {-2., 0., 0., 0.};
    ref.orbital_energies = {-1., 1.};
    std::vector<double> seeds(4);
    hf::RHFFrameResponseOptions options;
    options.relax_orbitals = false;
    options.matrix_blas = false;
    const auto baseline = hf::rhf_frame_response_cuda(system, ref, seeds, seeds, 0, options);

    // A correctly shaped prior frame: 256 s AOs, 128 occupied orbitals and
    // 256 electrons on 128 He atoms. Synthetic matrices are only used to
    // construct a reachable identity/cache payload, never physical forces.
    core::System old_system;
    old_system.electron_count = 256;
    for (unsigned i = 0; i < 128; ++i) {
      old_system.atoms.push_back({2, {double(i) * 3., 0., 0.}, 0});
      old_system.shells.push_back({i, 0, {{1., 1.}}});
      old_system.shells.push_back({i, 0, {{2., 1.}}});
    }
    auto old = ref;
    old.nbf = 256;
    old.nocc = 128;
    for (auto* matrix : {&old.coefficients, &old.overlap, &old.hcore, &old.fock,
                         &old.density, &old.weighted_density})
      matrix->assign(256 * 256, 0.);
    old.orbital_energies.assign(256, 1.);
    std::fill_n(old.orbital_energies.begin(), 128, -1.);
    for (unsigned i = 0; i < 256; ++i) {
      old.coefficients[i * 256 + i] = old.overlap[i * 256 + i] = 1.;
      old.fock[i * 256 + i] = old.hcore[i * 256 + i] = old.orbital_energies[i];
      old.density[i * 256 + i] = i < 128 ? 2. : 0.;
      old.weighted_density[i * 256 + i] = i < 128 ? -2. : 0.;
    }
    hf::RHFFrameResponseRecycle cache;
    const auto old_bound = hf::RHFFrameIdentity::required_storage_bytes(old_system, old) +
                           3 * sizeof(double) * 128 * 128;
    std::vector<double> direction(128 * 128, 1.);
    auto populate = [&] {
      require(cache.prepare(old_system, old, 0, false, "old", old_bound),
              "prior cache preparation failed");
      require(cache.capture(direction, direction), "prior cache capture failed");
    };
    populate();
    const auto retained = cache.storage_bytes();
    require(retained > baseline.numeric_capacity_bytes,
            "fixture must exceed the entire cache-free response allowance");
    options.recycling = &cache;

    // No additional room and one-byte-short must release; an exact fit must
    // preserve both the original identity and its populated solved direction.
    for (const auto extra : {std::size_t(0), retained, retained - 1}) {
      populate();
      options.maximum_bytes = baseline.numeric_capacity_bytes + extra;
      const auto result = hf::rhf_frame_response_cuda(system, ref, seeds, seeds, 0, options);
      const auto expected = extra >= retained ? retained : 0;
      require(cache.storage_bytes() == expected, "inactive cache release/preservation mismatch");
      require(result.recycle_capacity_bytes == expected, "inactive retained cache not charged");
      require(result.numeric_capacity_bytes == baseline.numeric_capacity_bytes + expected,
              "complete response capacity omitted or duplicated the cache");
      require(result.numeric_capacity_bytes <= options.maximum_bytes, "admitted over budget");
      if (expected) {
        require(cache.populated(), "unused cache lost its solved direction");
        require(cache.matches(old_system, old, 0, false, "old"),
                "inactive request rebound the prior cache identity");
      } else {
        require(!cache.populated(), "released cache remained populated");
      }
    }

    // Keep the existing active-relaxation path: a stale entry with no room
    // is cleared before a new owner can allocate, without increasing capacity.
    populate();
    options.relax_orbitals = true;
    options.maximum_bytes = baseline.numeric_capacity_bytes;
    const auto relaxed = hf::rhf_frame_response_cuda(system, ref, seeds, seeds, 0, options);
    require(cache.storage_bytes() == 0 && !cache.populated(),
            "active short-budget cache was retained");
    require(relaxed.recycle_capacity_bytes == 0 &&
                relaxed.numeric_capacity_bytes == baseline.numeric_capacity_bytes,
            "active fallback changed its cache-free admission");
    std::cout << "Inactive exact-fit, short/released and relaxed cache admission passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""
