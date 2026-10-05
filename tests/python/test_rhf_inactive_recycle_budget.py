"""Exercise actual RHF admission with a live, inactive caller-owned cache.

AO counting, direct-device resources and optional CUDA binding-capacity queries
are host doubles. Generated scalar-map queries, identity/cache owners, numerical
inverse storage and response admission are real; this bounded resource test does
not execute physical integrals or CUDA work.
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
    start = native.index("RHFFrameResponseResult rhf_frame_response_cuda_attempt(")
    end = native.index("\n  result.operator_hash = maps::orbital_action_hash;", start)
    admission = native[start:end]
    # Execute the real per-attempt admission under the public test signature;
    # no resident publication occurs before this extracted admission boundary.
    admission = (
        admission.replace(
            "rhf_frame_response_cuda_attempt(", "rhf_frame_response_cuda(", 1
        )
        .replace(
            ", bool& resident_values_prepared,\n    std::size_t& attempted_capacity",
            "",
            1,
        )
        .replace("  attempted_capacity = result.numeric_capacity_bytes;", "", 1)
    )
    setup_end = '            if (!inverse) result.preconditioner_reason = "unsafe DF diagonal or Cholesky";'
    assert admission.count(setup_end) == 1
    admission = admission.replace(setup_end, INVERSE_OBSERVATION + setup_end)
    assert (
        "result.numeric_capacity_bytes = checked_add(total, resident_budget);"
        in admission
    )
    (tmp_path / "generated_rhf_frame_response_cpu.hpp").write_text(cpu_header())
    source = tmp_path / "inactive_recycle.cpp"
    policy_start = native.index(
        "  if (recycling && result.prepared_contractions != prepare)"
    )
    policy_end = native.index("  result.setup_seconds =", policy_start)
    policy = native[policy_start:policy_end]
    source.write_text(
        PREFIX
        + admission
        + "\n  return result;\n}\n"
        + "bool retain_cache_after_setup(RHFFrameResponseRecycle* recycling, "
        "bool prepare, RHFFrameResponseResult& result) {\n"
        + policy
        + "return recycling != nullptr;\n}\n}\n"
        + MAIN
    )
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
// The prepared CUDA provider itself is not executed by this admission test.
// Deliberately nonzero metadata cost detects omitted table accounting; native
// descriptor tests independently qualify the exact generated table bound.
namespace generativeqc::scf::generated::rhf_frame {
std::size_t prepared_host_bytes() { return 4096; }
bool prepared_dimensions_fit(std::size_t o, std::size_t v) { return o && v; }
}
namespace generativeqc::hf {
std::size_t observed_inverse_calls{}, observed_cache_bytes{}, observed_live_lower_bound{};
using Clock = std::chrono::steady_clock;
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

INVERSE_OBSERVATION = r"""
            if (inverse) {
              ++observed_inverse_calls;
              observed_cache_bytes = options.recycling ? options.recycling->storage_bytes() : 0;
              observed_live_lower_bound = checked_add(observed_cache_bytes,
                  checked_add(preconditioner->storage_bytes(), inverse->owned_bytes()));
              // These three actual live owners alone must fit. No future
              // physical-owner or derivative allowance is counted as live here.
              require(observed_live_lower_bound <= options.maximum_bytes,
                      "live retained cache and inverse exceed response budget");
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
    options.maximum_bytes = (96ULL << 20) - 1;
    const auto baseline = hf::rhf_frame_response_cuda(system, ref, seeds, seeds, 0, options);

    const auto binding = (96ULL << 20) + scf::generated::rhf_frame::prepared_host_bytes();
    for (const auto extra : {binding - 1, binding}) {
      options.maximum_bytes = baseline.numeric_capacity_bytes + extra;
      const auto admitted = hf::rhf_frame_response_cuda(system, ref, seeds, seeds, 0, options);
      require(admitted.contraction_binding_bytes == (extra == binding ? binding : 0),
              "prepared exact/one-byte-short admission changed");
      require(admitted.numeric_capacity_bytes == baseline.numeric_capacity_bytes +
                  admitted.contraction_binding_bytes,
              "prepared table charge omitted or duplicated");
    }
    {
      // All optional cache storage, including the audited-image vector, comes
      // after the complete prepared-table charge. Test both execution policies.
      hf::RHFFrameResponseRecycle small;
      const auto retained = hf::RHFFrameIdentity::required_storage_bytes(system, ref) +
                            3 * sizeof(double);
      options.relax_orbitals = true;
      options.recycling = &small;
      for (const bool prepared : {false, true}) {
        options.maximum_bytes = baseline.numeric_capacity_bytes +
            (prepared ? binding : 0) + retained + sizeof(double);
        for (const bool setup_accepts : {false, true}) {
          small.clear();
          auto admitted = hf::rhf_frame_response_cuda(system, ref, seeds, seeds, 0, options);
          const auto charge = admitted.recycle_capacity_bytes;
          require(admitted.contraction_binding_bytes == (prepared ? binding : 0),
                  "cache displaced prepared reservation");
          require(charge == retained + sizeof(double), "cache/image charge incomplete");
          require(small.matches(system, ref, 0, prepared,
                                scf::generated::rhf_frame::orbital_action_hash),
                  "cache execution policy identity differs from admission");
          require(small.capture(std::vector<double>{1.}, std::vector<double>{2.}),
                  "fixture cache not populated");
          // Owner has already removed its binding reservation if setup refused.
          admitted.prepared_contractions = prepared && setup_accepts;
          if (prepared && !setup_accepts) {
            admitted.numeric_capacity_bytes -= admitted.contraction_binding_bytes;
            admitted.contraction_binding_bytes = 0;
          }
          const auto before = admitted.numeric_capacity_bytes;
          const auto keep = hf::retain_cache_after_setup(&small, prepared, admitted);
          const bool rejected = prepared && !setup_accepts;
          require(keep != rejected, "stale execution policy cache remained active");
          require(small.populated() != rejected, "cache clear/preserve mismatch");
          require(admitted.recycle_capacity_bytes == (rejected ? 0 : charge),
                  "cache allowance not retired after provider refusal");
          require(admitted.numeric_capacity_bytes == before - (rejected ? charge : 0),
                  "cache capacity missing or subtracted twice");
        }
        // One byte less keeps the prepared policy, but really releases a cache
        // that cannot retain the separately charged final physical image.
        options.maximum_bytes -= 1;
        const auto short_cache = hf::rhf_frame_response_cuda(system, ref, seeds, seeds, 0, options);
        require(short_cache.contraction_binding_bytes == (prepared ? binding : 0),
                "short cache changed physical binding policy");
        require(short_cache.recycle_capacity_bytes == 0 && small.storage_bytes() == 0,
                "one-byte-short active cache was retained");
      }
      options.recycling = nullptr;
      options.relax_orbitals = false;
      options.maximum_bytes = (96ULL << 20) - 1;
    }

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

    {
      // The entry payload fits, yet allocating the rank-512 inverse while
      // retaining the stale larger-frame cache exceeded the entire budget.
      constexpr std::size_t rank = 512;
      const auto make_inverse_data = [&] {
        return std::make_unique<hf::RHFFrameDFPreconditioner>(
            system, ref, rank, 1, std::vector<double>{2.}, std::vector<double>(rank, 0.));
      };
      auto raw = make_inverse_data();
      const auto raw_bytes = raw->storage_bytes();
      const auto inverse_bytes = response::LowRankPreconditioner::capacity_bytes(1, rank);
      options.maximum_bytes = baseline.numeric_capacity_bytes + raw_bytes + inverse_bytes;
      options.relax_orbitals = options.df_preconditioning = true;
      require(retained + raw_bytes <= options.maximum_bytes, "entry cache/raw payload must fit");
      hf::observed_inverse_calls = 0;
      const auto stale = hf::rhf_frame_response_cuda(
          system, ref, seeds, seeds, 0, options, std::move(raw));
      require(stale.df_preconditioned && hf::observed_inverse_calls == 1,
              "stale cache prevented otherwise admitted inverse");
      require(hf::observed_cache_bytes == 0 && cache.storage_bytes() == 0,
              "stale cache survived optional inverse allocation");
      require(stale.numeric_capacity_bytes == options.maximum_bytes,
              "stale-cache release changed exact inverse admission");

      // Matching populated cache remains useful when the complete combination
      // fits. If inverse+retained cache is one byte short, refuse the inverse
      // rather than allocating across the live-cache reservation.
      const auto current_bound = hf::RHFFrameIdentity::required_storage_bytes(system, ref) +
                                 3 * sizeof(double);
      for (const bool combined_fits : {true, false}) {
        cache.clear();
        require(cache.prepare(system, ref, 0, false,
                              scf::generated::rhf_frame::orbital_action_hash, current_bound),
                "matching cache preparation failed");
        require(cache.capture(std::vector<double>{1.}, std::vector<double>{2.}),
                "matching cache capture failed");
        const auto current = cache.storage_bytes();
        options.maximum_bytes = baseline.numeric_capacity_bytes + raw_bytes + inverse_bytes +
                                current;
        if (combined_fits) options.maximum_bytes += sizeof(double);
        else --options.maximum_bytes;
        hf::observed_inverse_calls = 0;
        const auto matching = hf::rhf_frame_response_cuda(
            system, ref, seeds, seeds, 0, options, make_inverse_data());
        require(matching.df_preconditioned == combined_fits &&
                    hf::observed_inverse_calls == std::size_t(combined_fits),
                "inverse ignored the already-live matching cache charge");
        require(cache.populated() && cache.storage_bytes() == current,
                "useful matching cache was discarded");
        require(matching.recycle_capacity_bytes == current + sizeof(double),
                "matching cache/image charge omitted or duplicated");
        require(matching.numeric_capacity_bytes == baseline.numeric_capacity_bytes +
                    matching.preconditioner_capacity_bytes + matching.recycle_capacity_bytes,
                "temporary retained-cache charge was not replaced exactly once");
        if (combined_fits)
          require(hf::observed_cache_bytes == current, "matching cache not live during setup");
        else
          require(matching.preconditioner_fallback, "refused inverse fallback not reported");
      }
      options.relax_orbitals = options.df_preconditioning = false;
    }

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
    std::cout << "Prepared exact/short, policy fallback and inactive cache admission passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""
