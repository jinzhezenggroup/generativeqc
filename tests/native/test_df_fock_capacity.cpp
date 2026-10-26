#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "scf/df_fock_capacity.hpp"
#include "scf/fock_prepared.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void test_resolver_envelopes() {
  using namespace generativeqc::scf;
  // Exercise the executable policy, including clipping, subbudgeting and
  // promotion. These are independently resolved records, not expectations
  // copied from the capacity composition's arithmetic.
  for (const auto n : {std::size_t{2}, std::size_t{96}, std::size_t{768}})
    for (const auto rank : {std::size_t{0}, n / 2})
      for (const auto source : {std::size_t{64} << 10, std::size_t{100} << 20})
        for (const auto exact_bound : {std::size_t{0}, std::size_t{16} << 20})
          for (const auto requested :
               {std::size_t{0}, std::size_t{1}, std::size_t{16} << 20, std::size_t{2} << 30})
            for (const bool forces : {false, true})
              for (const bool automatic : {false, true}) {
                const DfBudgetWorkload workload{n, n == 768 ? 3712U : 256U, 96, 1, 0, forces};
                const auto bound = df_fock_device_capacity(workload, requested, exact_bound, source,
                                                           rank, rank == 0, automatic);
                if (requested)
                  require(bound.retained_response_bytes == requested,
                          "explicit DF cap was enlarged or Direct was charged twice");
                for (const auto free : {1ULL, 400ULL << 20, 8ULL << 30, 30ULL << 30, 1ULL << 40})
                  for (const bool live : {false, true}) {
                    const auto base = resolve_df_budget(
                        workload, {free, std::max(free, 32ULL << 30), live}, requested);
                    if (!base.feasible) continue;
                    const auto exact = std::min(exact_bound, base.total_bytes / 2);
                    auto resolved = resolve_df_subbudget(workload, base, exact);
                    if (!resolved.feasible) continue;
                    if (automatic)
                      resolved = resolve_method_owned_df_resident_budget(
                          resolved, workload.nbf, workload.naux, rank, source, rank == 0);
                    require(exact <= bound.retained_response_bytes &&
                                resolved.total_bytes <= bound.retained_response_bytes - exact,
                            "mixed Direct/DF resolved allowance exceeded its native bound");
                    require(exact + source + workload.naux * workload.naux * sizeof(double) <=
                                bound.source_setup_bytes,
                            "initial DF source metric exceeded the separate setup phase");
                  }
              }
}

void test_setup_and_rejection() {
  using namespace generativeqc::scf;
  const DfBudgetWorkload shape{2, 256, 2, 1, 0, true};
  // A tiny explicit value cap does not bound the earlier source factory's
  // metric. Keep this failed-later preparation visible to the fleet planner.
  const auto capacity = df_fock_device_capacity(shape, 1024, 256, 4096, 0, true, true);
  require(capacity.retained_response_bytes == 1024 &&
              capacity.source_setup_bytes == 256 + 4096 + 256 * 256 * sizeof(double),
          "source setup was clipped to the later value/response cap");
  for (std::size_t field = 0; field < 5; ++field) {
    auto invalid = shape;
    if (field == 0) invalid.nbf = 0;
    if (field == 1) invalid.naux = 0;
    if (field == 2) invalid.atoms = 0;
    if (field == 3) invalid.batch = 2;
    if (field == 4) invalid.diis_history = 1;
    bool refused = false;
    try {
      (void)df_fock_device_capacity(invalid, 1024, 0, 4096, 0, true, true);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    require(refused, "non-method-owned DF shape was accepted");
  }
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  for (const bool metric_overflow : {false, true}) {
    auto invalid = shape;
    if (metric_overflow) invalid.naux = maximum;
    bool refused = false;
    try {
      (void)df_fock_device_capacity(invalid, 1024, 0, metric_overflow ? 0 : maximum, 0, true,
                                    false);
    } catch (const std::overflow_error&) {
      refused = true;
    }
    require(refused, "overflowing source setup wrapped into a small capacity");
  }
}

void test_native_system_query() {
  using namespace generativeqc;
  using namespace generativeqc::scf;
  core::System orbital;
  orbital.atoms = {{1, {0, 0, 0}}, {1, {0, 0, 1.4}}};
  orbital.shells = {{0, 0, {{0.8, 1.0}}}, {1, 0, {{0.6, 1.0}}}};
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  const auto exact = resolve_fock_build(spec, FockBackend::Cuda);
  const auto direct = prepared_fock_device_capacity(orbital, nullptr, exact, 1024, 512);
  require(direct.retained_response_bytes == 512 && direct.setup_bytes == 512,
          "Direct-only query did not respect its allocator leaf");
  spec.coulomb.approximation = spec.exchange.approximation = FockApproximation::DensityFitted;
  const auto fitted = resolve_fock_build(spec, FockBackend::Cuda);
  // The two factories use the same default auxiliary topology. This query
  // must run in the CPU build without CUDA discovery or scientific work.
  const auto implicit_aux = prepared_fock_device_capacity(orbital, nullptr, fitted, 1024, 0, 1);
  const auto explicit_aux = prepared_fock_device_capacity(orbital, &orbital, fitted, 1024, 0, 1);
  require(implicit_aux.retained_response_bytes == explicit_aux.retained_response_bytes &&
              implicit_aux.setup_bytes == explicit_aux.setup_bytes &&
              implicit_aux.retained_response_bytes == 1024 && implicit_aux.setup_bytes >= 1024,
          "default DF source topology changed capacity");
  auto larger_aux = orbital;
  larger_aux.shells.insert(larger_aux.shells.end(), orbital.shells.begin(), orbital.shells.end());
  const auto larger = prepared_fock_device_capacity(orbital, &larger_aux, fitted, 1, 0, 1);
  const auto smaller = prepared_fock_device_capacity(orbital, &orbital, fitted, 1, 0, 1);
  require(larger.setup_bytes > smaller.setup_bytes,
          "native source setup ignored the auxiliary topology");
  bool refused = false;
  try {
    (void)prepared_fock_device_capacity(orbital, nullptr, exact, 1024, 0);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "Direct provider without an allocator bound was accepted");
}
}  // namespace

int main() {
  test_resolver_envelopes();
  test_setup_and_rejection();
  test_native_system_query();
}
