// Exercise owner authentication, bounded refusal and teardown after the HF handle dies.
// Complete independent energy/force oracles live in test_full_df_cbs_native.py.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "cc/df_source.hpp"
#include "cc/df_source_response.hpp"
#include "hf/df_reference.hpp"
#include "molecule/basis.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
  try {
    require(std::getenv("SLURM_JOB_ID"), "probe requires a Slurm allocation");
    using namespace generativeqc;
    core::System orbital;
    orbital.atoms = {{1, {0, 0, -0.7}}, {1, {0, 0, 0.7}}};
    orbital.shells = {{0, 0, {{1.24, 1.0}}}, {1, 0, {{1.24, 1.0}}}};
    std::string detail;
    require(molecule::validate_and_normalize(orbital, detail) == GENERATIVEQC_STATUS_SUCCESS,
            "invalid test orbital basis");
    auto auxiliary = orbital;
    auxiliary.shells.push_back({0, 0, {{0.32, 1.0}}});
    auxiliary.shells.push_back({1, 0, {{0.32, 1.0}}});
    require(molecule::validate_and_normalize(auxiliary, detail) == GENERATIVEQC_STATUS_SUCCESS,
            "invalid test auxiliary basis");
    constexpr std::size_t budget = 1ULL << 30;
    scf::ScfOptions options;
    options.reference_memory_budget_bytes = budget;
    options.energy_tolerance = 1e-12;
    options.density_tolerance = 1e-11;
    auto owner = std::make_shared<hf::DFReference>();
    const auto solved = owner->solve(orbital, auxiliary, options, 0, nullptr);
    require(solved.converged && solved.reference, "DF reference did not converge");
    auto build = [&](const core::System& aux, const hf::PhysicalReference& reference,
                     double threshold, bool retain, std::shared_ptr<hf::DFReference> shared) {
      return cc::build_df_source_cuda(orbital, aux, reference, budget - owner->storage_bytes(),
                                      threshold, 0, 0, retain, nullptr, false, std::move(shared));
    };
    auto cached = build(auxiliary, *solved.reference, 1e-10, true, owner);
    const auto fresh = build(auxiliary, *solved.reference, 1e-10, false, {});
    require(cached.shared_reference_bytes == owner->storage_bytes() &&
                cached.retained_source_bytes > cached.shared_reference_bytes &&
                cached.metric_staging_bytes == 0 && fresh.metric_staging_bytes > 0,
            "source/metric ownership was not reused");
    const std::vector<double>* left[] = {&cached.boo,  &cached.bov,  &cached.bvv,  &cached.ovov,
                                         &cached.ovvo, &cached.oovv, &cached.ovoo, &cached.oooo};
    const std::vector<double>* right[] = {&fresh.boo,  &fresh.bov,  &fresh.bvv,  &fresh.ovov,
                                          &fresh.ovvo, &fresh.oovv, &fresh.ovoo, &fresh.oooo};
    double maximum = 0;
    for (unsigned block = 0; block < 8; ++block) {
      require(left[block]->size() == right[block]->size(), "factor shape changed");
      for (std::size_t i = 0; i < left[block]->size(); ++i)
        maximum = std::max(maximum, std::abs((*left[block])[i] - (*right[block])[i]));
    }
    require(maximum < 1e-11, "shared source changed physical factors");
    auto changed = auxiliary;
    changed.shells.back().primitives.front().exponent *= 1.01;
    const auto other_aux = build(changed, *solved.reference, 1e-10, false, owner);
    const auto other_threshold = build(auxiliary, *solved.reference, 1e-9, false, owner);
    const auto detached = *solved.reference;
    const auto other_frame = build(auxiliary, detached, 1e-10, false, owner);
    require(!other_aux.shared_reference_bytes && !other_threshold.shared_reference_bytes &&
                !other_frame.shared_reference_bytes,
            "mismatched source/frame identity reused the reference");
    bool refused = false;
    try {
      (void)cc::build_df_source_cuda(orbital, auxiliary, *solved.reference, 1, 1e-10, 0, 0, true,
                                     nullptr, false, owner);
    } catch (const std::length_error&) {
      refused = true;
    }
    require(refused, "tiny source allowance was not refused");
    std::weak_ptr<hf::DFReference> lifetime = owner;
    owner.reset();
    require(!lifetime.expired(), "response lost the shared reference owner");
    cc::DFFactorResponseResult seeds;
    seeds.source_identity = cached.source_identity;
    seeds.boo.assign(cached.boo.size(), 0.01);
    seeds.bov.assign(cached.bov.size(), 0.02);
    seeds.bvv.assign(cached.bvv.size(), 0.03);
    std::size_t rows = 0, finishes = 0;
    const auto reverse = cc::pullback_df_source_cuda(
        cached.response_state, seeds, [&](std::size_t, const double*, void*) { ++rows; },
        [&](const double*, const double*, void*) { ++finishes; }, budget);
    require(
        rows == solved.reference->nbf && finishes == 1 && reverse.numeric_capacity_bytes <= budget,
        "retained source pullback failed after reference release");
    cached.response_state.reset();
    require(lifetime.expired(), "shared reference lifetime leaked");
    std::printf("{\"lifecycle\":true,\"mismatch_fallbacks\":3,\"maximum_factor_error\":%.17g}\n",
                maximum);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
