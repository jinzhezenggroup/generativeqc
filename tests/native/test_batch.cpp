#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "generativeqc/generativeqc.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

generativeqc_system* make_hydrogen_system(generativeqc_context* context,
                                          const std::vector<std::array<double, 3>>& positions,
                                          int charge) {
  std::vector<generativeqc_atom> atoms;
  std::vector<generativeqc_shell> shells;
  std::vector<generativeqc_primitive> primitives;
  atoms.reserve(positions.size());
  shells.reserve(positions.size());
  primitives.reserve(positions.size() * 3);
  for (std::size_t atom = 0; atom < positions.size(); ++atom) {
    atoms.push_back({1, positions[atom][0], positions[atom][1], positions[atom][2]});
    const std::uint32_t offset = static_cast<std::uint32_t>(primitives.size());
    primitives.push_back({3.42525091, 0.15432897});
    primitives.push_back({0.62391373, 0.53532814});
    primitives.push_back({0.16885540, 0.44463454});
    shells.push_back({static_cast<std::uint32_t>(atom), 0, offset, 3});
  }
  generativeqc_system_descriptor descriptor{sizeof(generativeqc_system_descriptor),
                                            GENERATIVEQC_ABI_VERSION,
                                            atoms.data(),
                                            static_cast<std::uint32_t>(atoms.size()),
                                            shells.data(),
                                            static_cast<std::uint32_t>(shells.size()),
                                            primitives.data(),
                                            static_cast<std::uint32_t>(primitives.size()),
                                            charge,
                                            1};
  generativeqc_system* system = nullptr;
  require(generativeqc_system_create(context, &descriptor, &system) == GENERATIVEQC_STATUS_SUCCESS,
          "failed to create hydrogen test system");
  return system;
}

generativeqc_batch_item_result_descriptor output(double* forces, std::uint32_t count) {
  return {sizeof(generativeqc_batch_item_result_descriptor),
          GENERATIVEQC_ABI_VERSION,
          GENERATIVEQC_STATUS_INTERNAL_ERROR,
          0.0,
          forces,
          count,
          0,
          0.0,
          0.0,
          0,
          GENERATIVEQC_BACKEND_CPU_REFERENCE,
          0,
          0,
          0};
}

}  // namespace

int main() {
  try {
    generativeqc_context_descriptor context_descriptor{sizeof(generativeqc_context_descriptor),
                                                       GENERATIVEQC_ABI_VERSION, 0,
                                                       GENERATIVEQC_BACKEND_CPU_REFERENCE};
    generativeqc_context* context = nullptr;
    require(
        generativeqc_context_create(&context_descriptor, &context) == GENERATIVEQC_STATUS_SUCCESS,
        "batch context creation failed");

    generativeqc_system* h2 =
        make_hydrogen_system(context, {{{0.0, 0.0, -0.7}}, {{0.0, 0.0, 0.7}}}, 0);
    generativeqc_system* h3_plus = make_hydrogen_system(
        context, {{{-1.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}}, 1);
    generativeqc_system* h4 = make_hydrogen_system(
        context, {{{-1.0, -1.0, 0.0}}, {{-1.0, 1.0, 0.0}}, {{1.0, -1.0, 0.0}}, {{1.0, 1.0, 0.0}}},
        0);
    const std::array<const generativeqc_system*, 3> systems{{h2, h4, h3_plus}};

    generativeqc_method_descriptor method{sizeof(generativeqc_method_descriptor),
                                          GENERATIVEQC_ABI_VERSION,
                                          GENERATIVEQC_METHOD_RHF,
                                          100,
                                          8,
                                          1.0e-12,
                                          1.0e-10,
                                          1.0e-14};
    generativeqc_batch* batch = nullptr;
    require(generativeqc_batch_prepare(context, systems.data(), systems.size(), &method,
                                       GENERATIVEQC_BATCH_ENABLE_WARM_STARTS,
                                       &batch) == GENERATIVEQC_STATUS_SUCCESS,
            "batch preparation failed");
    require(generativeqc_batch_get_system_count(batch) == systems.size(),
            "batch system count is incorrect");
    require(generativeqc_batch_set_warm_start_updates(nullptr, 0) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                generativeqc_batch_set_warm_start_updates(batch, 2) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "warm-start update policy accepted an invalid argument");

    std::array<double, 6> h2_forces{};
    std::array<double, 12> h4_forces{};
    std::array<double, 9> h3_forces{};
    std::array<generativeqc_batch_item_result_descriptor, 3> first{{
        output(h2_forces.data(), h2_forces.size()),
        output(h4_forces.data(), h4_forces.size()),
        output(h3_forces.data(), h3_forces.size()),
    }};
    require(generativeqc_batch_execute(batch, nullptr, 0, first.data(), first.size()) ==
                GENERATIVEQC_STATUS_SUCCESS,
            "first batch execution failed structurally");
    std::array<generativeqc_shell_class_profile_entry, GENERATIVEQC_DIRECT_SHELL_CLASS_COUNT>
        profile{};
    require(generativeqc_batch_get_last_shell_class_profile(
                batch, profile.data(), profile.size()) == GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
            "a non-profiled batch unexpectedly published shell-class data");
    generativeqc_ppps_queue_profile ppps_profile{};
    require(generativeqc_batch_get_last_ppps_queue_profile(batch, &ppps_profile) ==
                    GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                generativeqc_batch_get_last_ppps_queue_profile(nullptr, &ppps_profile) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "a non-profiled batch unexpectedly published PPPS queue data");
    std::uint32_t eigensolver_diagnostic_count = 0;
    require(generativeqc_batch_get_last_eigensolver_diagnostics(batch, nullptr, 0,
                                                                &eigensolver_diagnostic_count) ==
                    GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                generativeqc_batch_get_last_eigensolver_diagnostics(
                    nullptr, nullptr, 0, &eigensolver_diagnostic_count) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                generativeqc_batch_get_last_eigensolver_diagnostics(
                    batch, nullptr, 1, &eigensolver_diagnostic_count) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "CPU batch unexpectedly published CUDA eigensolver evidence");
    std::uint32_t density_fitting_diagnostic_count = 0;
    require(generativeqc_batch_get_last_density_fitting_metric_diagnostics(
                batch, nullptr, 0, &density_fitting_diagnostic_count) ==
                    GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                generativeqc_batch_get_last_density_fitting_metric_diagnostics(
                    nullptr, nullptr, 0, &density_fitting_diagnostic_count) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                generativeqc_batch_get_last_density_fitting_metric_diagnostics(
                    batch, nullptr, 1, &density_fitting_diagnostic_count) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "CPU batch unexpectedly published CUDA DF metric evidence");
    std::uint32_t inactive_profile_count = 0;
    require(
        generativeqc_batch_get_last_inactive_eigensolver_profile(
            batch, nullptr, 0, &inactive_profile_count) == GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
            generativeqc_batch_get_last_inactive_eigensolver_profile(nullptr, nullptr, 0,
                                                                     &inactive_profile_count) ==
                GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
            generativeqc_batch_get_last_inactive_eigensolver_profile(
                batch, nullptr, 1, &inactive_profile_count) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
        "CPU batch unexpectedly published inactive eigensolver data");
    for (const auto& item : first) {
      require(item.status == GENERATIVEQC_STATUS_SUCCESS && item.converged == 1,
              "a valid first-run batch item failed");
      require(item.warm_start_used == 0, "first execution unexpectedly used a warm start");
    }
    require(std::abs(first[0].energy - (-1.11671432506255)) < 2.0e-9,
            "ragged batch H2 energy is incorrect");
    require(first[0].bucket_id != first[1].bucket_id && first[0].bucket_id != first[2].bucket_id &&
                first[1].bucket_id != first[2].bucket_id,
            "different workload shapes were not assigned distinct buckets");

    // Freeze the post-cold snapshots so every warm replay uses one fixed dm0,
    // matching controlled backend-comparison benchmark semantics.
    require(generativeqc_batch_set_warm_start_updates(batch, 0) == GENERATIVEQC_STATUS_SUCCESS,
            "failed to freeze batch warm-start snapshots");

    std::array<generativeqc_batch_item_result_descriptor, 3> second{{
        output(h2_forces.data(), h2_forces.size()),
        output(h4_forces.data(), h4_forces.size()),
        output(h3_forces.data(), h3_forces.size()),
    }};
    require(generativeqc_batch_execute(batch, nullptr, 0, second.data(), second.size()) ==
                GENERATIVEQC_STATUS_SUCCESS,
            "warm batch execution failed structurally");
    for (std::size_t i = 0; i < second.size(); ++i) {
      require(second[i].status == GENERATIVEQC_STATUS_SUCCESS && second[i].warm_start_used == 1,
              "prepared batch did not reuse a converged per-system density");
      require(second[i].iterations <= first[i].iterations,
              "warm start increased the SCF iteration count");
    }
    require(second[2].iterations < first[2].iterations,
            "the nontrivial H3+ workload did not benefit from its warm start");

    std::array<double, 6> invalid_h2_coordinates{0.0, 0.0, std::numeric_limits<double>::quiet_NaN(),
                                                 0.0, 0.0, 0.7};
    std::array<generativeqc_batch_input_descriptor, 3> inputs{{
        {sizeof(generativeqc_batch_input_descriptor), GENERATIVEQC_ABI_VERSION,
         invalid_h2_coordinates.data(), invalid_h2_coordinates.size()},
        {sizeof(generativeqc_batch_input_descriptor), GENERATIVEQC_ABI_VERSION, nullptr, 0},
        {sizeof(generativeqc_batch_input_descriptor), GENERATIVEQC_ABI_VERSION, nullptr, 0},
    }};
    std::array<generativeqc_batch_item_result_descriptor, 3> isolated{{
        output(h2_forces.data(), h2_forces.size()),
        output(h4_forces.data(), h4_forces.size()),
        output(h3_forces.data(), h3_forces.size()),
    }};
    require(generativeqc_batch_execute(batch, inputs.data(), inputs.size(), isolated.data(),
                                       isolated.size()) == GENERATIVEQC_STATUS_SUCCESS,
            "an item-level failure incorrectly aborted the batch call");
    require(isolated[0].status == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "invalid coordinates were not isolated to their item");
    require(isolated[1].status == GENERATIVEQC_STATUS_SUCCESS &&
                isolated[2].status == GENERATIVEQC_STATUS_SUCCESS,
            "one failed system prevented valid neighbors from completing");

    require(generativeqc_batch_clear_warm_starts(batch) == GENERATIVEQC_STATUS_SUCCESS,
            "failed to clear batch warm starts");
    std::array<generativeqc_batch_item_result_descriptor, 3> cold_again{{
        output(h2_forces.data(), h2_forces.size()),
        output(h4_forces.data(), h4_forces.size()),
        output(h3_forces.data(), h3_forces.size()),
    }};
    require(generativeqc_batch_execute(batch, nullptr, 0, cold_again.data(), cold_again.size()) ==
                GENERATIVEQC_STATUS_SUCCESS,
            "cold batch execution after clearing warm state failed");
    for (const auto& item : cold_again) {
      require(item.warm_start_used == 0, "cleared warm-start state was unexpectedly reused");
    }

    generativeqc_batch_destroy(batch);

    generativeqc_system* h2_stretched =
        make_hydrogen_system(context, {{{0.0, 0.0, -0.8}}, {{0.0, 0.0, 0.8}}}, 0);
    const std::array<const generativeqc_system*, 2> mp2_systems{{h2, h2_stretched}};
    generativeqc_method_descriptor mp2_method{sizeof(generativeqc_method_descriptor),
                                              GENERATIVEQC_ABI_VERSION,
                                              GENERATIVEQC_METHOD_MP2,
                                              100,
                                              8,
                                              1.0e-12,
                                              1.0e-12,
                                              0.0};
    generativeqc_batch* mp2_batch = nullptr;
    require(generativeqc_batch_prepare(context, mp2_systems.data(), mp2_systems.size(), &mp2_method,
                                       0, &mp2_batch) == GENERATIVEQC_STATUS_SUCCESS,
            "MP2 batch preparation failed");
    std::array<double, 6> failed_forces{};
    failed_forces.fill(123.0);
    std::array<double, 6> neighbor_forces{};
    auto failed_output = output(failed_forces.data(), failed_forces.size());
    failed_output.energy = 987.0;
    failed_output.iterations = 123;
    failed_output.energy_change = 456.0;
    failed_output.density_rms = 789.0;
    failed_output.converged = 1;
    failed_output.executed_backend = GENERATIVEQC_BACKEND_HYBRID_CUDA;
    failed_output.bucket_id = 77;
    failed_output.warm_start_used = 1;
    failed_output.warm_start_fallback = 1;
    std::array<generativeqc_batch_item_result_descriptor, 2> mp2_isolated{
        failed_output, output(neighbor_forces.data(), neighbor_forces.size())};
    std::array<double, 6> invalid_mp2_coordinates{
        0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.8};
    std::array<generativeqc_batch_input_descriptor, 2> mp2_inputs{{
        {sizeof(generativeqc_batch_input_descriptor), GENERATIVEQC_ABI_VERSION,
         invalid_mp2_coordinates.data(), invalid_mp2_coordinates.size()},
        {sizeof(generativeqc_batch_input_descriptor), GENERATIVEQC_ABI_VERSION, nullptr, 0},
    }};
    require(generativeqc_batch_execute(mp2_batch, mp2_inputs.data(), mp2_inputs.size(),
                                       mp2_isolated.data(),
                                       mp2_isolated.size()) == GENERATIVEQC_STATUS_SUCCESS,
            "MP2 item failure aborted the batch call");
    require(mp2_isolated[0].status == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "invalid MP2 coordinates were not isolated");
    require(mp2_isolated[0].energy == 987.0 && mp2_isolated[0].iterations == 123 &&
                mp2_isolated[0].energy_change == 456.0 && mp2_isolated[0].density_rms == 789.0 &&
                mp2_isolated[0].converged == 1 &&
                mp2_isolated[0].executed_backend == GENERATIVEQC_BACKEND_HYBRID_CUDA &&
                mp2_isolated[0].bucket_id == 77 && mp2_isolated[0].warm_start_used == 1 &&
                mp2_isolated[0].warm_start_fallback == 1,
            "failed MP2 item modified caller metadata");
    require(std::all_of(failed_forces.begin(), failed_forces.end(),
                        [](double value) { return value == 123.0; }),
            "failed MP2 item modified caller force storage");
    require(mp2_isolated[1].status == GENERATIVEQC_STATUS_SUCCESS,
            "failed MP2 item prevented its neighbor from succeeding");
    std::array<double, 6> recovered_h2_forces{};
    std::array<double, 6> recovered_stretched_forces{};
    std::array<generativeqc_batch_item_result_descriptor, 2> mp2_recovered{{
        output(recovered_h2_forces.data(), recovered_h2_forces.size()),
        output(recovered_stretched_forces.data(), recovered_stretched_forces.size()),
    }};
    require(generativeqc_batch_execute(mp2_batch, nullptr, 0, mp2_recovered.data(),
                                       mp2_recovered.size()) == GENERATIVEQC_STATUS_SUCCESS &&
                mp2_recovered[0].status == GENERATIVEQC_STATUS_SUCCESS &&
                mp2_recovered[1].status == GENERATIVEQC_STATUS_SUCCESS,
            "MP2 batch did not recover after an isolated failure");
    std::array<double, 1> short_forces{321.0};
    std::array<double, 6> valid_neighbor_forces{};
    auto short_output = output(short_forces.data(), short_forces.size());
    short_output.energy = 654.0;
    short_output.iterations = 42;
    std::array<generativeqc_batch_item_result_descriptor, 2> short_buffer{{
        short_output,
        output(valid_neighbor_forces.data(), valid_neighbor_forces.size()),
    }};
    require(generativeqc_batch_execute(mp2_batch, nullptr, 0, short_buffer.data(),
                                       short_buffer.size()) == GENERATIVEQC_STATUS_SUCCESS,
            "MP2 output-buffer failure aborted the batch call");
    require(short_buffer[0].status == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                short_buffer[0].energy == 654.0 && short_buffer[0].iterations == 42 &&
                short_forces[0] == 321.0,
            "MP2 output-buffer failure modified caller storage");
    require(short_buffer[1].status == GENERATIVEQC_STATUS_SUCCESS,
            "MP2 output-buffer failure prevented its neighbor from succeeding");
    generativeqc_batch_destroy(mp2_batch);
    generativeqc_system_destroy(h2_stretched);
    generativeqc_system_destroy(h2);
    generativeqc_system_destroy(h3_plus);
    generativeqc_system_destroy(h4);
    generativeqc_context_destroy(context);
    std::cout << "ragged batch, bucketing, isolation, and warm starts: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
