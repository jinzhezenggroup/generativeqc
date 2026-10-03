#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "generativeqc/generativeqc.h"
#include "molecule/basis.hpp"
#include "scf/cuda/direct_jk_plan.hpp"
#include "scf/cuda/scf_matrix_kernels.hpp"
#include "scf/cuda_batch.hpp"
#include "scf/cuda_direct_jk_device.hpp"
#include "scf/mean_field.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool cuda_device_available() {
  generativeqc_context_descriptor descriptor{sizeof(generativeqc_context_descriptor),
                                             GENERATIVEQC_ABI_VERSION, 0,
                                             GENERATIVEQC_BACKEND_CUDA};
  generativeqc_context* context = nullptr;
  const generativeqc_status status = generativeqc_context_create(&descriptor, &context);
  if (context != nullptr) generativeqc_context_destroy(context);
  return status == GENERATIVEQC_STATUS_SUCCESS;
}

/** Managed storage makes the independent linear-map oracle host-readable.
 * Only this test uses managed memory; the production anchors remain resident. */
template <class Value>
struct ManagedArray {
  Value* data{};
  explicit ManagedArray(std::size_t size) {
    require(cudaMallocManaged(&data, size * sizeof(Value)) == cudaSuccess,
            "incremental kernel test allocation failed");
  }
  ~ManagedArray() { cudaFree(data); }
  ManagedArray(const ManagedArray&) = delete;
  ManagedArray& operator=(const ManagedArray&) = delete;
};

void verify_linear_channels(unsigned spins, unsigned channels, bool with_hcore,
                            unsigned basis_size = 2) {
  using namespace generativeqc::scf::cuda_execution;
  constexpr unsigned batch = 3;
  const unsigned matrix = basis_size * basis_size;
  const auto density_elements = batch * spins * matrix;
  const auto channel_elements = batch * channels * matrix;
  ManagedArray<double> density(density_elements), anchor_density(density_elements),
      delta(density_elements), hcore(batch * matrix), output(channel_elements),
      anchor_output(channel_elements), maximum_delta(batch);
  ManagedArray<std::uint32_t> updates(batch);
  ManagedArray<std::uint8_t> active(batch), full(batch);
  std::fill_n(anchor_density.data, density_elements, 77.0);
  std::fill_n(anchor_output.data, channel_elements, 77.0);
  std::fill_n(delta.data, density_elements, 77.0);
  std::fill_n(output.data, channel_elements, 77.0);
  for (unsigned system = 0; system < batch; ++system) {
    active.data[system] = system == 1 ? 0 : 1;
    full.data[system] = 77;
    updates.data[system] = 0xffffffffU;
    maximum_delta.data[system] = 0.0;
    for (unsigned element = 0; element < matrix; ++element)
      hcore.data[system * matrix + element] = 0.3 + 0.1 * element;
  }
  const auto blocks = (batch * std::max(spins, channels) * matrix + 127) / 128;
  const auto linear_value = [&](const double* input, unsigned system, unsigned channel,
                                unsigned element) {
    double value = 0.0;
    for (unsigned spin = 0; spin < spins; ++spin)
      value += (channel + 1.0) * (spin + 0.5) * input[(system * spins + spin) * matrix + element];
    return value;
  };
  for (unsigned iteration = 0; iteration < 4; ++iteration) {
    for (unsigned element = 0; element < density_elements; ++element)
      density.data[element] = 0.02 * element + 0.01 * iteration;
    launch_prepare_incremental_direct_jk_kernel(
        blocks, 128, 0, nullptr, batch, spins, basis_size, 1, density.data,
        with_hcore ? hcore.data : nullptr, active.data, anchor_density.data, anchor_output.data,
        delta.data, updates.data, full.data, maximum_delta.data, channels);
    require(cudaDeviceSynchronize() == cudaSuccess, "incremental prepare kernel failed");
    for (unsigned system : {0U, 2U}) {
      require(full.data[system] == (iteration % 2 == 0), "periodic full/delta policy changed");
      for (unsigned channel = 0; channel < channels; ++channel)
        for (unsigned element = 0; element < matrix; ++element)
          output.data[(system * channels + channel) * matrix + element] =
              linear_value(delta.data, system, channel, element) +
              (with_hcore ? hcore.data[system * matrix + element] : 0.0);
    }
    launch_finalize_incremental_direct_jk_kernel(
        blocks, 128, 0, nullptr, batch, spins, basis_size, density.data,
        with_hcore ? hcore.data : nullptr, active.data, anchor_density.data, anchor_output.data,
        output.data, updates.data, full.data, channels);
    require(cudaDeviceSynchronize() == cudaSuccess, "incremental finalize kernel failed");
    for (unsigned system = 0; system < batch; ++system) {
      for (unsigned channel = 0; channel < channels; ++channel)
        for (unsigned element = 0; element < matrix; ++element) {
          const auto offset = (system * channels + channel) * matrix + element;
          const double expected =
              system == 1 ? 77.0
                          : linear_value(density.data, system, channel, element) +
                                (with_hcore ? hcore.data[system * matrix + element] : 0.0);
          require(std::abs(output.data[offset] - expected) < 1e-13 &&
                      std::abs(anchor_output.data[offset] - expected) < 1e-13,
                  "incremental channels mixed spin, system, hcore or refresh state");
        }
      for (unsigned element = 0; element < spins * matrix; ++element) {
        const auto offset = system * spins * matrix + element;
        require(anchor_density.data[offset] == (system == 1 ? 77.0 : density.data[offset]),
                "incremental density anchor advanced an inactive system or lost a spin");
      }
    }
    require(updates.data[1] == 0xffffffffU && full.data[1] == 77 && maximum_delta.data[1] == 0,
            "inactive incremental control state changed");
  }
}

generativeqc::core::System asymmetric_two_center(bool unrestricted) {
  generativeqc::core::System system;
  system.atoms = {
      {2, {0.0, 0.0, -0.60}},
      {1, {0.0, 0.0, 0.85}},
  };
  system.shells = {
      {0, 0, {{6.36242139, 0.15432897}, {1.15892300, 0.53532814}, {0.31364979, 0.44463454}}},
      {1, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}},
  };
  system.charge = unrestricted ? 0 : 1;
  system.multiplicity = unrestricted ? 2 : 1;
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "incremental Direct-J/K fixture normalization failed");
  return system;
}

generativeqc::scf::RhfBucketItem run_cached(generativeqc::scf::CudaRhfBucketPlan** plan,
                                            const generativeqc::core::System& system,
                                            const generativeqc::scf::ScfOptions& options,
                                            bool unrestricted) {
  const std::vector<generativeqc::core::System> systems{system};
  const std::vector<const std::vector<double>*> cold{nullptr};
  const auto rows =
      unrestricted
          ? generativeqc::scf::run_uhf_cuda_bucket_cached(plan, systems, options, cold, 0, false)
          : generativeqc::scf::run_rhf_cuda_bucket_cached(plan, systems, options, cold, 0, false);
  require(rows.size() == 1, "incremental Direct-J/K CUDA bucket returned wrong size");
  return rows.front();
}

void compare_final_state(const generativeqc::scf::ScfResult& baseline,
                         const generativeqc::scf::ScfResult& incremental, double tolerance) {
  require(baseline.converged && incremental.converged,
          "incremental Direct-J/K CUDA parity fixture did not converge");
  require(std::abs(baseline.energy - incremental.energy) < tolerance,
          "incremental Direct-J/K changed final energy");
  require(baseline.forces.size() == incremental.forces.size(),
          "incremental Direct-J/K changed force shape");
  for (std::size_t i = 0; i < baseline.forces.size(); ++i) {
    require(std::abs(baseline.forces[i] - incremental.forces[i]) < 10.0 * tolerance,
            "incremental Direct-J/K changed final forces");
  }
}

/** Resident provider arguments deliberately reject managed memory. */
struct DeviceStatus {
  int* data{};
  DeviceStatus() {
    require(cudaMalloc(&data, sizeof(int)) == cudaSuccess, "device status allocation failed");
  }
  ~DeviceStatus() { cudaFree(data); }
};

void verify_generated_work_observer() {
  using namespace generativeqc;
  constexpr std::size_t classes = scf::detail::kDirectQuartetShellClassCount;
  ManagedArray<unsigned long long> coulomb_counts(classes), exchange_counts(classes);
  DeviceStatus error;
  scf::CudaDirectJkPlan* raw{};
  scf::CudaDirectJkDiagnostic diagnostic;
  std::string detail;
  require(scf::create_cuda_direct_jk_plan(0, {asymmetric_two_center(false)}, 0, 1e-12,
                                          64U * 1024U * 1024U, &raw, diagnostic,
                                          detail) == GENERATIVEQC_STATUS_SUCCESS,
          "could not prepare generated-work observer control");
  std::unique_ptr<scf::CudaDirectJkPlan, decltype(&scf::destroy_cuda_direct_jk_plan)> plan(
      raw, scf::destroy_cuda_direct_jk_plan);
  require(raw->generated_exchange && raw->generated_exchange->shared->value_capability,
          "observer control lacks complete generated streaming sources");
  scf::FockBuildSpec spec;
  spec.derivative_order = 0;
  std::vector<double> density{0.6, 0.1, 0.1, 0.4}, coulomb, exchange, beta;
  require(scf::execute_cuda_direct_jk(raw, spec, density, {}, coulomb, exchange, beta, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "unobserved direct provider control failed");
  const auto expected_coulomb = coulomb;
  const auto expected_exchange = exchange;
  require(cudaMemcpy(raw->density, density.data(), density.size() * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess,
          "resident observer input upload failed");
  const auto execute_resident = [&] {
    if (scf::enqueue_cuda_direct_jk_device(raw, spec, raw->density, nullptr, density.size(),
                                           raw->coulomb, raw->alpha_exchange, nullptr, error.data,
                                           detail) != GENERATIVEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail);
    require(cudaDeviceSynchronize() == cudaSuccess, "observed resident provider execution failed");
    int numerical_error = 1;
    require(cudaMemcpy(&numerical_error, error.data, sizeof(int), cudaMemcpyDeviceToHost) ==
                    cudaSuccess &&
                numerical_error == 0,
            "observed resident provider reported a numerical failure");
    require(cudaMemcpy(coulomb.data(), raw->coulomb, coulomb.size() * sizeof(double),
                       cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(exchange.data(), raw->alpha_exchange, exchange.size() * sizeof(double),
                           cudaMemcpyDeviceToHost) == cudaSuccess,
            "observed resident provider output download failed");
  };
  auto& generated = *raw->generated_exchange;
  generated.shared->admitted_shell_counts = coulomb_counts.data;
  generated.admitted_shell_counts = exchange_counts.data;
  std::fill_n(coulomb_counts.data, classes, 0ULL);
  std::fill_n(exchange_counts.data, classes, 0ULL);
  execute_resident();
  require(coulomb_counts.data[0] > 0 && exchange_counts.data[0] > 0,
          "separate J/K observers did not count actual admitted tasks");
  for (std::size_t element = 0; element < density.size(); ++element)
    require(std::abs(coulomb[element] - expected_coulomb[element]) < 1e-13 &&
                std::abs(exchange[element] - expected_exchange[element]) < 1e-13,
            "intrusive admission counters changed raw J/K values");
  std::fill_n(coulomb_counts.data, classes, 0ULL);
  std::fill_n(exchange_counts.data, classes, 0ULL);
  require(cudaMemset(raw->density, 0, density.size() * sizeof(double)) == cudaSuccess,
          "zero-density observer input reset failed");
  execute_resident();
  require(exchange_counts.data[0] == 0, "zero-density K was counted before its screening gate");
  for (std::size_t element = 0; element < density.size(); ++element)
    require(coulomb[element] == 0.0 && exchange[element] == 0.0,
            "zero-density observed provider retained a previous raw matrix");
  for (std::size_t shell_class = 1; shell_class < classes; ++shell_class)
    require(coulomb_counts.data[shell_class] == 0 && exchange_counts.data[shell_class] == 0,
            "observer wrote a shell class absent from the control basis");
  generated.shared->admitted_shell_counts = nullptr;
  generated.admitted_shell_counts = nullptr;
}

void verify_case(bool unrestricted, double screening_tolerance, unsigned requested_interval) {
  const auto system = asymmetric_two_center(unrestricted);
  generativeqc::scf::ScfOptions baseline_options;
  baseline_options.max_iterations = 100;
  baseline_options.energy_tolerance = 1.0e-12;
  baseline_options.density_tolerance = 1.0e-10;
  baseline_options.screening_tolerance = screening_tolerance;
  baseline_options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  baseline_options.compute_forces = true;

  generativeqc::scf::CudaRhfBucketPlan* plan = nullptr;
  const auto baseline = run_cached(&plan, system, baseline_options, unrestricted);
  require(baseline.status == GENERATIVEQC_STATUS_SUCCESS && baseline.scf.converged,
          "baseline CUDA Direct-J/K fixture failed");

  auto incremental_options = baseline_options;
  incremental_options.incremental_direct_jk = true;
  incremental_options.incremental_direct_jk_rebuild_interval = requested_interval;
  const auto incremental = run_cached(&plan, system, incremental_options, unrestricted);
  require(incremental.status == GENERATIVEQC_STATUS_SUCCESS && incremental.scf.converged,
          "incremental CUDA Direct-J/K fixture failed");
  compare_final_state(baseline.scf, incremental.scf, screening_tolerance == 0.0 ? 5.0e-10 : 5.0e-8);

  const auto& diagnostic = incremental.scf.incremental_direct_jk;
  require(diagnostic.requested && diagnostic.active,
          "native CUDA did not activate requested incremental Direct-J/K");
  const std::uint64_t builds = diagnostic.anchor_full_builds + diagnostic.delta_builds;
  require(builds == incremental.scf.iterations && diagnostic.delta_builds != 0U,
          "native CUDA incremental build counters do not match SCF iterations");
  require(diagnostic.anchor_updates == diagnostic.delta_builds,
          "native CUDA incremental anchor-update semantics changed");
  require(diagnostic.max_abs_delta_density > 0.0,
          "native CUDA incremental path did not observe a density delta");
  require(diagnostic.post_scf_full_builds != 0U,
          "native CUDA incremental path skipped strict final full rebuilds");
  require(diagnostic.quartet_work_counters_valid,
          "fixed-topology incremental Direct-J/K work counters are not valid");
  require(diagnostic.full_admitted_shell_quartets <= diagnostic.full_candidate_shell_quartets &&
              diagnostic.delta_admitted_shell_quartets <= diagnostic.delta_candidate_shell_quartets,
          "incremental Direct-J/K admitted more shell quartets than it visited");
  require(diagnostic.full_admitted_quartet_tiles >= diagnostic.full_admitted_shell_quartets &&
              diagnostic.delta_admitted_quartet_tiles >= diagnostic.delta_admitted_shell_quartets,
          "incremental Direct-J/K tile census is smaller than its admitted shell-quartet census");
  require(
      diagnostic.full_candidate_shell_quartets ==
              diagnostic.full_admitted_shell_quartets + diagnostic.full_rejected_shell_quartets &&
          diagnostic.delta_candidate_shell_quartets ==
              diagnostic.delta_admitted_shell_quartets + diagnostic.delta_rejected_shell_quartets,
      "incremental Direct-J/K candidate/admitted/rejected counters do not reconcile");

  if (screening_tolerance == 0.0) {
    require(diagnostic.anchor_full_builds == 1U && diagnostic.periodic_rebuilds == 0U,
            "unscreened exact incremental mode rebuilt unexpectedly");
    require(diagnostic.full_rejected_shell_quartets == 0U &&
                diagnostic.delta_rejected_shell_quartets == 0U,
            "unscreened incremental Direct-J/K rejected shell quartets");
  } else {
    const std::uint64_t expected_full = (builds + 1U) / 2U;
    require(diagnostic.anchor_full_builds == expected_full,
            "screened incremental mode accumulated more than one delta update");
    require(diagnostic.periodic_rebuilds + 1U == diagnostic.anchor_full_builds,
            "screened incremental periodic rebuild counters disagree");
  }

  // Reuse the same incremental owner at changed geometry. The per-execution
  // anchor must restart with a full-density build instead of treating the new
  // geometry as a delta from the previous integral operator.
  auto changed_system = system;
  changed_system.atoms[1].position[2] += 0.03;
  generativeqc::scf::CudaRhfBucketPlan* changed_baseline_plan = nullptr;
  const auto changed_baseline =
      run_cached(&changed_baseline_plan, changed_system, baseline_options, unrestricted);
  generativeqc::scf::destroy_rhf_cuda_bucket_plan(changed_baseline_plan);
  const auto changed_incremental =
      run_cached(&plan, changed_system, incremental_options, unrestricted);
  require(changed_baseline.status == GENERATIVEQC_STATUS_SUCCESS &&
              changed_incremental.status == GENERATIVEQC_STATUS_SUCCESS,
          "changed-geometry CUDA incremental replay failed");
  compare_final_state(changed_baseline.scf, changed_incremental.scf,
                      screening_tolerance == 0.0 ? 5.0e-10 : 5.0e-8);
  require(changed_incremental.scf.incremental_direct_jk.anchor_full_builds != 0U,
          "changed geometry reused a stale incremental Direct-J/K anchor");
  if (screening_tolerance == 0.0 && requested_interval == 0U) {
    require(changed_incremental.scf.incremental_direct_jk.anchor_full_builds == 1U,
            "unscreened changed geometry established more than one full anchor");
  }

  // Switch the same cached owner back to the ordinary policy. The bucket
  // identity must rebuild rather than replaying an incremental captured Graph.
  const auto ordinary_again = run_cached(&plan, system, baseline_options, unrestricted);
  require(ordinary_again.status == GENERATIVEQC_STATUS_SUCCESS && ordinary_again.scf.converged &&
              !ordinary_again.scf.incremental_direct_jk.active,
          "cached CUDA plan leaked incremental Direct-J/K policy across executions");
  generativeqc::scf::destroy_rhf_cuda_bucket_plan(plan);
}

}  // namespace

int main(int argc, char** argv) {
  const bool linear_only = argc == 2 && std::string(argv[1]) == "--linear-kernels-only";
  if (argc != 1 && !linear_only) return EXIT_FAILURE;
  if (!cuda_device_available()) return 77;
  try {
    verify_linear_channels(1, 2, false);
    verify_linear_channels(2, 3, false);
    verify_linear_channels(2, 1, false);
    verify_linear_channels(2, 3, false, 7);
    verify_linear_channels(2, 1, false, 7);
    verify_linear_channels(2, 2, true, 7);
    if (linear_only) return EXIT_SUCCESS;
    verify_generated_work_observer();
    verify_linear_channels(1, 1, true);
    verify_linear_channels(2, 2, true);
    verify_case(false, 0.0, 0U);
    verify_case(true, 0.0, 0U);
    verify_case(false, 1.0e-12, 8U);
    verify_case(true, 1.0e-12, 8U);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EXIT_FAILURE;
  }
}
