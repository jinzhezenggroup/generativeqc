#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "molecule/basis.hpp"
#include "scf/cuda_fock_execution.hpp"
#include "scf/cuda_fock_provider.hpp"
#include "scf/fleet.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/mean_field.hpp"

namespace {
using namespace generativeqc::scf;
using generativeqc::core::System;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void close(double a, double b, double tolerance, const char* message) {
  if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > tolerance)
    throw std::runtime_error(std::string(message) + ": " + std::to_string(a) + " vs " +
                             std::to_string(b));
}
void matrix(const std::vector<double>& a, const std::vector<double>& b, const char* message) {
  require(a.size() == b.size(), message);
  for (std::size_t i = 0; i < a.size(); ++i) close(a[i], b[i], 2e-9, message);
}
template <class Function>
void rejected(Function fn, const char* message) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}
System fixture(bool polarized = false) {
  System out;
  out.atoms = {{1, {0.0, 0.1, -0.7}}, {1, {0.2, -0.1, 0.7}}};
  out.shells = {{0, 0, {{0.8, 0.7}, {0.2, 0.3}}}, {1, 0, {{0.6, 1.0}}}};
  if (polarized) out.shells.push_back({1, 1, {{0.4, 1.0}}});
  out.electron_count = 2;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(out, detail) == GENERATIVEQC_STATUS_SUCCESS,
      detail.c_str());
  return out;
}
DensityFittingScfData oracle(const System& orbital, const System& aux, double cutoff) {
  DensityFittingScfData out;
  out.raw = generativeqc::integrals::build_density_fitting_integrals(orbital, aux);
  out.metric_relative_threshold = cutoff;
  out.three_center = orthonormalize_density_fitting_three_center(
      out.raw.three_center, out.raw.nbf,
      factor_density_fitting_metric(out.raw.metric, out.raw.naux, cutoff));
  out.df_gradient_orbital = orbital;
  out.df_gradient_auxiliary = aux;
  out.df_gradient_budget = 8U * 1024U * 1024U;
  return out;
}

void composed_items() {
  const auto first = fixture(true);
  auto second = first;
  second.atoms[1].position[2] += 0.17;
  second.shells[0].primitives[0].exponent *= 1.2;
  auto aux1 = first, aux2 = second;
  // Redundant auxiliary functions exercise a truncated metric and its full
  // spectral response. The CPU integral evaluator supplies the independent oracle.
  aux1.shells.push_back(first.shells[0]);
  aux2.shells.push_back(second.shells[0]);
  constexpr double cutoff = 1e-8;
  const auto exact1 = generativeqc::integrals::build_integrals(first);
  const auto exact2 = generativeqc::integrals::build_integrals(second);
  const std::vector<DensityFittingScfData> data{oracle(first, aux1, cutoff),
                                                oracle(second, aux2, cutoff)};
  const auto n = exact1.nbf, m = n * n;
  CudaDirectJkPlan* raw{};
  CudaDirectJkDiagnostic info;
  std::string detail;
  require(create_cuda_direct_jk_plan(0, {first, second}, 1, 0.0, 8U * 1024U * 1024U, &raw, info,
                                     detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> direct(
      raw, &destroy_cuda_direct_jk_plan);
  std::vector<double> a(m), b(m), sa(m), sb(m);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      a[i * n + j] = std::cos(0.3 * i + 0.7 * j) / n;
      b[i * n + j] = std::sin(0.2 * i - 0.4 * j) / n;
    }
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      sa[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
      sb[i * n + j] = 0.5 * (b[i * n + j] + b[j * n + i]);
    }
  for (bool streamed : {false, true}) {
    CudaDensityFittingJkPlan* raw_df{};
    std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
    if (streamed) {
      CudaDensityFittingIntegralSource* source{};
      std::vector<double> metrics;
      std::size_t nbf{}, naux{};
      require(create_cuda_density_fitting_integral_source(0, {first, second}, {aux1, aux2}, &source,
                                                          metrics, nbf, naux,
                                                          detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      const auto status = create_cuda_density_fitting_jk_plan_from_source(
          0, &source, 2, nbf, naux, metrics, cutoff, 2, 3, &raw_df, diagnostics, detail);
      destroy_cuda_density_fitting_integral_source(source);
      require(status == GENERATIVEQC_STATUS_SUCCESS, detail.c_str());
    } else {
      auto metrics = data[0].raw.metric, tensors = data[0].raw.three_center;
      metrics.insert(metrics.end(), data[1].raw.metric.begin(), data[1].raw.metric.end());
      tensors.insert(tensors.end(), data[1].raw.three_center.begin(),
                     data[1].raw.three_center.end());
      require(create_cuda_density_fitting_jk_plan(0, 2, n, data[0].raw.naux, metrics, tensors,
                                                  cutoff, 0, &raw_df, diagnostics,
                                                  detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
    }
    std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)> df(
        raw_df, &destroy_cuda_density_fitting_jk_plan);
    require(diagnostics[0].effective_rank < data[0].raw.naux,
            "DF fixture did not truncate the metric");
    for (std::size_t item : {1U, 0U})
      for (bool uhf : {false, true})
        for (int j : {0, 1, 2})
          for (int k : {0, 1, 2}) {
            auto spec = make_hf_fock_spec(uhf ? FockSpin::Unrestricted : FockSpin::Restricted);
            spec.coulomb.present = j != 0;
            spec.exchange.present = k != 0;
            spec.coulomb.approximation =
                j == 2 ? FockApproximation::DensityFitted : FockApproximation::Exact;
            spec.exchange.approximation =
                k == 2 ? FockApproximation::DensityFitted : FockApproximation::Exact;
            spec.coulomb.coefficient = -0.7;
            spec.exchange.coefficient = 0.23;
            const auto& exact = item ? exact2 : exact1;
            auto cpu_provider = [&](int term) -> std::optional<CpuFockProviderView> {
              if (!term) return {};
              return term == 2 ? CpuFockProviderView(data[item]) : CpuFockProviderView(exact);
            };
            auto gpu_provider = [&](int term) -> std::optional<CudaFockProviderView> {
              if (!term) return {};
              return term == 2 ? CudaFockProviderView(df.get(), data[item], item)
                               : CudaFockProviderView(direct.get(), item);
            };
            const CpuFockPlanView cpu(resolve_fock_build(spec, FockBackend::Cpu, 0.0, cutoff), n, 6,
                                      cpu_provider(j), cpu_provider(k));
            const CudaFockPlanView gpu(resolve_fock_build(spec, FockBackend::Cuda, 0.0, cutoff), n,
                                       6, gpu_provider(j), gpu_provider(k));
            const auto expected = cpu.build(a, uhf ? b : std::vector<double>{});
            const auto actual = gpu.build(a, uhf ? b : std::vector<double>{});
            matrix(actual.coulomb, expected.coulomb, "composed CUDA J");
            matrix(actual.exchange_alpha, expected.exchange_alpha, "composed CUDA Ka");
            matrix(actual.exchange_beta, expected.exchange_beta, "composed CUDA Kb");
            matrix(gpu.energy_derivative(sa, uhf ? sb : std::vector<double>{}),
                   cpu.energy_derivative(sa, uhf ? sb : std::vector<double>{}),
                   "composed CUDA response");
            if (j == 2 || k == 2)
              rejected([&] { (void)gpu.energy_derivative(a, uhf ? b : std::vector<double>{}); },
                       "nonsymmetric DF response survived preflight");
          }
    auto spec = make_hf_fock_spec(FockSpin::Restricted);
    spec.coulomb.approximation = FockApproximation::DensityFitted;
    const auto strategy = resolve_fock_build(spec, FockBackend::Cuda, 0.0, cutoff);
    const CudaFockProviderView j(df.get(), data[0]), k(direct.get());
    rejected([&] { (void)CudaFockPlanView(strategy, n, 6, k, j); },
             "swapped CUDA providers accepted");
    auto wrong = data[0];
    wrong.metric_relative_threshold = 1e-3;
    rejected(
        [&] { (void)CudaFockPlanView(strategy, n, 6, CudaFockProviderView(df.get(), wrong), k); },
        "CUDA cutoff mismatch accepted");
    rejected(
        [&] {
          (void)CudaFockPlanView(strategy, n, 6, CudaFockProviderView(df.get(), data[0], 2), k);
        },
        "out-of-range CUDA DF item accepted");
    rejected(
        [&] { (void)CudaFockPlanView(strategy, n, 6, j, CudaFockProviderView(direct.get(), 2)); },
        "out-of-range CUDA direct item accepted");
  }
}

void molecular_endpoints() {
  const auto system = fixture();
  for (bool uhf : {false, true})
    for (int pair : {0, 1, 2, 3, 4}) {
      ScfOptions options;
      options.energy_tolerance = 1e-12;
      options.density_tolerance = 1e-10;
      options.screening_tolerance = 0.0;
      auto spec = make_hf_fock_spec(uhf ? FockSpin::Unrestricted : FockSpin::Restricted);
      spec.coulomb.approximation =
          pair & 1 ? FockApproximation::DensityFitted : FockApproximation::Exact;
      spec.exchange.approximation =
          pair & 2 ? FockApproximation::DensityFitted : FockApproximation::Exact;
      // Nonstandard coefficients force common independent execution even for
      // matching pairs; pure-J also represents a semilocal consumer's request.
      spec.exchange.coefficient *= 0.7;
      spec.exchange.present = pair != 4;
      // The full-strength Hartree-only model oscillates for this asymmetric
      // two-function fixture on both backends. Use an explicit weaker J for
      // the convergent absent-K endpoint; raw tests still cover arbitrary J.
      if (pair == 4) spec.coulomb.coefficient = 0.3;
      options.resolved_fock_build = resolve_fock_build(spec, FockBackend::Cpu, 0.0);
      const auto cpu = run_fock_strategy(system, nullptr, options, -1);
      options.resolved_fock_build = resolve_fock_build(spec, FockBackend::Cuda, 0.0);
      const auto gpu = run_fock_strategy(system, nullptr, options, 0);
      if (!cpu.converged || !gpu.converged)
        throw std::runtime_error("independent SCF convergence spin=" + std::to_string(uhf) +
                                 " pair=" + std::to_string(pair) +
                                 " CPU=" + std::to_string(cpu.converged) +
                                 " CUDA=" + std::to_string(gpu.converged) +
                                 " density=" + std::to_string(gpu.density_rms));
      close(gpu.energy, cpu.energy, 2e-10, "independent CUDA SCF energy");
      matrix(gpu.forces, cpu.forces, "independent CUDA SCF forces");
      const auto warm = run_fock_strategy(system, nullptr, options, 0, &gpu.density);
      require(warm.converged && warm.initial_density_used, "independent CUDA warm replay failed");
      close(warm.energy, gpu.energy, 2e-10, "independent CUDA warm energy");
      auto plus = system, minus = system;
      constexpr double step = 1e-4;
      plus.atoms[1].position[2] += step;
      minus.atoms[1].position[2] -= step;
      options.compute_forces = false;
      const auto ep = run_fock_strategy(plus, nullptr, options, 0, &gpu.density);
      const auto em = run_fock_strategy(minus, nullptr, options, 0, &gpu.density);
      require(ep.converged && em.converged && ep.forces.empty() && em.forces.empty(),
              "CUDA changed-geometry energy failed");
      close(gpu.forces[5], -(ep.energy - em.energy) / (2 * step), 2e-8,
            "CUDA stationary force finite difference");
    }
}

void ragged_replay() {
  const auto system = fixture();
  auto helium = system;
  helium.atoms = {{2, {0.0, 0.0, 0.0}}};
  helium.shells.resize(1);
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  spec.coulomb.approximation = FockApproximation::DensityFitted;
  ScfOptions options;
  options.resolved_fock_build = resolve_fock_build(spec, FockBackend::Cuda);
  FleetPlan fleet({system, helium, system}, GENERATIVEQC_METHOD_RHF, options, true, true, false,
                  false, 0);
  const auto first = fleet.execute({});
  for (const auto& item : first)
    require(item.status == GENERATIVEQC_STATUS_SUCCESS &&
                item.executed_backend == GENERATIVEQC_BACKEND_CUDA,
            "independent ragged CUDA execution failed");
  std::vector<std::optional<std::vector<double>>> coordinates(3);
  coordinates[1] = std::vector<double>{0.0};
  const auto isolated = fleet.execute(coordinates);
  require(isolated[1].status == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "malformed CUDA fleet item accepted");
  for (std::size_t i : {0U, 2U}) {
    require(isolated[i].status == GENERATIVEQC_STATUS_SUCCESS && isolated[i].warm_start_used,
            "malformed CUDA neighbor changed warm state");
    close(isolated[i].scf.energy, first[i].scf.energy, 2e-9, "ragged CUDA replay energy");
  }
  const auto recovered = fleet.execute({});
  require(recovered[1].status == GENERATIVEQC_STATUS_SUCCESS && recovered[1].warm_start_used,
          "CUDA rejected coordinates replaced the previous warm state");
}

void prepared_replay() {
  auto system = fixture();
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  spec.coulomb.approximation = FockApproximation::DensityFitted;
  ScfOptions options;
  options.resolved_fock_build = resolve_fock_build(spec, FockBackend::Cuda);
  std::unique_ptr<PreparedFockPlan> cache;
  const auto first = run_fock_strategy_cached(cache, system, nullptr, options, 0);
  require(first.converged && cache, "prepared CUDA source creation failed");
  const auto* retained = cache.get();
  const auto warm = run_fock_strategy_cached(cache, system, nullptr, options, 0, &first.density);
  require(warm.converged && cache.get() == retained, "identical CUDA replay rebuilt sources");
  close(warm.energy, first.energy, 2e-9, "retained CUDA source changed energy");
  // Iteration controls affect convergence, not cached integral identity.
  options.max_iterations += 20;
  (void)run_fock_strategy_cached(cache, system, nullptr, options, 0, &first.density);
  require(cache.get() == retained, "convergence controls invalidated integral sources");
  const auto info = cache->diagnostic();
  require(info.device_bytes >= info.direct.device_bytes && !info.fitted.empty() &&
              info.fitted_source.metric_staged_on_host,
          "prepared composite source diagnostics incomplete");
  // A failed replacement cannot release the last valid prepared source.
  options.density_fitting_memory_budget_bytes = 1;
  bool oom = false;
  try {
    (void)run_fock_strategy_cached(cache, system, nullptr, options, 0);
  } catch (const std::bad_alloc&) {
    oom = true;
  }
  require(oom && cache.get() == retained, "failed preparation replaced valid CUDA cache");
  options.density_fitting_memory_budget_bytes = 0;
  system.atoms[1].position[2] += 0.09;
  const auto moved = run_fock_strategy_cached(cache, system, nullptr, options, 0, &first.density);
  require(moved.converged && cache.get() != retained, "changed geometry reused stale CUDA source");
  const auto cold = run_fock_strategy(system, nullptr, options, 0);
  close(moved.energy, cold.energy, 2e-9, "changed-geometry cached source differs from cold source");
  auto changed = spec;
  changed.exchange.coefficient = -0.25;
  retained = cache.get();
  options.resolved_fock_build = resolve_fock_build(changed, FockBackend::Cuda);
  const auto model = run_fock_strategy_cached(cache, system, nullptr, options, 0);
  require(model.converged && cache.get() != retained, "changed coefficients reused old CUDA plan");
  require(!cache->matches(system, nullptr, *options.resolved_fock_build, 1, 0),
          "changed CUDA device accepted as compatible");
}

void retained_direct_derivative_reuse() {
  const auto system = fixture(true);
  const auto strategy = resolve_fock_build(make_rsh_primary_fock_spec(FockSpin::Restricted, 0.2),
                                           FockBackend::Cuda, 0.0);
  const auto correction = resolve_fock_build(
      make_rsh_correction_fock_spec(FockSpin::Restricted, 0.2, 0.5, 0.4), FockBackend::Cuda, 0.0);
  std::size_t primitives = 0;
  for (const auto& shell : system.shells) primitives += shell.primitives.size();
  const auto budget =
      cuda_direct_coulomb_device_bytes(1, generativeqc::molecule::ao_count(system),
                                       system.atoms.size(), system.shells.size(), primitives, 1);
  PreparedFockPlan plan(system, nullptr, strategy, 0, budget, 1);
  require(plan.strategy().spec.derivative_order == 0,
          "retained derivative capability changed value-side Fock identity");
  require(plan.diagnostic().direct.derivative_order == 1,
          "prepared Direct owner did not retain first-derivative capability");
  require(std::string(plan.diagnostic().direct.schedule).find("generated-shell-coulomb") !=
              std::string::npos,
          "retained derivative capability disabled generated value-J scheduling");

  const auto value = prepared_cuda_fock_binding(plan);
  const auto derivative = prepared_cuda_direct_derivative_binding(plan);
  require(value && derivative, "prepared Direct owner did not expose both value/derivative views");
  require(value.source_identity == derivative.source_identity,
          "prepared value/derivative bindings borrowed different Direct owners");
  require(derivative.maximum_derivative_order == 1 &&
              derivative.coordinates_per_item == 3 * system.atoms.size(),
          "prepared derivative binding reported the wrong capability");

  const auto n = generativeqc::molecule::ao_count(system);
  std::vector<double> density(n * n);
  for (std::size_t i = 0; i < n; ++i) density[i * n + i] = 1.0;
  std::vector<double> response;
  std::string detail;
  require(execute_prepared_cuda_direct_rsh_energy_derivatives(
              plan, correction, density, {}, response, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  require(response.size() == 3 * derivative.coordinates_per_item,
          "prepared fused RSH derivative returned the wrong shape");
  require(std::all_of(response.begin(), response.end(),
                      [](double value) { return std::isfinite(value); }),
          "prepared fused RSH derivative returned nonfinite values");

  double* device_density = nullptr;
  require(cudaMalloc(reinterpret_cast<void**>(&device_density), density.size() * sizeof(double)) ==
              cudaSuccess,
          "device density allocation failed");
  require(cudaMemcpyAsync(device_density, density.data(), density.size() * sizeof(double),
                          cudaMemcpyHostToDevice, derivative.stream) == cudaSuccess,
          "device density upload failed");
  int device_count = 0;
  require(cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0,
          "CUDA device count query failed");
  if (device_count > 1) {
    const int alternate_device = derivative.device_id == 0 ? 1 : 0;
    require(cudaSetDevice(alternate_device) == cudaSuccess,
            "alternate CUDA device selection failed");
  }
  std::vector<double> shell_full_range;
  require(execute_prepared_cuda_direct_shell_full_range_derivatives_device(
              plan, device_density, nullptr, density.size(), shell_full_range, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  require(shell_full_range.size() == 2 * derivative.coordinates_per_item,
          "prepared shell full-range derivative returned the wrong shape");

  auto j_spec = strategy.spec;
  j_spec.derivative_order = 1;
  j_spec.exchange.present = false;
  std::vector<double> expected_j;
  require(execute_cuda_direct_energy_derivative(plan.cuda_direct_source(), j_spec, density, {},
                                                expected_j, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  auto k_spec = strategy.spec;
  k_spec.derivative_order = 1;
  k_spec.coulomb.present = false;
  std::vector<double> expected_k;
  require(execute_cuda_direct_energy_derivative(plan.cuda_direct_source(), k_spec, density, {},
                                                expected_k, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  matrix(std::vector<double>(shell_full_range.begin(),
                             shell_full_range.begin() + derivative.coordinates_per_item),
         expected_j, "prepared shell J derivative");
  matrix(std::vector<double>(shell_full_range.begin() + derivative.coordinates_per_item,
                             shell_full_range.end()),
         expected_k, "prepared shell K derivative");

  std::vector<double> resident_response;
  require(execute_prepared_cuda_direct_rsh_energy_derivatives_device(
              plan, correction, device_density, nullptr, density.size(), resident_response,
              detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  int current_device = -1;
  require(cudaGetDevice(&current_device) == cudaSuccess && current_device == derivative.device_id,
          "resident prepared fused RSH derivative did not select the owning device");
  require(cudaFree(device_density) == cudaSuccess, "device density free failed");
  matrix(resident_response, response, "resident prepared fused RSH derivative");
}

void shell_rsh_range_derivative_reuse() {
  constexpr double omega = 0.31;
  constexpr double short_exchange = 0.22;
  constexpr double long_exchange = 0.67;
  for (unsigned angular : {0U, 1U, 2U}) {
    for (auto representation : {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL}) {
      auto system = fixture(false);
      system.shells = {{0, 0, {{0.8, 1.0}}}, {1, angular, {{0.6, 1.0}}}};
      system.basis_representation = representation;
      std::string detail;
      require(generativeqc::molecule::validate_and_normalize(system, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      const auto n = generativeqc::molecule::ao_count(system);
      const auto matrix_size = n * n;
      std::vector<double> alpha(matrix_size), beta(matrix_size);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
          alpha[i * n + j] = std::cos(0.31 * (i + j + 1)) / n;
          beta[i * n + j] = std::sin(0.27 * (i + j + 1)) / (2 * n);
        }

      std::size_t primitives = 0;
      for (const auto& shell : system.shells) primitives += shell.primitives.size();
      for (auto spin : {FockSpin::Restricted, FockSpin::Unrestricted}) {
        const auto strategy = resolve_fock_build(make_rsh_primary_fock_spec(spin, short_exchange),
                                                 FockBackend::Cuda, 0.0);
        const auto budget = cuda_direct_coulomb_device_bytes(1, n, system.atoms.size(),
                                                             system.shells.size(), primitives, 1);
        PreparedFockPlan plan(system, nullptr, strategy, 0, budget, 1);
        const auto binding = prepared_cuda_direct_derivative_binding(plan);
        require(binding, "range shell derivative plan did not retain Direct capability");

        std::vector<double> expected;
        require(execute_cuda_direct_rsh_energy_derivatives_item(
                    plan.cuda_direct_source(), 0, spin, strategy.spec.coulomb.coefficient,
                    strategy.spec.exchange.coefficient,
                    (spin == FockSpin::Restricted ? -0.5 : -1.0) * long_exchange, omega, alpha,
                    spin == FockSpin::Unrestricted ? beta : std::vector<double>{}, expected,
                    detail) == GENERATIVEQC_STATUS_SUCCESS,
                detail.c_str());

        double* device_alpha = nullptr;
        double* device_beta = nullptr;
        require(cudaMalloc(reinterpret_cast<void**>(&device_alpha), matrix_size * sizeof(double)) ==
                    cudaSuccess,
                "range shell alpha density allocation failed");
        require(cudaMemcpyAsync(device_alpha, alpha.data(), matrix_size * sizeof(double),
                                cudaMemcpyHostToDevice, binding.stream) == cudaSuccess,
                "range shell alpha density upload failed");
        if (spin == FockSpin::Unrestricted) {
          require(cudaMalloc(reinterpret_cast<void**>(&device_beta),
                             matrix_size * sizeof(double)) == cudaSuccess,
                  "range shell beta density allocation failed");
          require(cudaMemcpyAsync(device_beta, beta.data(), matrix_size * sizeof(double),
                                  cudaMemcpyHostToDevice, binding.stream) == cudaSuccess,
                  "range shell beta density upload failed");
        }
        std::vector<double> actual;
        require(
            execute_cuda_direct_shell_rsh_energy_derivatives_device(
                plan.cuda_direct_source(), spin, strategy.spec.coulomb.coefficient,
                strategy.spec.exchange.coefficient,
                (spin == FockSpin::Restricted ? -0.5 : -1.0) * long_exchange, omega, device_alpha,
                device_beta, matrix_size, actual, detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail.c_str());
        require(cudaFree(device_alpha) == cudaSuccess, "range shell alpha density free failed");
        if (device_beta)
          require(cudaFree(device_beta) == cudaSuccess, "range shell beta density free failed");
        matrix(actual, expected, "shell-native SR/LR derivative");
      }
    }
  }
}

void independent_reference_export() {
  // Calling the prepared entry directly exercises the independent host driver
  // even for a standard fitted HF pair. Its exported physical frame must not
  // inherit the compact solver's retained-state identity.
  const auto system = fixture(true);
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  spec.coulomb.approximation = spec.exchange.approximation = FockApproximation::DensityFitted;
  ScfOptions options;
  options.export_physical_reference = true;
  options.screening_tolerance = 0;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  options.resolved_fock_build = resolve_fock_build(spec, FockBackend::Cpu, 0);
  PreparedFockPlan cpu(system, &system, *options.resolved_fock_build, -1, 0);
  const auto expected = run_prepared_fock_strategy(cpu, options);
  require(expected.converged && expected.reference, "independent CPU export failed");
  options.resolved_fock_build = resolve_fock_build(spec, FockBackend::Cuda, 0);
  PreparedFockPlan gpu(system, &system, *options.resolved_fock_build, 0, 16U << 20);
  std::vector<double> seed;
  for (bool forces : {false, true, false}) {
    options.compute_forces = forces;
    const auto actual = run_prepared_fock_strategy(gpu, options, seed.empty() ? nullptr : &seed);
    require(actual.converged && actual.reference, "independent CUDA export failed");
    auto frame = *actual.reference;
    validate_physical_reference(frame);
    close(actual.energy, expected.energy, 1e-9, "independent exported energy");
    matrix(actual.density, expected.density, "independent exported density");
    matrix(frame.fock, expected.reference->fock, "independent exported physical Fock");
    matrix(frame.orbital_energies, expected.reference->orbital_energies,
           "independent exported canonical energies");
    if (forces)
      matrix(actual.forces, expected.forces, "independent exported complete forces");
    else
      require(actual.forces.empty(), "energy-only independent export computed forces");
    if (seed.empty()) seed = actual.density;
  }
}
}  // namespace
int main() {
  try {
    composed_items();
    molecular_endpoints();
    ragged_replay();
    prepared_replay();
    retained_direct_derivative_reuse();
    shell_rsh_range_derivative_reuse();
    independent_reference_export();
    std::cout << "CUDA common Fock composition: exact/DF/absent pairs, signed gradients, batch "
                 "items, SCF/replay/geometry PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
