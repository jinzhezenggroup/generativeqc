#include "scf/cuda_fock_execution.hpp"

#include "scf/cuda/metadata_upload.hpp"
#include "scf/cuda_density_fitting_device.hpp"
#include "scf/cuda_direct_jk_device.hpp"

namespace generativeqc::scf {
namespace {

bool exact_full_range(const FockTermSpec& term) noexcept {
  return !term.present ||
         (term.approximation == FockApproximation::Exact && term.op == FockOperator::FullRange);
}

bool exact_value_exchange(const FockTermSpec& term) noexcept {
  if (!term.present) return true;
  if (term.approximation != FockApproximation::Exact) return false;
  if (term.op == FockOperator::FullRange) return true;
  return (term.op == FockOperator::ShortRange || term.op == FockOperator::LongRange) &&
         term.omega >= 0.0;
}

bool fitted_full_range(const FockTermSpec& term) noexcept {
  return !term.present || (term.approximation == FockApproximation::DensityFitted &&
                           term.op == FockOperator::FullRange);
}

}  // namespace

PreparedCudaFockBinding prepared_cuda_fock_binding(const PreparedFockPlan& plan) noexcept {
  const auto& strategy = plan.strategy();
  if (strategy.backend != FockBackend::Cuda || strategy.spec.derivative_order != 0) return {};

  const bool exact =
      exact_full_range(strategy.spec.coulomb) && exact_value_exchange(strategy.spec.exchange);
  if (exact) {
    auto* source = plan.cuda_direct_source();
    if (!source) return {};
    const auto diagnostic = cuda_direct_jk_plan_diagnostic(source);
    if (!diagnostic.nbf) return {};
    return {cuda_direct_jk_device(source), cuda_direct_jk_stream(source), source, diagnostic.nbf};
  }

  const bool fitted =
      fitted_full_range(strategy.spec.coulomb) && fitted_full_range(strategy.spec.exchange);
  if (!fitted) return {};
  auto* source = plan.cuda_fitted_source();
  if (!source || !plan.diagnostic().nbf) return {};
  return {cuda_density_fitting_device(source), cuda_density_fitting_stream(source), source,
          plan.diagnostic().nbf};
}

PreparedCudaDirectDerivativeBinding prepared_cuda_direct_derivative_binding(
    const PreparedFockPlan& plan) noexcept {
  const auto& strategy = plan.strategy();
  if (strategy.backend != FockBackend::Cuda || strategy.spec.derivative_order != 0) return {};
  auto* source = plan.cuda_direct_source();
  if (!source) return {};
  const auto diagnostic = cuda_direct_jk_plan_diagnostic(source);
  if (!diagnostic.nbf || !diagnostic.coordinates_per_item || diagnostic.derivative_order < 1)
    return {};
  return {cuda_direct_jk_device(source),
          cuda_direct_jk_stream(source),
          source,
          diagnostic.nbf,
          diagnostic.coordinates_per_item,
          diagnostic.device_bytes,
          diagnostic.derivative_order};
}

generativeqc_status execute_prepared_cuda_direct_rsh_energy_derivatives(
    const PreparedFockPlan& plan, const ResolvedFockBuild& long_range_correction,
    const std::vector<double>& density, const std::vector<double>& beta,
    std::vector<double>& derivatives, std::string& detail) {
  const auto binding = prepared_cuda_direct_derivative_binding(plan);
  auto* source = plan.cuda_direct_source();
  const auto& primary = plan.strategy();
  const auto& p = primary.spec;
  const auto& c = long_range_correction.spec;
  if (!binding || !source) {
    detail = "prepared CUDA Fock owner did not retain Direct first-derivative capability";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  const bool valid_primary = primary.backend == FockBackend::Cuda && p.derivative_order == 0 &&
                             p.coulomb.present &&
                             p.coulomb.approximation == FockApproximation::Exact &&
                             p.coulomb.op == FockOperator::FullRange && p.exchange.present &&
                             p.exchange.approximation == FockApproximation::Exact &&
                             p.exchange.op == FockOperator::FullRange;
  const bool valid_correction =
      long_range_correction.backend == FockBackend::Cuda && c.derivative_order == 0 &&
      c.spin == p.spin && !c.coulomb.present && c.exchange.present &&
      c.exchange.approximation == FockApproximation::Exact &&
      c.exchange.op == FockOperator::LongRange && c.exchange.omega > 0.0 &&
      long_range_correction.screening_tolerance == primary.screening_tolerance;
  if (!valid_primary || !valid_correction) {
    detail = "prepared CUDA RSH derivative plans have incompatible scientific identity";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  return execute_cuda_direct_rsh_energy_derivatives_item(
      source, 0, p.spin, p.coulomb.coefficient, p.exchange.coefficient,
      p.exchange.coefficient + c.exchange.coefficient, c.exchange.omega, density, beta, derivatives,
      detail);
}

generativeqc_status execute_prepared_cuda_direct_rsh_energy_derivatives_device(
    const PreparedFockPlan& plan, const ResolvedFockBuild& long_range_correction,
    const double* density, const double* beta, std::size_t matrix_elements,
    std::vector<double>& derivatives, std::string& detail) {
  const auto binding = prepared_cuda_direct_derivative_binding(plan);
  auto* source = plan.cuda_direct_source();
  const auto& primary = plan.strategy();
  const auto& p = primary.spec;
  const auto& c = long_range_correction.spec;
  if (!binding || !source) {
    detail = "prepared CUDA Fock owner did not retain Direct first-derivative capability";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  const bool valid_primary = primary.backend == FockBackend::Cuda && p.derivative_order == 0 &&
                             p.coulomb.present &&
                             p.coulomb.approximation == FockApproximation::Exact &&
                             p.coulomb.op == FockOperator::FullRange && p.exchange.present &&
                             p.exchange.approximation == FockApproximation::Exact &&
                             p.exchange.op == FockOperator::FullRange;
  const bool valid_correction =
      long_range_correction.backend == FockBackend::Cuda && c.derivative_order == 0 &&
      c.spin == p.spin && !c.coulomb.present && c.exchange.present &&
      c.exchange.approximation == FockApproximation::Exact &&
      c.exchange.op == FockOperator::LongRange && c.exchange.omega > 0.0 &&
      long_range_correction.screening_tolerance == primary.screening_tolerance;
  if (!valid_primary || !valid_correction || matrix_elements != binding.nbf * binding.nbf) {
    detail = "prepared CUDA resident RSH derivative has incompatible scientific identity";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  auto status = execute_cuda_direct_shell_rsh_energy_derivatives_device(
      source, p.spin, p.coulomb.coefficient, p.exchange.coefficient,
      p.exchange.coefficient + c.exchange.coefficient, c.exchange.omega, density, beta,
      matrix_elements, derivatives, detail);
  if (status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
    return execute_cuda_direct_rsh_energy_derivatives_device(
        source, p.spin, p.coulomb.coefficient, p.exchange.coefficient,
        p.exchange.coefficient + c.exchange.coefficient, c.exchange.omega, density, beta,
        matrix_elements, derivatives, detail);
  }
  return status;
}

generativeqc_status execute_prepared_cuda_direct_shell_full_range_derivatives_device(
    const PreparedFockPlan& plan, const double* density, const double* beta,
    std::size_t matrix_elements, std::vector<double>& derivatives, std::string& detail) {
  const auto binding = prepared_cuda_direct_derivative_binding(plan);
  auto* source = plan.cuda_direct_source();
  const auto& strategy = plan.strategy();
  const auto& spec = strategy.spec;
  const bool valid =
      binding && source && strategy.backend == FockBackend::Cuda && spec.derivative_order == 0 &&
      spec.coulomb.present && spec.coulomb.approximation == FockApproximation::Exact &&
      spec.coulomb.op == FockOperator::FullRange &&
      (!spec.exchange.present || (spec.exchange.approximation == FockApproximation::Exact &&
                                  spec.exchange.op == FockOperator::FullRange)) &&
      matrix_elements == binding.nbf * binding.nbf;
  if (!valid) {
    detail = "prepared CUDA shell derivative has incompatible full-range scientific identity";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  return execute_cuda_direct_shell_full_range_derivatives_device(
      source, spec.spin, spec.coulomb.coefficient,
      spec.exchange.present ? spec.exchange.coefficient : 0.0, density, beta, matrix_elements,
      derivatives, detail);
}

generativeqc_status enqueue_prepared_cuda_fock(const PreparedFockPlan& plan, const double* density,
                                               const double* beta, std::size_t matrix_elements,
                                               double* coulomb, double* alpha_exchange,
                                               double* beta_exchange, int* numerical_error,
                                               bool mixed_coulomb, std::string& detail) {
  const auto binding = prepared_cuda_fock_binding(plan);
  if (!binding) {
    detail = "prepared CUDA Fock owner has no single-provider resident value execution";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }

  auto* exact = plan.cuda_direct_source();
  auto* fitted = plan.cuda_fitted_source();
  if (exact) {
    if (fitted) {
      detail = "prepared CUDA Fock facade refuses mixed resident providers";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    return mixed_coulomb ? enqueue_cuda_direct_jk_device_mixed_j(
                               exact, plan.strategy().spec, density, beta, matrix_elements, coulomb,
                               alpha_exchange, beta_exchange, numerical_error, detail)
                         : enqueue_cuda_direct_jk_device(exact, plan.strategy().spec, density, beta,
                                                         matrix_elements, coulomb, alpha_exchange,
                                                         beta_exchange, numerical_error, detail);
  }

  if (!fitted) {
    detail = "prepared CUDA Fock source became unavailable";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  if (mixed_coulomb) {
    detail = "prepared density-fitted CUDA Fock does not support mixed Coulomb precision";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  const auto& spec = plan.strategy().spec;
  const bool unrestricted = spec.spin == FockSpin::Unrestricted;
  const bool valid_outputs =
      (spec.coulomb.present ? coulomb != nullptr : coulomb == nullptr) &&
      (spec.exchange.present ? alpha_exchange != nullptr : alpha_exchange == nullptr) &&
      (spec.exchange.present && unrestricted ? beta_exchange != nullptr : beta_exchange == nullptr);
  if (matrix_elements != binding.nbf * binding.nbf || numerical_error == nullptr ||
      density == nullptr || (unrestricted ? beta == nullptr : beta != nullptr) || !valid_outputs) {
    detail = "prepared density-fitted CUDA Fock buffers, spin or dimensions are invalid";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  const auto reset = cudaMemsetAsync(numerical_error, 0, sizeof(*numerical_error), binding.stream);
  if (reset != cudaSuccess) {
    detail =
        std::string("reset prepared density-fitted CUDA Fock status: ") + cudaGetErrorString(reset);
    return cuda_execution::source_cuda_status(reset);
  }

  const JkTermSelection terms{spec.coulomb.present, spec.exchange.present};
  return spec.spin == FockSpin::Unrestricted
             ? execute_cuda_density_fitting_uhf_jk_device(fitted, density, beta, coulomb,
                                                          alpha_exchange, beta_exchange, detail,
                                                          terms, FockMatrixLayout::RowMajor)
             : execute_cuda_density_fitting_rhf_jk_device(fitted, density, coulomb, alpha_exchange,
                                                          detail, terms,
                                                          FockMatrixLayout::RowMajor);
}

generativeqc_status enqueue_prepared_cuda_exchange_correction(
    const PreparedFockPlan& plan, const ResolvedFockBuild& correction, const double* density,
    const double* beta, std::size_t matrix_elements, double* alpha_exchange, double* beta_exchange,
    int* numerical_error, std::string& detail) {
  const auto binding = prepared_cuda_fock_binding(plan);
  auto* source = plan.cuda_direct_source();
  const auto& primary = plan.strategy();
  const auto& spec = correction.spec;
  if (!binding || !source || correction.backend != FockBackend::Cuda ||
      spec.derivative_order != 0 || spec.spin != primary.spec.spin || spec.coulomb.present ||
      !spec.exchange.present || spec.exchange.approximation != FockApproximation::Exact ||
      spec.exchange.op != FockOperator::LongRange || spec.exchange.omega <= 0.0 ||
      correction.screening_tolerance != primary.screening_tolerance) {
    detail = "CUDA range correction is incompatible with the prepared primary Fock owner";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }

  return enqueue_cuda_direct_jk_device(source, spec, density, beta, matrix_elements, nullptr,
                                       alpha_exchange, beta_exchange, numerical_error, detail);
}

}  // namespace generativeqc::scf
