#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "molecule/nuclear_gradient.hpp"
#include "posthf/cuda_derivative.hpp"
#include "posthf/mp2_derivative.hpp"
#include "posthf/mp2_derivative_common.hpp"
#include "runtime/df_progress_trace.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda_one_electron_gradient.hpp"

namespace generativeqc::mp2 {
#if GENERATIVEQC_HAS_CUDA
namespace {
void check_cuda_derivative(generativeqc_status status, const std::string& detail) {
  if (status == GENERATIVEQC_STATUS_SUCCESS) return;
  const auto message = detail.empty() ? "CUDA conventional derivative failed" : detail;
  if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::length_error(message);
  if (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT) throw std::invalid_argument(message);
  throw std::runtime_error(message);
}

}  // namespace
#endif

std::vector<double> conventional_derivative_cuda(const core::System& system,
                                                 const hf::PhysicalReference& reference,
                                                 const LagrangianWeights& weights, int device_id,
                                                 std::size_t stage_budget) {
#if !GENERATIVEQC_HAS_CUDA
  (void)system;
  (void)reference;
  (void)weights;
  (void)device_id;
  (void)stage_budget;
  throw std::runtime_error("CUDA conventional derivative is unavailable in this build");
#else
  if (device_id < 0 || !stage_budget)
    throw std::invalid_argument("invalid CUDA conventional derivative request");
  runtime::df_progress::Scope trace("cuda_conventional_derivative");
  posthf::CudaEriDerivativeBatch batch(device_id, system, stage_budget, trace.enabled());
  auto derivative = detail::conventional_derivative_accumulate(
      system, reference, weights,
      [&](std::span<const double> overlap, std::span<const double> hcore) {
        std::vector<double> gradient;
        std::string detail;
        const auto status = scf::execute_cuda_one_electron_gradient(
            device_id, system, overlap, hcore, hcore,
            scf::cuda_policy::one_electron_derivative_mapping_requested(), stage_budget, gradient,
            detail);
        check_cuda_derivative(status, detail);
        molecule::add_nuclear_repulsion_gradient(system, gradient);
        return gradient;
      },
      [&](const std::array<std::size_t, 4>& shells, std::span<const double> local,
          std::span<double> gradient) { batch.append(shells, local, gradient); },
      [&](std::span<double> gradient) { batch.finish(gradient); });
  if (trace.enabled()) {
    // These completed consumer calls include record upload, primitive kernels,
    // result download, and stream/allocation lifetime. Host record construction
    // belongs to the enclosing shell callback, not consumer_ns.
    using runtime::df_progress::Scope;
    const auto& total = batch.diagnostic();
    Scope::number("batched_shells", batch.batched());
    Scope::number("batch_numeric_capacity_bytes", batch.numeric_capacity_bytes());
    Scope::number("primitive_records", total.primitive_records);
    Scope::number("consumer_calls", total.consumer_calls);
    Scope::number("consumer_ns", total.consumer_nanoseconds);
  }
  return derivative;
#endif
}

}  // namespace generativeqc::mp2
