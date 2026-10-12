#include "scf/cuda/hf_initial_guess.hpp"

#include <chrono>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "generativeqc/generativeqc.hpp"
#include "integrals/s_integrals.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/cuda/eigensolver.hpp"
#include "scf/cuda/rhf_bucket_internal.hpp"
#include "scf/preliminary_guess.hpp"
#include "scf/solver/eigen_frame.hpp"

namespace generativeqc::scf {
namespace {

void check_copy(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess)
    throw generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR, cudaGetErrorString(status));
}

reference::EigenResult seed_eigen(CudaRhfBucketPlan& plan, const CudaHfSeedWorkspace& workspace,
                                  const reference::Matrix& input, std::size_t dimension) {
  if (dimension != plan.nbf || input.size() != dimension * dimension)
    throw std::invalid_argument("CUDA HF seed eigen dimension changed");
  reference::EigenResult frame;
  frame.values.resize(dimension);
  frame.vectors.resize(input.size());
  auto& resources = plan.resources;
  int info{};
  try {
    check_copy(cudaMemcpyAsync(workspace.matrix, input.data(), input.size() * sizeof(double),
                               cudaMemcpyHostToDevice, resources.stream_));
    const auto status = cuda_execution::launch_solver(
        resources.eigensolver_view(), plan.eigensolver_diagnostic.ordinary_family,
        static_cast<int>(dimension), 1, workspace.matrix, workspace.scratch, workspace.eigenvalues,
        plan.lwork, workspace.solver_info, workspace.active);
    if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
    if (status != GENERATIVEQC_STATUS_SUCCESS)
      throw generativeqc::Error(status, "CUDA HF seed eigensolver launch failed");
    check_copy(cudaMemcpyAsync(frame.vectors.data(), workspace.matrix,
                               input.size() * sizeof(double), cudaMemcpyDeviceToHost,
                               resources.stream_));
    check_copy(cudaMemcpyAsync(frame.values.data(), workspace.eigenvalues,
                               dimension * sizeof(double), cudaMemcpyDeviceToHost,
                               resources.stream_));
    check_copy(cudaMemcpyAsync(&info, workspace.solver_info, sizeof(info), cudaMemcpyDeviceToHost,
                               resources.stream_));
    check_copy(cudaStreamSynchronize(resources.stream_));
  } catch (...) {
    // Drain borrowed host buffers before stack/vector storage can expire.
    (void)cudaStreamSynchronize(resources.stream_);
    throw;
  }
  if (info != 0)
    throw generativeqc::Error(GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                              "CUDA HF seed eigensolver did not return a successful frame");
  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t column = row + 1; column < dimension; ++column)
      std::swap(frame.vectors[row * dimension + column], frame.vectors[column * dimension + row]);
  // Preserve physical values and columns together in degenerate occupation
  // clusters. Ordering never substitutes for the unchanged residual gates.
  (void)solver::order_eigenpairs(frame.values, frame.vectors, dimension);
  solver::EigenFrameDiagnostic diagnostic;
  diagnostic.solver_info = info;
  std::string detail;
  if (!solver::validate_eigen_frame(input, nullptr, frame.values, frame.vectors, dimension,
                                    diagnostic, detail))
    throw generativeqc::Error(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, detail);
  return frame;
}

}  // namespace

void prepare_hf_cuda_initial_densities(
    CudaRhfBucketPlan& plan, const cuda_execution::HostBatch& host,
    const std::vector<core::System>& systems, const ScfOptions& options,
    const CudaHfSeedWorkspace& workspace,
    std::vector<initial_guess::PreliminaryDiagnostic>& diagnostics) {
  if (!options.preliminary_guess || systems.size() != host.warm_mask.size() ||
      diagnostics.size() != systems.size())
    throw std::invalid_argument("CUDA HF seed preparation requires a matching batch policy");
  const auto& policy = *options.preliminary_guess;
  const auto stream = plan.resources.stream_;
  const auto matrix_size = host.nbf * host.nbf;
  const initial_guess::EigenOperation eigen = [&](const auto& matrix, const auto* overlap,
                                                  const auto* orthogonalizer, auto dimension) {
    if (overlap || orthogonalizer)
      throw std::invalid_argument("CUDA HF seed eigen operation accepts symmetric solves only");
    return seed_eigen(plan, workspace, matrix, dimension);
  };
  for (std::size_t index = 0; index < systems.size(); ++index) {
    auto& diagnostic = diagnostics[index];
    if (host.warm_mask[index]) {
      diagnostic.outcome = initial_guess::PreliminaryOutcome::ExplicitDensity;
      continue;
    }
    const auto started = std::chrono::steady_clock::now();
    std::optional<std::vector<double>> candidate;
    try {
      initial_guess::validate_preliminary_target(systems[index], *options.resolved_fock_build,
                                                 options);
      diagnostic.preparation_numeric_capacity =
          initial_guess::preliminary_numeric_capacity(systems[index], policy);
      if (diagnostic.preparation_numeric_capacity > policy.maximum_numeric_bytes) {
        diagnostic.outcome = initial_guess::PreliminaryOutcome::BudgetSkipped;
      } else {
        runtime::CpuRetainedCapacity retained(hf_cuda_retained_host_numeric_bytes(plan));
        runtime::CpuRetainedCapacity retained_input(cuda_execution::host_batch_numeric_bytes(host));
        integrals::IntegralData target;
        target.nbf = host.nbf;
        target.overlap.resize(matrix_size);
        try {
          check_copy(cudaMemcpyAsync(target.overlap.data(), workspace.overlap + index * matrix_size,
                                     matrix_size * sizeof(double), cudaMemcpyDeviceToHost, stream));
          check_copy(cudaStreamSynchronize(stream));
          candidate = initial_guess::prepare_huckel_initial_density(systems[index], target, policy,
                                                                    diagnostic, eigen);
        } catch (...) {
          (void)cudaStreamSynchronize(stream);
          throw;
        }
      }
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const std::exception&) {
      diagnostic.outcome = initial_guess::PreliminaryOutcome::PreparationFailed;
      diagnostic.work_counters_complete = false;
    }
    if (candidate) {
      // Publish only after strict admission, without warm normalization. A
      // failed publication is fatal: the target arena may be partially written
      // and must not masquerade as an untouched Hcore fallback.
      try {
        check_copy(cudaMemcpyAsync(workspace.density + index * matrix_size, candidate->data(),
                                   matrix_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        check_copy(cudaStreamSynchronize(stream));
      } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
      }
    }
    diagnostic.preparation_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  }
}
}  // namespace generativeqc::scf
