#include "scf/cuda/resident_final_validation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "runtime/cuda_component_trace.hpp"
#include "tensor/cuda_square_linalg.hpp"

namespace generativeqc::scf::cuda_execution {
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("resident final validation: ") +
                             cudaGetErrorString(status));
}

struct Drain {
  cudaStream_t stream;
  bool pending{true};
  ~Drain() {
    if (pending) (void)cudaStreamSynchronize(stream);
  }
  void finish() {
    check(cudaStreamSynchronize(stream));
    runtime::cuda_trace::trace_counter("explicit_synchronizations", 1);
    pending = false;
  }
};

bool intersects(const void* first, std::size_t first_bytes, const void* second,
                std::size_t second_bytes) {
  const auto left = reinterpret_cast<std::uintptr_t>(first);
  const auto right = reinterpret_cast<std::uintptr_t>(second);
  if (first_bytes > std::numeric_limits<std::uintptr_t>::max() - left ||
      second_bytes > std::numeric_limits<std::uintptr_t>::max() - right)
    throw std::invalid_argument("resident final-validation address range overflows");
  return left < right + second_bytes && right < left + first_bytes;
}
}  // namespace

bool resident_final_state_products(MatrixLibraryResources resources,
                                   std::span<const cuda_df::ValidationInputs> spins,
                                   ResidentFinalValidationWorkspace workspace, bool canonical,
                                   double nuclear, solver::FinalStateDiagnostic& diagnostic,
                                   std::string& detail) {
  diagnostic = {};
  detail.clear();
  if (!resources.blas_ || spins.empty() || spins.size() > 2 || !spins[0].n ||
      spins[0].n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      spins[0].n > std::numeric_limits<std::size_t>::max() / sizeof(double) / spins[0].n ||
      !workspace.partial ||
      workspace.partial_count < resident_final_validation_partial_count(spins[0].n) ||
      !std::isfinite(nuclear))
    throw std::invalid_argument("invalid resident final-validation resources");
  const auto dimension = spins[0].n;
  const auto matrix_bytes = dimension * dimension * sizeof(double);
  const auto partial_count = resident_final_validation_partial_count(dimension);
  const auto partial_bytes = partial_count * sizeof(cuda_df::ValidationPartial);
  for (std::size_t index = 0; index < workspace.matrices.size(); ++index) {
    const auto* matrix = workspace.matrices[index];
    if (!matrix || intersects(matrix, matrix_bytes, workspace.partial, partial_bytes))
      throw std::invalid_argument("invalid resident final-validation scratch");
    for (std::size_t other = 0; other < index; ++other)
      if (intersects(matrix, matrix_bytes, workspace.matrices[other], matrix_bytes))
        throw std::invalid_argument("resident final-validation scratch aliases");
  }
  for (const auto& input : spins) {
    if (input.n != dimension || input.occupied > dimension ||
        (input.weight != 1 && input.weight != 2) || !input.values || !input.info)
      throw std::invalid_argument("invalid resident final-validation frame");
    const auto check_input = [&](const double* input_matrix) {
      if (!input_matrix || intersects(input_matrix, matrix_bytes, workspace.partial, partial_bytes))
        throw std::invalid_argument("resident final-validation inputs alias scratch");
      for (const auto* matrix : workspace.matrices)
        if (intersects(input_matrix, matrix_bytes, matrix, matrix_bytes))
          throw std::invalid_argument("resident final-validation inputs alias scratch");
    };
    for (const auto* input_matrix : {input.f, input.s, input.h, input.d, input.c})
      check_input(input_matrix);
    if (input.expected_density) check_input(input.expected_density);
    if (intersects(input.values, dimension * sizeof(double), workspace.partial, partial_bytes) ||
        intersects(input.info, sizeof(int), workspace.partial, partial_bytes) ||
        (input.generation &&
         intersects(input.generation, sizeof(std::uint64_t), workspace.partial, partial_bytes)))
      throw std::invalid_argument("resident final-validation scalar inputs alias scratch");
    for (const auto* matrix : workspace.matrices)
      if (intersects(input.values, dimension * sizeof(double), matrix, matrix_bytes) ||
          intersects(input.info, sizeof(int), matrix, matrix_bytes) ||
          (input.generation &&
           intersects(input.generation, sizeof(std::uint64_t), matrix, matrix_bytes)))
        throw std::invalid_argument("resident final-validation scalar inputs alias scratch");
  }
  cudaStreamCaptureStatus capture{};
  check(cudaStreamIsCapturing(resources.stream_, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::invalid_argument("resident final validation cannot run during capture");

  runtime::cuda_trace::TraceOperation trace("resident_final_state_validation", resources.stream_,
                                            {1, dimension});
  runtime::cuda_trace::trace_counter("spin_channels", spins.size());
  const int extent = static_cast<int>(dimension);
  const auto blocks = cuda_df::validation_block_count(dimension);
  auto* packet = workspace.partial + 3 * blocks;
  std::array<cuda_df::ValidationPartial, 2> observed{};
  // Pageable result packets remain alive until the exceptional/ordinary drain.
  Drain drain{resources.stream_};
  const auto gemm = [&](const double* left, const double* right, double* output,
                        bool transpose_left = false, bool transpose_right = false, int rank = -1) {
    if (rank == 0) {
      check(cudaMemsetAsync(output, 0, matrix_bytes, resources.stream_));
      return;
    }
    const auto status =
        tensor::cuda::square_panel_product(resources.blas_, extent, rank < 0 ? extent : rank, left,
                                           transpose_left, right, transpose_right, output);
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("resident final-validation cuBLAS product failed");
    runtime::cuda_trace::trace_counter("validation_gemms", 1);
  };
  const auto [first, second, third, fourth] = workspace.matrices;
  for (std::size_t spin = 0; spin < spins.size(); ++spin) {
    const auto& input = spins[spin];
    const bool operators = input.transposed_operators;
    gemm(input.f, input.c, first, operators);
    gemm(input.s, input.c, second, operators);
    gemm(input.c, second, third, true);
    if (canonical) gemm(input.c, first, fourth, true);
    cuda_df::launch_validation_eigen(resources.stream_, input, first, second, third,
                                     canonical ? fourth : nullptr, workspace.partial);
    cuda_df::launch_validation_columns(resources.stream_, dimension, input.occupied, input.c,
                                       nullptr, input.weight, second);
    gemm(second, input.c, first, false, true, static_cast<int>(input.occupied));
    gemm(input.d, input.s, second, operators, operators);
    gemm(second, input.d, third, false, operators);
    cuda_df::launch_validation_density(resources.stream_, input, first, second, third,
                                       workspace.partial + blocks);
    gemm(input.f, second, fourth, operators);
    // Preserve both products for approximately symmetric inputs; FDS cannot
    // stand in for SDF by a transpose shortcut at the shared symmetry tolerance.
    gemm(input.s, input.d, first, operators, operators);
    gemm(first, input.f, third, false, operators);
    cuda_df::launch_validation_commutator(resources.stream_, dimension, fourth, third,
                                          workspace.partial + 2 * blocks);
    cuda_df::launch_validation_finish(resources.stream_, workspace.partial, packet, 3, blocks);
    check(cudaPeekAtLastError());
    check(cudaMemcpyAsync(&observed[spin], packet, sizeof(*packet), cudaMemcpyDeviceToHost,
                          resources.stream_));
  }
  drain.finish();
  runtime::cuda_trace::trace_counter("device_to_host_bytes", spins.size() * sizeof(*packet));
  diagnostic.energy = nuclear;
  diagnostic.eigenframes.assign(spins.size(), {});
  for (std::size_t spin = 0; spin < spins.size(); ++spin) {
    const auto& raw = observed[spin];
    if (raw.invalid & cuda_df::validation_input_failure)
      throw std::runtime_error("resident final-validation frame has invalid solver/generation");
    const double scale = raw.norm_f * raw.norm_c + raw.norm_rhs;
    if (raw.invalid || !std::isfinite(scale) || !std::isfinite(raw.norm_residual) ||
        !std::isfinite(raw.norm_density) || !std::isfinite(raw.electrons) ||
        !std::isfinite(raw.energy)) {
      detail = "nonfinite resident final-state products or invalid orbital order";
      return false;
    }
    auto& eigen = diagnostic.eigenframes[spin];
    eigen.maximum_eigen_residual = raw.eigen;
    eigen.maximum_metric_error = raw.metric;
    eigen.scaled_eigen_residual = scale == 0 ? raw.norm_residual : raw.norm_residual / scale;
    diagnostic.maximum_density_error = std::max(diagnostic.maximum_density_error, raw.density);
    diagnostic.maximum_canonical_error =
        std::max(diagnostic.maximum_canonical_error, raw.canonical);
    diagnostic.maximum_idempotency_error =
        std::max(diagnostic.maximum_idempotency_error, raw.idempotency);
    diagnostic.maximum_commutator = std::max(diagnostic.maximum_commutator, raw.commutator);
    diagnostic.density_rms = std::max(diagnostic.density_rms, raw.norm_density / dimension);
    diagnostic.maximum_trace_error =
        std::max(diagnostic.maximum_trace_error,
                 std::abs(raw.electrons - spins[spin].weight * spins[spin].occupied));
    diagnostic.energy += raw.energy;
  }
  return true;
}
}  // namespace generativeqc::scf::cuda_execution
