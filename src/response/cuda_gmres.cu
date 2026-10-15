#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "response/cuda_gmres.hpp"
#include "response/cuda_gmres_kernels.cuh"

namespace generativeqc::response {
namespace {
namespace kernels = cuda_gmres_detail;
using kernels::State;

constexpr std::size_t state_bytes =
    ((sizeof(State) + alignof(double) - 1) / alignof(double)) * alignof(double);

bool add_size(std::size_t left, std::size_t right, std::size_t& output) noexcept {
  if (right > std::numeric_limits<std::size_t>::max() - left) return false;
  output = left + right;
  return true;
}

bool multiply_size(std::size_t left, std::size_t right, std::size_t& output) noexcept {
  if (right && left > std::numeric_limits<std::size_t>::max() / right) return false;
  output = left * right;
  return true;
}

bool interval(const void* pointer, std::size_t bytes, std::uintptr_t& end) noexcept {
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes && (!pointer || begin % alignof(double))) return false;
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) return false;
  end = begin + bytes;
  return true;
}

bool overlap(const void* first, std::size_t first_bytes, const void* second,
             std::size_t second_bytes) noexcept {
  return first_bytes && second_bytes &&
         reinterpret_cast<std::uintptr_t>(first) <
             reinterpret_cast<std::uintptr_t>(second) + second_bytes &&
         reinterpret_cast<std::uintptr_t>(second) <
             reinterpret_cast<std::uintptr_t>(first) + first_bytes;
}

unsigned blocks(std::size_t count) noexcept {
  return static_cast<unsigned>(std::max<std::size_t>(
      1, std::min<std::size_t>(65535, count / kernels::threads + (count % kernels::threads != 0))));
}

cudaError_t settle(cudaStream_t stream, cudaError_t error) noexcept {
  const auto completion = cudaStreamSynchronize(stream);
  return error == cudaSuccess ? completion : error;
}
}  // namespace

cudaError_t cuda_gmres_workspace(std::size_t dimension, const CudaGmresOptions& options,
                                 std::size_t& bytes) noexcept {
  bytes = 0;
  if (!options.restart || !options.max_iterations ||
      (options.reorthogonalizations != 1 && options.reorthogonalizations != 2) ||
      !std::isfinite(options.relative_tolerance) || options.relative_tolerance < 0 ||
      options.relative_tolerance >= 1 || !std::isfinite(options.absolute_tolerance) ||
      options.absolute_tolerance < 0 ||
      (options.relative_tolerance == 0 && options.absolute_tolerance == 0) ||
      !std::isfinite(options.breakdown_relative_tolerance) ||
      options.breakdown_relative_tolerance < 0 || options.breakdown_relative_tolerance >= 1 ||
      !std::isfinite(options.diagonal_floor) || options.diagonal_floor < 0)
    return cudaErrorInvalidValue;

  const auto restart = std::min({dimension, options.restart, options.max_iterations});
  std::size_t slots{}, vectors{}, leading{}, hessenberg{}, scalars{}, elements{}, total{};
  if (!multiply_size(restart, 2, slots) || !add_size(slots, 6, slots) ||
      !multiply_size(slots, dimension, vectors) || !add_size(restart, 1, leading) ||
      !multiply_size(leading, restart, hessenberg) || !multiply_size(restart, 4, scalars) ||
      !add_size(scalars, 1, scalars) || !add_size(scalars, hessenberg, scalars) ||
      !add_size(vectors, scalars, elements) || !multiply_size(elements, sizeof(double), total) ||
      !add_size(total, state_bytes, total) || total > options.max_workspace_bytes)
    return cudaErrorInvalidValue;
  bytes = total;
  return cudaSuccess;
}

cudaError_t solve_cuda_gmres(std::size_t dimension, const CudaGmresOptions& options,
                             CudaGmresAction action, void* context, const double* rhs,
                             const double* diagonal, double* solution, void* workspace,
                             std::size_t workspace_bytes, cudaStream_t stream,
                             CudaGmresResult& result) noexcept {
  result = {};
  std::size_t required{}, vector_bytes{}, end{};
  if (cuda_gmres_workspace(dimension, options, required) != cudaSuccess ||
      workspace_bytes < required || !multiply_size(dimension, sizeof(double), vector_bytes) ||
      (dimension && !action) || !interval(workspace, workspace_bytes, end) ||
      !interval(rhs, vector_bytes, end) || !interval(solution, vector_bytes, end) ||
      (diagonal && !interval(diagonal, vector_bytes, end)) ||
      overlap(workspace, workspace_bytes, rhs, vector_bytes) ||
      overlap(workspace, workspace_bytes, solution, vector_bytes) ||
      (diagonal && overlap(workspace, workspace_bytes, diagonal, vector_bytes)) ||
      overlap(solution, vector_bytes, rhs, vector_bytes) ||
      (diagonal && overlap(solution, vector_bytes, diagonal, vector_bytes)))
    return cudaErrorInvalidValue;
  result.workspace_bytes = required;

  const auto restart = std::min({dimension, options.restart, options.max_iterations});
  const auto leading = restart + 1;
  const unsigned grid = blocks(dimension);
  auto* state = static_cast<State*>(workspace);
  auto* base = reinterpret_cast<double*>(static_cast<unsigned char*>(workspace) + state_bytes);
  auto* residual = base + dimension;
  auto* work = residual + dimension;
  auto* candidate = work + dimension;
  auto* image = candidate + dimension;
  auto* basis = image + dimension;
  auto* directions = basis + leading * dimension;
  auto* hessenberg = directions + restart * dimension;
  auto* cosine = hessenberg + leading * restart;
  auto* sine = cosine + restart;
  auto* target = sine + restart;
  auto* coefficients = target + leading;

  auto error = cudaMemsetAsync(workspace, 0, required, stream);
  if (error != cudaSuccess) return settle(stream, error);
  if (dimension) {
    error = cudaMemcpyAsync(residual, rhs, vector_bytes, cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) return settle(stream, error);
  }
  kernels::validate<<<grid, kernels::threads, 0, stream>>>(rhs, diagonal, dimension,
                                                           options.diagonal_floor, state);
  kernels::norm<<<1, kernels::threads, 0, stream>>>(rhs, dimension, &state->rhs_norm, state,
                                                    CudaGmresStatus::nonfinite_input);
  kernels::initial<<<1, 1, 0, stream>>>(state, options.relative_tolerance,
                                        options.absolute_tolerance);

  int checkpoint{};
  auto inspect = [&]() noexcept {
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return settle(stream, status);
    status = cudaMemcpyAsync(&checkpoint, &state->code, sizeof(checkpoint), cudaMemcpyDeviceToHost,
                             stream);
    result.checkpoint_bytes += sizeof(checkpoint);
    return settle(stream, status);
  };
  auto finish = [&](const double* accepted) noexcept {
    State terminal{};
    auto status =
        cudaMemcpyAsync(&terminal, state, sizeof(terminal), cudaMemcpyDeviceToHost, stream);
    status = settle(stream, status);
    if (status != cudaSuccess) return status;
    result.status = terminal.code ? static_cast<CudaGmresStatus>(terminal.code)
                                  : CudaGmresStatus::iteration_limit;
    result.rhs_norm = terminal.rhs_norm;
    result.residual_norm = terminal.residual_norm;
    result.tolerance = terminal.tolerance;
    if (result.converged() && dimension) {
      status = cudaMemcpyAsync(solution, accepted, vector_bytes, cudaMemcpyDeviceToDevice, stream);
      return settle(stream, status);
    }
    return cudaSuccess;
  };

  error = inspect();
  if (error != cudaSuccess) return error;
  if (checkpoint) return finish(base);

  while (result.iterations < options.max_iterations) {
    error = cudaMemsetAsync(hessenberg, 0, (leading * restart + 4 * restart + 1) * sizeof(double),
                            stream);
    if (error != cudaSuccess) return settle(stream, error);
    kernels::cycle<<<1, 1, 0, stream>>>(state, target);
    kernels::normalize<<<grid, kernels::threads, 0, stream>>>(residual, basis, dimension,
                                                              &state->beta, state, false);
    for (std::size_t column = 0; column < restart && result.iterations < options.max_iterations;
         ++column) {
      auto* direction = directions + column * dimension;
      kernels::precondition<<<grid, kernels::threads, 0, stream>>>(
          basis + column * dimension, diagonal, direction, dimension, state);
      error = cudaGetLastError();
      if (error != cudaSuccess) return settle(stream, error);
      error = action(context, direction, work, dimension, stream);
      ++result.operator_actions;
      if (diagonal) ++result.preconditioner_actions;
      if (error != cudaSuccess) return settle(stream, error);
      kernels::norm<<<1, kernels::threads, 0, stream>>>(work, dimension, &state->norm_before, state,
                                                        CudaGmresStatus::nonfinite_operator);
      for (unsigned pass = 0; pass < options.reorthogonalizations; ++pass) {
        for (std::size_t row = 0; row <= column; ++row) {
          kernels::dot<<<1, kernels::threads, 0, stream>>>(basis + row * dimension, work, dimension,
                                                           state);
          kernels::orthogonalize<<<grid, kernels::threads, 0, stream>>>(basis + row * dimension,
                                                                        work, dimension, state);
          kernels::accumulate<<<1, 1, 0, stream>>>(hessenberg + column * leading + row, state);
        }
      }
      kernels::norm<<<1, kernels::threads, 0, stream>>>(work, dimension, &state->norm_after, state,
                                                        CudaGmresStatus::numerical_failure);
      kernels::arnoldi<<<1, 1, 0, stream>>>(hessenberg + column * leading + column + 1, state,
                                            options.breakdown_relative_tolerance);
      kernels::normalize<<<grid, kernels::threads, 0, stream>>>(
          work, basis + (column + 1) * dimension, dimension, &state->norm_after, state, true);
      kernels::rotate<<<1, 1, 0, stream>>>(hessenberg, cosine, sine, target, coefficients, leading,
                                           column, state);
      kernels::candidate<<<grid, kernels::threads, 0, stream>>>(
          base, directions, coefficients, candidate, dimension, column + 1, state);
      error = cudaGetLastError();
      if (error != cudaSuccess) return settle(stream, error);
      error = action(context, candidate, image, dimension, stream);
      ++result.operator_actions;
      if (error != cudaSuccess) return settle(stream, error);
      kernels::residual<<<grid, kernels::threads, 0, stream>>>(rhs, image, residual, dimension,
                                                               state);
      kernels::norm<<<1, kernels::threads, 0, stream>>>(residual, dimension, &state->residual_norm,
                                                        state, CudaGmresStatus::nonfinite_operator);
      kernels::accept<<<1, 1, 0, stream>>>(state);
      ++result.iterations;
      error = inspect();
      if (error != cudaSuccess) return error;
      if (checkpoint) return finish(candidate);
    }
    if (result.iterations < options.max_iterations) {
      error = cudaMemcpyAsync(base, candidate, vector_bytes, cudaMemcpyDeviceToDevice, stream);
      if (error != cudaSuccess) return settle(stream, error);
      ++result.restarts;
    }
  }
  return finish(candidate);
}
}  // namespace generativeqc::response
