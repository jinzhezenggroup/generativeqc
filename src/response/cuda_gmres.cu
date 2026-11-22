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

struct IterationGraph {
  cudaGraph_t graph{};
  cudaGraphExec_t execution{};
  ~IterationGraph() {
    if (execution) (void)cudaGraphExecDestroy(execution);
    if (graph) (void)cudaGraphDestroy(graph);
  }
};

struct Panels {
  Panels(std::size_t dimension, const CudaGmresOptions& options, void* workspace) noexcept
      : restart(std::min({dimension, options.restart, options.max_iterations})),
        leading(restart + 1),
        grid(blocks(dimension)),
        state(static_cast<State*>(workspace)),
        base(reinterpret_cast<double*>(static_cast<unsigned char*>(workspace) + state_bytes)),
        residual(base + dimension),
        work(residual + dimension),
        candidate(work + dimension),
        image(candidate + dimension),
        basis(image + dimension),
        directions(basis + leading * dimension),
        hessenberg(directions + restart * dimension),
        cosine(hessenberg + leading * restart),
        sine(cosine + restart),
        target(sine + restart),
        coefficients(target + leading) {}

  std::size_t restart, leading;
  unsigned grid;
  State* state;
  double *base, *residual, *work, *candidate, *image, *basis, *directions, *hessenberg, *cosine,
      *sine, *target, *coefficients;
};
}  // namespace

cudaError_t cuda_gmres_workspace(std::size_t dimension, const CudaGmresOptions& options,
                                 std::size_t& bytes) noexcept {
  bytes = 0;
  if (!options.restart || !options.max_iterations ||
      options.max_iterations > (std::numeric_limits<std::size_t>::max() - 1) / 2 ||
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

namespace {
bool admit(std::size_t dimension, const CudaGmresOptions& options, CudaGmresAction action,
           const double* rhs, const double* diagonal, double* solution, void* workspace,
           std::size_t workspace_bytes, const double* initial_guess,
           std::size_t& required) noexcept {
  std::size_t vector_bytes{}, end{};
  if (cuda_gmres_workspace(dimension, options, required) != cudaSuccess ||
      workspace_bytes < required || !multiply_size(dimension, sizeof(double), vector_bytes) ||
      (dimension && !action) || !interval(workspace, workspace_bytes, end) ||
      !interval(rhs, vector_bytes, end) || !interval(solution, vector_bytes, end) ||
      (diagonal && !interval(diagonal, vector_bytes, end)) ||
      overlap(workspace, workspace_bytes, rhs, vector_bytes) ||
      overlap(workspace, workspace_bytes, solution, vector_bytes) ||
      (diagonal && overlap(workspace, workspace_bytes, diagonal, vector_bytes)) ||
      (initial_guess && (!interval(initial_guess, vector_bytes, end) ||
                         overlap(workspace, workspace_bytes, initial_guess, vector_bytes) ||
                         (initial_guess != solution &&
                          overlap(solution, vector_bytes, initial_guess, vector_bytes)))) ||
      overlap(solution, vector_bytes, rhs, vector_bytes) ||
      (diagonal && overlap(solution, vector_bytes, diagonal, vector_bytes)))
    return false;
  return true;
}

cudaError_t initialize(const Panels& panels, std::size_t required, std::size_t dimension,
                       const CudaGmresOptions& options, CudaGmresAction action, void* context,
                       const double* rhs, const double* diagonal, const double* initial_guess,
                       const int* upstream_status, cudaStream_t stream) noexcept {
  const auto vector_bytes = dimension * sizeof(double);
  auto error = cudaMemsetAsync(panels.state, 0, required, stream);
  if (error != cudaSuccess) return error;
  if (dimension) {
    error = cudaMemcpyAsync(panels.residual, rhs, vector_bytes, cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) return error;
  }
  kernels::validate<<<panels.grid, kernels::threads, 0, stream>>>(
      rhs, diagonal, dimension, options.diagonal_floor, upstream_status, panels.state);
  kernels::norm<<<1, kernels::threads, 0, stream>>>(rhs, dimension, &panels.state->rhs_norm,
                                                    panels.state, CudaGmresStatus::nonfinite_input);
  const bool seeded = initial_guess && dimension;
  if (seeded) {
    error =
        cudaMemcpyAsync(panels.base, initial_guess, vector_bytes, cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) return error;
    if ((error = action(context, panels.base, panels.image, dimension, stream)) != cudaSuccess)
      return error;
    kernels::residual<<<panels.grid, kernels::threads, 0, stream>>>(
        rhs, panels.image, panels.residual, dimension, panels.state);
    kernels::norm<<<1, kernels::threads, 0, stream>>>(panels.residual, dimension,
                                                      &panels.state->residual_norm, panels.state,
                                                      CudaGmresStatus::nonfinite_operator);
  }
  kernels::initial<<<1, 1, 0, stream>>>(panels.state, options.relative_tolerance,
                                        options.absolute_tolerance, seeded);
  return cudaGetLastError();
}

cudaError_t build_iteration_graph(const Panels& panels, std::size_t dimension,
                                  const CudaGmresOptions& options, CudaGmresAction action,
                                  void* context, const double* rhs, const double* diagonal,
                                  cudaStream_t stream, bool device_gate,
                                  IterationGraph& loop) noexcept {
  auto error = cudaGraphCreate(&loop.graph, 0);
  if (error != cudaSuccess) return error;
  cudaGraphConditionalHandle condition{};
  error = cudaGraphConditionalHandleCreate(&condition, loop.graph, device_gate ? 0 : 1,
                                           cudaGraphCondAssignDefault);
  if (error != cudaSuccess) return error;
  cudaGraphNode_t gate{};
  if (device_gate) {
    auto* state = panels.state;
    void* arguments[]{&state, &condition};
    cudaKernelNodeParams launch{};
    launch.func = reinterpret_cast<void*>(kernels::start_iteration);
    launch.gridDim = launch.blockDim = dim3(1);
    launch.kernelParams = arguments;
    error = cudaGraphAddKernelNode(&gate, loop.graph, nullptr, 0, &launch);
    if (error != cudaSuccess) return error;
  }
  cudaGraphNodeParams parameters{};
  parameters.type = cudaGraphNodeTypeConditional;
  parameters.conditional.handle = condition;
  parameters.conditional.type = cudaGraphCondTypeWhile;
  parameters.conditional.size = 1;
  cudaGraphNode_t iteration{};
  error = cudaGraphAddNode(&iteration, loop.graph, device_gate ? &gate : nullptr,
                           device_gate ? 1 : 0, &parameters);
  if (error != cudaSuccess) return error;
  const auto body = parameters.conditional.phGraph_out[0];
  error = cudaStreamBeginCaptureToGraph(stream, body, nullptr, nullptr, 0,
                                        cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) return error;
  const auto record = [&]() noexcept {
    const auto [restart, leading, grid, state, base, residual, work, candidate, image, basis,
                directions, hessenberg, cosine, sine, target, coefficients] = panels;
    // Action bindings are invariant across restart columns. The old A*x panel
    // is dead after the residual is formed, so it can stage the next A*z input.
    kernels::cycle<<<1, 1, 0, stream>>>(state, target);
    kernels::prepare_direction<<<grid, kernels::threads, 0, stream>>>(
        residual, basis, base, candidate, diagonal, directions, image, dimension, state);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if ((status = action(context, image, work, dimension, stream)) != cudaSuccess) return status;
    kernels::norm<<<1, kernels::threads, 0, stream>>>(work, dimension, &state->norm_before, state,
                                                      CudaGmresStatus::nonfinite_operator);
    kernels::orthogonalize<<<1, kernels::threads, 0, stream>>>(
        basis, work, hessenberg, dimension, leading, options.reorthogonalizations, state);
    kernels::norm<<<1, kernels::threads, 0, stream>>>(work, dimension, &state->norm_after, state,
                                                      CudaGmresStatus::numerical_failure);
    kernels::arnoldi<<<1, 1, 0, stream>>>(hessenberg, leading, state,
                                          options.breakdown_relative_tolerance);
    kernels::normalize<<<grid, kernels::threads, 0, stream>>>(work, basis, dimension, state);
    kernels::rotate<<<1, 1, 0, stream>>>(hessenberg, cosine, sine, target, coefficients, leading,
                                         state);
    kernels::candidate<<<grid, kernels::threads, 0, stream>>>(base, directions, coefficients,
                                                              candidate, dimension, state);
    if ((status = cudaGetLastError()) != cudaSuccess) return status;
    if ((status = action(context, candidate, image, dimension, stream)) != cudaSuccess)
      return status;
    kernels::residual<<<grid, kernels::threads, 0, stream>>>(rhs, image, residual, dimension,
                                                             state);
    kernels::norm<<<1, kernels::threads, 0, stream>>>(residual, dimension, &state->residual_norm,
                                                      state, CudaGmresStatus::nonfinite_operator);
    kernels::complete_iteration<<<1, 1, 0, stream>>>(state, restart, options.max_iterations,
                                                     condition);
    return cudaGetLastError();
  };
  error = record();
  cudaGraph_t recorded{};
  const auto captured = cudaStreamEndCapture(stream, &recorded);
  if (error != cudaSuccess) return error;
  if (captured != cudaSuccess) return captured;
  if (recorded != body) return cudaErrorInvalidResourceHandle;
  return cudaGraphInstantiate(&loop.execution, loop.graph);
}
}  // namespace

CudaGmresReplay::~CudaGmresReplay() {
  if (execution_) (void)cudaGraphExecDestroy(execution_);
}

cudaError_t CudaGmresReplay::prepare(std::size_t dimension, const CudaGmresOptions& options,
                                     CudaGmresAction action, void* context, const double* rhs,
                                     const double* diagonal, double* solution, void* workspace,
                                     std::size_t workspace_bytes, cudaStream_t stream) noexcept {
  std::size_t required{};
  if (receipt_ || !admit(dimension, options, action, rhs, diagonal, solution, workspace,
                         workspace_bytes, nullptr, required))
    return cudaErrorInvalidValue;
  const Panels panels(dimension, options, workspace);
  if (dimension) {
    IterationGraph loop;
    const auto error = build_iteration_graph(panels, dimension, options, action, context, rhs,
                                             diagonal, stream, true, loop);
    if (error != cudaSuccess) return error;
    execution_ = loop.execution;
    loop.execution = nullptr;
  }
  stream_ = stream;
  options_ = options;
  rhs_ = rhs;
  diagonal_ = diagonal;
  solution_ = solution;
  workspace_ = workspace;
  dimension_ = dimension;
  required_ = required;
  receipt_ = static_cast<const CudaGmresDeviceReceipt*>(panels.state);
  return cudaSuccess;
}

cudaError_t CudaGmresReplay::enqueue(const int* upstream_status) const noexcept {
  if (!receipt_) return cudaErrorInvalidResourceHandle;
  const Panels panels(dimension_, options_, workspace_);
  auto error = initialize(panels, required_, dimension_, options_, nullptr, nullptr, rhs_,
                          diagonal_, nullptr, upstream_status, stream_);
  if (error != cudaSuccess || !dimension_) return error;
  if ((error = cudaGraphLaunch(execution_, stream_)) != cudaSuccess) return error;
  kernels::copy_converged<<<panels.grid, kernels::threads, 0, stream_>>>(
      panels.base, panels.candidate, solution_, dimension_, panels.state);
  return cudaGetLastError();
}

cudaError_t solve_cuda_gmres(std::size_t dimension, const CudaGmresOptions& options,
                             CudaGmresAction action, void* context, const double* rhs,
                             const double* diagonal, double* solution, void* workspace,
                             std::size_t workspace_bytes, cudaStream_t stream,
                             CudaGmresResult& result, const double* initial_guess) noexcept {
  result = {};
  std::size_t required{};
  if (!admit(dimension, options, action, rhs, diagonal, solution, workspace, workspace_bytes,
             initial_guess, required))
    return cudaErrorInvalidValue;
  result.workspace_bytes = required;
  const Panels panels(dimension, options, workspace);
  auto error = initialize(panels, required, dimension, options, action, context, rhs, diagonal,
                          initial_guess, nullptr, stream);
  if (initial_guess && dimension) ++result.operator_actions;
  if (error != cudaSuccess) return settle(stream, error);

  int checkpoint{};
  error = cudaMemcpyAsync(&checkpoint, &panels.state->code, sizeof(checkpoint),
                          cudaMemcpyDeviceToHost, stream);
  result.checkpoint_bytes += sizeof(checkpoint);
  if ((error = settle(stream, error)) != cudaSuccess) return error;
  IterationGraph loop;
  if (!checkpoint &&
      ((error = build_iteration_graph(panels, dimension, options, action, context, rhs, diagonal,
                                      stream, false, loop)) != cudaSuccess ||
       (error = cudaGraphLaunch(loop.execution, stream)) != cudaSuccess))
    return settle(stream, error);
  CudaGmresDeviceReceipt terminal{};
  error = cudaMemcpyAsync(&terminal, static_cast<const CudaGmresDeviceReceipt*>(panels.state),
                          sizeof(terminal), cudaMemcpyDeviceToHost, stream);
  if ((error = settle(stream, error)) != cudaSuccess) return error;
  result.status = terminal.code ? static_cast<CudaGmresStatus>(terminal.code)
                                : CudaGmresStatus::iteration_limit;
  result.rhs_norm = terminal.rhs_norm;
  result.residual_norm = terminal.residual_norm;
  result.tolerance = terminal.tolerance;
  result.iterations = terminal.iterations;
  result.restarts = terminal.restarts;
  result.operator_actions += 2 * terminal.iterations;
  result.preconditioner_actions = diagonal ? terminal.iterations : 0;
  if (result.converged() && dimension) {
    error = cudaMemcpyAsync(solution, checkpoint ? panels.base : panels.candidate,
                            dimension * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    return settle(stream, error);
  }
  return cudaSuccess;
}
}  // namespace generativeqc::response
