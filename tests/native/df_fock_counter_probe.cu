// Exercise the production assembly owner: capture is not execution, replay is.
#include <cuda_runtime.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>

#include "scf/cuda/df_scf_kernels.hpp"
#include "scf/cuda/scf_state_kernels.hpp"

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main() {
  using generativeqc::scf::cuda_df::launch_assemble_rhf_fock_kernel;
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  double* values{};
  std::uint64_t* count{};
  check(cudaMalloc(&values, 16 * sizeof(double)));
  check(cudaMalloc(&count, sizeof(std::uint64_t)));
  const std::array<double, 12> inputs{1, 2, 3, 4, 4, 3, 2, 1, 2, 4, 6, 8};
  check(cudaMemcpyAsync(values, inputs.data(), sizeof(inputs), cudaMemcpyHostToDevice, stream));
  auto reset = [&] { check(cudaMemsetAsync(count, 0, sizeof(std::uint64_t), stream)); };
  auto read = [&] {
    std::uint64_t result{};
    check(cudaMemcpyAsync(&result, count, sizeof(result), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return result;
  };
  auto assemble = [&](std::uint64_t* counter) {
    launch_assemble_rhf_fock_kernel(2, 32, 0, stream, 4, values, values + 4, values + 8,
                                    values + 12, counter);
    check(cudaGetLastError());
  };
  reset();
  check(cudaStreamSynchronize(stream));
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  assemble(count);
  assemble(count);
  check(cudaStreamEndCapture(stream, &graph));
  assert(read() == 0);
  check(cudaGraphInstantiate(&executable, graph, 0));
  for (unsigned i = 0; i < 3; ++i) check(cudaGraphLaunch(executable, stream));
  assemble(count);
  assert(read() == 7);
  std::array<double, 4> output{};
  check(cudaMemcpy(output.data(), values + 12, sizeof(output), cudaMemcpyDeviceToHost));
  for (unsigned i = 0; i < output.size(); ++i)
    assert(output[i] == inputs[i] + inputs[i + 4] - 0.5 * inputs[i + 8]);
  // A new solve resets the same persistent address before replay. Callers
  // without census ownership retain the null-counter compatibility path.
  reset();
  check(cudaGraphLaunch(executable, stream));
  assemble(nullptr);
  assert(read() == 2);
  check(cudaGraphExecDestroy(executable));
  check(cudaGraphDestroy(graph));

  // Direct work is masked per item. Changing the mask before graph replay
  // must change the executed census without any dependency on SCF history.
  std::uint64_t* direct_counts{};
  std::uint8_t* active{};
  check(cudaMalloc(&direct_counts, 2 * sizeof(std::uint64_t)));
  check(cudaMalloc(&active, 2));
  check(cudaMemsetAsync(direct_counts, 0, 2 * sizeof(std::uint64_t), stream));
  check(cudaMemsetAsync(active, 1, 2, stream));
  check(cudaStreamSynchronize(stream));
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  generativeqc::scf::cuda_execution::launch_count_fock_application_kernel(1, 32, 0, stream, 2,
                                                                          active, direct_counts);
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&executable, graph, 0));
  check(cudaGraphLaunch(executable, stream));
  check(cudaMemsetAsync(active, 0, 1, stream));
  check(cudaGraphLaunch(executable, stream));
  std::array<std::uint64_t, 2> direct{};
  check(cudaMemcpyAsync(direct.data(), direct_counts, sizeof(direct), cudaMemcpyDeviceToHost,
                        stream));
  check(cudaStreamSynchronize(stream));
  assert(direct[0] == 1 && direct[1] == 2);
  check(cudaGraphExecDestroy(executable));
  check(cudaGraphDestroy(graph));
  check(cudaFree(active));
  check(cudaFree(direct_counts));
  check(cudaFree(count));
  check(cudaFree(values));
  check(cudaStreamDestroy(stream));
}
