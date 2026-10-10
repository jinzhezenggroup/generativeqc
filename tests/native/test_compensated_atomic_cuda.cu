#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "runtime/compensated_atomic.cuh"
#include "runtime/cuda_resources.cuh"
#include "scf/aot_shell_registry.hpp"
#include "scf/generated_shell_task.hpp"

namespace {

__global__ void scatter(const double* values, std::size_t count,
                        generativeqc::runtime::CompensatedOutput output) {
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    atomicAdd(output + 0U, values[index]);
}

/** Closed-form dyadic reference avoids reproducing the device summation.
 * Shuffle the same cancellation-heavy multiset across launches and block
 * shapes; residual recovery must cover both magnitude orderings and signs.
 */
void check_cancellation(cudaStream_t stream) {
  using namespace generativeqc::runtime;
  constexpr std::size_t copies = 8192;
  constexpr std::array<double, 5> terms{0x1p60, 1.0, -0x1p60, -0.5, 0.25};
  std::vector<double> values;
  for (std::size_t copy = 0; copy < copies; ++copy)
    values.insert(values.end(), terms.begin(), terms.end());
  OwnedCudaBuffer<double> input(0, values.size(), stream), output(0, 2, stream);
  std::mt19937 random(2019);
  for (unsigned repeat = 0; repeat < 24; ++repeat) {
    std::shuffle(values.begin(), values.end(), random);
    cuda_resource_check(cudaMemcpyAsync(input.get(), values.data(), values.size() * sizeof(double),
                                        cudaMemcpyHostToDevice, stream));
    cuda_resource_check(cudaMemsetAsync(output.get(), 0, 2 * sizeof(double), stream));
    scatter<<<1 + repeat * 13, 32 + 32 * (repeat % 8), 0, stream>>>(
        input.get(), values.size(), {output.get(), output.get() + 1});
    cuda_resource_check(cudaGetLastError());
    std::array<double, 2> actual{};
    cuda_resource_check(cudaMemcpyAsync(actual.data(), output.get(), sizeof(actual),
                                        cudaMemcpyDeviceToHost, stream));
    cuda_resource_check(cudaStreamSynchronize(stream));
    if (actual[0] + actual[1] != copies * 0.75)
      throw std::runtime_error("compensated scatter lost exact dyadic sum");
  }
}

/** Nonfinite arithmetic cannot be repaired into a plausible finite output. */
void check_nonfinite(cudaStream_t stream) {
  using namespace generativeqc::runtime;
  OwnedCudaBuffer<double> input(0, 2, stream), output(0, 2, stream);
  for (double value :
       {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::max()}) {
    const std::array<double, 2> values{value, value};
    cuda_resource_check(cudaMemcpyAsync(input.get(), values.data(), sizeof(values),
                                        cudaMemcpyHostToDevice, stream));
    cuda_resource_check(cudaMemsetAsync(output.get(), 0, 2 * sizeof(double), stream));
    scatter<<<1, 32, 0, stream>>>(input.get(), 2, {output.get(), output.get() + 1});
    cuda_resource_check(cudaGetLastError());
    std::array<double, 2> actual{};
    cuda_resource_check(cudaMemcpyAsync(actual.data(), output.get(), sizeof(actual),
                                        cudaMemcpyDeviceToHost, stream));
    cuda_resource_check(cudaStreamSynchronize(stream));
    if (std::isfinite(actual[0] + actual[1]))
      throw std::runtime_error("compensated scatter hid nonfinite arithmetic");
  }
}

/** The unnormalized coincident ssss integral is pi^(5/2)/4. A unit RHF
 * density therefore gives pi^(5/2)/8. Starting its Fock entry at 2^60 and
 * cancelling that value after the generated writer exposes lost atomics;
 * independent nonzero sentinels also protect the task's physical offset.
 */
void check_generated_fock(cudaStream_t stream) {
  using namespace generativeqc::runtime;
  namespace generated = generativeqc::scf::generated;
  using generativeqc::scf::detail::GeneratedPrimitivePairData;
  using generativeqc::scf::detail::GeneratedShellTask;
  cudaDeviceProp properties{};
  cuda_resource_check(cudaGetDeviceProperties(&properties, 0));
  generated::select_profile_for_device(0, properties.major, properties.minor);
  if ((generated::enabled_fock_shell_class_mask() & 1U) == 0U) return;

  GeneratedShellTask task{};
  for (unsigned center = 0; center < 4; ++center) task.primitive_end[center] = 1;
  task.matrix_order = 1;
  task.density_offset = task.spin_offset = 1;
  const GeneratedPrimitivePairData pair{2.0, 0.5, {0.0, 0.0, 0.0}, 1.0, 0.5, 0.5};
  const std::array<std::int64_t, 2> offsets{0, 1};
  const std::array<double, 3> positions{};
  const std::array<double, 2> density{0.0, 1.0};
  const double coefficient = 1.0, cancellation = -0x1p60;
  OwnedCudaBuffer<GeneratedShellTask> tasks(0, 1, stream);
  OwnedCudaBuffer<GeneratedPrimitivePairData> pairs(0, 1, stream);
  OwnedCudaBuffer<std::int64_t> pair_offsets(0, 2, stream);
  OwnedCudaBuffer<double> coordinates(0, 3, stream), coefficients(0, 1, stream),
      densities(0, 2, stream), sum(0, 2, stream), correction(0, 2, stream),
      cancel_value(0, 1, stream);
  OwnedCudaBuffer<std::uint32_t> queue(0, 3, stream);
  const auto upload = [&](void* destination, const void* source, std::size_t bytes) {
    cuda_resource_check(
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
  };
  upload(tasks.get(), &task, sizeof(task));
  upload(pairs.get(), &pair, sizeof(pair));
  upload(pair_offsets.get(), offsets.data(), sizeof(offsets));
  upload(coordinates.get(), positions.data(), sizeof(positions));
  upload(coefficients.get(), &coefficient, sizeof(coefficient));
  upload(densities.get(), density.data(), sizeof(density));
  upload(cancel_value.get(), &cancellation, sizeof(cancellation));
  for (bool compensated : {false, true}) {
    const std::array<std::uint32_t, 3> queue_state{0, 1, 0};
    const std::array<double, 2> initial_sum{17.0, compensated ? 0x1p60 : 0.0};
    const std::array<double, 2> initial_correction{23.0, 0.0};
    upload(queue.get(), queue_state.data(), sizeof(queue_state));
    upload(sum.get(), initial_sum.data(), sizeof(initial_sum));
    upload(correction.get(), initial_correction.data(), sizeof(initial_correction));
    cuda_resource_check(generated::launch_shell_class_fock(
        0, stream, false, 4, tasks.get(), queue.get(), pair_offsets.get(), pairs.get(),
        coefficients.get(), coordinates.get(), 0.0, nullptr, densities.get(),
        {sum.get(), compensated ? correction.get() : nullptr}, queue.get() + 1, queue.get() + 2));
    if (compensated) {
      scatter<<<1, 1, 0, stream>>>(cancel_value.get(), 1, {sum.get() + 1, correction.get() + 1});
      cuda_resource_check(cudaGetLastError());
    }
    std::array<double, 2> actual_sum{}, actual_correction{};
    cuda_resource_check(cudaMemcpyAsync(actual_sum.data(), sum.get(), sizeof(actual_sum),
                                        cudaMemcpyDeviceToHost, stream));
    cuda_resource_check(cudaMemcpyAsync(actual_correction.data(), correction.get(),
                                        sizeof(actual_correction), cudaMemcpyDeviceToHost, stream));
    cuda_resource_check(cudaStreamSynchronize(stream));
    const double expected = std::pow(std::acos(-1.0), 2.5) / 8.0;
    if (std::fabs(actual_sum[1] + actual_correction[1] - expected) > 1e-12 ||
        actual_sum[0] != 17.0 || actual_correction[0] != 23.0)
      throw std::runtime_error("generated Fock lost its compensated output or physical offset");
  }
  std::puts("generated compensated Fock analytic gate passed");
}
}  // namespace

int main() {
  cudaStream_t stream{};
  try {
    generativeqc::runtime::cuda_resource_check(cudaSetDevice(0));
    generativeqc::runtime::cuda_resource_check(cudaStreamCreate(&stream));
    check_cancellation(stream);
    check_nonfinite(stream);
    check_generated_fock(stream);
    generativeqc::runtime::cuda_resource_check(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception& failure) {
    if (stream) cudaStreamDestroy(stream);
    std::fprintf(stderr, "%s\n", failure.what());
    return 1;
  }
}
