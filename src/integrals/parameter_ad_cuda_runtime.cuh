// CUDA scheduling for the same compiler-owned scalar integral expressions.
// Ordered source tasks have disjoint value slots; pullbacks use private lane
// vectors followed by a deterministic reduction. No floating-point atomics or
// host integral evaluation enter either derivative order.
#include <stdexcept>
#include <vector>

namespace integral_ad {
constexpr unsigned cuda_lanes = 32;

static void checked(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class T>
struct DeviceArray {
  T* pointer{};
  explicit DeviceArray(std::size_t count) { checked(cudaMalloc(&pointer, count * sizeof(T))); }
  ~DeviceArray() {
    if (pointer) cudaFree(pointer);
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
};

template <class Scalar>
__global__ void execute_tasks(const Scalar* xyz, const Scalar* alpha, const Scalar* raw,
                              const Scalar* seeds, Scalar* result, Scalar* scratch, int* status) {
  const unsigned lane = threadIdx.x;
  Scalar* output = seeds ? result + lane * parameter_count : result;
  status[lane] = execute<Scalar>(xyz, alpha, raw, seeds, output, scratch + lane * scratch_count,
                                 lane, cuda_lanes);
}

template <class Scalar>
__global__ void reduce_tasks(Scalar* result, bool derivative) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (derivative) {
    if (index >= parameter_count) return;
    Scalar value = 0;
    for (unsigned lane = 0; lane < cuda_lanes; ++lane)
      value += result[lane * parameter_count + index];
    result[index] = value;
  } else if (index == 0) {
    Scalar value = 0;
    for (unsigned lane = 0; lane < cuda_lanes; ++lane) value += result[value_count - 1 + lane];
    result[value_count - 1] = value;
  }
}

template <class Scalar>
static int run_cuda(int device, const Scalar* xyz, const Scalar* alpha, const Scalar* raw,
                    const Scalar* seeds, Scalar* output) {
  checked(cudaSetDevice(device));
  const auto input_count = parameter_count + (seeds ? value_count : 0);
  const auto output_count = seeds ? parameter_count * cuda_lanes : value_count + cuda_lanes - 1;
  DeviceArray<Scalar> inputs(input_count), results(output_count),
      scratch(scratch_count * cuda_lanes);
  DeviceArray<int> status(cuda_lanes);
  auto* exponents = inputs.pointer + 3 * nat;
  auto* coefficients = exponents + nprim;
  auto* weights = coefficients + nprim;
  checked(cudaMemcpy(inputs.pointer, xyz, 3 * nat * sizeof(Scalar), cudaMemcpyHostToDevice));
  checked(cudaMemcpy(exponents, alpha, nprim * sizeof(Scalar), cudaMemcpyHostToDevice));
  checked(cudaMemcpy(coefficients, raw, nprim * sizeof(Scalar), cudaMemcpyHostToDevice));
  if (seeds)
    checked(cudaMemcpy(weights, seeds, value_count * sizeof(Scalar), cudaMemcpyHostToDevice));
  checked(cudaMemset(results.pointer, 0, output_count * sizeof(Scalar)));
  execute_tasks<<<1, cuda_lanes>>>(inputs.pointer, exponents, coefficients,
                                   seeds ? weights : nullptr, results.pointer, scratch.pointer,
                                   status.pointer);
  checked(cudaGetLastError());
  reduce_tasks<<<(parameter_count + 127) / 128, 128>>>(results.pointer, seeds != nullptr);
  checked(cudaGetLastError());
  int statuses[cuda_lanes];
  checked(cudaMemcpy(statuses, status.pointer, sizeof(statuses), cudaMemcpyDeviceToHost));
  for (int value : statuses)
    if (value) return value;
  checked(cudaMemcpy(output, results.pointer,
                     (seeds ? parameter_count : value_count) * sizeof(Scalar),
                     cudaMemcpyDeviceToHost));
  return 0;
}
}  // namespace integral_ad

extern "C" int integral_ad_execute(int device, const double* xyz, const double* alpha,
                                   const double* coefficients, const double* seeds,
                                   double* output) {
  try {
    return integral_ad::run_cuda(device, xyz, alpha, coefficients, seeds, output);
  } catch (...) {
    return 2;
  }
}

#ifdef GQC_INTEGRAL_AD_SECOND
extern "C" int integral_ad_directional(int device, const double* xyz, const double* alpha,
                                       const double* coefficients, const double* direction,
                                       const double* seeds, const double* seed_direction,
                                       double* output) {
  using namespace integral_ad;
  try {
    std::vector<Directional> inputs(parameter_count), weights(seeds ? value_count : 0);
    for (int i = 0; i < 3 * nat; ++i) inputs[i] = {xyz[i], direction[i]};
    for (int i = 0; i < nprim; ++i) {
      inputs[3 * nat + i] = {alpha[i], direction[3 * nat + i]};
      inputs[3 * nat + nprim + i] = {coefficients[i], direction[3 * nat + nprim + i]};
    }
    if (seeds)
      for (std::size_t i = 0; i < value_count; ++i)
        weights[i] = {seeds[i], seed_direction ? seed_direction[i] : 0.0};
    std::vector<Directional> result(seeds ? parameter_count : value_count);
    int status =
        run_cuda(device, inputs.data(), inputs.data() + 3 * nat, inputs.data() + 3 * nat + nprim,
                 seeds ? weights.data() : nullptr, result.data());
    if (status) return status;
    for (std::size_t i = 0; i < result.size(); ++i) output[i] = result[i].tangent;
    return 0;
  } catch (...) {
    return 2;
  }
}
#endif
