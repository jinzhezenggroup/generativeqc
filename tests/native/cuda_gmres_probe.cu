#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "response/cuda_gmres.hpp"

namespace {
namespace response = generativeqc::response;
constexpr double sentinel = 314159.25;

void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};

template <class Value>
struct Device {
  Value* value{};
  std::size_t count;
  explicit Device(std::size_t size) : count(size) {
    if (count) check(cudaMalloc(reinterpret_cast<void**>(&value), count * sizeof(Value)));
  }
  ~Device() { cudaFree(value); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void upload(const std::vector<Value>& input, cudaStream_t stream) {
    if (input.size() != count) throw std::runtime_error("probe upload shape mismatch");
    if (count)
      check(cudaMemcpyAsync(value, input.data(), count * sizeof(Value), cudaMemcpyHostToDevice,
                            stream));
  }
};

__global__ void matvec(const double* matrix, const double* input, double* output, std::size_t count,
                       bool poison, double factor) {
  const std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= count) return;
  double value = 0;
  for (std::size_t column = 0; column < count; ++column)
    value += matrix[row * count + column] * input[column];
  output[row] = poison ? nan("") : factor * value;
}

struct Operator {
  const double* matrix;
  bool poison;
  bool fail;
  bool inconsistent;
  bool fail_second;
  bool poison_second;
  unsigned calls{};
};

cudaError_t action(void* context, const double* input, double* output, std::size_t count,
                   cudaStream_t stream) noexcept {
  auto& owner = *static_cast<Operator*>(context);
  ++owner.calls;
  if (owner.fail || (owner.fail_second && owner.calls == 2)) return cudaErrorInvalidValue;
  matvec<<<static_cast<unsigned>((count + 127) / 128), 128, 0, stream>>>(
      owner.matrix, input, output, count, owner.poison || (owner.poison_second && owner.calls == 2),
      owner.inconsistent && owner.calls == 2 ? 2.0 : 1.0);
  return cudaGetLastError();
}

void require(bool condition) {
  if (!condition) throw std::runtime_error("GMRES admission contract failed");
}

void contract() {
  response::CudaGmresOptions options;
  std::size_t bytes{};
  require(response::cuda_gmres_workspace(7, options, bytes) == cudaSuccess && bytes > 0);
  const auto required = bytes;
  options.max_workspace_bytes = required - 1;
  require(response::cuda_gmres_workspace(7, options, bytes) == cudaErrorInvalidValue && !bytes);
  options.max_workspace_bytes = required;
  require(response::cuda_gmres_workspace(7, options, bytes) == cudaSuccess && bytes == required);
  options = {};
  options.max_workspace_bytes = std::numeric_limits<std::size_t>::max();
  require(response::cuda_gmres_workspace(std::numeric_limits<std::size_t>::max(), options, bytes) ==
          cudaErrorInvalidValue);
  for (unsigned variant = 0; variant < 10; ++variant) {
    options = {};
    switch (variant) {
      case 0:
        options.restart = 0;
        break;
      case 1:
        options.max_iterations = 0;
        break;
      case 2:
        options.reorthogonalizations = 3;
        break;
      case 3:
        options.relative_tolerance = 1;
        break;
      case 4:
        options.absolute_tolerance = -1;
        break;
      case 5:
        options.breakdown_relative_tolerance = 1;
        break;
      case 6:
        options.diagonal_floor = -1;
        break;
      case 7:
        options.relative_tolerance = std::numeric_limits<double>::quiet_NaN();
        break;
      case 8:
        options.absolute_tolerance = std::numeric_limits<double>::infinity();
        break;
      case 9:
        options.absolute_tolerance = options.relative_tolerance = 0;
        break;
    }
    require(response::cuda_gmres_workspace(7, options, bytes) == cudaErrorInvalidValue && !bytes);
  }
  options = {};
  response::CudaGmresResult result;
  require(response::solve_cuda_gmres(7, options, action, nullptr, nullptr, nullptr, nullptr,
                                     nullptr, 0, nullptr, result) == cudaErrorInvalidValue);
  std::cout << "{\"contract\":\"passed\"}\n";
}

std::vector<double> read(std::size_t count) {
  std::vector<double> result(count);
  for (auto& value : result)
    if (!(std::cin >> value) || !std::isfinite(value))
      throw std::runtime_error("probe requires complete finite input");
  return result;
}

void solve(int argc, char** argv) {
  if (argc != 13) throw std::runtime_error("solve needs eleven arguments");
  const auto count = std::stoull(argv[2]);
  if (count > 2048) throw std::runtime_error("probe dimension exceeds 2048");
  response::CudaGmresOptions options;
  options.restart = std::stoull(argv[3]);
  options.max_iterations = std::stoull(argv[4]);
  options.reorthogonalizations = static_cast<unsigned>(std::stoul(argv[5]));
  options.relative_tolerance = std::stod(argv[6]);
  options.absolute_tolerance = std::stod(argv[7]);
  options.breakdown_relative_tolerance = std::stod(argv[8]);
  options.diagonal_floor = std::stod(argv[9]);
  const bool fitted = std::stoi(argv[10]) != 0;
  const std::string mode = argv[11];
  options.max_workspace_bytes = std::stoull(argv[12]);
  auto matrix = read(count * count);
  auto rhs = read(count);
  auto diagonal = fitted ? read(count) : std::vector<double>{};
  if (mode == "nan_rhs" && count) rhs[0] = std::numeric_limits<double>::quiet_NaN();
  if (mode == "nan_diagonal" && fitted && count)
    diagonal[0] = std::numeric_limits<double>::quiet_NaN();
  std::size_t bytes{};
  check(response::cuda_gmres_workspace(count, options, bytes));
  Stream stream;
  Device<double> device_matrix(matrix.size()), device_rhs(count), device_diagonal(diagonal.size()),
      solution(count);
  Device<unsigned char> workspace(bytes + 8);
  device_matrix.upload(matrix, stream.value);
  device_rhs.upload(rhs, stream.value);
  device_diagonal.upload(diagonal, stream.value);
  const std::vector<double> initial(count, sentinel);
  solution.upload(initial, stream.value);
  Operator owner{device_matrix.value,           mode == "nan_action",  mode == "fail_action",
                 mode == "inconsistent_action", mode == "fail_second", mode == "nan_second"};
  response::CudaGmresResult result;
  auto* output = mode == "alias" ? device_rhs.value : solution.value;
  auto* scratch = mode == "overlap" ? reinterpret_cast<unsigned char*>(device_rhs.value)
                                    : workspace.value + (mode == "unaligned" ? 1 : 0);
  const auto error = response::solve_cuda_gmres(
      count, options, action, &owner, device_rhs.value, device_diagonal.value, output, scratch,
      bytes - (mode == "short" ? 1 : 0), stream.value, result);
  check(cudaStreamSynchronize(stream.value));
  std::vector<double> observed(count);
  if (count)
    check(cudaMemcpy(observed.data(), solution.value, count * sizeof(double),
                     cudaMemcpyDeviceToHost));
  const bool untouched = observed == initial;
  if ((!result.converged() || error != cudaSuccess) && !untouched)
    throw std::runtime_error("failed solve modified public output");
  std::cout << std::setprecision(17) << "{\"cuda_error\":" << static_cast<int>(error)
            << ",\"status\":" << static_cast<int>(result.status)
            << ",\"converged\":" << (result.converged() ? "true" : "false")
            << ",\"untouched\":" << (untouched ? "true" : "false")
            << ",\"iterations\":" << result.iterations
            << ",\"operator_actions\":" << result.operator_actions
            << ",\"preconditioner_actions\":" << result.preconditioner_actions
            << ",\"restarts\":" << result.restarts << ",\"workspace_bytes\":" << bytes
            << ",\"checkpoint_bytes\":" << result.checkpoint_bytes
            << ",\"rhs_norm\":" << result.rhs_norm << ",\"residual_norm\":" << result.residual_norm
            << ",\"tolerance\":" << result.tolerance << ",\"solution\":";
  if (error == cudaSuccess && result.converged()) {
    std::cout << '[';
    for (std::size_t index = 0; index < count; ++index) {
      if (index) std::cout << ',';
      std::cout << observed[index];
    }
    std::cout << ']';
  } else {
    std::cout << "null";
  }
  std::cout << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "contract")
      contract();
    else if (argc > 1 && std::string(argv[1]) == "solve")
      solve(argc, argv);
    else
      throw std::runtime_error("expected contract or solve");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
