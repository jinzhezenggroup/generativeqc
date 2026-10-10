// Standalone real-device driver. Link unchanged production translation units.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "scf/cuda/matrix_library.hpp"

namespace matrix = generativeqc::scf::cuda_execution;
namespace {
constexpr double sentinel = -314159.25;
int injected_create = 0;
bool injected_memory_error = false;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class T>
struct DeviceBuffer {
  T* data{};
  explicit DeviceBuffer(std::size_t count) { check(cudaMalloc(&data, count * sizeof(T))); }
  ~DeviceBuffer() {
    if (data) (void)cudaFree(data);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
__global__ void reset_output(double* output, std::size_t size) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) output[i] = sentinel;
}

void owner_probe(cudaStream_t stream, const std::string& name) {
  matrix::MatrixLibraryOwner owner;
  injected_create = name == "allocation" ? CUBLAS_STATUS_ALLOC_FAILED : 0;
  injected_memory_error = name == "memory-error";
  if (name == "capture") check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  const auto status = owner.prepare(stream, name == "small" ? 16 : 17);
  injected_create = 0;
  injected_memory_error = false;
  if (name == "capture") {
    cudaGraph_t graph{};
    check(cudaStreamEndCapture(stream, &graph));
    if (graph) check(cudaGraphDestroy(graph));
  }
  const bool enabled = owner.library_enabled();
  const auto retained = owner.retained_bytes();
  bool pass = false;
  if (name == "small" || name == "allocation")
    pass = status == GENERATIVEQC_STATUS_SUCCESS && !enabled && retained == 0;
  else if (name == "admit")
    pass = status == GENERATIVEQC_STATUS_SUCCESS && enabled && retained <= owner.kProviderAllowance;
  else if (name == "memory-error")
    pass = status == GENERATIVEQC_STATUS_CUDA_ERROR && !enabled;
  else if (name == "capture")
    pass = status == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !enabled;
  if (name != "capture")
    pass = pass && owner.prepare(stream, 17) == GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  owner.reset();
  owner.reset();
  pass = pass && !owner.library_enabled() && owner.retained_bytes() == 0;
  // Reuse after reset is a real allocation/lifetime probe, not a copied owner.
  pass =
      pass && owner.prepare(stream, 16) == GENERATIVEQC_STATUS_SUCCESS && !owner.library_enabled();
  owner.reset();
  std::cout << "{\"kind\":\"owner\",\"probe\":\"" << name << "\",\"status\":" << status
            << ",\"enabled\":" << (enabled ? "true" : "false") << ",\"retained_bytes\":" << retained
            << ",\"injected\":"
            << (name == "allocation" || name == "memory-error" ? "true" : "false")
            << ",\"pass\":" << (pass ? "true" : "false") << "}\n";
}

void run_case(const std::filesystem::path& folder, int id, int n, int batch, int spins,
              bool ordinary, bool left_spin, bool right_spin, bool transpose, bool graph_mode,
              bool library, double scale, cudaStream_t stream, matrix::MatrixLibraryOwner& owner) {
  const std::size_t matrix_size = static_cast<std::size_t>(n) * n;
  const std::size_t left_size = batch * (left_spin ? spins : 1) * matrix_size;
  const std::size_t right_size = batch * (right_spin ? spins : 1) * matrix_size;
  const std::size_t output_size = batch * spins * matrix_size;
  std::vector<double> left(left_size), right(right_size), output(output_size * 2);
  std::vector<std::uint8_t> active(batch);
  std::ifstream input(folder / (std::to_string(id) + ".input"), std::ios::binary);
  input.read(reinterpret_cast<char*>(left.data()), left.size() * sizeof(double));
  input.read(reinterpret_cast<char*>(right.data()), right.size() * sizeof(double));
  input.read(reinterpret_cast<char*>(active.data()), active.size());
  require(input.good() && input.peek() == std::char_traits<char>::eof(), "bad fixture length");
  DeviceBuffer<double> dl(left_size), dr(right_size), dout(output_size);
  DeviceBuffer<std::uint8_t> da(batch);
  check(cudaMemcpyAsync(dl.data, left.data(), left_size * sizeof(double), cudaMemcpyHostToDevice,
                        stream));
  check(cudaMemcpyAsync(dr.data, right.data(), right_size * sizeof(double), cudaMemcpyHostToDevice,
                        stream));
  check(cudaMemcpyAsync(da.data, active.data(), active.size(), cudaMemcpyHostToDevice, stream));
  check(cudaStreamSynchronize(stream));
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin));
  check(cudaEventCreate(&end));
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  int status = 0;
  float milliseconds = 0;
  const auto enqueue = [&] {
    reset_output<<<static_cast<unsigned>((output_size + 255) / 256), 256, 0, stream>>>(dout.data,
                                                                                       output_size);
    check(cudaPeekAtLastError());
    status = ordinary ? matrix::launch_matrix_product(owner.view(), batch, n, dl.data, transpose,
                                                      dr.data, da.data, dout.data, library, scale)
                      : matrix::launch_spin_matrix_product(owner.view(), batch, spins, n, dl.data,
                                                           left_spin, transpose, dr.data,
                                                           right_spin, da.data, dout.data, library);
  };
  if (graph_mode) {
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    enqueue();
    const auto capture_status = cudaStreamEndCapture(stream, &graph);
    if (capture_status != cudaSuccess) status = 10000 + static_cast<int>(capture_status);
    if (!status) {
      const auto instantiate_status = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
      if (instantiate_status != cudaSuccess) status = 20000 + static_cast<int>(instantiate_status);
    }
  }
  if (!status) {
    for (int repeat = 0; repeat < 2; ++repeat) {
      check(cudaEventRecord(begin, stream));
      if (graph_mode)
        check(cudaGraphLaunch(executable, stream));
      else
        enqueue();
      check(cudaEventRecord(end, stream));
      check(cudaEventSynchronize(end));
      float elapsed{};
      check(cudaEventElapsedTime(&elapsed, begin, end));
      milliseconds += elapsed;
      check(cudaMemcpyAsync(output.data() + repeat * output_size, dout.data,
                            output_size * sizeof(double), cudaMemcpyDeviceToHost, stream));
      check(cudaStreamSynchronize(stream));
    }
  }
  if (executable) check(cudaGraphExecDestroy(executable));
  if (graph) check(cudaGraphDestroy(graph));
  check(cudaEventDestroy(begin));
  check(cudaEventDestroy(end));
  if (!status) {
    std::ofstream file(folder / (std::to_string(id) + ".output"), std::ios::binary);
    file.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(double));
    require(file.good(), "failed to write output");
  }
  std::cout << "{\"kind\":\"case\",\"id\":" << id << ",\"status\":" << status
            << ",\"device_ms_reset_and_product_two_runs\":" << milliseconds
            << ",\"semantic_products\":" << 2 * batch * spins
            << ",\"library_submissions_per_execution\":" << (library ? (ordinary ? 1 : spins) : 0)
            << ",\"owner_retained_bytes\":" << owner.retained_bytes() << "}\n";
  std::cout.flush();
}
}  // namespace

// Diagnostic fault injection: no allocation exhaustion and no production edits.
extern "C" cublasStatus_t __real_cublasCreate_v2(cublasHandle_t*);
extern "C" cublasStatus_t __wrap_cublasCreate_v2(cublasHandle_t* handle) {
  if (injected_create) {
    *handle = nullptr;
    return static_cast<cublasStatus_t>(injected_create);
  }
  return __real_cublasCreate_v2(handle);
}
extern "C" cudaError_t __real_cudaMemGetInfo(std::size_t*, std::size_t*);
extern "C" cudaError_t __wrap_cudaMemGetInfo(std::size_t* free, std::size_t* total) {
  if (injected_memory_error) return cudaErrorUnknown;
  return __real_cudaMemGetInfo(free, total);
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: matrix-qualification INPUT_FOLDER");
    int count{}, device{}, driver{}, runtime{};
    check(cudaGetDeviceCount(&count));
    require(count > 0, "no visible CUDA device");
    check(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device));
    check(cudaDriverGetVersion(&driver));
    check(cudaRuntimeGetVersion(&runtime));
    std::cout << "{\"kind\":\"device\",\"ordinal\":" << device << ",\"name\":\"" << properties.name
              << "\",\"major\":" << properties.major << ",\"minor\":" << properties.minor
              << ",\"driver\":" << driver << ",\"runtime\":" << runtime
              << ",\"total_memory\":" << properties.totalGlobalMem << "}\n";
    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (const auto* name : {"small", "admit", "allocation", "memory-error", "capture"})
      owner_probe(stream, name);
    {
      matrix::MatrixLibraryOwner owner;
      require(owner.prepare(stream, 97) == GENERATIVEQC_STATUS_SUCCESS && owner.library_enabled(),
              "real library owner not admitted; forced primitive comparison unavailable");
      std::ifstream commands(std::filesystem::path(argv[1]) / "cases.txt");
      require(commands.good(), "missing cases.txt");
      int id{}, n{}, batch{}, spins{}, ordinary{}, left{}, right{}, transpose{}, graph{}, library{};
      double scale{};
      while (commands >> id >> n >> batch >> spins >> ordinary >> left >> right >> transpose >>
             graph >> library >> scale)
        run_case(argv[1], id, n, batch, spins, ordinary, left, right, transpose, graph, library,
                 scale, stream, owner);
      require(commands.eof(), "malformed cases.txt");
      std::ifstream maps("/proc/self/maps");
      std::ofstream saved_maps(std::filesystem::path(argv[1]) / "loaded-maps.txt");
      saved_maps << maps.rdbuf();
      require(maps.good() && saved_maps.good(), "could not retain actually loaded library maps");
    }
    check(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
