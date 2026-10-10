#include <iostream>
#include <stdexcept>

namespace {
bool refuse_optional_allocation = false;
bool simulate_allowance_pressure = false;
unsigned pressure_queries{};
std::size_t pressure_start_free{};
}  // namespace

extern "C" cudaError_t __real_cudaMalloc(void**, std::size_t);
extern "C" cudaError_t __real_cudaMemGetInfo(std::size_t*, std::size_t*);

extern "C" cudaError_t __wrap_cudaMalloc(void** pointer, std::size_t bytes) {
  if (refuse_optional_allocation && bytes == (4ULL << 20)) {
    *pointer = nullptr;
    return cudaErrorMemoryAllocation;
  }
  return __real_cudaMalloc(pointer, bytes);
}

extern "C" cudaError_t __wrap_cudaMemGetInfo(std::size_t* free, std::size_t* total) {
  const auto status = __real_cudaMemGetInfo(free, total);
  if (status != cudaSuccess || !simulate_allowance_pressure) return status;
  if (pressure_queries == 0)
    pressure_start_free = *free;
  else
    *free = pressure_start_free - ((pressure_queries == 1 ? 98ULL : 94ULL) << 20);
  ++pressure_queries;
  return status;
}

#include "tensor/cuda_contraction.cuh"

/** Real-device ownership and injected allocation failures, not timing evidence. */
int main() {
  try {
    cudaStream_t stream{};
    generativeqc_tensor::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto ledger = std::make_shared<generativeqc::runtime::DeviceResourceLedger>();
    ledger->limit = 4ULL << 20;
    generativeqc_tensor::cuda_check(cudaGetDevice(&ledger->device));
    ledger->journal = std::make_shared<generativeqc::runtime::DeviceAllocationJournal>();
    ledger->journal->limit = 8;
    ledger->journal->events.reserve(8);
    generativeqc::runtime::active_device_resource_ledger = ledger;
    generativeqc::tensor::CudaContractionContext context;
    if (!context.prepare(stream, 4ULL << 20) || context.workspace_bytes() != (4ULL << 20) ||
        context.retained_bytes() > context.kProviderAllowance)
      throw std::runtime_error("explicit workspace was not admitted under the provider ceiling");
    if (ledger->live != (4ULL << 20) || ledger->peak != ledger->live || ledger->allocations != 1 ||
        ledger->journal->events.size() != 1 || ledger->journal->events[0].kind != 0)
      throw std::runtime_error("owned workspace bypassed the active allocation ledger/journal");
    cublasMath_t math_mode{};
    generativeqc_tensor::blas_check(cublasGetMathMode(context.handle(), &math_mode));
    if (math_mode != CUBLAS_PEDANTIC_MATH)
      throw std::runtime_error("workspace preparation changed the pedantic math contract");
    const auto prepared_generation = context.generation();
    if (!context.release_workspace() || context.workspace_bytes() ||
        context.generation() == prepared_generation)
      throw std::runtime_error("workspace release did not invalidate the old binding generation");
    if (ledger->live || ledger->journal->events.size() != 2 ||
        ledger->journal->events[1].kind != 1 ||
        ledger->journal->events[0].generation != ledger->journal->events[1].generation)
      throw std::runtime_error("workspace release left an unmatched allocation generation");
    const auto released_generation = context.generation();
    if (context.release_workspace() || context.generation() != released_generation)
      throw std::runtime_error("zero-workspace release is not idempotent");
    context.reset();
    ledger->limit = 0;
    if (!context.prepare(stream, 4ULL << 20) || context.workspace_bytes() || ledger->live ||
        ledger->rejected != 1)
      throw std::runtime_error("ledger pressure did not preserve the zero-workspace provider");
    context.reset();
    generativeqc::runtime::active_device_resource_ledger.reset();
    refuse_optional_allocation = true;
    if (!context.prepare(stream, 4ULL << 20) || context.workspace_bytes())
      throw std::runtime_error("optional workspace OOM rejected the zero-workspace provider");
    context.reset();
    refuse_optional_allocation = false;
    simulate_allowance_pressure = true;
    if (!context.prepare(stream, 4ULL << 20) || context.workspace_bytes() ||
        pressure_queries != 3 || context.retained_bytes() > context.kProviderAllowance)
      throw std::runtime_error(
          "optional workspace allowance pressure rejected the zero-workspace provider");
    context.reset();
    simulate_allowance_pressure = false;
    if (!context.prepare(stream) || context.workspace_bytes())
      throw std::runtime_error("ordinary preparation changed its zero-workspace contract");
    context.reset();
    bool bound_refused = false;
    try {
      (void)context.prepare(stream, (4ULL << 20) + 1);
    } catch (const std::invalid_argument&) {
      bound_refused = true;
    }
    if (!bound_refused || context.prepared() || context.workspace_bytes())
      throw std::runtime_error("oversized optional request acquired provider state");
    // The default stream is valid too; its release must drain before detaching.
    if (!context.prepare(nullptr, 4ULL << 20) || !context.release_workspace())
      throw std::runtime_error("default-stream workspace could not be released");
    context.reset();
    generativeqc_tensor::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    bool capture_refused = false;
    try {
      (void)context.prepare(stream, 4ULL << 20);
    } catch (const std::logic_error&) {
      capture_refused = true;
    }
    cudaGraph_t graph{};
    generativeqc_tensor::cuda_check(cudaStreamEndCapture(stream, &graph));
    generativeqc_tensor::cuda_check(cudaGraphDestroy(graph));
    if (!capture_refused || context.prepared() || context.workspace_bytes())
      throw std::runtime_error("capture preparation acquired provider state");
    generativeqc_tensor::cuda_check(cudaStreamDestroy(stream));
    std::cout
        << "owned workspace/allowance/release/generation/idempotence/zero-default/extent/capture "
           "checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
#include <cuda_runtime.h>
