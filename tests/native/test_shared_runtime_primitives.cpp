#include <array>
#include <cassert>
#include <cmath>
#include <limits>

#include "runtime/cpu_isa.hpp"
#include "runtime/cuda_atomic.cuh"
#include "runtime/parallel_executor.hpp"

namespace {
struct Work {
  std::array<unsigned int, 8> values{};
};

void body(void* context, std::size_t chunk) noexcept {
  auto& work = *static_cast<Work*>(context);
  work.values[chunk] = static_cast<unsigned int>(chunk + 1);
}

void dispatch(void* context, std::size_t chunks, void (*callback)(void*, std::size_t) noexcept,
              void* work) {
  assert(context == work);
  for (std::size_t chunk = chunks; chunk > 0; --chunk) {
    callback(work, chunk - 1);
  }
}
}  // namespace

int main() {
  using generativeqc::runtime::atomic_add_fp64;
  generativeqc::runtime::ChunkExecutor executor;
  assert(!executor.enabled());
  assert(executor.worker_count == 1);
  assert(generativeqc::runtime::CpuIsa::kBaseline != generativeqc::runtime::CpuIsa::kAvx2Fma);
  Work work;
  executor = {&work, 2, dispatch};
  assert(executor.enabled());
  executor.dispatch_chunks(executor.pool_context, work.values.size(), body, &work);
  for (std::size_t chunk = 0; chunk < work.values.size(); ++chunk) {
    assert(work.values[chunk] == chunk + 1);
  }
  executor.dispatch_chunks(executor.pool_context, 0, body, &work);

  double value = 1.25;
  assert(atomic_add_fp64(&value, 2.5) == 1.25);
  assert(value == 3.75);
  assert(atomic_add_fp64(&value, -3.75) == 3.75);
  assert(value == 0.0);
  value = std::numeric_limits<double>::quiet_NaN();
  assert(std::isnan(atomic_add_fp64(&value, 1.0)));
  assert(std::isnan(value));
  value = std::numeric_limits<double>::infinity();
  assert(std::isinf(atomic_add_fp64(&value, 1.0)));
  assert(std::isinf(value));
#if defined(GENERATIVEQC_FORCE_FP64_ATOMIC_CAS) || __CUDA_ARCH__ < 600
  value = 1.0;
  cuda_atomic_double::interfere = true;
  cuda_atomic_double::attempts = 0;
  assert(atomic_add_fp64(&value, 2.0) == 7.0);
  assert(value == 9.0);
  assert(cuda_atomic_double::attempts >= 2);
#endif
}
