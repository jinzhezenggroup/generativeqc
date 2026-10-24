namespace generativeqc::dft::hessian_aot {
inline bool incomplete_closure = false;
int launch(const double*, std::size_t, const long long*, const double*, double*, int*,
           cudaStream_t) {
  return 0;
}
std::span<const Program> programs() noexcept {
  static constexpr unsigned angular[]{0, 0};
  static constexpr std::uint32_t components[]{0};
  static const Program programs[]{{"s",
                                   "s",
                                   Kind::integral,
                                   {},
                                   13,
                                   components,
                                   2,
                                   9,
                                   nullptr,
                                   nullptr,
                                   launch,
                                   {"overlap", angular, "full_range", 0}},
                                  {"t",
                                   "t",
                                   Kind::integral,
                                   {},
                                   13,
                                   components,
                                   2,
                                   9,
                                   nullptr,
                                   nullptr,
                                   launch,
                                   {"kinetic", angular, "full_range", 0}},
                                  {"v",
                                   "v",
                                   Kind::integral,
                                   {},
                                   19,
                                   components,
                                   3,
                                   12,
                                   nullptr,
                                   nullptr,
                                   launch,
                                   {"nuclear_attraction", angular, "full_range", 0}}};
  return {programs, incomplete_closure ? 2U : 3U};
}
}  // namespace generativeqc::dft::hessian_aot
namespace failure {
std::size_t allocation_size{}, skipped{}, hits{}, fail_after{};
}
void* operator new(std::size_t size) {
  if (size == failure::allocation_size && failure::skipped++ == failure::fail_after) {
    ++failure::hits;
    throw std::bad_alloc();
  }
  if (auto* p = std::malloc(size)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
int main() {
  using namespace generativeqc::integrals;
  const CudaOneElectronShape shape{2, 4, 2};
  const CudaOneElectronShell shells[]{{0, 0, 0, 2, 0, 1}, {1, 0, 2, 2, 1, 1}};
  CudaOneElectronOptions options;
  options.parallel_streams = 0;
  std::size_t scalar{}, parallel{};
  assert(cuda_one_electron_workspace(shape, shells, options, scalar) == cudaSuccess);
  options.parallel_streams = 32;
  assert(cuda_one_electron_workspace(shape, shells, options, parallel) == cudaSuccess);
  assert(parallel > scalar);
  // Each optional vector growth allocation can fail independently. This uses
  // the production task type rather than a guessed byte count.
  for (std::size_t capacity = 1; capacity <= 16; capacity *= 2) {
    failure::allocation_size = capacity * sizeof(ParallelTask);
    failure::skipped = failure::hits = failure::fail_after = 0;
    std::size_t bytes = 123;
    assert(cuda_one_electron_workspace(shape, shells, options, bytes) == cudaSuccess);
    assert(failure::hits == 1 && bytes == scalar);
  }
  failure::allocation_size = 0;
  // Budget/task refusals preserve the validated scalar plan too.
  for (const auto cap : {scalar, parallel - 1, parallel}) {
    options.max_workspace_bytes = cap;
    std::size_t bytes{};
    assert(cuda_one_electron_workspace(shape, shells, options, bytes) == cudaSuccess);
    assert(bytes == (cap < parallel ? scalar : parallel));
  }
  options.max_workspace_bytes = parallel;
  options.max_parallel_tasks = 1;
  std::size_t bytes{};
  assert(cuda_one_electron_workspace(shape, shells, options, bytes) == cudaSuccess &&
         bytes == scalar);
  options.max_parallel_tasks = 2048;
  // A scalar arena returned by a transient optional sizing allocation failure
  // remains usable when preparing later with the original, larger budget.
  alignas(double) double basis[100]{}, charges[2]{};
  CudaOneElectronReference reference{
      basis, charges, 0, reinterpret_cast<void*>(1), +[](void*) noexcept { return cudaSuccess; },
      basis};
  void* arena{};
  assert(cudaMalloc(&arena, parallel) == cudaSuccess);
  {
    CudaOneElectronResponse response;
    assert(response.prepare(shape, shells, options, reference, arena, scalar) == cudaSuccess);
    assert(response.work().workspace_bytes == scalar);
    for (const auto allocation :
         {sizeof(ParallelTask), 32 * sizeof(cudaStream_t), 33 * sizeof(cudaEvent_t)}) {
      failure::allocation_size = allocation;
      failure::skipped = failure::hits = failure::fail_after = 0;
      assert(response.prepare(shape, shells, options, reference, arena, parallel) == cudaSuccess);
      assert(failure::hits == 1 && response.work().parallel_tasks == 0);
      assert(stub::streams == 0 && stub::events == 0);
    }
    failure::allocation_size = 0;
    assert(response.prepare(shape, shells, options, reference, arena, parallel) == cudaSuccess);
    assert(response.work().parallel_tasks != 0);
    stub::graph_error = cudaErrorUnknown;
    assert(response.prepare(shape, shells, options, reference, arena, parallel) ==
           cudaErrorUnknown);
    stub::graph_error = cudaSuccess;
    auto revoked = reference;
    revoked.authenticate = +[](void*) noexcept { return cudaErrorInvalidResourceHandle; };
    assert(response.prepare(shape, shells, options, revoked, arena, scalar) ==
           cudaErrorInvalidResourceHandle);
  }
  assert(stub::streams == 0 && stub::events == 0);
  cudaFree(arena);
  // Mandatory metadata OOM and scientific closure/work refusals are not hidden.
  options.max_parallel_tasks = 2048;
  failure::allocation_size = sizeof(Job);
  failure::skipped = failure::hits = 0;
  assert(cuda_one_electron_workspace(shape, shells, options, bytes) == cudaErrorMemoryAllocation);
  assert(bytes == 0 && failure::hits == 1);
  failure::allocation_size = 0;
  generativeqc::dft::hessian_aot::incomplete_closure = true;
  assert(cuda_one_electron_workspace(shape, shells, options, bytes) == cudaErrorNotSupported &&
         bytes == 0);
  generativeqc::dft::hessian_aot::incomplete_closure = false;
  options.max_jobs = 1;
  assert(cuda_one_electron_workspace(shape, shells, options, bytes) == cudaErrorInvalidValue &&
         bytes == 0);
  options.max_jobs = 100000;
  options.max_projection_terms = 1;
  assert(cuda_one_electron_workspace(shape, shells, options, bytes) == cudaErrorInvalidValue &&
         bytes == 0);
  std::cout << "PASS optional host metadata fallback; scalar=" << scalar << " parallel=" << parallel
            << '\n';
}
