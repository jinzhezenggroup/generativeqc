#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "scf/solver/cpu_target_eigen.hpp"
#include "tensor/cpu_linalg.hpp"
namespace tensor = generativeqc::tensor;
static int calls, info, local_calls, global_calls, current_threads = 7, requested_threads;
static const double* expected_matrix;
static bool fail_allocation;
void* operator new(std::size_t size) {
  if (fail_allocation) throw std::bad_alloc();
  if (void* pointer = std::malloc(size ? size : 1)) return pointer;
  throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
static void require(bool value) {
  if (!value) std::abort();
}
extern "C" int openblas_set_num_threads_local(int n) {
  ++local_calls;
  int old = current_threads;
  current_threads = n;
  return old;
}
extern "C" int openblas_get_num_threads() { return current_threads; }
extern "C" void openblas_set_num_threads(int n) {
  ++global_calls;
  current_threads = n;
}
extern "C" int LAPACKE_dsyevd(int layout, char job, char uplo, int n, double* a, int lda,
                              double* w) {
  require(layout == 101 && job == 'V' && uplo == 'L' && n == 2 && lda == 2 &&
          a == expected_matrix && current_threads == requested_threads);
  ++calls;
  if (info) return info;
  require(a[0] == 2 && a[1] == 1 && a[2] == 1 && a[3] == 2);
  w[0] = 1;
  w[1] = 3;
  const double q = 1 / std::sqrt(2.);
  a[0] = q;
  a[1] = q;
  a[2] = -q;
  a[3] = q;
  return 0;
}
extern "C" void cblas_dgemm(int, int, int, int, int, int, double, const double*, int, const double*,
                            int, double, double*, int) {
  std::abort();
}
static void solve(tensor::CpuLinalgPlan plan) {
  std::vector<double> matrix{2, 1, 1, 2};
  expected_matrix = matrix.data();
  requested_threads = plan.provider_threads;
  auto result = tensor::cpu_symmetric_eigen(std::move(matrix), 2, plan);
  require(result.values == std::vector<double>({1, 3}) &&
          result.vectors.data() == expected_matrix && current_threads == 7);
}
int main() {
  using namespace tensor;
  const CpuLinalgPlan scalar{CpuLinalgProvider::scalar, CpuLinalgThreadOwnership::task_parallel, 1};
  for (int status : {0, -3, 4}) {
    info = status;
    const CpuLinalgPlan explicit_provider{CpuLinalgProvider::openblas,
                                          CpuLinalgThreadOwnership::provider_parallel, 3};
    bool failed = false;
    try {
      solve(explicit_provider);
    } catch (const std::invalid_argument& e) {
      failed = true;
      require(status < 0 &&
              std::string_view(e.what()) == "OpenBLAS symmetric eigensolver rejected an argument");
    } catch (const std::runtime_error& e) {
      failed = true;
      require(status > 0 &&
              std::string_view(e.what()) == "OpenBLAS symmetric eigensolver did not converge");
    }
    require(failed == (status != 0) && current_threads == 7);
  }
  info = 0;
#if GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS
  solve({});
  require(calls == 4 && local_calls == 8 && global_calls == 0);
#else
  const int before_automatic = calls;
  auto automatic = cpu_symmetric_eigen({2, 1, 1, 2}, 2);
  require(calls == before_automatic && std::abs(automatic.values[0] - 1) < 1e-14 &&
          std::abs(automatic.values[1] - 3) < 1e-14);
  bool explicit_rejected = false;
  try {
    solve({CpuLinalgProvider::openblas, CpuLinalgThreadOwnership::task_parallel, 1});
  } catch (const std::runtime_error&) {
    explicit_rejected = true;
  }
  require(explicit_rejected && calls == before_automatic && global_calls == 6 && local_calls == 0);
#endif
  {
    std::vector<double> matrix{2, 1, 1, 2};
    const int before_calls = calls, before_local = local_calls, before_global = global_calls;
    bool allocation_failed = false;
    fail_allocation = true;
    try {
      (void)cpu_symmetric_eigen(
          std::move(matrix), 2,
          {CpuLinalgProvider::openblas, CpuLinalgThreadOwnership::provider_parallel, 3});
    } catch (const std::bad_alloc&) {
      allocation_failed = true;
    }
    fail_allocation = false;
    require(allocation_failed && calls == before_calls && local_calls == before_local &&
            global_calls == before_global && current_threads == 7);
  }
  const int before = calls;
  auto target = generativeqc::scf::solver::cpu_target_eigen({2, 1, 1, 2}, nullptr, nullptr, 2);
  require(calls == before && std::abs(target.values[0] - 1) < 1e-14);
  bool rejected = false;
  try {
    (void)cpu_symmetric_eigen({2, 1, 1, 2}, 2, {}, 1e-14);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected && calls == before);
  auto strict = cpu_symmetric_eigen({1e6, 1e-9, 1e-9, 1e6}, 2, scalar, 1e-14);
  require(strict.values[0] < 1e6 && strict.values[1] > 1e6 && calls == before);
  std::printf("owned row-major LAPACKE: calls=%d local=%d global=%d; scalar policy retained\n",
              calls, local_calls, global_calls);
}
