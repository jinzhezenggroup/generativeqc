#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

#include "solver/cpu/symmetric_eigen.hpp"
using namespace generativeqc::solver::cpu;
static bool forbid = false;
static int calls = 0;
static int expected_info = 0;
void* operator new(size_t n) {
  assert(!forbid);
  if (auto p = malloc(n)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { free(p); }
void operator delete(void* p, size_t) noexcept { free(p); }
static LapackInt spy(LapackInt layout, char vectors, char triangle, LapackInt n, double* matrix,
                     LapackInt lda, double* values, double*, LapackInt lwork, LapackInt*,
                     LapackInt liwork) {
  ++calls;
  assert(layout == 102 && vectors == 'V' && triangle == 'L' && n == 2 && lda == 2 && lwork == 81 &&
         liwork == 28);
  matrix[0] = 23.0;
  values[0] = 29.0;
  return expected_info;
}
int main() {
  PreparedSymmetricEigen p;
  assert(prepare_borrowed_symmetric_eigen(5, p));
  assert(p.required().doubles == 81 && p.required().integers == 28);
  auto saved = p;
  for (size_t bad :
       {size_t(0), size_t(32767), size_t(2147483647), std::numeric_limits<size_t>::max()}) {
    assert(!prepare_borrowed_symmetric_eigen(bad, p));
    assert(p.maximum_order() == saved.maximum_order() &&
           p.required().doubles == saved.required().doubles &&
           p.required().integers == saved.required().integers && p.family() == saved.family());
  }
  PreparedSymmetricEigen edge;
  assert(prepare_borrowed_symmetric_eigen(32766, edge));
  alignas(64) unsigned char arena[8192]{};
  auto a = (double*)arena;
  auto w = (double*)(arena + 1024);
  auto work = (double*)(arena + 2048);
  auto iw = (LapackInt*)(arena + 4096);
  SymmetricEigenStorage storage{work, 81, iw, 28};
  SymmetricEigenWorkBinding b;
  assert(bind_symmetric_eigen_work(p, storage, b));
  BorrowedSymmetricEigenProblem problem{2, a, 4, w, 2};
  forbid = true;
  for (int info : {-5, 0, 2}) {
    expected_info = info;
    auto r = execute_symmetric_eigen(p, spy, problem, b);
    assert(r.submitted() && r.info == info && a[0] == 23.0 && w[0] == 29.0);
  }
  assert(calls == 3);
  auto invalid = [&](auto pp, auto bb) {
    auto r = execute_symmetric_eigen(p, spy, pp, bb);
    assert(!r.submitted() && r.info == 0 && calls == 3);
  };
  for (size_t n : {size_t(0), size_t(6), std::numeric_limits<size_t>::max()}) {
    auto q = problem;
    q.n = n;
    invalid(q, b);
  }
  auto q = problem;
  q.matrix_capacity = 3;
  invalid(q, b);
  q = problem;
  q.value_capacity = 1;
  invalid(q, b);
  q = problem;
  q.matrix = nullptr;
  invalid(q, b);
  q = problem;
  q.values = nullptr;
  invalid(q, b);
  q = problem;
  q.matrix = (double*)(arena + 1);
  invalid(q, b);
  q = problem;
  q.values = (double*)(arena + 1025);
  invalid(q, b);
  q = problem;
  q.matrix = (double*)(std::numeric_limits<uintptr_t>::max() - 7);
  invalid(q, b);
  auto bb = b;
  bb.counts.doubles++;
  invalid(problem, bb);
  bb = b;
  bb.counts.integers++;
  invalid(problem, bb);
  q = problem;
  q.values = a + 1;
  invalid(q, b);
  q = problem;
  q.matrix = work;
  invalid(q, b);
  q = problem;
  q.matrix = (double*)iw;
  invalid(q, b);
  q = problem;
  q.values = work;
  invalid(q, b);
  q = problem;
  q.values = (double*)iw;
  invalid(q, b);
  bb = b;
  bb.integer_work = (LapackInt*)work;
  invalid(problem, bb);
  for (int c = 0; c < 6; c++) {
    auto s = storage;
    switch (c) {
      case 0:
        s.double_capacity = 80;
        break;
      case 1:
        s.integer_capacity = 27;
        break;
      case 2:
        s.work = nullptr;
        break;
      case 3:
        s.integer_work = nullptr;
        break;
      case 4:
        s.work = (double*)(arena + 1);
        break;
      case 5:
        s.integer_work = (LapackInt*)work;
        break;
    }
    auto out = b;
    assert(!bind_symmetric_eigen_work(p, s, out));
    assert(out.work == b.work && out.integer_work == b.integer_work && out.counts.doubles == 81);
  }
  forbid = false;
  puts("borrowed exact-work/rejection/no-allocation/raw-info checks passed");
}
