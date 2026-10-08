#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

#include "solver/cpu/generalized_eigen.hpp"

namespace eigen = generativeqc::solver;
namespace tensor = generativeqc::tensor;
static bool forbid_allocation;
void* operator new(std::size_t n) {
  if (forbid_allocation) std::abort();
  if (auto* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

const tensor::CpuLinalgPlan scalar{tensor::CpuLinalgProvider::scalar,
                                   tensor::CpuLinalgThreadOwnership::task_parallel, 1};

void known_generalized_frame(eigen::GeneralizedEigenBasis basis, std::size_t capacity) {
  // Independent prescribed construction: S=L L^T, F=L [[2,1],[1,2]] L^T.
  // Both admitted basis representations must produce eigenvalues 1 and 3.
  const std::array<double, 4> f{8, 10, 10, 26}, s{4, 2, 2, 10};
  const std::array<double, 4> l{2, 0, 1, 3}, x{.5, -1. / 6., 0, 1. / 3.};
  const auto& transform = basis == eigen::GeneralizedEigenBasis::canonical_x ? x : l;
  std::array<double, 4> reduced = f, temporary{}, coefficients{};
  const eigen::GeneralizedEigenDomain domain{2, 1, capacity,
                                             eigen::GeneralizedEigenLayout::row_major, 1};
  const eigen::GeneralizedEigenMatrices matrices{reduced.data(),
                                                 transform.data(),
                                                 temporary.data(),
                                                 reduced.data(),
                                                 reduced.data(),
                                                 4,
                                                 4,
                                                 4,
                                                 4,
                                                 4};
  forbid_allocation = true;
  assert(eigen::reduce_generalized_eigen(
             basis, eigen::cpu::GeneralizedEigenLowering{domain, matrices, scalar}) == 0);
  forbid_allocation = false;
  for (std::size_t i = 0; i != 4; ++i)
    assert(std::abs(reduced[i] - std::array<double, 4>{2, 1, 1, 2}[i]) < 1e-14);
  auto frame = tensor::cpu_symmetric_eigen(std::vector<double>(reduced.begin(), reduced.end()), 2,
                                           scalar, 1e-14);
  assert(std::abs(frame.values[0] - 1) < 1e-14 && std::abs(frame.values[1] - 3) < 1e-14);
  auto recovery = matrices;
  recovery.reduced = frame.vectors.data();
  recovery.coefficients = basis == eigen::GeneralizedEigenBasis::canonical_x ? coefficients.data()
                                                                             : frame.vectors.data();
  forbid_allocation = true;
  assert(eigen::recover_generalized_eigen(
             basis, eigen::cpu::GeneralizedEigenLowering{domain, recovery, scalar}) == 0);
  forbid_allocation = false;
  const auto* c = recovery.coefficients;
  for (std::size_t row = 0; row != 2; ++row)
    for (std::size_t col = 0; col != 2; ++col) {
      double fc = 0, sc = 0, gram = 0;
      for (std::size_t k = 0; k != 2; ++k) {
        fc += f[row * 2 + k] * c[k * 2 + col];
        sc += s[row * 2 + k] * c[k * 2 + col];
        for (std::size_t j = 0; j != 2; ++j) gram += c[k * 2 + row] * s[k * 2 + j] * c[j * 2 + col];
      }
      assert(std::abs(fc - sc * frame.values[col]) < 1e-13);
      assert(std::abs(gram - (row == col ? 1.0 : 0.0)) < 1e-13);
    }
}

void binding_boundaries() {
  using Basis = eigen::GeneralizedEigenBasis;
  using Layout = eigen::GeneralizedEigenLayout;
  const eigen::GeneralizedEigenDomain domain{2, 1, 5, Layout::row_major, 1};
  assert(domain.valid() && domain.matrix_extent() == 4 && domain.value_extent() == 2);
  auto invalid = domain;
  invalid.solves = 6;
  assert(!invalid.valid());
  invalid = domain;
  invalid.order = std::numeric_limits<std::size_t>::max();
  assert(!invalid.valid());
  invalid = domain;
  invalid.layout = static_cast<Layout>(123);
  assert(!invalid.valid());
  std::array<double, 8> input{.5, 0, 0, 2}, output{}, scratch{};
  eigen::GeneralizedEigenMatrices matrices{
      input.data(), input.data(), scratch.data(), output.data(), output.data(), 4, 4, 4, 4, 4};
  // The old canonical consumer permits X==F. Preserve this valid alias.
  assert(eigen::reduce_generalized_eigen(Basis::canonical_x, eigen::cpu::GeneralizedEigenLowering{
                                                                 domain, matrices, scalar}) == 0);
  assert(output[0] == .125 && output[3] == 8);
  for (int change = 0; change != 5; ++change) {
    auto bad = matrices;
    if (change == 0) bad.temporary = output.data() + 1;
    if (change == 1) bad.basis = output.data();
    if (change == 2) bad.input_elements = 3;
    if (change == 3) bad.temporary_elements = 3;
    if (change == 4) bad.reduced_elements = 3;
    const auto before_output = output, before_scratch = scratch;
    assert(eigen::reduce_generalized_eigen(
               Basis::canonical_x, eigen::cpu::GeneralizedEigenLowering{domain, bad, scalar}) != 0);
    assert(output == before_output && scratch == before_scratch);
  }
  matrices.basis = nullptr;
  matrices.temporary = nullptr;
  matrices.basis_elements = matrices.temporary_elements = 0;
  forbid_allocation = true;
  assert(eigen::reduce_generalized_eigen(
             Basis::identity, eigen::cpu::GeneralizedEigenLowering{domain, matrices, scalar}) == 0);
  assert(eigen::recover_generalized_eigen(
             Basis::identity, eigen::cpu::GeneralizedEigenLowering{domain, matrices, scalar}) == 0);
  forbid_allocation = false;
  assert(std::equal(input.begin(), input.begin() + 4, output.begin()));
  // Oversized actual submissions still fail before touching borrowed storage.
  for (const auto invalid_work :
       {eigen::GeneralizedEigenDomain{2, std::size_t(std::numeric_limits<int>::max()) + 1,
                                      std::numeric_limits<std::size_t>::max(), Layout::row_major},
        eigen::GeneralizedEigenDomain{50000, std::size_t(std::numeric_limits<int>::max()),
                                      std::numeric_limits<std::size_t>::max(),
                                      Layout::row_major}}) {
    assert(!invalid_work.valid());
    const auto before = output;
    assert(eigen::reduce_generalized_eigen(
               Basis::identity,
               eigen::cpu::GeneralizedEigenLowering{invalid_work, matrices, scalar}) != 0);
    assert(eigen::recover_generalized_eigen(
               Basis::identity,
               eigen::cpu::GeneralizedEigenLowering{invalid_work, matrices, scalar}) != 0);
    assert(output == before);
  }
  auto wrong_layout = domain;
  wrong_layout.layout = Layout::column_major;
  assert(eigen::reduce_generalized_eigen(
             Basis::identity,
             eigen::cpu::GeneralizedEigenLowering{wrong_layout, matrices, scalar}) != 0);
}

int main() {
  // Only one 2x2 matrix is consumed. The unused solve-storage tail must not
  // become a vendor batch bound or an allocation/active-matrix byte count.
  const auto above_vendor = std::size_t(std::numeric_limits<int>::max()) + 1;
  const auto overflow_tail = std::numeric_limits<std::size_t>::max() / sizeof(double) / 4 + 1;
  for (const auto capacity : {std::size_t(5), above_vendor, overflow_tail}) {
    known_generalized_frame(eigen::GeneralizedEigenBasis::canonical_x, capacity);
    known_generalized_frame(eigen::GeneralizedEigenBasis::lower_cholesky, capacity);
  }
  binding_boundaries();
}
