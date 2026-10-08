#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace generativeqc::solver {

/** Basis data is explicit: X is an admitted canonical orthogonalizer; L is an
 * admitted cached lower Cholesky factor. This service neither constructs nor
 * accepts an overlap, and makes no conditioning or publication decision. */
enum class GeneralizedEigenBasis { identity, canonical_x, lower_cholesky };
enum class GeneralizedEigenLayout { row_major, column_major };
enum class GeneralizedEigenOperand { input, basis, temporary, reduced, coefficients };

struct GeneralizedEigenDomain {
  std::size_t order{};
  std::size_t solves{};
  std::size_t capacity{};
  GeneralizedEigenLayout layout{GeneralizedEigenLayout::column_major};
  // Informational owner cardinality: spin solves can exceed physical systems.
  std::size_t physical_systems{};

  bool valid() const noexcept {
    const auto limit = std::numeric_limits<std::size_t>::max() / sizeof(double);
    // Capacity describes unused availability too. Only submitted solves enter
    // the vendor integer domain or the active matrix byte extent.
    return order && order <= std::size_t(std::numeric_limits<int>::max()) && solves &&
           solves <= capacity && solves <= std::size_t(std::numeric_limits<int>::max()) &&
           (layout == GeneralizedEigenLayout::row_major ||
            layout == GeneralizedEigenLayout::column_major) &&
           order <= limit / order && solves <= limit / (order * order);
  }
  std::size_t matrix_elements() const noexcept { return order * order; }
  std::size_t matrix_extent() const noexcept { return matrix_elements() * solves; }
  std::size_t value_extent() const noexcept { return order * solves; }
};

/** A phase binds only storage live during that phase. In particular canonical
 * CPU transform scratch need not survive into ordinary spectral execution.
 * Extents are elements, not allocation promises. The caller retains lifetime,
 * layout conversion, admission and the already-selected spectral binding. */
struct GeneralizedEigenMatrices {
  const double* input{};
  const double* basis{};
  double* temporary{};
  double* reduced{};
  double* coefficients{};
  std::size_t input_elements{}, basis_elements{}, temporary_elements{};
  std::size_t reduced_elements{}, coefficient_elements{};
};

namespace generalized_eigen_detail {
inline bool span(const void* p, std::size_t n) noexcept {
  const auto address = reinterpret_cast<std::uintptr_t>(p);
  return p && address % alignof(double) == 0 && n &&
         n <= std::numeric_limits<std::size_t>::max() / sizeof(double) &&
         address <= std::numeric_limits<std::uintptr_t>::max() - n * sizeof(double);
}
inline bool disjoint(const void* a, const void* b, std::size_t n) noexcept {
  const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
  return x <= y ? y - x >= n * sizeof(double) : x - y >= n * sizeof(double);
}
inline bool validate(const GeneralizedEigenDomain& domain, GeneralizedEigenBasis basis,
                     const GeneralizedEigenMatrices& m, bool recovery) noexcept {
  if (!domain.valid()) return false;
  const auto n = domain.matrix_extent();
  if (!span(m.reduced, n) || m.reduced_elements < n) return false;
  if (basis == GeneralizedEigenBasis::lower_cholesky)
    return span(m.basis, n) && m.basis_elements >= n && disjoint(m.basis, m.reduced, n) &&
           (recovery ? m.coefficients == m.reduced && m.coefficient_elements >= n
                     : m.input == m.reduced && m.input_elements >= n);
  const auto* input = recovery ? m.reduced : m.input;
  auto* output = recovery ? m.coefficients : m.reduced;
  const auto input_count = recovery ? m.reduced_elements : m.input_elements;
  const auto output_count = recovery ? m.coefficient_elements : m.reduced_elements;
  if (!span(input, n) || !span(output, n) || input_count < n || output_count < n) return false;
  if (basis == GeneralizedEigenBasis::identity)
    return input == output || disjoint(input, output, n);
  if (basis != GeneralizedEigenBasis::canonical_x || !span(m.basis, n) || m.basis_elements < n ||
      !disjoint(m.basis, output, n))
    return false;
  if (recovery) return disjoint(input, output, n);
  return span(m.temporary, n) && m.temporary_elements >= n && disjoint(m.temporary, m.basis, n) &&
         disjoint(m.temporary, input, n) && disjoint(m.temporary, output, n);
}
}  // namespace generalized_eigen_detail

/** These are independent numerical phases, not a host iteration driver.
 * Ordinary spectral execution uses the existing prepared symmetric-eigen
 * service between them. Owners may symmetrize/validate after reduction and
 * compact successful device peers before recovery. Lowerings contain primitive
 * tensor operations only; no callback enters a method-owned numerical body. */
template <class Lowering>
auto reduce_generalized_eigen(GeneralizedEigenBasis basis, Lowering&& lowering) {
  using Operand = GeneralizedEigenOperand;
  if (!lowering.valid(basis, false)) return lowering.invalid();
  if (basis == GeneralizedEigenBasis::identity) return lowering.identity(false);
  if (basis == GeneralizedEigenBasis::canonical_x) {
    const auto status =
        lowering.multiply(false, Operand::input, Operand::basis, Operand::temporary);
    if (!lowering.success(status)) return status;
    return lowering.multiply(true, Operand::basis, Operand::temporary, Operand::reduced);
  }
  const auto status = lowering.triangular(false, false);
  if (!lowering.success(status)) return status;
  return lowering.triangular(true, true);
}

template <class Lowering>
auto recover_generalized_eigen(GeneralizedEigenBasis basis, Lowering&& lowering) {
  using Operand = GeneralizedEigenOperand;
  if (!lowering.valid(basis, true)) return lowering.invalid();
  if (basis == GeneralizedEigenBasis::identity) return lowering.identity(true);
  if (basis == GeneralizedEigenBasis::canonical_x)
    return lowering.multiply(false, Operand::basis, Operand::reduced, Operand::coefficients);
  return lowering.triangular(false, true);
}

}  // namespace generativeqc::solver
