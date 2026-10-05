#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "runtime/execution_precision.hpp"

namespace generativeqc::tensor {

using runtime::PrecisionDtype;

inline std::size_t contraction_product(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("native contraction size overflow");
  return a * b;
}

/** Runtime-shape projection of common.lowering_contract.OperandLayout.
 * Modes are the existing TensorIR einsum ordinals, not matrix axis names.
 * Dense batches and unbatched padded matrix views are supported. Other packing
 * remains explicit, budgeted TensorIR work outside this provider binding.
 */
struct ContractionOperand {
  static constexpr std::size_t kMaximumRank = 8;
  std::size_t rank{};
  std::array<int, kMaximumRank> modes{};
  std::array<std::size_t, kMaximumRank> shape{}, strides{};
  PrecisionDtype dtype{PrecisionDtype::Fp64};

  static ContractionOperand dense(std::initializer_list<int> labels,
                                  std::initializer_list<std::size_t> extents,
                                  PrecisionDtype dtype) {
    if (labels.size() != extents.size() || labels.size() > kMaximumRank)
      throw std::invalid_argument("native contraction operand rank mismatch");
    ContractionOperand result;
    result.rank = labels.size();
    result.dtype = dtype;
    std::copy(labels.begin(), labels.end(), result.modes.begin());
    std::copy(extents.begin(), extents.end(), result.shape.begin());
    std::size_t stride = 1;
    for (auto i = result.rank; i != 0; --i) {
      if (!result.shape[i - 1] || result.modes[i - 1] < 0)
        throw std::invalid_argument("native contraction requires positive extents and modes");
      result.strides[i - 1] = stride;
      stride = contraction_product(stride, result.shape[i - 1]);
    }
    return result;
  }

  std::size_t elements() const {
    std::size_t result = 1;
    for (std::size_t i = 0; i != rank; ++i) result = contraction_product(result, shape[i]);
    return result;
  }

  /** A flattened row/column view with explicit padding between rows. Only
   * this matrix cut may carry padding; inner axes retain their original order.
   * This describes a borrowed view, without packing or allocating a tensor. */
  static ContractionOperand matrix_view(std::initializer_list<int> labels,
                                        std::initializer_list<std::size_t> extents,
                                        PrecisionDtype dtype, std::size_t row_axes,
                                        std::size_t leading_dimension) {
    auto view = dense(labels, extents, dtype);
    if (row_axes > view.rank) throw std::invalid_argument("invalid matrix view cut");
    std::size_t columns = 1;
    for (auto i = row_axes; i < view.rank; ++i)
      columns = contraction_product(columns, view.shape[i]);
    if (leading_dimension < columns) throw std::invalid_argument("overlapping native matrix rows");
    auto stride = leading_dimension;
    for (auto i = row_axes; i != 0; --i) {
      view.strides[i - 1] = stride;
      stride = contraction_product(stride, view.shape[i - 1]);
    }
    return view;
  }

  std::size_t storage_elements() const {
    std::size_t last = 0;
    for (std::size_t i = 0; i < rank; ++i) {
      const auto offset = contraction_product(shape[i] - 1, strides[i]);
      if (offset > std::numeric_limits<std::size_t>::max() - last - 1)
        throw std::length_error("native contraction view address overflow");
      last += offset;
    }
    return last + 1;
  }
};

/** A compiler-owned einsum plus one admitted physical matrix implementation.
 * The template identities come from the canonical LoweringRequest. They refer
 * to the symbolic AOT template, whose representative extents are NOT the
 * runtime semantic identity: resolved operand modes/shapes below are also
 * required for binding compatibility. No provider identifier enters science.
 */
struct ContractionRequest {
  std::string_view scientific_identity, semantic_template_identity, precision_identity;
  std::array<ContractionOperand, 3> operands;
  runtime::PrecisionDirective precision;
  PrecisionDtype publication_dtype{PrecisionDtype::Fp64};
  char a_trans{'N'}, b_trans{'N'};
  std::size_t batches{1}, m{}, n{}, k{};
  double coefficient{1};
  // Zero leading dimensions preserve the dense AOT descriptor ABI. Explicit
  // row strides currently support unbatched matrices; padding is never packed
  // or read as scientific data. beta==0 must not read uninitialized output.
  std::array<std::size_t, 3> leading_dimensions{};
  double beta{};

  std::size_t leading_dimension(std::size_t operand) const {
    if (leading_dimensions[operand]) return leading_dimensions[operand];
    return operand == 0 ? (a_trans == 'N' ? k : m) : operand == 1 ? (b_trans == 'N' ? n : k) : n;
  }

  std::size_t output_elements() const {
    return contraction_product(batches, contraction_product(m, n));
  }
  std::size_t summands() const { return contraction_product(output_elements(), k); }

  /** Logical work counts do not require a matrix factorization or packing.
   * Call validate_affine before using an externally supplied descriptor. */
  std::size_t affine_output_elements() const { return operands[2].elements(); }
  std::size_t affine_summands() const {
    auto count = affine_output_elements();
    for (std::size_t axis = 0; axis < operands[0].rank; ++axis)
      if (!mode_extent(2, operands[0].modes[axis]))
        count = contraction_product(count, operands[0].shape[axis]);
    return count;
  }

  /** Validate the original binary einsum on arbitrary nonoverlapping affine
   * views. This does not require that a matrix recipe implements the layout;
   * general providers consume these same axes without inserting pack nodes.
   * Diagonals, broadcast strides and one-sided reductions remain unsupported. */
  void validate_affine() const {
    for (const auto identity :
         {scientific_identity, semantic_template_identity, precision_identity})
      if (identity.size() != 64 ||
          identity.find_first_not_of("0123456789abcdef") != std::string_view::npos)
        throw std::invalid_argument("native contraction requires compiler identities");
    if ((precision.storage_dtype != PrecisionDtype::Fp64 &&
         precision.storage_dtype != PrecisionDtype::Fp32) ||
        precision.math_mode != runtime::kStrictPrecisionMathMode ||
        precision.storage_dtype != precision.compute_dtype ||
        precision.compute_dtype != precision.accumulation_dtype ||
        publication_dtype != precision.storage_dtype)
      throw std::invalid_argument(
          "native affine candidate does not implement requested arithmetic");
    if (!std::isfinite(coefficient) || !std::isfinite(beta))
      throw std::invalid_argument("native affine coefficient is not finite");
    for (const auto& view : operands) {
      if (view.rank > ContractionOperand::kMaximumRank || view.dtype != precision.storage_dtype)
        throw std::invalid_argument("native affine operand rank or dtype is invalid");
      std::array<std::pair<std::size_t, std::size_t>, ContractionOperand::kMaximumRank> axes{};
      for (std::size_t axis = 0; axis < view.rank; ++axis) {
        if (!view.shape[axis] || view.modes[axis] < 0)
          throw std::invalid_argument("native affine requires positive extents and modes");
        if (!view.strides[axis] ||
            view.strides[axis] > std::size_t(std::numeric_limits<std::int64_t>::max()) ||
            view.shape[axis] > std::size_t(std::numeric_limits<std::int64_t>::max()))
          throw std::invalid_argument(
              "native affine requires positive signed-64-bit strides/extents");
        axes[axis] = {view.strides[axis], view.shape[axis]};
      }
      std::sort(axes.begin(), axes.begin() + view.rank);
      std::size_t span = 1;
      for (std::size_t axis = 0; axis < view.rank; ++axis) {
        const auto [stride, extent] = axes[axis];
        if (extent == 1) continue;
        if (stride < span) throw std::invalid_argument("native affine view overlaps itself");
        const auto offset = contraction_product(extent - 1, stride);
        if (offset > std::numeric_limits<std::size_t>::max() - span)
          throw std::length_error("native affine address overflow");
        span += offset;
      }
      const auto bytes =
          contraction_product(span, publication_dtype == PrecisionDtype::Fp64 ? 8 : 4);
      if (bytes > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::length_error("native affine address range overflow");
    }
    for (const auto& view : operands)
      for (std::size_t axis = 0; axis < view.rank; ++axis) {
        const auto mode = view.modes[axis];
        const auto a = mode_extent(0, mode), b = mode_extent(1, mode), c = mode_extent(2, mode);
        if ((a && b && a != b) || (a && c && a != c) || (b && c && b != c) || (c && !a && !b) ||
            (!c && (!a || !b)))
          throw std::invalid_argument("native affine has incompatible semantic modes");
      }
    (void)affine_summands();
  }

  /** Fail before resource preparation; unsupported arithmetic is never widened
   * or narrowed. FP32 compute with FP64 accumulation needs another candidate.
   */
  void validate() const {
    validate_affine();
    if (std::max({batches, m, n, k}) > std::size_t(std::numeric_limits<int>::max()) || !batches ||
        !m || !n || !k || !std::isfinite(coefficient) || !std::isfinite(beta) ||
        (a_trans != 'N' && a_trans != 'T') || (b_trans != 'N' && b_trans != 'T'))
      throw std::invalid_argument("native matrix candidate dimensions/coefficient are invalid");
    const std::array<std::size_t, 3> counts{contraction_product(batches, contraction_product(m, k)),
                                            contraction_product(batches, contraction_product(k, n)),
                                            output_elements()};
    for (std::size_t i = 0; i != operands.size(); ++i) {
      if (operands[i].rank > ContractionOperand::kMaximumRank)
        throw std::invalid_argument("native contraction operand rank exceeds bound");
      const auto columns = i == 0   ? (a_trans == 'N' ? k : m)
                           : i == 1 ? (b_trans == 'N' ? n : k)
                                    : n;
      const auto ld = leading_dimension(i);
      if (ld < columns || ld > std::size_t(std::numeric_limits<int>::max()) ||
          (batches != 1 && ld != columns))
        throw std::invalid_argument("native contraction leading dimension unsupported");
      std::size_t stride = 1, extent = 1;
      bool padded = false;
      for (auto axis = operands[i].rank; axis != 0; --axis) {
        // A unit column axis may precede the matrix cut. Do not mistake that
        // axis for a padded row merely because its extent product is one.
        if (!padded && ld != columns && extent == columns && operands[i].strides[axis - 1] == ld) {
          stride = ld;
          padded = true;
        }
        if (!operands[i].shape[axis - 1] || operands[i].strides[axis - 1] != stride)
          throw std::invalid_argument("native matrix strides differ from its semantic view");
        stride = contraction_product(stride, operands[i].shape[axis - 1]);
        extent = contraction_product(extent, operands[i].shape[axis - 1]);
      }
      if (ld != columns && !padded && counts[i] > columns)
        throw std::invalid_argument("native matrix view omits its row padding");
      if (operands[i].dtype != precision.storage_dtype || operands[i].elements() != counts[i])
        throw std::invalid_argument("native matrix candidate does not match semantic operands");
      const auto bytes = contraction_product(operands[i].storage_elements(),
                                             publication_dtype == PrecisionDtype::Fp64 ? 8 : 4);
      if (bytes > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::length_error("native contraction address range overflow");
    }
    validate_modes();
    (void)summands();
  }

 private:
  std::size_t mode_extent(std::size_t operand, int mode) const {
    std::size_t found = 0;
    const auto& view = operands[operand];
    for (std::size_t axis = 0; axis < view.rank; ++axis) {
      if (view.modes[axis] != mode) continue;
      if (found) throw std::invalid_argument("native contraction cannot implement diagonal modes");
      found = view.shape[axis];
    }
    return found;
  }

  // Check the physical recipe against the original semantic labels, including
  // collapsed axes. Equal element counts alone cannot detect a swapped axis or
  // a batch mistaken for a reduction. All scans are bounded by three ranks.
  void validate_modes() const {
    struct Group {
      std::array<int, ContractionOperand::kMaximumRank> modes{};
      std::size_t size{}, extent{1};
      void add(int mode, std::size_t n) {
        modes[size++] = mode;
        extent = contraction_product(extent, n);
      }
    } batch, rows, columns, reduction;
    const auto extent = [&](std::size_t operand, int mode) { return mode_extent(operand, mode); };
    const auto& output = operands[2];
    for (std::size_t i = 0; i != output.rank; ++i) {
      const auto mode = output.modes[i];
      auto& group = extent(0, mode) && extent(1, mode) ? batch : extent(0, mode) ? rows : columns;
      group.add(mode, output.shape[i]);
    }
    for (std::size_t i = 0; i != operands[0].rank; ++i)
      if (!extent(2, operands[0].modes[i]))
        reduction.add(operands[0].modes[i], operands[0].shape[i]);
    const auto ordered = [&](const ContractionOperand& view, const Group& first,
                             const Group& second) {
      std::size_t axis = 0;
      for (const auto* group : std::array<const Group*, 3>{&batch, &first, &second})
        for (std::size_t i = 0; i != group->size; ++i)
          if (axis >= view.rank || view.modes[axis++] != group->modes[i]) return false;
      return axis == view.rank;
    };
    if (batch.extent != batches || rows.extent != m || columns.extent != n ||
        reduction.extent != k || !ordered(output, rows, columns) ||
        !ordered(operands[0], a_trans == 'N' ? rows : reduction,
                 a_trans == 'N' ? reduction : rows) ||
        !ordered(operands[1], b_trans == 'N' ? reduction : columns,
                 b_trans == 'N' ? columns : reduction))
      throw std::invalid_argument("native matrix layout does not implement the semantic einsum");
  }
};

/** Immutable dense runtime domain of a compiler-emitted einsum template.
 * Exact-shape plan providers cannot bind this domain. Its shape-polymorphic
 * implementations accept any positive subextent, with the same mode ordering,
 * arithmetic, update semantics and packed physical layout. Validation uses
 * fixed-size stack storage; no descriptor cache grows with the tile count. */
struct BoundedContractionDomain {
  ContractionRequest maximum;

  explicit BoundedContractionDomain(ContractionRequest upper) : maximum(upper) {
    maximum.validate();
    if (maximum.batches != 1 || maximum.leading_dimensions != std::array<std::size_t, 3>{})
      throw std::invalid_argument("bounded contraction requires packed unbatched matrices");
  }

  void validate(const ContractionRequest& actual) const {
    actual.validate();
    if (actual.scientific_identity != maximum.scientific_identity ||
        actual.semantic_template_identity != maximum.semantic_template_identity ||
        actual.precision_identity != maximum.precision_identity ||
        actual.precision.storage_dtype != maximum.precision.storage_dtype ||
        actual.batches != maximum.batches || actual.a_trans != maximum.a_trans ||
        actual.b_trans != maximum.b_trans || actual.coefficient != maximum.coefficient ||
        actual.beta != maximum.beta || actual.leading_dimensions != maximum.leading_dimensions)
      throw std::invalid_argument("bounded contraction template changed; prepare again");
    for (std::size_t i = 0; i < maximum.operands.size(); ++i) {
      const auto& a = actual.operands[i];
      const auto& b = maximum.operands[i];
      if (a.rank != b.rank || a.modes != b.modes)
        throw std::invalid_argument("bounded contraction mode order changed");
      for (std::size_t j = 0; j < a.rank; ++j)
        if (a.shape[j] > b.shape[j])
          throw std::length_error("contraction exceeds its prepared runtime domain");
    }
  }
};

}  // namespace generativeqc::tensor
