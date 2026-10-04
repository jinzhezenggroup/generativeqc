#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string_view>

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
 * This first native adapter accepts dense input/output views only. Packing
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

  std::size_t output_elements() const {
    return contraction_product(batches, contraction_product(m, n));
  }
  std::size_t summands() const { return contraction_product(output_elements(), k); }

  /** Fail before resource preparation; unsupported arithmetic is never widened
   * or narrowed. FP32 compute with FP64 accumulation needs another candidate.
   */
  void validate() const {
    for (const auto identity :
         {scientific_identity, semantic_template_identity, precision_identity}) {
      if (identity.size() != 64 ||
          identity.find_first_not_of("0123456789abcdef") != std::string_view::npos)
        throw std::invalid_argument("native contraction requires compiler identities");
    }
    if (precision.math_mode != runtime::kStrictPrecisionMathMode ||
        precision.storage_dtype != precision.compute_dtype ||
        precision.compute_dtype != precision.accumulation_dtype ||
        publication_dtype != precision.storage_dtype)
      throw std::invalid_argument(
          "native matrix candidate does not implement requested arithmetic");
    if (std::max({batches, m, n, k}) > std::size_t(std::numeric_limits<int>::max()) || !batches ||
        !m || !n || !k || !std::isfinite(coefficient) || (a_trans != 'N' && a_trans != 'T') ||
        (b_trans != 'N' && b_trans != 'T'))
      throw std::invalid_argument("native matrix candidate dimensions/coefficient are invalid");
    const std::array<std::size_t, 3> counts{contraction_product(batches, contraction_product(m, k)),
                                            contraction_product(batches, contraction_product(k, n)),
                                            output_elements()};
    for (std::size_t i = 0; i != operands.size(); ++i) {
      if (operands[i].rank > ContractionOperand::kMaximumRank)
        throw std::invalid_argument("native contraction operand rank exceeds bound");
      std::size_t stride = 1;
      for (auto axis = operands[i].rank; axis != 0; --axis) {
        if (!operands[i].shape[axis - 1] || operands[i].strides[axis - 1] != stride)
          throw std::invalid_argument("native matrix candidate requires explicit dense packing");
        stride = contraction_product(stride, operands[i].shape[axis - 1]);
      }
      if (operands[i].dtype != precision.storage_dtype || operands[i].elements() != counts[i])
        throw std::invalid_argument("native matrix candidate does not match semantic operands");
      const auto bytes =
          contraction_product(counts[i], publication_dtype == PrecisionDtype::Fp64 ? 8 : 4);
      if (bytes > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::length_error("native contraction address range overflow");
    }
    validate_modes();
    (void)summands();
  }

 private:
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
    const auto extent = [&](std::size_t operand, int mode) {
      std::size_t found = 0;
      const auto& view = operands[operand];
      for (std::size_t i = 0; i != view.rank; ++i) {
        if (view.modes[i] != mode) continue;
        if (found)
          throw std::invalid_argument("native matrix candidate cannot implement diagonal modes");
        found = view.shape[i];
      }
      return found;
    };
    for (const auto& view : operands) {
      for (std::size_t i = 0; i != view.rank; ++i) {
        const auto a = extent(0, view.modes[i]), b = extent(1, view.modes[i]),
                   c = extent(2, view.modes[i]);
        if (view.modes[i] < 0 || (a && b && a != b) || (a && c && a != c) || (b && c && b != c) ||
            (c && !a && !b) || (!c && (!a || !b)))
          throw std::invalid_argument("native matrix recipe has incompatible semantic modes");
      }
    }
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

}  // namespace generativeqc::tensor
