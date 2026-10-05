#pragma once

#include "tensor/native_contraction.hpp"

namespace generativeqc::tensor {

/** Physical cuBLASLt recipe for an existing affine contraction. This record is
 * provider metadata, not a second operation: resolved modes and precision in
 * ContractionRequest remain authoritative. No packing or symmetry is inferred. */
struct CublasLtMatrixRecipe {
  struct Layout {
    std::size_t rows{}, columns{}, ld{}, batch_stride{};
    bool row_major{};
  };
  std::array<Layout, 3> layouts;
  std::size_t batches{1};

  static CublasLtMatrixRecipe from(const ContractionRequest& request) {
    request.validate_affine();
    // Membership identifies M/N/K, even when all extents happen to be equal.
    std::array<int, 8> mode{};
    std::array<std::size_t, 8> count{};
    for (std::size_t operand = 0; operand < 3; ++operand) {
      const auto& view = request.operands[operand];
      if (view.rank < 2 || view.rank > 3)
        throw std::invalid_argument("cuBLASLt adapter requires rank-2/3 matrix views");
      for (std::size_t axis = 0; axis < view.rank; ++axis) {
        const auto label = view.modes[axis];
        unsigned membership = 0;
        for (std::size_t other = 0; other < 3; ++other) {
          const auto& peer = request.operands[other];
          if (std::find(peer.modes.begin(), peer.modes.begin() + peer.rank, label) !=
              peer.modes.begin() + peer.rank)
            membership |= 1U << other;
        }
        // Count each label once, from the first operand in its membership.
        if ((membership & ((1U << operand) - 1)) == 0) {
          ++count[membership];
          mode[membership] = label;
        }
      }
    }
    if (count[3] != 1 || count[5] != 1 || count[6] != 1 || count[7] > 1 || count[1] || count[2] ||
        count[4])
      throw std::invalid_argument("cuBLASLt requires one M/N/K mode and optional batch");
    CublasLtMatrixRecipe result;
    const std::array<int, 3> row_modes{mode[5], mode[3], mode[5]},
        column_modes{mode[3], mode[6], mode[6]};
    for (std::size_t operand = 0; operand < 3; ++operand) {
      const auto& view = request.operands[operand];
      auto& out = result.layouts[operand];
      std::size_t rs{}, cs{};
      for (std::size_t axis = 0; axis < view.rank; ++axis) {
        if (view.modes[axis] == row_modes[operand]) {
          out.rows = view.shape[axis];
          rs = view.strides[axis];
        } else if (view.modes[axis] == column_modes[operand]) {
          out.columns = view.shape[axis];
          cs = view.strides[axis];
        } else if (count[7] && view.modes[axis] == mode[7]) {
          result.batches = view.shape[axis];
          out.batch_stride = view.strides[axis];
        }
      }
      if ((out.columns == 1 || cs == 1) && (out.rows == 1 || rs >= out.columns)) {
        out.row_major = true;
        out.ld = out.rows > 1 ? rs : out.columns;
      } else if ((out.rows == 1 || rs == 1) && (out.columns == 1 || cs >= out.rows)) {
        out.ld = out.columns > 1 ? cs : out.rows;
      } else {
        throw std::invalid_argument("cuBLASLt requires native row/column matrix strides");
      }
      // validate_affine already proved these products/sums fit the address space.
      const auto span = (out.rows - 1) * rs + (out.columns - 1) * cs + 1;
      if (result.batches > 1 && out.batch_stride < span)
        throw std::invalid_argument("cuBLASLt matrix batches overlap");
      if (std::max({out.rows, out.columns, out.ld, result.batches}) >
          std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("cuBLASLt matrix dimensions exceed native bound");
    }
    return result;
  }
};

}  // namespace generativeqc::tensor
