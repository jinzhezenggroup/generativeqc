#pragma once

#include "tensor/native_contraction.hpp"

namespace generativeqc::tensor {

/** Physical matrix recipe for an existing affine contraction. This record is
 * execution metadata, not a second operation: resolved modes and precision in
 * ContractionRequest remain authoritative. No packing or symmetry is inferred. */
struct MatrixContractionRecipe {
  struct Layout {
    std::size_t rows{}, columns{}, ld{}, batch_stride{};
    bool row_major{};
  };
  std::array<Layout, 3> layouts;
  std::size_t batches{1};

  static MatrixContractionRecipe from(const ContractionRequest& request) {
    request.validate_affine();
    // Derive ordered M/N/K/batch groups from the original modes. Flattening
    // is legal only when each operand proves the same group address order;
    // equal element counts cannot prove a transpose or reduction permutation.
    struct Group {
      std::array<int, ContractionOperand::kMaximumRank> modes{};
      std::size_t size{};
      void add(int mode) { modes[size++] = mode; }
    } rows, columns, reduction, batch;
    const auto contains = [&](std::size_t operand, int mode) {
      const auto& view = request.operands[operand];
      return std::find(view.modes.begin(), view.modes.begin() + view.rank, mode) !=
             view.modes.begin() + view.rank;
    };
    const auto& output = request.operands[2];
    for (std::size_t axis = 0; axis < output.rank; ++axis) {
      const int mode = output.modes[axis];
      (contains(0, mode) && contains(1, mode) ? batch
       : contains(0, mode)                    ? rows
                                              : columns)
          .add(mode);
    }
    const auto& left = request.operands[0];
    for (std::size_t axis = 0; axis < left.rank; ++axis)
      if (!contains(2, left.modes[axis])) reduction.add(left.modes[axis]);
    if (!rows.size || !columns.size || !reduction.size)
      throw std::invalid_argument("affine matrix requires nonempty M/N/K groups");
    const auto collapse = [](const ContractionOperand& view, const Group& group) {
      std::size_t extent = 1, stride = 0;
      for (auto index = group.size; index != 0; --index) {
        const auto axis =
            std::find(view.modes.begin(), view.modes.begin() + view.rank, group.modes[index - 1]) -
            view.modes.begin();
        if (axis == view.rank)
          throw std::invalid_argument("affine matrix mode group is incomplete");
        const auto size = view.shape[axis], physical = view.strides[axis];
        if (size != 1) {
          if (!stride) stride = physical;
          if (physical != contraction_product(extent, stride))
            throw std::invalid_argument("affine matrix mode group is not physically contiguous");
        }
        extent = contraction_product(extent, size);
      }
      return std::pair{extent, stride ? stride : 1};
    };
    MatrixContractionRecipe result;
    const std::array<const Group*, 3> row_groups{&rows, &reduction, &rows},
        column_groups{&reduction, &columns, &columns};
    for (std::size_t operand = 0; operand < 3; ++operand) {
      const auto& view = request.operands[operand];
      auto& out = result.layouts[operand];
      const auto [row_extent, rs] = collapse(view, *row_groups[operand]);
      const auto [column_extent, cs] = collapse(view, *column_groups[operand]);
      out.rows = row_extent;
      out.columns = column_extent;
      const auto [batches, batch_stride] = collapse(view, batch);
      result.batches = batches;
      out.batch_stride = batch.size ? batch_stride : 0;
      if ((out.columns == 1 || cs == 1) && (out.rows == 1 || rs >= out.columns)) {
        out.row_major = true;
        out.ld = out.rows > 1 ? rs : out.columns;
      } else if ((out.rows == 1 || rs == 1) && (out.columns == 1 || cs >= out.rows)) {
        out.ld = out.columns > 1 ? cs : out.rows;
      } else {
        throw std::invalid_argument("affine matrix requires native row/column matrix strides");
      }
      // validate_affine already proved these products/sums fit the address space.
      const auto span = (out.rows - 1) * rs + (out.columns - 1) * cs + 1;
      if (result.batches > 1 && out.batch_stride < span)
        throw std::invalid_argument("affine matrix matrix batches overlap");
      if (std::max({out.rows, out.columns, out.ld, result.batches}) >
          std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("affine matrix matrix dimensions exceed native bound");
    }
    return result;
  }
};

}  // namespace generativeqc::tensor
