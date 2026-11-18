#pragma once

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"

namespace generativeqc::dft {

/** Optional exact first-order AO tiles for a single borrowed native frame.
 * Admission owns only a whole-tile prefix; the numeric budget includes packed
 * basis bytes, point coordinates and all four AO components. Metadata is bounded
 * by the admitted tiles. Every hit checks input content and the complete floating
 * environment, and restores the evaluator's resulting environment. Weights are
 * always read live. Misses/tails use the original evaluator, not an approximation.
 * The borrowing nuclear owner still authenticates geometry/epochs around reads;
 * this cache neither replaces that validation nor caches the generic XC reader.
 */
class HostAoPrefix {
 public:
  HostAoPrefix(const AoBasis& basis, const MolecularGrid& grid, std::size_t tile)
      : basis_(basis),
        grid_(grid),
        tile_(tile),
        atoms_(basis.natom),
        primitives_(basis.nprimitive),
        basis_count_(basis.nao),
        basis_size_(basis.packed.size()),
        grid_count_(grid.point_count()) {
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (!tile_ || !atoms_ || !primitives_ || !basis_count_ || atoms_ > maximum / 3 ||
        primitives_ > maximum / 2 || basis_count_ > maximum / 16 || grid_count_ > maximum / 3 ||
        2 * primitives_ > maximum - 3 * atoms_ ||
        16 * basis_count_ > maximum - 3 * atoms_ - 2 * primitives_ ||
        basis_size_ != 3 * atoms_ + 2 * primitives_ + 16 * basis_count_)
      throw std::invalid_argument("invalid native AO-prefix shape");
    per_point_ = 4 * basis_count_ + 3;
    validate_shape();
  }

  /** Allocate before the first read. Return actual newly owned numeric bytes.
   * Insufficient capacity or optional allocation failure keeps streaming.
   * An admitted prefix never grows; repeated admission is a no-op.
   */
  std::size_t admit(std::size_t maximum_bytes) {
    validate_shape();
    if (reads_) throw std::logic_error("native AO-prefix admission follows a read");
    if (storage_) return 0;
    const auto capacity = maximum_bytes / sizeof(double);
    if (basis_size_ > capacity) return 0;
    auto points = std::min(grid_count_, (capacity - basis_size_) / per_point_);
    if (points < grid_count_) points -= points % tile_;
    if (!points) return 0;
    const auto elements = basis_size_ + points * per_point_;
    try {
      std::vector<Entry> entries(points / tile_ + (points % tile_ != 0));
      auto storage = std::make_unique<double[]>(elements);
      std::copy(basis_.packed.begin(), basis_.packed.end(), storage.get());
      entries_ = std::move(entries);
      storage_ = std::move(storage);
    } catch (const std::bad_alloc&) {
      return 0;
    }
    points_ = points;
    elements_ = elements;
    return numeric_bytes();
  }

  /** Read [jet,point,AO] and current weights. A failed or nonfinite evaluation
   * never publishes a ready entry; downstream numerical checks remain in force.
   * Shape mutations are rejected before divisions or borrowed pointer arithmetic.
   */
  void read(std::size_t begin, std::size_t count, std::span<double> ao, std::span<double> weights) {
    validate_shape();
    if (begin > grid_count_ || count > grid_count_ - begin ||
        count > std::numeric_limits<std::size_t>::max() / basis_count_ / 4 ||
        ao.size() != 4 * count * basis_count_ || weights.size() != count)
      throw std::invalid_argument("native AO-prefix tile shape mismatch");
    ++reads_;
    if (!count) return;
    std::fenv_t before{};
    const bool environment_valid = std::fegetenv(&before) == 0;
    const bool cacheable =
        storage_ && environment_valid && begin % tile_ == 0 && begin < points_ &&
        count <= points_ - begin && count == std::min(tile_, grid_count_ - begin) &&
        std::memcmp(storage_.get(), basis_.packed.data(), basis_size_ * sizeof(double)) == 0;
    // Tail offsets can exceed the allocation even without dereferencing them.
    // Form retained pointers only after the complete prefix bounds check.
    auto* saved_points = cacheable ? storage_.get() + basis_size_ + 3 * begin : nullptr;
    auto* saved_ao =
        cacheable ? storage_.get() + basis_size_ + 3 * points_ + 4 * begin * basis_count_ : nullptr;
    auto* entry = cacheable ? &entries_[begin / tile_] : nullptr;
    const auto* points = grid_.points().data() + 3 * begin;
    if (entry && entry->ready && std::memcmp(&entry->before, &before, sizeof(before)) == 0 &&
        std::memcmp(saved_points, points, 3 * count * sizeof(double)) == 0 &&
        std::fesetenv(&entry->after) == 0) {
      std::copy_n(saved_ao, ao.size(), ao.begin());
      cached_points_ += count;
    } else {
      basis_.evaluate(points, count, 1, 0, basis_count_, ao.data(), ao.size());
      evaluated_points_ += count;
      std::fenv_t after{};
      if (entry && !entry->ready && std::fegetenv(&after) == 0 &&
          std::all_of(ao.begin(), ao.end(), [](double value) { return std::isfinite(value); })) {
        std::copy_n(points, 3 * count, saved_points);
        std::copy(ao.begin(), ao.end(), saved_ao);
        std::memcpy(&entry->before, &before, sizeof(before));
        std::memcpy(&entry->after, &after, sizeof(after));
        entry->ready = true;
      }
    }
    std::copy_n(grid_.weights().begin() + begin, count, weights.begin());
  }

  std::size_t numeric_bytes() const noexcept { return elements_ * sizeof(double); }
  std::size_t prefix_points() const noexcept { return points_; }
  std::size_t reads() const noexcept { return reads_; }
  std::size_t evaluated_points() const noexcept { return evaluated_points_; }
  std::size_t cached_points() const noexcept { return cached_points_; }

 private:
  struct Entry {
    std::fenv_t before{}, after{};
    bool ready{};
  };

  void validate_shape() const {
    if (basis_.natom != atoms_ || basis_.nprimitive != primitives_ || basis_.nao != basis_count_ ||
        basis_.packed.size() != basis_size_ || grid_.point_count() != grid_count_ ||
        grid_.points().size() != 3 * grid_count_)
      throw std::invalid_argument("native AO-prefix borrowed shape changed");
  }

  const AoBasis& basis_;
  const MolecularGrid& grid_;
  std::size_t tile_{}, atoms_{}, primitives_{}, basis_count_{}, basis_size_{}, grid_count_{};
  std::size_t per_point_{}, points_{}, elements_{};
  std::unique_ptr<double[]> storage_;
  std::vector<Entry> entries_;
  std::size_t reads_{}, evaluated_points_{}, cached_points_{};
};

}  // namespace generativeqc::dft
