#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string_view>

#include "dft/host_hessian_aot.hpp"

namespace generativeqc::dft {

struct HostGgaResponseShape {
  std::size_t basis_functions{}, points{}, spins{}, tile_points{};
};

/** A prepared owner's real AO values/first derivatives and fixed grid weights.
 * The reader fills component/point/AO-major storage, components (value,x,y,z),
 * for the requested consecutive points. It must not retain the output spans.
 * Grid construction, AO representation and their identity belong to the caller.
 */
using HostGgaTileReader =
    std::function<void(std::size_t, std::size_t, std::span<double>, std::span<double>)>;

struct HostGgaResponseWork {
  /** Actions/point_visits count physical directions; tile_reads and scalar
   * point slots count actual shared provider work, not padded lane estimates.
   */
  std::size_t actions_started{}, actions_completed{}, point_visits{}, tile_reads{};
  /** Submitted scalar point slots and exact replayed slots. A failed scalar
   * batch may terminate before evaluating every submitted slot.
   */
  std::size_t xc_evaluated_points{}, cached_xc_points{};
};

/** Bounded, synchronous, fixed-grid semilocal AO Fock response, not a Hessian.
 * RKS uses one total-density matrix; UKS uses two occupation-one spin matrices.
 * Density/direction/output are spin/row/column-major in the same real AO basis.
 * The immutable reference density is borrowed. validate_current must reject a
 * dead or changed stationary owner, including geometry/grid/method/spin changes.
 * Exact linked XC and coefficient-response identities are mandatory; no JIT or
 * reference fallback exists. Nonnegative spin densities and compatible zero
 * gradients use the linked graph's exact production vacuum branch. Nonfinite
 * derivatives outside that branch fail, never acquire a density/tangent floor.
 *
 * workspace_bytes bounds owned heap numeric panels, including transactional
 * output staging, not borrowed state, callback/provider storage, compiled scalar
 * stack, allocator overhead or process RSS. The mandatory workspace is tile-bounded.
 * On supported CPUs, ordered SIMD contractions reuse coefficient-input scratch
 * before those inputs become live. Insufficient tile capacity or unavailable
 * SIMD retains scalar fields; neither path allocates an additional panel.
 * Optional exact point-partial storage uses a separately admitted byte allowance,
 * never an implicit full-grid allocation. Calls on a single owner may not overlap/reenter.
 * Output is untouched on failure and may alias the direction, but must not overlap
 * the borrowed reference density.
 */
class HostGgaXcResponse {
 public:
  static std::size_t workspace_bytes(HostGgaResponseShape shape);
  /** Additional caller-owned FP64 scratch for 1--4 same-frame directions.
   * One direction uses the original owner arena and needs zero extra bytes.
   * Includes batch staging, packed inputs, fields, coefficients and AO packing;
   * the enclosing response owner must price it before admitting other caches.
   */
  static std::size_t batch_workspace_bytes(HostGgaResponseShape shape, std::size_t directions);
  /** Largest whole-tile point prefix fitting the additional numeric allowance.
   * Stores exact linked XC inputs/outputs, not AO matrices or a response tensor.
   */
  static std::size_t primal_cache_bytes(HostGgaResponseShape shape, std::size_t maximum_bytes);

  HostGgaXcResponse(HostGgaResponseShape shape, std::size_t budget_bytes,
                    std::span<const double> density, std::string_view xc_identity,
                    std::string_view coefficient_response_identity, HostGgaTileReader reader,
                    std::function<void()> validate_current);

  void apply(std::span<const double> direction, std::span<double> output);
  /** Synchronous same-frame independent directions, not a coupled/block solve.
   * Inputs/outputs are [direction,spin,row,column]. Output publishes atomically
   * across all directions and may alias the inputs, never the borrowed density.
   * Workspace must cover batch_workspace_bytes and be disjoint from inputs,
   * outputs, density and this owner's storage. It remains caller-owned and is
   * NOT included in scratch_bytes(). No batch allocation or cross-call field
   * cache is hidden here; insufficient capacity fails before source work.
   */
  void apply_batch(std::span<const double> directions, std::span<double> output,
                   std::span<double> workspace);
  /** Allocate only, before any action. The first successful action prepares the
   * bank; later actions authenticate every input bit before reuse. An incomplete
   * first preparation is never reused. Only optional allocation failure or an
   * insufficient allowance selects the unchanged streamed evaluator.
   */
  bool admit_primal_cache(std::size_t maximum_bytes);
  [[nodiscard]] std::size_t scratch_bytes() const noexcept {
    return (storage_count_ + cache_count_) * sizeof(double);
  }
  [[nodiscard]] HostGgaResponseWork work() const noexcept { return work_; }

 private:
  HostGgaResponseShape shape_;
  std::span<const double> density_;
  const host_hessian_aot::Program* xc_{};
  const host_hessian_aot::Program* coefficients_{};
  HostGgaTileReader reader_;
  std::function<void()> validate_current_;
  std::array<std::size_t, 5> feature_inputs_{}, values_{}, tangents_{};
  std::array<std::size_t, 6> gradients_{}, gradient_tangents_{};
  std::unique_ptr<double[]> storage_, cache_;
  std::size_t storage_count_{}, cache_count_{}, cache_points_{};
  bool cache_ready_{};
  std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
  HostGgaResponseWork work_;
};

}  // namespace generativeqc::dft
