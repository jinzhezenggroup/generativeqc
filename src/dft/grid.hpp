#ifndef GENERATIVEQC_DFT_GRID_HPP
#define GENERATIVEQC_DFT_GRID_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "core/types.hpp"

namespace generativeqc::dft {

/** Table-free molecular quadrature. Version 1 preserves the historical
 * one-Bohr fallback exactly. Version 2 is a resolved production contract and
 * requires an explicit radius for every element that is materialized. */
struct GridSpec {
  std::uint32_t version{1};
  std::size_t radial_points{48};
  std::size_t angular_polar{16};
  std::size_t angular_azimuth{32};
  unsigned partition_iterations{3};
  double coincident_tolerance{1.0e-12};
  /** In v1 zero selects the one-Bohr reference radius. In v2 zero is
   * unsupported/fail-closed. Slot zero is unused. Fixed storage preserves
   * immutable identity without borrowed pointers. */
  std::array<double, 119> element_radii{};
  bool operator==(const GridSpec&) const = default;
};

/** Pure prescription validation, shared by preparation and materialization. */
void validate_grid_spec(const GridSpec& spec);

/** Borrowed immutable CUDA quadrature. The opaque lifetime token keeps the
 * device allocation alive without exposing CUDA runtime types from this
 * backend-neutral header. A view is valid only on its recorded device. */
struct CudaMolecularGridView {
  int device{-1};
  const double *points{}, *weights{};
  std::size_t point_count{}, device_bytes{};
  std::uint64_t owner{};
  std::shared_ptr<const void> lifetime;

  explicit operator bool() const noexcept {
    return device >= 0 && points != nullptr && weights != nullptr && point_count != 0 &&
           device_bytes != 0 && owner != 0 && lifetime != nullptr;
  }
};

/** Owned molecular grid in atom-radial-polar-azimuth order. The ordinary
 * constructor is the independent CPU reference; CUDA preparation uses the
 * generated, bounded factory, retains the same host export contract and may
 * additionally retain the exact generated points/weights on their source
 * device for zero-copy downstream consumers. */
class MolecularGrid {
 public:
  explicit MolecularGrid(const core::System& system, GridSpec spec = {});
  /** Materialize with compiler-generated CUDA kernels. No CPU partition
   * fallback; throws on unsupported input, allocation or normalization error. */
  static MolecularGrid from_cuda(const core::System& system, GridSpec spec,
                                 int device,\n bool retain_device = true);

  const GridSpec& spec() const noexcept { return spec_; }
  const core::System& system() const noexcept { return system_; }
  std::size_t point_count() const noexcept { return weights_.size(); }
  const std::vector<double>& points() const noexcept { return points_; }
  const std::vector<double>& weights() const noexcept { return weights_; }
  const std::vector<std::uint32_t>& owners() const noexcept { return owners_; }
  /** Borrow the exact CUDA-generated points/weights when this grid originated
   * from from_cuda(). CPU grids return an empty view. */
  CudaMolecularGridView cuda_view() const noexcept {
    if (!cuda_storage_) return {};
    return {cuda_device_,       cuda_points_, cuda_weights_, point_count(),
            cuda_device_bytes_, cuda_owner_,  cuda_storage_};
  }
  /** Explicit derivative export of the atomic measure before partitioning.
   * Reconstruct only quadrature rules, not Becke weights, on request; energy
   * execution retains no additional point-sized array. */
  std::vector<double> atomic_weights() const;
  /** Contract dE/dw(point) directly with the analytic nuclear response of
   * the exact materialized Becke weights. Grid points translate with their
   * owner atoms; this routine differentiates only the partition weights, not
   * point coordinates or the element/radial/angular atomic measure. */
  std::vector<double> contract_weight_derivative(std::span<const double> weight_sensitivity) const;

 private:
  struct Deferred {};
  MolecularGrid(const core::System& system, GridSpec spec, Deferred);
  static std::pair<std::vector<double>, std::vector<double>> legendre_rule(std::size_t count);
  core::System system_;
  GridSpec spec_;
  std::vector<double> points_;
  std::vector<double> weights_;
  std::vector<std::uint32_t> owners_;
  std::shared_ptr<const void> cuda_storage_;
  const double *cuda_points_{}, *cuda_weights_{};
  std::size_t cuda_device_bytes_{};
  std::uint64_t cuda_owner_{};
  int cuda_device_{-1};
};

/** Pure shape query for the retained full-grid CUDA points/weights allocation. */
std::size_t cuda_resident_grid_bytes(std::size_t points);

/** Pure shape query for peak CUDA quadrature preparation. The peak includes
 * the retained full-grid points/weights because they coexist with bounded
 * quadrature scratch until the generated grid has been validated. */
std::size_t cuda_quadrature_bytes(std::size_t atoms, std::size_t points);

}  // namespace generativeqc::dft

#endif
