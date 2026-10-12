#pragma once

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "dft/ao_grid_work.hpp"
#include "dft/cuda_ks.hpp"
#include "dft/cuda_stationary_binding.hpp"
#include "runtime/resource_ledger.hpp"

namespace generativeqc::dft {
struct CudaStationaryResources {
  std::size_t device_upper_bytes{}, stationary_numeric_bytes{}, grid_numeric_bytes{};
  std::size_t grid_provider_retained_bytes{}, grid_prepare_device_delta{};
  std::size_t grid_provider_allowance_bytes{};
  std::size_t minimum_device_bytes{};
  std::array<std::uint64_t, 24> stationary_work{};
  AoGridWork grid_work;
};

/** Complete native Direct stationary source controller, not an admission rule.
 * Reuses the generated profile and native grid/integral owners. It never exports
 * dense D/W or sums sources on the host. The owning calculation supplies the
 * authenticated resident frame and must remain locked/alive through completion.
 * Prepared owners share their immutable scientific rule with descriptor sizing;
 * the private endpoint also permits focused scientific/resource qualification.
 * Arena checks alone never establish complete endpoint performance or RSS. */
class CudaStationaryController {
 public:
  CudaStationaryController(const core::System& system, const AoBasis& basis,
                           const MolecularGrid& grid, std::string_view profile, int device,
                           std::size_t tile_points, std::size_t maximum_device_bytes);
  ~CudaStationaryController();
  CudaStationaryController(const CudaStationaryController&) = delete;
  CudaStationaryController& operator=(const CudaStationaryController&) = delete;

  generativeqc_status gradient(const CudaKsResidentDensityBinding& density,
                               const CudaKsResidentStationaryWeightsBinding& weights,
                               const scf::PreparedFockPlan& fock,
                               const std::function<bool()>& token_current,
                               std::vector<double>& output, std::string& detail,
                               bool allow_spatial_default = false);
  const CudaStationaryResources& resources() const noexcept { return resources_; }

 private:
  void close() noexcept;
  int device_{};
  const CudaStationaryBinding* binding_{};
  CudaMolecularGridView quadrature_;
  std::size_t aos_{}, tile_points_{}, points_per_atom_{};
  double coincident_tolerance_{};
  std::vector<double> centers_, charges_;
  void *grid_{}, *stationary_{};
  std::optional<runtime::DeviceProviderAllowance> grid_provider_allowance_;
  CudaStationaryResources resources_;
};
}  // namespace generativeqc::dft
