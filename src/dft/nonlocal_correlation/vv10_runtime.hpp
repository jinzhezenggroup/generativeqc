#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "dft/nonlocal_correlation/vv10_capacity.hpp"
#include "generativeqc/generativeqc.h"

#if GENERATIVEQC_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace generativeqc::dft {
struct GridTaskView;
}

namespace generativeqc::dft::nlc {

enum class Vv10Variant : std::int32_t { vv10 = 1, rvv10 = 2 };

struct Vv10Parameters {
  Vv10Variant variant{Vv10Variant::vv10};
  double b{};
  double c{};
  double coefficient{1.0};
  bool operator==(const Vv10Parameters&) const = default;
};

struct Vv10ResourceUsage {
  std::uint64_t workspace_bytes{};
  std::uint64_t host_workspace_bytes{};
  std::uint64_t device_workspace_bytes{};
  std::uint64_t maximum_bytes{};
  std::uint64_t pair_evaluations{};
  std::uint64_t tiles{};
  std::uint32_t point_count{};
  std::uint32_t tile_points{};
};

class Vv10Plan {
 public:
  static std::unique_ptr<Vv10Plan> prepare(generativeqc_backend backend, int device_id,
                                           std::uint32_t point_count, std::uint32_t tile_points,
                                           Vv10Parameters parameters, std::uint64_t maximum_bytes,
                                           std::string& detail, generativeqc_status& status);

  [[nodiscard]] generativeqc_backend backend() const noexcept { return backend_; }
  [[nodiscard]] int device_id() const noexcept { return device_id_; }
  [[nodiscard]] const Vv10Parameters& parameters() const noexcept { return parameters_; }
  [[nodiscard]] const Vv10ResourceUsage& resources() const noexcept { return resources_; }

  // Actual successful CPU pair work; zero before execution, after failure, or
  // for CUDA execution. resources().pair_evaluations remains the dense bound.
  [[nodiscard]] std::uint64_t last_execution_pair_evaluations() const noexcept {
    return last_execution_pair_evaluations_;
  }

  // weighted_potential_only is an internal molecular E/V contract: feature
  // outputs at exactly zero weights may be zeroed because the caller only
  // consumes weight * vrho/vsigma. Raw and geometry consumers must not opt in.
  // Unsupported variants/shapes or unsafe numerical ranges retain dense work.
  generativeqc_status execute(std::span<const double> coordinates, std::span<const double> weights,
                              std::span<const double> density,
                              std::span<const double> density_gradient, double& energy,
                              std::span<double> vrho, std::span<double> vsigma,
                              std::span<double> point_derivative,
                              std::span<double> weight_derivative, std::string& detail,
                              bool weighted_potential_only = false);

 private:
  Vv10Plan(generativeqc_backend backend, int device_id, Vv10Parameters parameters,
           Vv10ResourceUsage resources)
      : backend_(backend), device_id_(device_id), parameters_(parameters), resources_(resources) {}

  generativeqc_backend backend_{};
  int device_id_{};
  Vv10Parameters parameters_{};
  Vv10ResourceUsage resources_{};
  std::uint64_t last_execution_pair_evaluations_{};
  std::vector<double> omega_;
  std::vector<double> kappa_;
  std::vector<double> weighted_density_;
  std::vector<double> domega_drho_;
  std::vector<double> domega_dsigma_;
  std::vector<double> dkappa_drho_;
};

#if GENERATIVEQC_HAS_CUDA
/** Enqueue one resident fixed-grid evaluation on the caller stream.
 * All scientific inputs and outputs are device-resident. No allocation,
 * host transfer or synchronization occurs here. energy may alias workspace
 * because it is published only after all pair kernels have consumed scratch.
 * point_derivative may alias density_gradient: local-scale construction is
 * the last gradient reader and precedes every point-derivative writer on stream.
 * Mixed execution narrows bounded pair algebra only. Storage, local scales,
 * coordinate differences, weighted contractions and reductions remain FP64.
 * Optional arithmetic_work has two device counters (FP32/strict pair values),
 * reset on this stream; an exact-key owner must include arithmetic in its key.
 */
void enqueue_vv10_cuda_device(const Vv10CudaDeviceLayout& layout, Vv10Parameters parameters,
                              int device_id, cudaStream_t stream, const double* coordinates,
                              const double* weights, const double* density,
                              const double* density_gradient, void* workspace,
                              std::size_t workspace_bytes, double* energy, double* vrho,
                              double* vsigma, double* point_derivative, double* weight_derivative,
                              int* numerical_error, const int* cache_mismatch = nullptr,
                              bool mixed_pair_arithmetic = false,
                              std::uint64_t* arithmetic_work = nullptr);

/** Apply the molecular fixed-grid domain policy without changing point extent.
 * Inputs are validated before screening. rho<threshold becomes zero weight,
 * rho=1 and grad-rho=0 so inactive points disappear from both pair domains
 * while local scales remain finite. No allocation, transfer or fence occurs.
 * Effective density and gradient may respectively alias their inputs; each
 * point's inputs are loaded before its thread writes any effective output.
 */
void enqueue_vv10_molecular_domain_cuda(cudaStream_t stream, std::size_t point_count,
                                        double density_threshold, const double* weights,
                                        const double* density, const double* density_gradient,
                                        double* effective_weights, double* effective_density,
                                        double* effective_density_gradient, int* numerical_error);

/** Gather total rho and grad-rho from one resident GridTaskView tile.
 * The tile is written into [offset, offset+npoint) of caller-owned full-grid
 * buffers on the borrowed stream. No allocation, transfer or fence occurs.
 */
void enqueue_vv10_collect_total_features_cuda(cudaStream_t stream,
                                              const generativeqc::dft::GridTaskView& view,
                                              std::size_t offset, std::size_t total_points,
                                              double* density, double* density_gradient,
                                              int* numerical_error);

/** Pack point derivatives into the shared [6,N] stationary seed layout.
 * vrho/vsigma/weight derivatives already occupy seed rows 0,1,5. This fills
 * x/y/z rows 2,3,4 from the native AoS point derivative and poisons all seed
 * rows with NaN when any upstream asynchronous error flag is set.
 */
void enqueue_vv10_pack_force_seeds_cuda(cudaStream_t stream, std::size_t point_count,
                                        const double* effective_weights,
                                        const double* point_derivative, double* seeds,
                                        const int* collect_error, const int* domain_error,
                                        const int* pair_error, const int* cache_mismatch = nullptr);

/** Count an actual pair sweep or exact-key reuse on device, without a fence. */
void enqueue_vv10_cache_work_cuda(cudaStream_t stream, const int* mismatch,
                                  std::uint64_t* counters);

/** Compare the complete VV10 pair ingredient key against retained scratch.
 * Coordinates, weights, parameters, device and schedule are immutable/checked
 * by the owning force plan. Slots 0..5 retain the exact generated local scales
 * and weighted density; the dead energy-terms panel at slot 6 retains rho.
 * This is bitwise equality, not a tolerance or fingerprint. No allocation or
 * synchronization occurs; mismatch is a caller-owned, initially zero flag.
 */
void enqueue_vv10_force_key_compare_cuda(cudaStream_t stream, std::size_t point_count,
                                         Vv10Parameters parameters, const double* weights,
                                         const double* density, const double* gradient,
                                         const double* retained_workspace, const int* domain_error,
                                         const int* pair_error, int* mismatch);

/** Preserve rho and validity in dead energy/index scratch after seed packing. */
void enqueue_vv10_store_force_key_cuda(cudaStream_t stream, std::size_t point_count,
                                       const double* density, double* workspace,
                                       const int* collect_error, const int* domain_error,
                                       const int* pair_error);

void execute_vv10_cuda(const double* coordinates, const double* weights, const double* density,
                       const double* density_gradient, std::size_t point_count,
                       std::size_t tile_points, Vv10Parameters parameters, int device_id,
                       double& energy, double* vrho, double* vsigma, double* point_derivative,
                       double* weight_derivative);
#endif

}  // namespace generativeqc::dft::nlc
