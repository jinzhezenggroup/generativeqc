#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <vector>

#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/topology.hpp"

namespace generativeqc::scf::cuda_execution {

struct ShellPairDensityBounds;

/** Optional geometry owner for the generated pure-J consumer. It borrows the
 * direct provider's stream and public basis metadata, and owns bounded shell
 * topology, Cartesian transforms and scratch. No quartet list is materialized.
 */
struct GeneratedCoulombPlan {
  DeviceBatch batch{};
  cudaStream_t stream{};
  std::vector<void*> allocations;
  std::size_t device_bytes{}, host_preparation_bytes{};
  std::uint64_t class_mask{};
  unsigned worker_blocks{};
  double screening{};
  double *density{}, *coulomb{}, *temporary{}, *total_density{}, *zero{}, *schwarz{},
      *shell_bounds{};
  std::uint8_t* active{};
  std::uint32_t* heads{};
  const std::uint32_t* pair_order{};
  const std::uint32_t* pair_class_offsets{};
  GeneratedShellPairStream* topology{};
  ~GeneratedCoulombPlan();
};

/** Unsupported angular classes or insufficient optional capacity return null.
 * CUDA execution failures propagate; only allocation failure selects fallback.
 */
std::unique_ptr<GeneratedCoulombPlan> prepare_generated_coulomb(const HostBatch& host,
                                                                DeviceBatch borrowed,
                                                                cudaStream_t stream, int device,
                                                                double screening,
                                                                std::size_t budget);

/** Enqueue raw J from total spin density. Inputs and result use public AO order.
 * The same stream owns every transform, scatter and projection; no host copies.
 */
cudaError_t enqueue_generated_coulomb(GeneratedCoulombPlan& plan, const double* density,
                                      const double* beta, double* coulomb);

/** Optional raw-K owner layered on the generated-J geometry/topology owner.
 * Value-only direct CUDA plans prefer this owner when the supported shell
 * classes and optional device budget admit it. Density screening uses the same
 * shell-pair reductions as Direct HF; no range-separated operator is represented here.
 */
struct GeneratedExchangePlan {
  std::unique_ptr<GeneratedCoulombPlan> shared;
  std::vector<void*> allocations;
  std::size_t device_bytes{}, host_preparation_bytes{};
  double *public_spin{}, *direct_spin{}, *direct_exchange{};
  double *density_temporary{}, *fock_temporary{}, *public_exchange{};
  ShellPairDensityBounds* shell_pair_density_bounds{};
  double *system_density_bounds{}, *system_pair_density_bounds{};
  std::uint32_t* heads{};
  GeneratedShellPairStream* topology{};
  ~GeneratedExchangePlan();
};

/** Prepare the generated J+full-range-K owner within one explicit budget.
 * Unsupported classes or insufficient optional capacity return null.
 */
std::unique_ptr<GeneratedExchangePlan> prepare_generated_exchange(const HostBatch& host,
                                                                  DeviceBatch borrowed,
                                                                  cudaStream_t stream, int device,
                                                                  double screening,
                                                                  std::size_t budget);

/** Enqueue positive raw K in public AO order. UHF returns independent alpha/beta
 * matrices. The caller owns output buffers on the same device/stream.
 */
cudaError_t enqueue_generated_exchange(GeneratedExchangePlan& plan, bool unrestricted,
                                       const double* alpha, const double* beta,
                                       double* alpha_exchange, double* beta_exchange);

}  // namespace generativeqc::scf::cuda_execution
