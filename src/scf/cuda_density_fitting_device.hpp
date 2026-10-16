#pragma once

#include <cuda_runtime_api.h>

#include "scf/cuda_density_fitting.hpp"

namespace generativeqc::scf {
/** Borrow the DF owner's ordinary stream for device density/XC consumers.
 * The plan owns its lifetime and must outlive all enqueued work. This preserves
 * ordering without staging densities or matrices through the host. */
cudaStream_t cuda_density_fitting_stream(const CudaDensityFittingJkPlan* plan);
/** Device ordinal in Slurm/process visibility; null returns -1. */
int cuda_density_fitting_device(const CudaDensityFittingJkPlan* plan) noexcept;

/** Immutable full-rank value sources, never the owner's mutable J/K scratch.
 * Whitening is row-major [out_aux,source_aux]; the physical raw-metric inverse
 * is W^T W. The plan and its stream must survive all borrowed work. Source
 * identity is a lifetime capability, not a stationary KS state proof. */
struct CudaDfReadonlySource {
  int device_id{-1};
  cudaStream_t stream{};
  CudaDensityFittingIntegralSource* integrals{};
  const double* whitening{};
  std::size_t nao{}, naux{};
  std::uint64_t identity{};
  bool operator==(const CudaDfReadonlySource&) const = default;
  explicit operator bool() const noexcept {
    return device_id >= 0 && stream && integrals && whitening && nao && naux && identity;
  }
};

/** Borrow one complete single-system full-rank source; unsupported/streamed
 * host-only whitening, missing raw sources and a different current device fail
 * closed. No allocation, transfer, scratch writer or projection revocation. */
CudaDfReadonlySource cuda_density_fitting_readonly_source(
    const CudaDensityFittingJkPlan* plan) noexcept;

/** Additional immutable metric-domain evidence for a full-rank second response.
 * Eigenvalues remain on device in ascending order. This lease has the same
 * lifetime identity as the raw/whitening source; no mutable J/K scratch is lent.
 * The consumer still enforces its stronger cutoff-separation contract. */
struct CudaDfReadonlyMetric {
  const double* eigenvalues{};
  double relative_threshold{};
  std::size_t naux{};
  std::uint64_t identity{};
  bool operator==(const CudaDfReadonlyMetric&) const = default;
  explicit operator bool() const noexcept { return eigenvalues && naux && identity; }
};

/** Missing, truncated, cutoff-unstable or wrong-current-device owners return an
 * empty lease. This getter performs no numerical transfer or scratch write. */
CudaDfReadonlyMetric cuda_density_fitting_readonly_metric(
    const CudaDensityFittingJkPlan* plan) noexcept;

}  // namespace generativeqc::scf
