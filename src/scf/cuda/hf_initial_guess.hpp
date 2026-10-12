#pragma once

#include <vector>

#include "scf/types.hpp"

namespace generativeqc::scf {
struct CudaRhfBucketPlan;
namespace cuda_execution {
struct HostBatch;
}

/** Borrowed idle target buffers. No new device arena or cuSOLVER owner is
 * allocated; the call completes before ordinary SCF/graph replay begins. */
struct CudaHfSeedWorkspace {
  double* overlap{};
  double* density{};
  double* matrix{};
  double* scratch{};
  double* eigenvalues{};
  int* solver_info{};
  const std::uint8_t* active{};
};

/** Build only cold Hückel seeds against the GPU target's actual overlap.
 * Warm inputs are authoritative. Failed preparation leaves Hcore untouched;
 * allocation failures propagate. Diagnostics include serialized atomic work. */
void prepare_hf_cuda_initial_densities(
    CudaRhfBucketPlan& plan, const cuda_execution::HostBatch& host,
    const std::vector<core::System>& systems, const ScfOptions& options,
    const CudaHfSeedWorkspace& workspace,
    std::vector<initial_guess::PreliminaryDiagnostic>& diagnostics);
}  // namespace generativeqc::scf
