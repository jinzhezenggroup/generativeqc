#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "dft/cuda_ks_final_state.hpp"
#include "dft/cuda_wb97mv_grid.hpp"
#include "response/cuda_cpks.hpp"

namespace generativeqc::methods {
class PreparedBatch;
}

namespace generativeqc::dft {

struct CudaKsWb97mvResponseOptions {
  CudaWb97mvGridOptions grid{};
  std::size_t max_device_bytes{512ULL << 20};
  std::size_t max_affine_term_visits{100'000'000};
  std::uint64_t max_direct_eri_evaluations{16'777'216};
  double reference_absolute_tolerance{3e-9};
  double reference_relative_tolerance{3e-9};
};

struct CudaKsWb97mvResponseWork {
  std::size_t device_bytes{}, affine_reference_terms{}, affine_action_terms{};
  std::uint64_t reference_quartets{}, reference_eris{}, response_quartets{}, response_eris{};
  std::uint64_t actions{};
  CudaWb97mvGridWork grid{};
};

/** Authenticated fixed-geometry physical induced Fock for native WB97M-V KS.
 * Preparation verifies the complete model and reconstructs the actual accepted
 * F from H + exact/fitted full-range J/K + exact LR correction + XC/VV10. DF
 * consumes the existing compiler energy VJP and immutable raw/metric sources,
 * not occupied factors. Signed/indefinite coupled spin directions are legal.
 * Exact actions use a separate unscreened geometry-only Direct owner on the
 * producer stream; neither SCF scratch nor occupied projection leases change.
 *
 * Preparation may allocate; actions never allocate, compile, switch devices or
 * execute Python/CPU numerical work. Resource and semantic-work bounds are
 * explicit. The batch must outlive this nonmovable owner, with exclusive access.
 * Each action reauthenticates before execution/publication; failures preserve
 * output before publication. This is an induced-Fock component, not a nuclear
 * Hessian: moving-grid, integral and ECP source derivatives are not supplied.
 */
class CudaKsWb97mvResponse {
 public:
  CudaKsWb97mvResponse();
  ~CudaKsWb97mvResponse();
  CudaKsWb97mvResponse(const CudaKsWb97mvResponse&) = delete;
  CudaKsWb97mvResponse& operator=(const CudaKsWb97mvResponse&) = delete;
  CudaKsWb97mvResponse(CudaKsWb97mvResponse&&) = delete;
  CudaKsWb97mvResponse& operator=(CudaKsWb97mvResponse&&) = delete;

  generativeqc_status prepare(methods::PreparedBatch& batch, std::size_t item,
                              const CudaKsFinalStateToken& expected,
                              const CudaKsWb97mvResponseOptions& options, std::string& detail);
  cudaError_t apply(const double* direction, double* output, cudaStream_t stream) noexcept;
  CudaKsWb97mvResponseWork work() const noexcept;
  /** Directly compatible with solve_cuda_cpks, passing this owner as context. */
  static cudaError_t induced_fock(void* context, const double* direction, double* output,
                                  const response::CudaCpksShape& shape,
                                  cudaStream_t stream) noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::dft
