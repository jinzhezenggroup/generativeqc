#pragma once

#include <span>

#include "integrals/cuda_df_nuclear_response.hpp"

namespace generativeqc::dft {

/** Private dense tables, in Cartesian-column order. The consumer owns their
 * storage and does not publish them until the transaction audit passes. */
struct CudaDfFrozenTables {
  double *hessian{}, *explicit_fock{}, *auxiliary_rhs{};
};

/** Sweep the full-range Cartesian s/p source once, retaining only final
 * contracted derivatives. NotSupported and pre-launch OOM permit raw replay;
 * any error after submission fails the transaction. The caller authenticates
 * the basis, accepted density and fitted charge on this stream. */
cudaError_t cuda_wb97mv_df_frozen_sweep(const scf::CudaDfReadonlyBasis& basis,
                                        std::span<const integrals::CudaDfNuclearShell> orbital,
                                        std::span<const integrals::CudaDfNuclearShell> auxiliary,
                                        const double* density, const double* charge,
                                        CudaDfFrozenTables output, std::size_t max_scratch_bytes,
                                        int* status, cudaStream_t stream) noexcept;

}  // namespace generativeqc::dft
