#pragma once

#include "scf/cuda/matrix_library.hpp"

namespace generativeqc::scf::cuda_execution {

/** Occupied C diag(f) C^T, or signed C diag(f eps) C^T when energies are present.
 * Scaled and masked-output spans must already be charged and stream-live; they
 * may not alias inputs/output. Inactive systems publish no bytes. */
generativeqc_status launch_structured_density(
    MatrixLibraryResources resources, int systems, int spins, int dimension,
    const std::int32_t* occupied, const double* coefficients, const double* energies,
    const std::uint8_t* active, double* output, double* scaled, std::size_t scaled_elements,
    double* unweighted = nullptr, std::size_t unweighted_elements = 0);

/** Preserve per-system eigenvalue-domain rejection without host synchronization.
 * Invalid/inactive systems retain output bytes and invalid systems set failed. */
generativeqc_status launch_structured_orthogonalizer(MatrixLibraryResources resources, int systems,
                                                     int dimension, const double* vectors,
                                                     const double* eigenvalues,
                                                     const std::uint8_t* active, double* output,
                                                     std::uint8_t* failed, double* scaled,
                                                     std::size_t scaled_elements);

/** One public/direct basis stage. Sparse shell spans retain the original bounded
 * executor; dense stages use their frozen rectangular binding. The final Fock
 * stage adds hcore during masked publication, without another matrix staging. */
generativeqc_status launch_structured_basis(
    MatrixLibraryResources resources, tensor::StructuredOperation operation, int systems, int spins,
    int dimension, int direct_dimension, const double* transform, const double* input,
    const double* hcore, const std::uint8_t* active, double* output, double* scratch,
    std::size_t scratch_elements, const std::int32_t* shell_spans = nullptr);
}  // namespace generativeqc::scf::cuda_execution
