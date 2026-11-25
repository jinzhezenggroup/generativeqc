#pragma once

#include <algorithm>
#include <cstddef>

#include "runtime/bounded_workspace.hpp"

namespace generativeqc::api {
struct HostOrbitalCapacity {
  std::size_t output_elements;
  std::size_t numeric_bytes;
};

/** Detached verified frame, staged coefficients and caller output only.
 * The physical owner's verification/eigensolver scratch is not part of this bound.
 */
inline HostOrbitalCapacity host_orbital_capacity(std::size_t basis, std::size_t spins) {
  using runtime::size_add;
  using runtime::size_mul;
  const auto elements = size_mul(spins, size_mul(basis, basis));
  return {elements,
          size_mul(size_add(size_mul(5, elements), size_mul(spins, basis)), sizeof(double))};
}

/** Model the native workspace for a CPU physical-state read with no weighted
 * density, plus the subsequent transactional C export. Retained F/D/S/Hcore
 * already belong to the prepared batch; the caller's C is separately charged
 * by host_orbital_capacity. These are numeric payloads, not allocator overhead.
 */
inline std::size_t host_orbital_workspace_bytes(std::size_t basis, std::size_t spins) {
  using runtime::size_add;
  using runtime::size_mul;
  // read_final_state keeps X and the previous spin eigenvectors during Jacobi.
  const auto diagonalization = size_add(spins, 4);
  // validate_ks_final_state keeps the candidate and copies C and F; the
  // determinant validator is spin-serial and has at most seven live products.
  const auto validation = size_add(8, size_mul(3, spins));
  // Copying the verified D/F/C overlaps the candidate, validation C/F and X.
  const auto verification_copy = size_add(1, size_mul(6, spins));
  // After the read, only verified D/F/C and the publication staging C remain.
  const auto staging = size_mul(4, spins);
  const auto peak_matrices = std::max({diagonalization, validation, verification_copy, staging});
  const auto matrices = size_mul(peak_matrices, size_mul(basis, basis));
  // At most three sets of eigenvalues coexist. Also allow one overlap spectrum
  // and the Jacobi sorting permutation, irrespective of their shorter lifetime.
  const auto spectra = size_mul(size_add(size_mul(3, spins), 1), basis);
  return size_add(size_mul(size_add(matrices, spectra), sizeof(double)),
                  size_mul(basis, sizeof(std::size_t)));
}
}  // namespace generativeqc::api
