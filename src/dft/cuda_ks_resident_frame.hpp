#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace generativeqc::dft {

/** Complete canonical frame borrowed from a scalar-validated live KS owner.
 * C is spin-major, column-major [AO,orbital]; energies are [spin,orbital].
 * D/F are spin-major row-major matrices and S is one shared row-major matrix.
 * No detached host snapshot is created. Pointers remain valid only under the
 * exact token, on the producer device/stream, without a concurrent owner call.
 * Response consumers must reacquire this binding before execution/publication;
 * copying owner/epoch/generation labels does not authenticate a future use.
 * This host-only declaration does not load a CUDA runtime or grant capability.
 */
struct CudaKsResidentOrbitalFrameBinding {
  int device_id{-1};
  const double* coefficients{};
  const double* orbital_energies{};
  const double* density{};
  const double* fock{};
  const double* overlap{};
  std::size_t nao{};
  unsigned spins{};
  std::array<std::size_t, 2> occupied{};
  void* stream{};
  std::uint64_t owner{}, solve_epoch{}, generation{};

  bool operator==(const CudaKsResidentOrbitalFrameBinding&) const = default;

  explicit operator bool() const noexcept {
    return device_id >= 0 && coefficients && orbital_energies && density && fock && overlap &&
           nao && (spins == 1 || spins == 2) && occupied[0] <= nao && occupied[1] <= nao &&
           (spins == 2 || occupied[1] == 0) && stream && owner && solve_epoch && generation;
  }
};

}  // namespace generativeqc::dft
