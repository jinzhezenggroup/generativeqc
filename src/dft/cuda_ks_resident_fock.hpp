#pragma once

#include <cstddef>
#include <cstdint>

namespace generativeqc::scf {
class PreparedFockPlan;
struct ResolvedFockBuild;
}  // namespace generativeqc::scf

namespace generativeqc::dft {

/** Live prepared sources and accepted F/H from the scalar-validated KS owner.
 * This grants read access to immutable source/model metadata, not permission to
 * overwrite provider scratch. A response consumer must use separate storage,
 * retain the batch, exclude concurrent batch access and reborrow this binding
 * before execution/publication. The declaration is usable without CUDA headers.
 */
struct CudaKsResidentFockBinding {
  int device_id{-1};
  const scf::PreparedFockPlan* primary{};
  const scf::ResolvedFockBuild* correction{};
  const double *hcore{}, *fock{};
  std::size_t nao{};
  unsigned spins{};
  void* stream{};
  std::uint64_t owner{}, solve_epoch{}, generation{};

  bool operator==(const CudaKsResidentFockBinding&) const = default;

  explicit operator bool() const noexcept {
    return device_id >= 0 && primary && hcore && fock && nao && (spins == 1 || spins == 2) &&
           stream && owner && solve_epoch && generation;
  }
};

}  // namespace generativeqc::dft
