#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "hf/rhf_frame_identity.hpp"
#include "response/native_gmres.hpp"

namespace generativeqc::hf {

/** Caller-owned rank-one solved-direction subspace for repeated exact response.
 *
 * Retain x and its independently audited exact A*x, normalized by ||A*x||.
 * A new RHS is projected onto that image; GMRES still computes a fresh physical
 * residual of the proposed initial guess. Reuse requires bitwise equality of
 * the complete reference/source and the operator/device/provider policy. In
 * particular, a nearby geometry or approximately equal orbitals never match.
 * This mutable workspace is deliberately non-reentrant, has no global cache,
 * and does not serialize images across builds/devices.
 */
class RHFFrameResponseRecycle {
 public:
  void clear() noexcept { entry_.reset(); }
  /** Admit all retained/projection storage before the physical solve. Rejected
   * stale/short-budget state is really released. Allocation refusal disables
   * this accelerator and never changes physical-solver admission.
   */
  bool prepare(const core::System& system, const PhysicalReference& ref, int device,
               bool matrix_blas, const char* operator_hash, std::size_t maximum_bytes) {
    if (!ref.nocc || ref.nbf <= ref.nocc || !system.ecp_terms.empty()) {
      clear();
      return false;
    }
    if (entry_ && !matches(system, ref, device, matrix_blas, operator_hash)) entry_.reset();
    try {
      const auto dimension = posthf::checked_mul(ref.nocc, ref.nbf - ref.nocc);
      const auto bound = posthf::checked_add(RHFFrameIdentity::required_storage_bytes(system, ref),
                                             posthf::checked_mul(3 * sizeof(double), dimension));
      if (!dimension || bound > maximum_bytes) {
        entry_.reset();
        return false;
      }
      if (!entry_)
        entry_ =
            std::make_unique<Entry>(system, ref, device, matrix_blas, operator_hash, dimension);
      if (storage_bytes() > maximum_bytes) {
        entry_.reset();
        return false;
      }
      return true;
    } catch (const std::bad_alloc&) {
      entry_.reset();
      return false;
    } catch (const std::overflow_error&) {
      entry_.reset();
      return false;
    }
  }

  [[nodiscard]] bool matches(const core::System& system, const PhysicalReference& ref, int device,
                             bool matrix_blas, const char* operator_hash) const {
    return entry_ && entry_->device == device && entry_->matrix_blas == matrix_blas &&
           entry_->operator_hash == operator_hash && entry_->identity.matches(system, ref);
  }
  [[nodiscard]] bool populated() const { return entry_ && entry_->populated; }
  [[nodiscard]] std::size_t storage_bytes() const {
    if (!entry_) return 0;
    auto bytes = entry_->identity.storage_bytes();
    for (const auto* array : {&entry_->direction, &entry_->image, &entry_->guess})
      bytes = posthf::checked_add(bytes, posthf::checked_mul(array->capacity(), sizeof(double)));
    return bytes;
  }

  /** Empty means no safe proposal. These operations contain no physical work;
   * their storage and wall time are included by the response owner.
   */
  std::span<const double> initial_guess(std::span<const double> rhs) {
    if (!populated() || rhs.size() != entry_->image.size()) return {};
    long double coefficient = 0, norm2 = 0;
    for (std::size_t i = 0; i < rhs.size(); ++i) {
      coefficient += static_cast<long double>(entry_->image[i]) * rhs[i];
      norm2 += static_cast<long double>(entry_->image[i]) * entry_->image[i];
    }
    if (!(norm2 > 0)) return {};
    coefficient /= norm2;
    for (std::size_t i = 0; i < rhs.size(); ++i) {
      entry_->guess[i] = static_cast<double>(coefficient * entry_->direction[i]);
      if (!std::isfinite(entry_->guess[i])) return {};
    }
    return entry_->guess;
  }

  /** Publish only after the response owner's independent residual/stationarity
   * and complete derivative gates. Failed endpoints never publish a new image.
   */
  bool capture(std::span<const double> solution, std::span<const double> exact_image) {
    if (!entry_ || solution.size() != entry_->direction.size() ||
        exact_image.size() != solution.size())
      return false;
    for (double value : exact_image)
      if (!std::isfinite(value)) return false;
    double norm = 0;
    try {
      norm = response::stable_norm(exact_image);
    } catch (const std::overflow_error&) {
      return false;
    }
    if (!(norm > 0) || !std::isfinite(norm)) return false;
    entry_->populated = false;
    for (std::size_t i = 0; i < solution.size(); ++i) {
      entry_->direction[i] = solution[i] / norm;
      entry_->image[i] = exact_image[i] / norm;
      if (!std::isfinite(entry_->direction[i]) || !std::isfinite(entry_->image[i])) return false;
    }
    entry_->populated = true;
    return true;
  }

 private:
  struct Entry {
    Entry(const core::System& system, const PhysicalReference& ref, int d, bool blas,
          const char* hash, std::size_t n)
        : identity(system, ref),
          device(d),
          matrix_blas(blas),
          operator_hash(hash),
          direction(n),
          image(n),
          guess(n) {}
    RHFFrameIdentity identity;
    int device{};
    bool matrix_blas{}, populated{};
    std::string operator_hash;
    std::vector<double> direction, image, guess;
  };
  std::unique_ptr<Entry> entry_;
};

}  // namespace generativeqc::hf
