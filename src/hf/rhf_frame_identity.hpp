#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hf/reference.hpp"
#include "molecule/basis_geometry_identity.hpp"
#include "posthf/capacity.hpp"

namespace generativeqc::hf {

/** Exact immutable reference/source binding for numerical response accelerators.
 * No hash collision, shape-only match, approximate orbital comparison or live
 * pointer identity is accepted. Operator-specific policies/hashes remain an
 * additional binding for accelerators that retain actual operator images.
 */
class RHFFrameIdentity {
 public:
  static std::size_t required_storage_bytes(const core::System& system,
                                            const PhysicalReference& reference) {
    std::size_t words = 0;
    visit(reference, [&](std::uint64_t) {
      words = posthf::checked_add(words, 1);
      return true;
    });
    return posthf::checked_add(molecule::BasisGeometryIdentity::required_storage_bytes(system),
                               posthf::checked_mul(words, sizeof(std::uint64_t)));
  }

  RHFFrameIdentity(const core::System& system, const PhysicalReference& reference)
      : source_(system),
        charge_(system.charge),
        electrons_(system.electron_count),
        multiplicity_(system.multiplicity) {
    const auto total = required_storage_bytes(system, reference);
    words_.reserve((total - molecule::BasisGeometryIdentity::required_storage_bytes(system)) /
                   sizeof(std::uint64_t));
    visit(reference, [&](std::uint64_t word) {
      words_.push_back(word);
      return true;
    });
  }

  [[nodiscard]] bool matches(const core::System& system, const PhysicalReference& reference) const {
    // This response owner is all-electron and closed-shell. Do not reuse an
    // identity outside that domain even if a supplied matrix happens to match.
    if (!system.ecp_terms.empty() || charge_ != system.charge ||
        electrons_ != system.electron_count || multiplicity_ != system.multiplicity ||
        !source_.matches(system))
      return false;
    std::size_t i = 0;
    return visit(reference,
                 [&](std::uint64_t word) { return i < words_.size() && words_[i++] == word; }) &&
           i == words_.size();
  }
  [[nodiscard]] std::size_t storage_bytes() const {
    return posthf::checked_add(source_.storage_bytes(),
                               posthf::checked_mul(words_.capacity(), sizeof(std::uint64_t)));
  }

 private:
  template <class Sink>
  static bool visit(const PhysicalReference& ref, Sink&& sink) {
    if (!sink(1) || !sink(ref.nbf) || !sink(ref.nocc) ||
        !sink(std::bit_cast<std::uint64_t>(ref.energy)))
      return false;
    for (const auto* values : {&ref.coefficients, &ref.fock, &ref.overlap, &ref.density, &ref.hcore,
                               &ref.orbital_energies, &ref.weighted_density}) {
      if (!sink(values->size())) return false;
      for (double value : *values)
        if (!sink(std::bit_cast<std::uint64_t>(value))) return false;
    }
    return true;
  }
  molecule::BasisGeometryIdentity source_;
  int charge_{}, electrons_{};
  unsigned multiplicity_{};
  std::vector<std::uint64_t> words_;
};

}  // namespace generativeqc::hf
