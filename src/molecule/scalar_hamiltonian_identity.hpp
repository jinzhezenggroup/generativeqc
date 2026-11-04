#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "molecule/basis_geometry_identity.hpp"

namespace generativeqc::molecule {

/** Immutable scalar molecular source, including the complete ECP operator.
 * Coulomb-only caches retain BasisGeometryIdentity. Physical reference/response
 * owners must also bind ionic populations and every potential parameter, even
 * when supplied orbitals or matrices happen to be bitwise unchanged. Method,
 * fitting and backend policies remain additional consumer-specific bindings. */
class ScalarHamiltonianIdentity {
 public:
  explicit ScalarHamiltonianIdentity(const core::System& system) : geometry_(system) {
    const auto geometry_bytes = BasisGeometryIdentity::required_storage_bytes(system);
    words_.reserve((required_storage_bytes(system) - geometry_bytes) / sizeof(std::uint64_t));
    visit(system, [&](std::uint64_t word) {
      words_.push_back(word);
      return true;
    });
  }

  /** Bound the complete logical payload without allocating a snapshot. */
  static std::size_t required_storage_bytes(const core::System& system) {
    const auto geometry_bytes = BasisGeometryIdentity::required_storage_bytes(system);
    const auto maximum = std::min(
        std::vector<std::uint64_t>{}.max_size(),
        (std::numeric_limits<std::size_t>::max() - geometry_bytes) / sizeof(std::uint64_t));
    std::size_t words = 0;
    visit(system, [&](std::uint64_t) {
      if (words == maximum) throw std::length_error("scalar Hamiltonian identity is too large");
      ++words;
      return true;
    });
    return geometry_bytes + words * sizeof(std::uint64_t);
  }

  [[nodiscard]] bool matches(const core::System& system) const noexcept {
    if (!geometry_.matches(system)) return false;
    std::size_t index = 0;
    return visit(system,
                 [&](std::uint64_t word) {
                   return index < words_.size() && words_[index++] == word;
                 }) &&
           index == words_.size();
  }

  [[nodiscard]] std::size_t storage_bytes() const noexcept {
    return geometry_.storage_bytes() + words_.capacity() * sizeof(std::uint64_t);
  }

 private:
  template <class Sink>
  static bool visit(const core::System& system, Sink&& sink) {
    if (!sink(1) || !sink(static_cast<std::uint64_t>(system.charge)) ||
        !sink(static_cast<std::uint64_t>(system.electron_count)) || !sink(system.multiplicity) ||
        !sink(system.ecp_terms.size()))
      return false;
    for (const auto& term : system.ecp_terms)
      if (!sink(term.atom_index) || !sink(static_cast<std::uint64_t>(term.channel)) ||
          !sink(term.power) || !sink(std::bit_cast<std::uint64_t>(term.exponent)) ||
          !sink(std::bit_cast<std::uint64_t>(term.coefficient)))
        return false;
    return true;
  }

  BasisGeometryIdentity geometry_;
  std::vector<std::uint64_t> words_;
};

}  // namespace generativeqc::molecule
