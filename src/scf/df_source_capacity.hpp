#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "core/types.hpp"
#include "molecule/basis.hpp"
#include "molecule/basis_geometry_identity.hpp"

namespace generativeqc::scf {
namespace df_source_capacity {
inline std::size_t add(std::size_t a, std::size_t b) {
  constexpr auto limit = static_cast<std::size_t>(INT64_MAX);
  if (a > limit || b > limit - a) throw std::overflow_error("DF source capacity overflow");
  return a + b;
}
inline std::size_t mul(std::size_t a, std::size_t b) {
  constexpr auto limit = static_cast<std::size_t>(INT64_MAX);
  if (a && b > limit / a) throw std::overflow_error("DF source capacity overflow");
  return a * b;
}

struct Shape {
  std::size_t atoms{}, shells{}, primitives{}, public_aos{}, cartesian_aos{};
  std::size_t system_bytes{}, identity_bytes{};
};
inline Shape shape(const core::System& system) {
  Shape p;
  p.atoms = system.atoms.size();
  p.shells = system.shells.size();
  p.system_bytes = add(sizeof(core::System), mul(p.atoms, sizeof(core::Atom)));
  p.system_bytes = add(p.system_bytes, mul(p.shells, sizeof(core::Shell)));
  p.system_bytes = add(p.system_bytes, mul(system.ecp_terms.size(), sizeof(core::EcpTerm)));
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > 4)
      throw std::invalid_argument("DF source capacity supports basis metadata through g");
    const auto l = static_cast<std::size_t>(shell.angular_momentum);
    const auto cart = (l + 1) * (l + 2) / 2;
    p.cartesian_aos = add(p.cartesian_aos, cart);
    p.public_aos =
        add(p.public_aos,
            system.basis_representation == GENERATIVEQC_BASIS_SPHERICAL ? 2 * l + 1 : cart);
    p.primitives = add(p.primitives, shell.primitives.size());
  }
  p.system_bytes = add(p.system_bytes, mul(p.primitives, sizeof(core::Primitive)));
  // BasisGeometryIdentity's two-pass immutable word inventory.
  p.identity_bytes = mul(add(add(4, mul(5, p.atoms)), add(mul(3, p.shells), mul(2, p.primitives))),
                         sizeof(std::uint64_t));
  return p;
}

struct ObjectSizes {
  std::size_t source{}, host_batch{}, public_expansion{};
};
struct Plan {
  std::size_t atoms{}, shells{}, primitives{}, cartesian_aos{}, public_aos{}, pairs{};
  std::size_t host_metadata_bytes{}, numeric_bytes{}, device_bytes{};
};

/** Single-source construction, before copies, packing, device selection or allocation.
 * Count logical numeric storage, not allocator/driver overhead. The source
 * reserves its incrementally packed arrays and upload-ownership list. Legacy
 * warm/pair payloads remain charged even if a later metadata-only packer omits
 * them. This is deliberately conservative, not an observed physical peak.
 */
inline Plan plan(const Shape& orbital, const Shape& auxiliary, ObjectSizes sizes) {
  if (!orbital.public_aos || !auxiliary.public_aos || orbital.atoms != auxiliary.atoms ||
      !sizes.source || !sizes.host_batch || !sizes.public_expansion)
    throw std::invalid_argument("invalid DF source capacity shape");
  Plan p;
  p.atoms = orbital.atoms;
  p.shells = add(add(orbital.shells, auxiliary.shells), 1);
  p.primitives = add(add(orbital.primitives, auxiliary.primitives), 1);
  p.cartesian_aos = add(add(orbital.cartesian_aos, auxiliary.cartesian_aos), 1);
  p.public_aos = add(orbital.public_aos, auxiliary.public_aos);
  p.pairs = mul(p.shells, add(p.shells, 1)) / 2;
  // The legacy packer also forms a checked quartet count (no quartet array).
  (void)mul(p.pairs, add(p.pairs, 1));
  const auto offsets = mul(add(p.shells, 1), 3 * sizeof(std::int64_t));
  const auto ao_bytes = add(20, mul(11, molecule::kMaximumAoExpansionTerms));
  p.host_metadata_bytes = add(2 * sizeof(std::int64_t), mul(32, p.atoms));
  p.host_metadata_bytes = add(p.host_metadata_bytes, add(mul(5, p.shells), offsets));
  p.host_metadata_bytes = add(p.host_metadata_bytes, mul(ao_bytes, p.cartesian_aos));
  p.host_metadata_bytes = add(p.host_metadata_bytes, mul(2 * sizeof(double), p.primitives));
  const auto transforms = mul(sizes.public_expansion, p.public_aos);
  const auto metric = mul(sizeof(double), mul(auxiliary.public_aos, auxiliary.public_aos));
  p.device_bytes = add(add(p.host_metadata_bytes, transforms), metric);

  // Two argument copies (initializer-list and vector) coexist with one combined
  // system. Its shell vector is reserved before either insert and the dummy.
  auto host = mul(3, add(orbital.system_bytes, auxiliary.system_bytes));
  host = add(host, sizeof(core::System) + sizeof(core::Shell) + sizeof(core::Primitive));
  host = add(host, add(sizes.source, sizes.host_batch));
  host = add(host, p.host_metadata_bytes);
  // Five two-entry non-uploaded prefix arrays can each retain their one-entry
  // predecessor during growth. Pair arrays are pre-reserved; the primitive
  // offset's initial entry is alive while its final reservation is allocated.
  host = add(host, 5 * 3 * sizeof(std::int64_t) + sizeof(std::int32_t) + sizeof(std::uint8_t));
  host = add(host, add(mul(20, p.pairs), 2 * sizeof(std::int64_t)));
  host = add(host, mul(sizeof(double), mul(p.cartesian_aos, p.cartesian_aos)));
  host = add(host, 2 * sizeof(molecule::BasisGeometryIdentity));
  host = add(host, add(orbital.identity_bytes, auxiliary.identity_bytes));
  host = add(host, 2 * sizeof(std::int64_t) + 20 * sizeof(void*));
  // Packed public transforms and the two per-item outputs coexist.
  host = add(host, mul(2, transforms));
  host = add(host, metric);
  // Fixed maximum-shell expansion scratch for l<=4: two Cartesian component
  // lists, the g polynomial, nested expansion vectors and their old/new storage.
  // libstdc++/libc++ growth is covered by three simultaneous nominal payloads;
  // allocation-event tests cover this small bounded helper separately.
  constexpr std::size_t cart = 15;
  const auto expansion_scratch =
      add(mul(2 * cart, sizeof(molecule::CartesianComponent)),
          add(mul(cart, sizeof(double)),
              mul(3 * cart, add(sizeof(molecule::AoExpansion),
                                mul(cart, sizeof(molecule::CartesianExpansionTerm))))));
  host = add(host, expansion_scratch);
  p.numeric_bytes = add(host, p.device_bytes);
  return p;
}

inline Plan plan(const core::System& orbital, const core::System& auxiliary, ObjectSizes sizes) {
  return plan(shape(orbital), shape(auxiliary), sizes);
}

/** Reserve the upload inventory once. HostBatch is kept a template here so
 * this planner and its overflow checks remain usable without CUDA headers.
 */
template <class HostBatch>
void reserve_upload_metadata(HostBatch& h, const Plan& p) {
  h.atom_offsets.reserve(2);
  h.atom_systems.reserve(p.atoms);
  h.atomic_numbers.reserve(p.atoms);
  h.positions.reserve(mul(3, p.atoms));
  h.shell_atoms.reserve(p.shells);
  h.shell_angular.reserve(p.shells);
  h.shell_ao_offsets.reserve(add(p.shells, 1));
  h.shell_direct_ao_offsets.reserve(add(p.shells, 1));
  h.shell_primitive_offsets.reserve(add(p.shells, 1));
  h.ao_shells.reserve(p.cartesian_aos);
  h.ao_term_counts.reserve(p.cartesian_aos);
  h.ao_term_angular.reserve(mul(3 * molecule::kMaximumAoExpansionTerms, p.cartesian_aos));
  h.ao_term_coefficients.reserve(mul(molecule::kMaximumAoExpansionTerms, p.cartesian_aos));
  h.direct_ao_shells.reserve(p.cartesian_aos);
  h.direct_ao_angular.reserve(mul(3, p.cartesian_aos));
  h.direct_ao_coefficients.reserve(p.cartesian_aos);
  h.primitive_exponents.reserve(p.primitives);
  h.primitive_coefficients.reserve(p.primitives);
}
}  // namespace df_source_capacity
}  // namespace generativeqc::scf
