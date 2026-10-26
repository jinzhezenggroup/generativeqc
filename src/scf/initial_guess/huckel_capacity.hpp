#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"
#include "scf/initial_guess/huckel.hpp"

namespace generativeqc::scf::initial_guess {

/** Metadata for one expanded contraction, with no coefficients or coordinates. */
struct HuckelShellShape {
  std::size_t atom_index;
  unsigned angular_momentum;
  std::size_t primitives;
};

struct HuckelCapacityShape {
  std::size_t target_aos{}, atomic_aos{}, atoms{}, primitives{};
};

/** Share occupied atomic-channel counting between native preparation and dry runs.
 * Accessors borrow either the prepared System or compact ABI arrays. Scratch is
 * one counter per atom; no integral, density, atomic SCF or device is constructed.
 * Empty occupied subspaces remain valid shapes: execution retains its bounded
 * Hcore fallback when no atomic seed can be built.
 */
template <class AtomicNumber, class ShellShape>
HuckelCapacityShape huckel_capacity_shape(std::size_t atoms, std::size_t shells, int representation,
                                          AtomicNumber atomic_number, ShellShape shell_shape) {
  if (!atoms || !shells || atoms > std::numeric_limits<std::uint32_t>::max() ||
      (representation != GENERATIVEQC_BASIS_CARTESIAN &&
       representation != GENERATIVEQC_BASIS_SPHERICAL))
    throw std::invalid_argument("invalid Huckel capacity topology");
  std::vector<std::size_t> atomic_aos(atoms, 0);
  for (std::size_t atom = 0; atom < atoms; ++atom)
    (void)huckel_atomic_configuration(atomic_number(atom));
  HuckelCapacityShape shape{0, 0, atoms, 0};
  for (std::size_t index = 0; index < shells; ++index) {
    const auto shell = shell_shape(index);
    const auto aos = molecule::shell_ao_count(shell.angular_momentum, representation);
    if (shell.atom_index >= atoms || !shell.primitives || !aos)
      throw std::invalid_argument("invalid Huckel capacity shell");
    const auto occupied = huckel_atomic_configuration(atomic_number(shell.atom_index));
    if (shell.angular_momentum == 0 || (shell.angular_momentum == 1 && occupied.second))
      atomic_aos[shell.atom_index] =
          runtime::size_add(atomic_aos[shell.atom_index], 2 * shell.angular_momentum + 1);
    shape.target_aos = runtime::size_add(shape.target_aos, aos);
    shape.primitives = runtime::size_add(shape.primitives, shell.primitives);
  }
  shape.atomic_aos = *std::max_element(atomic_aos.begin(), atomic_aos.end());
  return shape;
}

/** Use identical shape rules for the actual native target, without copying it. */
inline HuckelCapacityShape huckel_capacity_shape(const core::System& system) {
  return huckel_capacity_shape(
      system.atoms.size(), system.shells.size(), system.basis_representation,
      [&](std::size_t index) { return system.atoms[index].atomic_number; },
      [&](std::size_t index) {
        const auto& shell = system.shells[index];
        return HuckelShellShape{shell.atom_index, shell.angular_momentum, shell.primitives.size()};
      });
}

/** Conservative preparation payload, shared by admission and the planner.
 * Assembly and strict admission are serialized. Sixteen FP64 target matrices
 * cover the larger fourteen-matrix admission peak, including the incoming seed,
 * nested products and validation copies. Cached atomic frames and one atomic
 * integral/DIIS owner have separate allowances. Target integrals, allocator
 * overhead and process RSS belong to other inventories.
 */
inline std::size_t huckel_numeric_capacity(const HuckelCapacityShape& shape,
                                           std::size_t diis_history) {
  if (!shape.target_aos || !shape.atoms || !shape.primitives || !diis_history || diis_history > 16)
    throw std::invalid_argument("invalid Huckel capacity shape or DIIS history");
  const auto n2 = runtime::size_mul(shape.target_aos, shape.target_aos);
  const auto atomic2 = runtime::size_mul(shape.atomic_aos, shape.atomic_aos);
  auto bytes = runtime::size_add(16U << 20, runtime::size_mul(16 * sizeof(double), n2));
  bytes = runtime::size_add(bytes, runtime::size_mul(24, runtime::size_mul(atomic2, atomic2)));
  bytes = runtime::size_add(
      bytes, runtime::size_mul(8, runtime::size_mul(atomic2, 200 + 2 * diis_history)));
  bytes = runtime::size_add(
      bytes, runtime::size_mul(512, runtime::size_add(shape.target_aos, shape.atoms)));
  bytes = runtime::size_add(bytes, runtime::size_mul(128, shape.primitives));
  if (bytes > static_cast<std::uint64_t>(INT64_MAX))
    throw std::overflow_error("Huckel capacity exceeds portable int64 scope");
  return bytes;
}

}  // namespace generativeqc::scf::initial_guess
