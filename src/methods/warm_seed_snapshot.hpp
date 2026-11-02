#pragma once

#include <new>
#include <stdexcept>

#include "methods/warm_seed_capacity.hpp"

namespace generativeqc::methods {

/** Detached scientific seeds, with no prepared owner, context or device lifetime. */
struct WarmSeedSnapshot {
  std::vector<std::optional<scf::HfWarmState>> states;
  std::size_t numeric_bytes{}, largest_density_bytes{};
};

inline void validate_warm_seed_shape(const BatchWarmStarts& owner, std::size_t index,
                                     std::uint32_t atoms, const scf::HfWarmState& state) {
  if (state.density.size() != owner.warm_density_size(index) ||
      state.coordinates.size() != runtime::size_mul(atoms, 3))
    throw std::invalid_argument("warm seed dimensions do not match the prepared owner");
}

/** Bound all slots before export, including cold slots whose presence can change.
 * The underlying export may materialize resident CUDA seeds, but no scientific
 * equations or policy are reimplemented here. Failed copies publish no snapshot. */
inline WarmSeedSnapshot capture_warm_seeds(const BatchWarmStarts& owner,
                                           std::span<const std::uint32_t> atom_counts,
                                           std::uint64_t maximum_numeric_bytes) {
  const auto bound = warm_seed_capacity(owner, atom_counts);
  if (bound.total_bytes > maximum_numeric_bytes) throw std::bad_alloc();
  WarmSeedSnapshot snapshot;
  snapshot.states.resize(atom_counts.size());
  for (std::size_t i = 0; i < atom_counts.size(); ++i) {
    const auto& source = owner.warm_state(i);
    if (!source) continue;
    validate_warm_seed_shape(owner, i, atom_counts[i], *source);
    snapshot.states[i] = source;
    const auto density = runtime::size_mul(source->density.size(), sizeof(double));
    const auto coordinates = runtime::size_mul(source->coordinates.size(), sizeof(double));
    snapshot.numeric_bytes =
        runtime::size_add(snapshot.numeric_bytes, runtime::size_add(density, coordinates));
    snapshot.largest_density_bytes = std::max(snapshot.largest_density_bytes, density);
  }
  return snapshot;
}

/** Preserve the existing all-or-nothing physical restore and reusable snapshot.
 * The frontend retains same-model identity; dimensions alone never certify it. */
inline void restore_warm_seeds(BatchWarmStarts& owner, std::span<const std::uint32_t> atom_counts,
                               const WarmSeedSnapshot& snapshot) {
  if (snapshot.states.size() != atom_counts.size())
    throw std::invalid_argument("warm seed snapshot item count changed");
  for (std::size_t i = 0; i < atom_counts.size(); ++i)
    if (snapshot.states[i]) validate_warm_seed_shape(owner, i, atom_counts[i], *snapshot.states[i]);
  owner.restore_warm_states(snapshot.states);
}

}  // namespace generativeqc::methods
