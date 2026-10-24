#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/types.hpp"
#include "methods/method.hpp"

namespace generativeqc::methods::geometry {

/** Pack host geometry metadata without evaluating any scientific operator. */
inline std::vector<double> coordinates(std::span<const core::Atom> atoms) {
  std::vector<double> result;
  result.reserve(3 * atoms.size());
  for (const auto& atom : atoms)
    result.insert(result.end(), atom.position.begin(), atom.position.end());
  return result;
}

inline std::vector<double> coordinates(const core::System& system) {
  return coordinates(system.atoms);
}

inline bool valid_coordinates(const std::vector<double>& values, const core::System& system) {
  return values.size() == 3 * system.atoms.size() &&
         std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

/** Caller validates the complete coordinate list before mutating an owner. */
inline void set_coordinates(core::System& system, const std::vector<double>& values) {
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
    std::copy_n(values.begin() + 3 * atom, 3, system.atoms[atom].position.begin());
}

/** Bind one ragged item before backend-specific source/owner preparation.
 * Failed geometry retains its requested placement and has no publishable energy. */
inline core::System bind_batch_item(const core::System& source, const Coordinates& updates,
                                    std::size_t index, BatchItemResult& result,
                                    generativeqc_backend backend, std::size_t bucket) {
  result.bucket_id = bucket;
  result.calculation.executed_backend = backend;
  result.calculation.energy = std::numeric_limits<double>::quiet_NaN();
  auto target = source;
  if (!updates.empty() && updates.at(index)) {
    if (!valid_coordinates(*updates[index], target))
      throw std::invalid_argument("invalid batch item coordinates");
    set_coordinates(target, *updates[index]);
  }
  return target;
}

}  // namespace generativeqc::methods::geometry
