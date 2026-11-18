#pragma once

#include <array>
#include <cstddef>

namespace generativeqc::integrals {

/** Real scalar Coulomb ERIs have pair exchange and within-pair symmetry.
 * These are index schedules, not recurrence or scientific-weight owners.
 * Repeated indices reduce the orbit; consume each distinct ordered tuple once.
 */
inline std::array<std::array<std::size_t, 4>, 8> eri_permutations(
    const std::array<std::size_t, 4>& indices) {
  const auto [first, second, third, fourth] = indices;
  return {{{first, second, third, fourth},
           {second, first, third, fourth},
           {first, second, fourth, third},
           {second, first, fourth, third},
           {third, fourth, first, second},
           {fourth, third, first, second},
           {third, fourth, second, first},
           {fourth, third, second, first}}};
}

template <class Consumer>
void for_each_eri_permutation(const std::array<std::size_t, 4>& indices, Consumer&& consume) {
  const auto permutations = eri_permutations(indices);
  for (std::size_t current = 0; current < permutations.size(); ++current) {
    bool repeated = false;
    for (std::size_t previous = 0; previous < current; ++previous)
      repeated = repeated || permutations[previous] == permutations[current];
    if (!repeated) consume(permutations[current]);
  }
}

/** One representative of every real-ERI orbit, in triangular pair order. */
template <class Consumer>
void for_each_unique_eri_quartet(std::size_t count, Consumer&& consume) {
  for (std::size_t first = 0; first < count; ++first)
    for (std::size_t second = 0; second <= first; ++second)
      for (std::size_t third = 0; third <= first; ++third)
        for (std::size_t fourth = 0; fourth <= third; ++fourth) {
          if (first == third && second < fourth) continue;
          consume(std::array<std::size_t, 4>{first, second, third, fourth});
        }
}

}  // namespace generativeqc::integrals
