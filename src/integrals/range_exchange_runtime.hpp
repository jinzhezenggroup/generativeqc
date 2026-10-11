#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace generativeqc::integrals {

/** Bounded public-AO consumer of the shared CPU/CUDA range recurrence.
 * Packed coefficients already include contraction and Cartesian/public-basis
 * normalization. Independent slots preserve repeated-center derivatives for
 * primitive qualification; endpoint mode seeds each physical atom together.
 * Translation recovers the final atom/slot, never a displaced-energy oracle.
 * Numeric publication is transactional and no AO-rank-four tensor is retained.
 */
template <class Program>
int range_exchange_records(const double* packed, std::size_t natom, std::size_t nprimitive,
                           std::size_t nao, const std::int64_t* indices, const double* weights,
                           std::size_t count, bool independent_slots, double* output,
                           std::uint64_t* visits) {
  using namespace generativeqc::scf::cuda_execution;
  if (!packed || !indices || !weights || !output || !visits || !natom || natom > 8 || !nao ||
      nao > 64 || !nprimitive || nprimitive > 128 || !count || count > 4096 ||
      (independent_slots && count != 1))
    return 1;
  const double* primitives = packed + 3 * natom;
  const double* aos = primitives + 2 * nprimitive;
  std::array<double, 24> candidate{};
  std::uint64_t records = 0;
  std::uint64_t evaluations = 0;
  for (std::size_t quartet = 0; quartet < count; ++quartet) {
    if (!std::isfinite(weights[quartet])) return 1;
    const double* rows[4];
    double centers[12];
    unsigned owners[4];
    for (unsigned slot = 0; slot < 4; ++slot) {
      const auto index = indices[4 * quartet + slot];
      if (index < 0 || static_cast<std::size_t>(index) >= nao) return 1;
      rows[slot] = aos + 16 * index;
      const auto* row = rows[slot];
      if (row[0] < 0 || row[0] >= natom || row[1] < 0 || row[2] < 1 ||
          row[1] + row[2] > nprimitive || row[3] < 1 || row[3] > 3)
        return 1;
      for (unsigned field = 0; field < 4; ++field)
        if (!std::isfinite(row[field]) || std::floor(row[field]) != row[field]) return 1;
      owners[slot] = static_cast<unsigned>(row[0]);
      for (unsigned axis = 0; axis < 3; ++axis) {
        centers[3 * slot + axis] = packed[3 * owners[slot] + axis];
        if (!std::isfinite(centers[3 * slot + axis])) return 1;
      }
    }
    unsigned masks[4]{}, targets[4]{}, seeds = 0;
    if (independent_slots) {
      for (unsigned slot = 0; slot < 3; ++slot) {
        masks[seeds] = 1U << slot;
        targets[seeds++] = slot;
      }
    } else {
      for (unsigned slot = 0; slot < 4; ++slot) {
        if (owners[slot] == owners[3]) continue;
        unsigned seed = 0;
        while (seed < seeds && targets[seed] != owners[slot]) ++seed;
        if (seed == seeds) targets[seeds++] = owners[slot];
        masks[seed] |= 1U << slot;
      }
    }
    const unsigned final_target = independent_slots ? 3 : owners[3];
    for (unsigned first = 0; first < rows[0][3]; ++first)
      for (unsigned second = 0; second < rows[1][3]; ++second)
        for (unsigned third = 0; third < rows[2][3]; ++third)
          for (unsigned fourth = 0; fourth < rows[3][3]; ++fourth) {
            const unsigned terms[4]{first, second, third, fourth};
            Angular angular[4];
            double component_weight = weights[quartet];
            for (unsigned slot = 0; slot < 4; ++slot) {
              const auto* term = rows[slot] + 4 + 4 * terms[slot];
              unsigned powers[3];
              for (unsigned axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(term[axis]) || term[axis] < 0 || term[axis] > 3 ||
                    std::floor(term[axis]) != term[axis])
                  return 1;
                powers[axis] = static_cast<unsigned>(term[axis]);
              }
              angular[slot] = {powers[0], powers[1], powers[2]};
              if (angular_total(angular[slot]) > 3 || !std::isfinite(term[3])) return 1;
              component_weight *= term[3];
            }
            for (unsigned first_primitive = 0; first_primitive < rows[0][2]; ++first_primitive)
              for (unsigned second_primitive = 0; second_primitive < rows[1][2]; ++second_primitive)
                for (unsigned third_primitive = 0; third_primitive < rows[2][2]; ++third_primitive)
                  for (unsigned fourth_primitive = 0; fourth_primitive < rows[3][2];
                       ++fourth_primitive) {
                    const unsigned local[4]{first_primitive, second_primitive, third_primitive,
                                            fourth_primitive};
                    double exponents[4], weight = component_weight;
                    for (unsigned slot = 0; slot < 4; ++slot) {
                      const auto* primitive =
                          primitives + 2 * (static_cast<unsigned>(rows[slot][1]) + local[slot]);
                      exponents[slot] = primitive[0];
                      if (!std::isfinite(exponents[slot]) || exponents[slot] <= 0 ||
                          !std::isfinite(primitive[1]))
                        return 1;
                      weight *= primitive[1];
                    }
                    ++records;
                    if (!std::isfinite(weight)) return 2;
                    for (unsigned seed = 0; seed < seeds; ++seed) {
                      ++evaluations;
                      const auto result =
                          Program::evaluate(exponents, centers, angular, masks[seed]);
                      const double derivative[3]{result.derivative_x, result.derivative_y,
                                                 result.derivative_z};
                      if (!std::isfinite(result.value)) return 2;
                      for (unsigned axis = 0; axis < 3; ++axis) {
                        const double value = weight * derivative[axis];
                        if (!std::isfinite(value)) return 2;
                        candidate[3 * targets[seed] + axis] += value;
                        candidate[3 * final_target + axis] -= value;
                      }
                    }
                  }
          }
  }
  const auto values = independent_slots ? 12 : 3 * natom;
  for (std::size_t index = 0; index < values; ++index)
    if (!std::isfinite(candidate[index])) return 2;
  std::copy_n(candidate.data(), values, output);
  visits[0] = records;
  visits[1] = evaluations;
  return 0;
}
}  // namespace generativeqc::integrals
