#pragma once

#include "dft/dispersion/d4_eeq_response.hpp"
#include "dft/dispersion/d4_fixed_charge_response.hpp"

#if defined(__CUDACC__)
#define GENERATIVEQC_D4_SECOND_HD __host__ __device__
#else
#define GENERATIVEQC_D4_SECOND_HD
#endif

namespace generativeqc::dft::dispersion {

/** Bounded complete EEQ D4 directional workspace, in doubles. */
GENERATIVEQC_D4_SECOND_HD inline std::size_t d4_eeq_direction_workspace_elements(int atoms) {
  const auto response = eeq_direction_workspace_elements(atoms);
  return response ? response + d4_direction_workspace_elements(atoms) + 12 * atoms : 0;
}

namespace d4_detail {
/** Receives the physical charges plus charge direction from the same solve. */
struct CompleteDirectionSeed {
  int atoms;
  const std::int32_t* numbers;
  const double *xyz, *direction;
  const D4Parameters* parameters;
  D4Tables tables;
  double *scratch, *gradient, *action;

  GENERATIVEQC_D4_SECOND_HD D4Status operator()(const double* charges, const double* charge_dot,
                                                double* seed, double* seed_dot) const {
    return d4_fixed_charge_direction(
        atoms, numbers, xyz, charges, direction, charge_dot, *parameters, tables, scratch,
        d4_direction_workspace_elements(atoms), gradient, action, seed, seed_dot);
  }
};
}  // namespace d4_detail

/** Complete D4-BJ/ATM/EEQ geometry direction; all buffers are host or device
 * according to the calling backend. No output is written on failure. Only
 * fixed total charge and the molecular, nonperiodic reference model are used. */
GENERATIVEQC_D4_SECOND_HD inline D4Status d4_eeq_direction(
    int atoms, const std::int32_t* numbers, const double* xyz, const double* direction,
    double total_charge, const D4Parameters& parameters, D4Tables tables, EEQTables eeq_tables,
    double* scratch, std::size_t scratch_count, double* gradient, double* action) {
  using namespace d4_detail;
  const auto required = d4_eeq_direction_workspace_elements(atoms);
  if (!required || parameters.reference_model != D4ReferenceModel::eeq)
    return D4Status::unsupported;
  if (scratch_count < required) return D4Status::invalid_argument;
  const auto cartesian_bytes = 3 * static_cast<std::size_t>(atoms) * sizeof(double);
  const void* buffers[] = {scratch,
                           gradient,
                           action,
                           numbers,
                           xyz,
                           direction,
                           &parameters,
                           tables.elements,
                           tables.references,
                           tables.reference_c6,
                           eeq_tables.elements,
                           eeq_tables.charge_elements};
  const std::size_t bytes[] = {required * sizeof(double),
                               cartesian_bytes,
                               cartesian_bytes,
                               static_cast<std::size_t>(atoms) * sizeof(*numbers),
                               cartesian_bytes,
                               cartesian_bytes,
                               sizeof(parameters),
                               tables.element_count * sizeof(*tables.elements),
                               tables.reference_count * sizeof(*tables.references),
                               tables.reference_c6_count * sizeof(double),
                               eeq_tables.element_count * sizeof(*eeq_tables.elements),
                               eeq_tables.element_count * sizeof(*eeq_tables.charge_elements)};
  if (tables.element_count != kD4TableElementCount ||
      tables.reference_count != kD4TableReferenceCount ||
      tables.reference_c6_count != kD4PackedReferenceC6Count ||
      eeq_tables.element_count != kD4TableElementCount)
    return D4Status::unsupported;
  Range ranges[12];
  for (int index = 0; index < 12; ++index) {
    if (!range(buffers[index], bytes[index], ranges[index]) ||
        reinterpret_cast<std::uintptr_t>(buffers[index]) %
            (index == 3 ? alignof(std::int32_t) : alignof(double)))
      return D4Status::invalid_argument;
    for (int previous = 0; previous < index && previous < 3; ++previous)
      if (overlaps(ranges[index], ranges[previous])) return D4Status::invalid_argument;
  }

  // The EEQ callback holds the explicit response fixed until the matching
  // implicit pullback finishes, then both contributions publish together.
  const auto eeq_capacity = eeq_direction_workspace_elements(atoms);
  double* fixed_scratch = scratch + eeq_capacity;
  double* fixed_gradient = fixed_scratch + d4_direction_workspace_elements(atoms);
  double* fixed_action = fixed_gradient + 3 * atoms;
  double* charge_gradient = fixed_action + 3 * atoms;
  double* charge_action = charge_gradient + 3 * atoms;
  const CompleteDirectionSeed seed{atoms,         numbers,        xyz,
                                   direction,     &parameters,    tables,
                                   fixed_scratch, fixed_gradient, fixed_action};
  const auto status =
      eeq_directional_pullback(atoms, numbers, xyz, total_charge, eeq_tables, direction, scratch,
                               eeq_capacity, seed, charge_gradient, charge_action);
  if (status != D4Status::success) return status;
  for (int index = 0; index < 3 * atoms; ++index) {
    fixed_gradient[index] += charge_gradient[index];
    fixed_action[index] += charge_action[index];
    if (!finite(fixed_gradient[index]) || !finite(fixed_action[index]))
      return D4Status::numerical_failure;
  }
  for (int index = 0; index < 3 * atoms; ++index) {
    gradient[index] = fixed_gradient[index];
    action[index] = fixed_action[index];
  }
  return D4Status::success;
}

}  // namespace generativeqc::dft::dispersion

#undef GENERATIVEQC_D4_SECOND_HD
