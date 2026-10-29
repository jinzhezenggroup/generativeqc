#pragma once

#include <type_traits>

#include "posthf/capacity.hpp"
#include "scf/cuda/rhf_bucket_internal.hpp"

namespace generativeqc::scf {
using posthf::checked_add;
using posthf::checked_mul;
/** Only the existing public-AO ERI producer's immutable inputs are copied.
 * Quartet queues, density-aware bounds, Cartesian J/K projections and all
 * solver arrays are intentionally absent from this source-only owner. */
template <class Visit>
void visit_rhf_source_fields(const CudaRhfBucketPlan& owner, cuda_execution::DeviceBatch& target,
                             Visit visit) {
  const auto& h = owner.topology;
  const auto& l = owner.layout;
  visit(target.atom_offsets, l.atom_offsets, h.atom_offsets.size());
  visit(target.atom_systems, l.atom_systems, h.atom_systems.size());
  visit(target.atomic_numbers, l.atomic_numbers, h.atomic_numbers.size());
  // Dynamic positions are removed from the host topology after packing.
  visit(target.positions, l.positions, checked_mul(owner.total_atoms, 3));
  visit(target.system_shell_offsets, l.system_shell_offsets, h.system_shell_offsets.size());
  visit(target.shell_atoms, l.shell_atoms, h.shell_atoms.size());
  visit(target.shell_angular, l.shell_angular, h.shell_angular.size());
  visit(target.shell_ao_offsets, l.shell_ao_offsets, h.shell_ao_offsets.size());
  visit(target.shell_primitive_offsets, l.shell_primitive_offsets,
        h.shell_primitive_offsets.size());
  visit(target.ao_shells, l.ao_shells, h.ao_shells.size());
  visit(target.ao_term_counts, l.ao_term_counts, h.ao_term_counts.size());
  visit(target.ao_term_angular, l.ao_term_angular, h.ao_term_angular.size());
  visit(target.ao_term_coefficients, l.ao_term_coefficients, h.ao_term_coefficients.size());
  visit(target.primitive_exponents, l.primitive_exponents, h.primitive_exponents.size());
  visit(target.primitive_coefficients, l.primitive_coefficients, h.primitive_coefficients.size());
}

/** Borrow exactly the incumbent compact source's inputs from the live bucket.
 * No allocation, copy, kernel, device selection or ownership transfer occurs. */
inline cuda_execution::DeviceBatch borrow_rhf_source_batch(const CudaRhfBucketPlan& owner) {
  cuda_execution::DeviceBatch batch{};
  batch.batch_size = 1;
  batch.nbf = static_cast<std::int32_t>(owner.nbf);
  batch.direct_nbf = batch.nbf;
  batch.total_atoms = static_cast<std::int64_t>(owner.total_atoms);
  batch.total_shells = static_cast<std::int64_t>(owner.total_shells);
  visit_rhf_source_fields(owner, batch, [&](auto*& pointer, auto offset, auto count) {
    using T =
        std::remove_const_t<std::remove_pointer_t<std::remove_reference_t<decltype(pointer)>>>;
    const auto bytes = checked_mul(count, sizeof(T));
    if (offset > owner.layout.bytes || bytes > owner.layout.bytes - offset)
      throw std::logic_error("borrowed RHF source input exceeds its admitted arena");
    pointer = reinterpret_cast<std::remove_reference_t<decltype(pointer)>>(
        static_cast<unsigned char*>(owner.resources.arena_) + offset);
  });
  return batch;
}
}  // namespace generativeqc::scf
