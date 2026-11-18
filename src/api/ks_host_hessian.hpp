#pragma once

#include <cstddef>
#include <cstdint>

#include "generativeqc/generativeqc.h"

/** Private synchronous CPU bridge, not a new stable C ABI. Capacities bound
 * derivative-owned numerical storage/reservations, not the borrowed SCF/Fock/
 * basis/grid producer, its final-state export scratch, BLAS or process RSS.
 * integral_bytes admits both nuclear integral sources; grid_bytes admits moving
 * geometry/XC sources. Fixed-grid induced response belongs to response_bytes.
 * Work ceilings cover all requested nuclear columns, not just one column.
 */
struct generativeqc_ks_host_hessian_options_v1 {
  std::size_t struct_size{sizeof(generativeqc_ks_host_hessian_options_v1)};
  std::uint32_t abi_version{GENERATIVEQC_ABI_VERSION};
  const generativeqc_atom* expected_atoms{};
  std::size_t expected_atom_count{};
  std::int32_t expected_charge{}, expected_multiplicity{};
  std::size_t numeric_bytes{512ULL << 20};
  std::size_t integral_bytes{64ULL << 20}, grid_bytes{128ULL << 20};
  std::size_t response_bytes{128ULL << 20}, ecp_bytes{128ULL << 20};
  std::size_t dispersion_bytes{64ULL << 20}, output_bytes{64ULL << 20};
  std::size_t max_eri_primitives{100'000'000}, max_map_terms{1ULL << 40};
  std::size_t max_grid_visits{1ULL << 40}, max_d3_pairs{16'777'216};
  std::size_t restart{30}, max_iterations{200};
  std::size_t gmres_bytes{64ULL << 20}, true_residual_every{1}, stagnation_window{25};
  unsigned reorthogonalizations{2};
  double relative_tolerance{1e-10}, absolute_tolerance{1e-12};
  double breakdown_tolerance{1e-14}, stagnation_tolerance{1e-14};
  double symmetry_absolute_tolerance{1e-7}, symmetry_relative_tolerance{1e-7};
};

/** Work is cumulative over completed columns, including both ECP source stages.
 * Failed requests publish neither this receipt nor any result panel. Numerical
 * reservations may conservatively charge shared geometry scratch twice; they
 * are not empirical peak RSS or whole-program instruction counts.
 */
struct generativeqc_ks_host_hessian_receipt_v1 {
  std::size_t struct_size{sizeof(generativeqc_ks_host_hessian_receipt_v1)};
  std::uint32_t abi_version{GENERATIVEQC_ABI_VERSION};
  std::uint64_t numeric_reservation_bytes{}, frame_bytes{}, geometry_bytes{}, output_bytes{};
  std::uint64_t fixed_bound_bytes{}, gmres_workspace_bytes{}, completed_directions{};
  std::uint64_t response_iterations{}, induced_fock_actions{};
  std::uint64_t first_eri_primitives{}, curvature_eri_primitives{};
  std::uint64_t first_one_electron_primitives{}, curvature_one_electron_primitives{};
  std::uint64_t source_xc_point_visits{}, curvature_xc_point_visits{}, response_xc_point_visits{};
  std::uint64_t ecp_kernel_blocks{}, ecp_radial_layers{}, ecp_ao_primitive_points{};
  std::uint64_t ecp_contraction_terms{}, d3_pair_visits{}, d3_reference_terms{};
  double response_residual_norm{}, response_relative_residual{};
  double electronic_energy{}, total_energy{}, d3_energy{}, source_energy_error{};
  double raw_symmetry_error{}, execution_seconds{};
};

/** Canonical orbitals for occupied-subspace tracking, not a second electronic
 * response provider. Only C crosses the wire; no AO, Fock or grid is exported.
 * The reservation includes the detached verified frame, staged C and caller C,
 * but excludes the producer's verification/eigensolver scratch and Python copies.
 */
struct generativeqc_ks_host_orbitals_receipt_v1 {
  std::size_t struct_size{sizeof(generativeqc_ks_host_orbitals_receipt_v1)};
  std::uint32_t abi_version{GENERATIVEQC_ABI_VERSION};
  std::uint64_t basis{}, spins{}, alpha_occupied{}, beta_occupied{};
  std::uint64_t output_elements{}, numeric_reservation_bytes{};
};

extern "C" {
GENERATIVEQC_API std::uint32_t generativeqc_ks_hessian_cpu_version_v1();
/** Exact linked electronic or supported native total MethodIR selector.
 * Shape, stationary-state and resource gates remain separate.
 */
GENERATIVEQC_API std::uint32_t generativeqc_ks_hessian_cpu_method_v1(const char* identity);
/** A null direction requests the raw row-major full Hessian. Otherwise both
 * panels have 3*atoms entries. All caller panels/descriptors must be disjoint;
 * borrowed native source aliases refuse. The current accepted batch supplies
 * every electronic and native geometry-only source. No Python/JIT/oracle runs.
 */
GENERATIVEQC_API generativeqc_status generativeqc_ks_hessian_cpu_v1(
    generativeqc_batch* batch, std::size_t index,
    const generativeqc_ks_host_hessian_options_v1* options, const double* direction,
    std::size_t direction_count, double* output, std::size_t output_count,
    generativeqc_ks_host_hessian_receipt_v1* receipt);
/** Same expected-geometry/charge/spin and owner gates as the Hessian bridge.
 * Only options' descriptor, expected geometry and numeric_bytes are consumed.
 * Null output/count=0 queries the bounded shape without exporting a frame.
 * Otherwise C is spin-major then row-major [AO,orbital]. Refusal publishes neither
 * C nor the receipt. This performs no SCF, JIT, grid reconstruction or response.
 */
GENERATIVEQC_API generativeqc_status generativeqc_ks_orbitals_cpu_v1(
    generativeqc_batch* batch, std::size_t index,
    const generativeqc_ks_host_hessian_options_v1* options, double* output,
    std::size_t output_count, generativeqc_ks_host_orbitals_receipt_v1* receipt);
}
