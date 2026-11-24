#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "api/ks_host_hessian.hpp"
#include "api/ks_host_orbital_capacity.hpp"
#include "methods/method.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/df_progress_trace.hpp"
#include "runtime/host_resource_ledger.hpp"

#if GENERATIVEQC_HAS_HOST_HESSIAN
#include "dft/host_ks_nuclear_hessian.hpp"
#include "dft/semilocal_family.hpp"
#include "methods/dft_method.hpp"
#include "methods/named_ks_catalog.hpp"
#include "molecule/basis.hpp"
#endif

namespace {
using generativeqc::runtime::size_add;
using generativeqc::runtime::size_mul;

bool overlap(const void* first, std::size_t first_bytes, const void* second,
             std::size_t second_bytes) {
  if (!first_bytes || !second_bytes) return false;
  const auto first_address = reinterpret_cast<std::uintptr_t>(first);
  const auto second_address = reinterpret_cast<std::uintptr_t>(second);
  return first_address <= second_address ? second_address - first_address < first_bytes
                                         : first_address - second_address < second_bytes;
}

#if GENERATIVEQC_HAS_HOST_HESSIAN
using namespace generativeqc;

void require(generativeqc_status status, const std::string& detail) {
  if (status != GENERATIVEQC_STATUS_SUCCESS) throw methods::MethodError(status, detail);
}

void budget(std::size_t needed, std::size_t limit, const char* detail) {
  if (needed > limit) throw std::length_error(detail);
}

/** Report the planner's finite work estimate so an explicit retry can be sized. */
void work_budget(std::size_t needed, std::size_t limit, const char* detail) {
  if (needed > limit)
    throw std::length_error(std::string(detail) + " (planned=" + std::to_string(needed) +
                            ", cap=" + std::to_string(limit) + ')');
}

bool supported_dispersion(const dft::host_hessian_aot::MethodBinding& binding,
                          const methods::named_ks::Dispersion& correction) {
  const bool bare_only = binding.point_domain == dft::host_hessian_aot::PointDomain::lda_rks ||
                         binding.point_domain == dft::host_hessian_aot::PointDomain::lda_uks;
  return !correction.kind || (!bare_only && correction.kind == 1 && !correction.s9 &&
                              !correction.cn_cutoff && !correction.pair_cutoff &&
                              !correction.pair_switch_width && !correction.composite_profile);
}

/** Resolve the exact total program to its linked electronic closure and correction. */
const dft::host_hessian_aot::MethodBinding* linked_program(
    std::string_view identity, methods::named_ks::Dispersion& correction) {
  correction = {};
  if (const auto* binding = dft::host_hessian_aot::find_method(identity)) return binding;
  const auto programs = methods::named_ks::methods();
  const auto found = std::find_if(programs.begin(), programs.end(), [&](const auto& program) {
    return program.identity == identity;
  });
  if (found == programs.end() || !found->cpu || found->bound_basis || found->nonlocal.variant ||
      found->range_omega)
    return nullptr;
  for (const auto& binding : dft::host_hessian_aot::method_bindings()) {
    if (!supported_dispersion(binding, found->dispersion)) continue;
    const auto bare = std::find_if(programs.begin(), programs.end(), [&](const auto& program) {
      return program.identity == binding.identity;
    });
    if (bare == programs.end() || bare->spin != found->spin || bare->domain != found->domain)
      continue;
    const auto first_components = methods::named_ks::components(*bare);
    const auto second_components = methods::named_ks::components(*found);
    const auto first_exchange = methods::named_ks::exchanges(*bare);
    const auto second_exchange = methods::named_ks::exchanges(*found);
    if (std::equal(first_components.begin(), first_components.end(), second_components.begin(),
                   second_components.end(),
                   [](const auto& first, const auto& second) {
                     return std::string_view(first.component_id) == second.component_id &&
                            first.coefficient == second.coefficient;
                   }) &&
        std::equal(first_exchange.begin(), first_exchange.end(), second_exchange.begin(),
                   second_exchange.end(), [](const auto& first, const auto& second) {
                     return first.operator_kind == second.operator_kind &&
                            first.coefficient == second.coefficient && first.omega == second.omega;
                   })) {
      correction = found->dispersion;
      return &binding;
    }
  }
  return nullptr;
}

void validate_linked_system(const core::System& system,
                            const dft::host_hessian_aot::MethodBinding& binding,
                            const methods::named_ks::Dispersion& correction) {
  // The shared AO pack and nuclear integral owner handle Cartesian monomials
  // and real-spherical polynomial expansions. The point binding selects J
  // and the XC closure, not an AO representation.
  if (!supported_dispersion(binding, correction))
    throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                               "CPU Hessian requires an admitted uncut two-body D3(BJ), ATM off");
  if (binding.point_domain != dft::host_hessian_aot::PointDomain::linked_ad &&
      (!system.ecp_terms.empty() ||
       std::any_of(system.atoms.begin(), system.atoms.end(),
                   [](const auto& atom) { return atom.ecp_core != 0; })))
    throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                               "native CPU LDA/PBE/PBE0 Hessian requires an all-electron system");
}

struct MandatoryReservation {
  std::size_t frame_bytes{}, panel_bytes{}, assembly_bytes{}, total_bytes{};
  integrals::EcpDirectionLimits first_ecp, second_ecp;
  dft::dispersion::HostD3ColumnPlan d3;
};

/** The base reservation charges whole grid/response caps, not their later sampled usage.
 * Optional density-dependent storage is neither promised nor sized here. */
MandatoryReservation mandatory_reservation(const core::System& system, std::size_t aos,
                                           std::size_t spins, std::size_t columns,
                                           bool snapshot_directions, bool corrected,
                                           std::size_t first_bytes, std::size_t curvature_bytes,
                                           const generativeqc_ks_host_hessian_options_v1& options) {
  MandatoryReservation result;
  const auto atoms = system.atoms.size();
  const auto dimension = size_mul(3, atoms);
  budget(size_add(first_bytes, curvature_bytes), options.integral_bytes,
         "CPU Hessian integral storage cap exceeded");
  result.first_ecp = integrals::ecp_direction_plan(system).limits;
  result.second_ecp = integrals::ecp_curvature_plan(system).limits;
  budget(std::max(result.first_ecp.workspace_bytes, result.second_ecp.workspace_bytes),
         options.ecp_bytes, "CPU Hessian ECP storage cap exceeded");
  result.d3 = dft::dispersion::HostD3Column::plan(atoms);
  if (corrected) {
    budget(result.d3.numeric_bytes, options.dispersion_bytes,
           "CPU Hessian D3 storage cap exceeded");
    work_budget(size_mul(columns, result.d3.pair_visits), options.max_d3_pairs,
                "CPU Hessian complete D3 pair-work cap exceeded");
  }
  const auto output_elements = size_mul(dimension, columns);
  const auto fields = size_add(2, size_mul(2, dimension));
  result.panel_bytes = size_mul(
      size_add(size_mul(snapshot_directions ? 3 : 2, output_elements), size_add(dimension, fields)),
      sizeof(double));
  budget(result.panel_bytes, options.output_bytes, "CPU Hessian output/staging cap exceeded");
  result.frame_bytes =
      size_mul(size_mul(spins, size_add(size_mul(4, size_mul(aos, aos)), aos)), sizeof(double));
  result.assembly_bytes = dft::HostKsNuclearHessian::assembly_bytes(atoms, aos, spins);
  for (const auto bytes :
       {result.frame_bytes, result.panel_bytes, result.assembly_bytes, options.response_bytes,
        options.grid_bytes, curvature_bytes, result.second_ecp.workspace_bytes,
        corrected ? result.d3.numeric_bytes : std::size_t{0}})
    result.total_bytes = size_add(result.total_bytes, bytes);
  return result;
}

/** Each ECP owner reports its last completed source, not a cumulative total. */
void add_ecp(generativeqc_ks_host_hessian_receipt_v1& receipt,
             const integrals::EcpDirectionWork& work) {
  receipt.ecp_kernel_blocks = size_add(receipt.ecp_kernel_blocks, work.kernel_blocks);
  receipt.ecp_radial_layers = size_add(receipt.ecp_radial_layers, work.radial_layers);
  receipt.ecp_ao_primitive_points =
      size_add(receipt.ecp_ao_primitive_points, work.ao_primitive_points);
  receipt.ecp_contraction_terms = size_add(receipt.ecp_contraction_terms, work.contraction_terms);
}

void reject_sources(const void* data, std::size_t bytes,
                    const methods::detail::KsHostDerivativeSources& source) {
  const auto reject = [&](const auto& values) {
    if (overlap(data, bytes, values.data(), size_mul(values.size(), sizeof(*values.data()))))
      throw std::invalid_argument("CPU Hessian caller storage aliases a borrowed source");
  };
  reject(source.system->atoms);
  reject(source.system->shells);
  reject(source.system->ecp_terms);
  for (const auto& shell : source.system->shells) reject(shell.primitives);
  reject(source.fock->one_electron().overlap);
  reject(source.fock->one_electron().hcore);
  reject(source.basis->packed);
  reject(source.grid->points());
  reject(source.grid->weights());
  reject(source.grid->owners());
  const auto reject_object = [&](const auto* object) {
    if (overlap(data, bytes, object, sizeof(*object)))
      throw std::invalid_argument("CPU Hessian caller storage aliases a borrowed owner");
  };
  reject_object(source.system);
  reject_object(source.fock);
  reject_object(source.basis);
  reject_object(source.grid);
  reject_object(source.dispersion);
}

void validate_sources(generativeqc_batch& batch, std::size_t index,
                      const dft::CudaKsFinalStateToken& token,
                      const methods::detail::KsHostDerivativeSources& source,
                      const generativeqc_ks_host_hessian_options_v1& options) {
  methods::detail::KsHostDerivativeSources current;
  std::string detail;
  require(methods::detail::borrow_dft_host_derivative_sources(*batch.plan, index, token, current,
                                                              detail),
          detail);
  if (current.system != source.system || current.fock != source.fock ||
      current.basis != source.basis || current.grid != source.grid ||
      current.dispersion != source.dispersion)
    throw std::invalid_argument("CPU Hessian native source owner changed");
  if (source.system->atoms.size() != options.expected_atom_count ||
      source.system->charge != options.expected_charge ||
      source.system->multiplicity != static_cast<unsigned>(options.expected_multiplicity))
    throw std::invalid_argument(
        "CPU Hessian requested shape/charge/multiplicity disagrees with its owner");
  for (std::size_t atom = 0; atom < options.expected_atom_count; ++atom) {
    const auto& expected = options.expected_atoms[atom];
    const auto& actual = source.system->atoms[atom];
    if (actual.atomic_number != expected.atomic_number || actual.position[0] != expected.x ||
        actual.position[1] != expected.y || actual.position[2] != expected.z)
      throw std::invalid_argument("CPU Hessian requested geometry disagrees with its native owner");
  }
}

const dft::host_hessian_aot::MethodBinding& live_binding(
    const dft::CudaKsFinalStateToken& token,
    const methods::detail::KsHostDerivativeSources& source) {
  const dft::host_hessian_aot::MethodBinding* binding{};
  for (const auto& candidate : dft::host_hessian_aot::method_bindings())
    if (candidate.functional == token.identity.model.functional &&
        candidate.scf_domain == token.identity.model.scf_domain_version &&
        candidate.spins == token.identity.model.spins &&
        candidate.exact_exchange == -token.identity.determinant.model.spec.exchange.coefficient *
                                        (candidate.spins == 1 ? 2.0 : 1.0))
      binding = &candidate;
  if (!binding)
    throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                               "CPU Hessian has no exact linked electronic MethodIR");
  validate_linked_system(*source.system, *binding, *source.dispersion);
  if (dft::host_hessian_aot::semilocal_only(binding->point_domain) &&
      (token.identity.model.semilocal_exchange_scale != 1.0 ||
       token.identity.model.semilocal_correlation_scale != 1.0))
    throw methods::MethodError(
        GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
        "native CPU semilocal Hessian requires canonical unscaled electronics");
  return *binding;
}

std::size_t minimum_bytes_for_shape(const core::System& system, std::size_t aos,
                                    const dft::host_hessian_aot::MethodBinding& binding,
                                    std::size_t directions, bool dense, bool corrected,
                                    const generativeqc_ks_host_hessian_options_v1& options) {
  const auto atoms = system.atoms.size();
  if (!atoms || !aos) throw std::invalid_argument("CPU Hessian requires a nonempty system/basis");
  const auto first =
      dft::HostKsNuclearIntegrals::direction_bytes(atoms, aos, binding.spins, binding.identity);
  const auto curvature = dft::HostKsNuclearIntegralCurvature::column_bytes(
      atoms, aos, binding.spins, binding.identity);
  return mandatory_reservation(system, aos, binding.spins, dense ? size_mul(3, atoms) : directions,
                               !dense && directions > 1, corrected, first, curvature, options)
      .total_bytes;
}

generativeqc_ks_host_hessian_receipt_v1 execute(
    generativeqc_batch& batch, std::size_t index,
    const generativeqc_ks_host_hessian_options_v1& requested, std::span<const double> external,
    std::span<double> output, const generativeqc_ks_host_hessian_receipt_v1* receipt,
    std::size_t recycle_width, bool sequential) {
  const auto started = std::chrono::steady_clock::now();
  const bool matrix = external.empty();
  const auto dimension = size_mul(3, batch.atom_counts[index]);
  const auto columns = matrix ? dimension : external.size() / dimension;
  const bool multivector = !matrix && columns > 1;
  runtime::df_progress::Scope progress(matrix        ? "cpu_hessian"
                                       : multivector ? "cpu_hessian_vector_batch"
                                                     : "cpu_hessian_vector");
  runtime::df_progress::number("columns", columns);
  dft::CudaKsFinalStateToken token;
  std::string detail;
  require(methods::detail::dft_final_state_token(*batch.plan, index, token, detail), detail);
  methods::detail::KsHostDerivativeSources source;
  require(methods::detail::borrow_dft_host_derivative_sources(*batch.plan, index, token, source,
                                                              detail),
          detail);
  reject_sources(output.data(), output.size_bytes(), source);
  reject_sources(receipt, sizeof(*receipt), source);
  reject_sources(&requested, sizeof(requested), source);
  reject_sources(external.data(), external.size_bytes(), source);
  auto options = requested;
  if (batch.cpu_hessian_workspace_bytes)
    options.numeric_bytes = std::min(options.numeric_bytes, batch.cpu_hessian_workspace_bytes);
  reject_sources(options.expected_atoms,
                 size_mul(options.expected_atom_count, sizeof(generativeqc_atom)), source);
  const auto* binding = &live_binding(token, source);
  const bool semilocal_only = dft::host_hessian_aot::semilocal_only(binding->point_domain);
  if (recycle_width && (!semilocal_only || binding->spins != 1))
    throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                               "recycled CPU Hessian requires linked LDA/PBE RKS response");
  if (sequential && (!semilocal_only || binding->spins != 1))
    throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                               "sequential CPU Hessian requires linked LDA/PBE RKS response");
  const auto& correction = *source.dispersion;
  const auto atoms = source.system->atoms.size(), basis = source.basis->nao;
  const auto spins = token.identity.model.spins;
  runtime::df_progress::number("atoms", atoms);
  runtime::df_progress::number("basis", basis);
  if (size_mul(atoms, 3) != dimension)
    throw std::invalid_argument("CPU Hessian batch and native geometry disagree");
  const auto validate = [&] { validate_sources(batch, index, token, source, options); };
  validate();
  std::optional<dft::dispersion::D3Parameters> bj;
  if (correction.kind) {
    bj = dft::dispersion::D3Parameters{correction.s6, correction.s8, correction.a1, correction.a2,
                                       correction.s9};
  }
  constexpr auto eri_schedule = dft::HostNuclearEriSchedule::SymmetryUnique;
  const auto first = dft::HostKsNuclearIntegrals::plan(*source.basis, spins, eri_schedule,
                                                       dft::HostNuclearFirstSchedule::Direction,
                                                       binding->identity);
  const auto curvature_plan = [&](auto schedule) {
    return dft::HostKsNuclearIntegralCurvature::plan(*source.basis, spins, schedule, eri_schedule,
                                                     binding->identity);
  };
  const auto curvature = curvature_plan(dft::HostNuclearIntegralCurvatureSchedule::Column);
  const auto mandatory =
      mandatory_reservation(*source.system, basis, spins, columns, multivector, bj.has_value(),
                            first.numeric_bytes, curvature.numeric_bytes, options);
  auto reservation = mandatory.total_bytes;
  budget(reservation, options.numeric_bytes,
         "CPU Hessian aggregate numerical reservation exceeded");
  const auto frame_bytes = mandatory.frame_bytes, panel_bytes = mandatory.panel_bytes;
  const auto& d3_plan = mandatory.d3;
  const auto fields = size_add(2, size_mul(2, dimension));
  const auto curvature_matrix =
      matrix ? curvature_plan(dft::HostNuclearIntegralCurvatureSchedule::Matrix)
             : dft::HostNuclearIntegralPlan{};
  const bool projected_basis =
      matrix && dft::HostKsNuclearIntegralCurvature::projected_matrix_supported(*source.basis, {});
  const auto projected_matrix =
      projected_basis ? curvature_plan(dft::HostNuclearIntegralCurvatureSchedule::ProjectedMatrix)
                      : curvature_matrix;
  const auto curvature_lower =
      matrix ? std::min(size_mul(columns, curvature.eri_primitives),
                        std::min(curvature_matrix.eri_primitives, projected_matrix.eri_primitives))
             : size_mul(columns, curvature.eri_primitives);
  work_budget(curvature_lower, options.max_eri_primitives,
              "CPU Hessian complete nuclear primitive-work cap exceeded (lower bound)");
  work_budget(matrix ? std::min(size_mul(columns, curvature.map_terms), curvature_matrix.map_terms)
                     : size_mul(columns, curvature.map_terms),
              options.max_map_terms, "CPU Hessian integral-map work cap exceeded (lower bound)");
  const auto base_plan =
      dft::HostGridGeometry::plan(*source.basis, *source.grid, token.identity.model.tile_points);
  const auto atomic_weight_bytes = size_mul(source.grid->weights().size(), sizeof(double));
  budget(size_add(base_plan.numeric_bytes, atomic_weight_bytes), options.grid_bytes,
         "CPU Hessian geometry cap exceeded");
  const dft::host_hessian_aot::Program* motion{};
  for (const auto& program : dft::host_hessian_aot::programs())
    if (program.name == "ao_first_motion") motion = &program;
  if (!motion)
    throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                               "CPU Hessian has no linked AO motion closure");
  dft::HostGridGeometry geometry(*source.basis, *source.grid, token.identity.model.tile_points,
                                 base_plan, motion->identity, validate);
  const auto column_plan = dft::HostGridColumnGeometry::plan(geometry);
  const auto geometry_bytes =
      size_add(atomic_weight_bytes, size_add(base_plan.numeric_bytes, column_plan.numeric_bytes));
  budget(geometry_bytes, options.grid_bytes, "CPU Hessian column geometry cap exceeded");
  const auto atomic_weights = source.grid->atomic_weights();
  dft::HostGridColumnGeometry column_geometry(geometry, atomic_weights, column_plan, validate);
  const auto xc_first = dft::HostGgaNuclearXc::plan(geometry, spins);
  const auto xc_second = dft::HostGgaNuclearColumn::plan(column_geometry, spins);
  budget(size_add(geometry_bytes, size_add(xc_first.numeric_bytes, xc_second.numeric_bytes)),
         options.grid_bytes, "CPU Hessian moving-grid/XC storage cap exceeded");
  work_budget(size_mul(columns, size_add(xc_first.map_terms, xc_second.map_terms)),
              options.max_map_terms, "CPU Hessian complete XC map-work cap exceeded");
  work_budget(size_mul(columns, size_add(xc_first.ao_point_visits, xc_second.ao_point_visits)),
              options.max_grid_visits, "CPU Hessian complete moving-grid work cap exceeded");
  dft::VerifiedKsFinalState state;
  require(methods::detail::read_dft_final_state(*batch.plan, index, token, true, state, detail),
          detail);
  dft::HostKsHessianLimits limits;
  limits.max_numeric_bytes = options.numeric_bytes - frame_bytes - panel_bytes - geometry_bytes;
  limits.max_optional_numeric_bytes = options.numeric_bytes - reservation;
  limits.max_assembly_numeric_bytes = size_add(
      mandatory.assembly_bytes, (matrix || multivector) ? options.output_bytes - panel_bytes : 0);
  limits.max_nuclear_pairs =
      atoms % 2 ? size_mul(atoms, (atoms - 1) / 2) : size_mul(atoms / 2, atoms - 1);
  limits.response.max_numeric_bytes = options.response_bytes;
  limits.response.recycle_rhs_groups = recycle_width > 0;
  limits.response.perturbation_map_terms = limits.response.assembly_map_terms =
      options.max_map_terms;
  limits.response.integrals.limits = first;
  if (matrix)
    limits.response.coordinate_source_bytes =
        options.integral_bytes - first.numeric_bytes - curvature.numeric_bytes;
  limits.response.xc = xc_first;
  limits.response.ecp = mandatory.first_ecp;
  limits.integrals.limits = curvature;
  limits.xc = xc_second;
  limits.ecp = mandatory.second_ecp;
  limits.dispersion = {d3_plan.numeric_bytes, d3_plan.pair_visits, d3_plan.reference_terms};
  limits.energy_absolute_tolerance = limits.energy_relative_tolerance = 2e-9;
  dft::HostKsNuclearHessian owner(
      state, *source.system, *source.fock, column_geometry, binding->identity, bj, limits,
      [&](const auto& identity) {
        if (identity != token.identity)
          throw std::invalid_argument("CPU Hessian frame identity changed");
        validate();
      });
  response::GmresOptions solver;
  solver.restart = options.restart;
  solver.max_iterations = options.max_iterations;
  solver.reorthogonalize = options.reorthogonalizations;
  solver.relative_tolerance = options.relative_tolerance;
  solver.absolute_tolerance = options.absolute_tolerance;
  solver.max_workspace_bytes = options.gmres_bytes;
  solver.true_residual_every = options.true_residual_every;
  solver.stagnation_window = options.stagnation_window;
  solver.breakdown_tolerance = options.breakdown_tolerance;
  solver.stagnation_tolerance = options.stagnation_tolerance;
  const auto base_reserved = owner.resources(solver).reserved_numeric_bytes;
  const auto requested_width =
      recycle_width ? owner.admit_response_batch(solver, recycle_width) : 1;
  if (recycle_width && requested_width != recycle_width)
    throw std::length_error("CPU Hessian requested recycle group does not fit");
  const bool coordinate_sources = matrix && owner.admit_coordinate_sources(solver);
  std::size_t remaining_integral{};
  if (coordinate_sources) {
    const auto panel = dft::HostKsNuclearIntegrals::plan(
        *source.basis, spins, eri_schedule, dft::HostNuclearFirstSchedule::CoordinatePanel,
        binding->identity);
    remaining_integral = options.integral_bytes - first.numeric_bytes - curvature.numeric_bytes -
                         panel.numeric_bytes;
  }
  const dft::HostNuclearIntegralPlan matrix_allowance{
      remaining_integral, curvature_matrix.one_electron_primitives, options.max_eri_primitives,
      options.max_map_terms};
  auto selected_matrix = curvature_matrix;
  bool matrix_integrals = false;
  bool projected_integrals = false;
  if (coordinate_sources && projected_basis &&
      projected_matrix.eri_primitives < curvature_matrix.eri_primitives) {
    matrix_integrals = owner.admit_integral_matrix(
        matrix_allowance, dft::HostNuclearIntegralCurvatureSchedule::ProjectedMatrix);
    if (matrix_integrals) {
      selected_matrix = projected_matrix;
      projected_integrals = true;
    }
  }
  if (coordinate_sources && !matrix_integrals)
    matrix_integrals = owner.admit_integral_matrix(
        matrix_allowance, dft::HostNuclearIntegralCurvatureSchedule::Matrix);
  const auto first_passes = coordinate_sources ? 1 : columns;
  const auto curvature_visits = matrix_integrals ? selected_matrix.eri_primitives
                                                 : size_mul(columns, curvature.eri_primitives);
  work_budget(size_add(size_mul(first_passes, first.eri_primitives), curvature_visits),
              options.max_eri_primitives,
              "CPU Hessian complete nuclear primitive-work cap exceeded");
  const auto curvature_terms =
      matrix_integrals
          ? size_add(selected_matrix.map_terms,
                     size_mul(columns, dft::HostKsNuclearIntegrals::density_response_terms(
                                           *source.basis, spins, binding->identity)))
          : size_mul(columns, curvature.map_terms);
  work_budget(size_add(size_mul(first_passes, first.map_terms), curvature_terms),
              options.max_map_terms, "CPU Hessian complete integral-map work cap exceeded");
  const auto response_width =
      recycle_width           ? requested_width
      : matrix && !sequential ? owner.admit_response_batch(solver)
      : multivector ? owner.admit_response_batch(solver, std::min(columns, std::size_t{4}))
                    : 1;
  owner.admit_response_cache(solver);
  const auto resources = owner.resources(solver);
  if (!resources.admitted)
    throw std::length_error("CPU Hessian response/GMRES workspace admission failed");
  reservation = size_add(reservation, resources.reserved_numeric_bytes - base_reserved);
  budget(reservation, options.numeric_bytes,
         "CPU Hessian aggregate numerical reservation exceeded");
  runtime::df_progress::number("admitted_first_eri_primitives",
                               size_mul(first_passes, first.eri_primitives));
  runtime::df_progress::number("admitted_curvature_eri_primitives", curvature_visits);
  runtime::df_progress::number(
      "admitted_xc_map_terms",
      size_mul(columns, size_add(xc_first.map_terms, xc_second.map_terms)));
  const char* integral_schedule = "column";
  if (matrix_integrals) integral_schedule = projected_integrals ? "projected_matrix" : "matrix";
  runtime::df_progress::label("frozen_integral_schedule", integral_schedule);
  runtime::df_progress::label("response_rhs_schedule", recycle_width ? "recycled"
                                                       : sequential  ? "sequential"
                                                                     : "independent");
  runtime::df_progress::number("independent_response_width", recycle_width ? 1 : response_width);
  if (recycle_width) runtime::df_progress::number("recycled_response_width", response_width);
  if (coordinate_sources) {
    runtime::df_progress::Scope preparation("cpu_hessian_prepare_first_sources");
    owner.prepare_coordinate_sources();
  }
  if (matrix_integrals) {
    runtime::df_progress::Scope preparation("cpu_hessian_prepare_frozen_integrals");
    owner.prepare_integral_matrix();
    runtime::df_progress::number("completed_eri_matrix_primitives",
                                 owner.work().integrals.eri_matrix_primitives);
    runtime::df_progress::number("completed_eri_matrix_coordinates",
                                 owner.work().integrals.eri_matrix_coordinates);
    runtime::df_progress::number("completed_eri_matrix_second_partials",
                                 owner.work().integrals.eri_matrix_second_partials);
  }
  std::vector<double> right(dimension), column(fields), candidate(output.size());
  std::vector<double> directions;
  if (multivector)
    directions.assign(external.begin(), external.end());
  else if (!matrix)
    std::copy(external.begin(), external.end(), right.begin());
  auto result = api::empty_host_hessian_receipt();
  result.numeric_reservation_bytes = reservation;
  result.frame_bytes = frame_bytes;
  result.geometry_bytes = geometry_bytes;
  result.output_bytes = panel_bytes;
  result.fixed_bound_bytes = resources.fixed_bound_bytes;
  result.gmres_workspace_bytes = resources.gmres_bytes;
  const auto require_response = [](const dft::HostKsHessianDiagnostic& diagnostic,
                                   std::size_t coordinate) {
    if (!diagnostic.converged()) {
      const auto& perturbation = diagnostic.response;
      const auto& iterative = perturbation.response;
      std::ostringstream message;
      message.precision(17);
      message << "CPU Hessian nuclear response failed at column " << coordinate
              << "; perturbation status=" << static_cast<int>(perturbation.status)
              << "; GMRES status=" << response::gmres_status_name(iterative.status);
      if (perturbation.status == dft::HostKsPerturbationStatus::physical_residual_failed) {
        message << "; true physical residual=" << perturbation.physical_residual_norm
                << "; relative physical residual=" << perturbation.relative_physical_residual;
      } else {
        // The D/W verification never runs after a failed solve. Its zero-initialized
        // fields are not measurements; report the solver's separate residual instead.
        message << "; final physical residual not evaluated";
        if ((iterative.status == response::GmresStatus::max_iterations ||
             iterative.status == response::GmresStatus::breakdown ||
             iterative.status == response::GmresStatus::stagnation) &&
            std::isfinite(iterative.residual_norm))
          message << "; true GMRES residual=" << iterative.residual_norm
                  << "; relative GMRES residual=" << iterative.relative_residual;
      }
      throw methods::MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, message.str());
    }
  };
  std::vector<dft::HostKsHessianDiagnostic> group;
  for (std::size_t coordinate = 0; coordinate < columns; ++coordinate) {
    runtime::df_progress::Scope column_progress("cpu_hessian_column");
    runtime::df_progress::number("column_index", coordinate);
    dft::HostKsHessianDiagnostic diagnostic;
    std::span<const double> values = column;
    if (response_width > 1) {
      const auto local = coordinate % response_width;
      if (!local) {
        const auto count = std::min(response_width, columns - coordinate);
        if (matrix)
          group = owner.apply_coordinates(coordinate, count, solver);
        else
          group = owner.apply_directions(std::span<const double>(directions)
                                             .subspan(coordinate * dimension, count * dimension),
                                         solver);
        for (std::size_t index = 0; index < group.size(); ++index)
          require_response(group[index], coordinate + index);
      }
      diagnostic = group[local];
      values = owner.batch_column(local);
    } else {
      if (matrix) {
        std::fill(right.begin(), right.end(), 0);
        right[coordinate] = 1;
      } else if (multivector) {
        const auto direction =
            std::span<const double>(directions).subspan(coordinate * dimension, dimension);
        std::copy(direction.begin(), direction.end(), right.begin());
      }
      diagnostic = owner.apply(right, column, solver);
      require_response(diagnostic, coordinate);
    }
    result.response_iterations =
        size_add(result.response_iterations, diagnostic.response.response.iterations);
    result.response_residual_norm =
        std::max(result.response_residual_norm, diagnostic.response.physical_residual_norm);
    result.response_relative_residual =
        std::max(result.response_relative_residual, diagnostic.response.relative_physical_residual);
    result.source_energy_error = std::max(result.source_energy_error, diagnostic.energy_error);
    result.electronic_energy = diagnostic.verified_energy;
    result.total_energy = values[0];
    result.d3_energy = values[0] - diagnostic.verified_energy;
    for (std::size_t row = 0; row < dimension; ++row)
      candidate[matrix ? row * dimension + coordinate : coordinate * dimension + row] =
          values[2 + dimension + row];
    add_ecp(result, diagnostic.first_ecp_work);
    add_ecp(result, diagnostic.curvature_ecp_work);
    result.d3_pair_visits = size_add(result.d3_pair_visits,
                                     size_add(diagnostic.dispersion_work.coordination_pairs,
                                              size_add(diagnostic.dispersion_work.energy_pairs,
                                                       diagnostic.dispersion_work.reverse_pairs)));
    result.d3_reference_terms =
        size_add(result.d3_reference_terms, diagnostic.dispersion_work.reference_terms);
    ++result.completed_directions;
    runtime::df_progress::number("completed_directions", result.completed_directions);
  }
  const auto work = owner.work();
  result.induced_fock_actions = work.response.induced_fock_actions;
  result.first_eri_primitives = work.response.integrals.eri_primitives;
  result.curvature_eri_primitives = work.integrals.eri_primitives;
  result.first_one_electron_primitives = work.response.integrals.one_electron_primitives;
  result.curvature_one_electron_primitives = work.integrals.one_electron_primitives;
  result.source_xc_point_visits = work.response.xc.point_visits;
  result.curvature_xc_point_visits = work.xc.point_visits;
  result.response_xc_point_visits = work.response.response_xc_point_visits;
  if (matrix) {
    double maximum{};
    for (const auto value : candidate) maximum = std::max(maximum, std::abs(value));
    std::size_t worst_row{}, worst_column{};
    for (std::size_t row = 0; row < dimension; ++row)
      for (std::size_t coordinate = 0; coordinate < row; ++coordinate) {
        const auto error = std::abs(candidate[row * dimension + coordinate] -
                                    candidate[coordinate * dimension + row]);
        if (error > result.raw_symmetry_error) {
          result.raw_symmetry_error = error;
          worst_row = row;
          worst_column = coordinate;
        }
      }
    const auto tolerance =
        options.symmetry_absolute_tolerance + options.symmetry_relative_tolerance * maximum;
    if (result.raw_symmetry_error > tolerance) {
      std::ostringstream message;
      message.precision(17);
      message << "CPU Hessian raw symmetry gate failed; no matrix repair is applied"
              << "; max |H_ij - H_ji|=" << result.raw_symmetry_error
              << " Eh/Bohr^2; tolerance=" << tolerance << " Eh/Bohr^2"
              << "; worst atom-major Cartesian indices (zero-based)=" << worst_row << ','
              << worst_column;
      throw methods::MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, message.str());
    }
  }
  validate();
  if (!matrix) {
    const auto snapshot =
        multivector ? std::span<const double>(directions) : std::span<const double>(right);
    if (!std::equal(snapshot.begin(), snapshot.end(), external.begin()))
      throw std::invalid_argument("CPU Hessian external direction changed before publication");
  }
  result.execution_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  std::copy(candidate.begin(), candidate.end(), output.begin());
  return result;
}
#endif
}  // namespace

void generativeqc::api::validate_host_hessian_system(const core::System& system,
                                                     std::string_view identity) {
#if GENERATIVEQC_HAS_HOST_HESSIAN
  methods::named_ks::Dispersion correction;
  if (const auto* binding = linked_program(identity, correction)) {
    validate_linked_system(system, *binding, correction);
    return;
  }
#else
  (void)system;
  (void)identity;
#endif
  throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                             "CPU Hessian has no exact linked MethodIR");
}

extern "C" std::uint32_t generativeqc_ks_hessian_cpu_version_v1() {
  return GENERATIVEQC_HAS_HOST_HESSIAN ? 1 : 0;
}

extern "C" std::uint32_t generativeqc_ks_hessian_cpu_method_v1(const char* identity) {
#if GENERATIVEQC_HAS_HOST_HESSIAN
  methods::named_ks::Dispersion correction;
  return identity && linked_program(identity, correction) ? 1 : 0;
#else
  (void)identity;
  return 0;
#endif
}

namespace {
generativeqc_status cpu_hessian_minimum_bytes(
    generativeqc_context* context, const generativeqc_system* system, const char* method,
    bool dense, std::size_t directions, const generativeqc_ks_host_hessian_options_v1* options,
    std::size_t* minimum_bytes) {
  if (!context || !system || !method || !directions || !options || !minimum_bytes)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  if (!generativeqc::api::valid_descriptor(options)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (overlap(minimum_bytes, sizeof(*minimum_bytes), options, sizeof(*options)) ||
      overlap(minimum_bytes, sizeof(*minimum_bytes), system, sizeof(*system)) ||
      system->correlation_auxiliary || system->ks_auxiliary || system->scf_auxiliary ||
      !options->grid_bytes || !options->response_bytes || !options->output_bytes ||
      !options->integral_bytes || !options->ecp_bytes || !options->dispersion_bytes ||
      !options->max_d3_pairs)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
#if GENERATIVEQC_HAS_HOST_HESSIAN
    using namespace generativeqc;
    methods::named_ks::Dispersion correction;
    const auto* named = methods::named_ks::find_method(method);
    const auto* binding =
        linked_program(named ? named->identity : std::string_view(method), correction);
    if (!binding)
      throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                                 "CPU Hessian has no exact linked MethodIR");
    const auto& source = system->data;
    const bool corrected = correction.kind != 0;
    validate_linked_system(source, *binding, correction);
    *minimum_bytes = minimum_bytes_for_shape(source, molecule::ao_count(source), *binding,
                                             directions, dense, corrected, *options);
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    (void)dense;
    context->last_detail = "CPU Hessian requires a linked native host AOT package; no fallback";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}
}  // namespace

extern "C" generativeqc_status generativeqc_ks_hessian_cpu_minimum_bytes_v1(
    generativeqc_context* context, const generativeqc_system* system, const char* method,
    std::uint32_t dense, const generativeqc_ks_host_hessian_options_v1* options,
    std::size_t* minimum_bytes) {
  if (dense > 1) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  return cpu_hessian_minimum_bytes(context, system, method, dense != 0, 1, options, minimum_bytes);
}

extern "C" generativeqc_status generativeqc_ks_hvp_cpu_batch_minimum_bytes_v1(
    generativeqc_context* context, const generativeqc_system* system, const char* method,
    std::size_t directions, const generativeqc_ks_host_hessian_options_v1* options,
    std::size_t* minimum_bytes) {
  return cpu_hessian_minimum_bytes(context, system, method, false, directions, options,
                                   minimum_bytes);
}

extern "C" generativeqc_status generativeqc_ks_hvp_cpu_prepared_minimum_bytes_v1(
    generativeqc_batch* batch, std::size_t index, std::size_t directions,
    const generativeqc_ks_host_hessian_options_v1* options, std::size_t* minimum_bytes) {
  if (!batch || !batch->context || !batch->plan || !directions || !options || !minimum_bytes)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  auto* context = batch->context;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  if (!generativeqc::api::valid_descriptor(options)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (index >= batch->atom_counts.size() || !batch->atom_counts[index] ||
      !options->expected_atoms || options->expected_atom_count != batch->atom_counts[index] ||
      options->expected_multiplicity < 1 || !options->grid_bytes || !options->response_bytes ||
      !options->output_bytes || !options->integral_bytes || !options->ecp_bytes ||
      !options->dispersion_bytes || !options->max_d3_pairs)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    const auto geometry_bytes = size_mul(options->expected_atom_count, sizeof(generativeqc_atom));
    if (overlap(minimum_bytes, sizeof(*minimum_bytes), options, sizeof(*options)) ||
        overlap(minimum_bytes, sizeof(*minimum_bytes), batch, sizeof(*batch)) ||
        overlap(minimum_bytes, sizeof(*minimum_bytes), options->expected_atoms, geometry_bytes))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
#if GENERATIVEQC_HAS_HOST_HESSIAN
    using namespace generativeqc;
    auto host_workspace =
        batch->host_lease ? batch->host_lease->use_workspace() : std::unique_lock<std::mutex>{};
    dft::CudaKsFinalStateToken token;
    std::string detail;
    require(methods::detail::dft_final_state_token(*batch->plan, index, token, detail), detail);
    methods::detail::KsHostDerivativeSources source;
    require(methods::detail::borrow_dft_host_derivative_sources(*batch->plan, index, token, source,
                                                                detail),
            detail);
    reject_sources(minimum_bytes, sizeof(*minimum_bytes), source);
    validate_sources(*batch, index, token, source, *options);
    const auto& binding = live_binding(token, source);
    *minimum_bytes = minimum_bytes_for_shape(*source.system, source.basis->nao, binding, directions,
                                             false, source.dispersion->kind != 0, *options);
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    context->last_detail = "CPU Hessian requires a linked native host AOT package; no fallback";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

extern "C" generativeqc_status generativeqc_ks_hessian_cpu_default_options_v1(
    generativeqc_ks_host_hessian_options_v1* options) {
  if (!options) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto map_work_cap = static_cast<std::size_t>(std::min<std::uint64_t>(1ULL << 40, SIZE_MAX));
  *options = {.struct_size = sizeof(*options),
              .abi_version = GENERATIVEQC_ABI_VERSION,
              .expected_atoms = nullptr,
              .expected_atom_count = 0,
              .expected_charge = 0,
              .expected_multiplicity = 0,
              .numeric_bytes = 512ULL << 20,
              .integral_bytes = 64ULL << 20,
              .grid_bytes = 128ULL << 20,
              .response_bytes = 128ULL << 20,
              .ecp_bytes = 128ULL << 20,
              .dispersion_bytes = 64ULL << 20,
              .output_bytes = 64ULL << 20,
              .max_eri_primitives = 100'000'000,
              .max_map_terms = map_work_cap,
              .max_grid_visits = map_work_cap,
              .max_d3_pairs = 16'777'216,
              .restart = 30,
              .max_iterations = 200,
              .gmres_bytes = 64ULL << 20,
              .true_residual_every = 1,
              .stagnation_window = 25,
              .reorthogonalizations = 2,
              .relative_tolerance = 1e-10,
              .absolute_tolerance = 1e-12,
              .breakdown_tolerance = 1e-14,
              .stagnation_tolerance = 1e-14,
              .symmetry_absolute_tolerance = 1e-7,
              .symmetry_relative_tolerance = 1e-7};
  return GENERATIVEQC_STATUS_SUCCESS;
}

namespace {
generativeqc_status dispatch_cpu_hessian(generativeqc_batch* batch, std::size_t index,
                                         const generativeqc_ks_host_hessian_options_v1* options,
                                         const double* direction, std::size_t direction_count,
                                         double* output, std::size_t output_count,
                                         generativeqc_ks_host_hessian_receipt_v1* receipt,
                                         std::size_t recycle_width, bool sequential = false,
                                         bool external_batch = false) {
  if (!batch || !batch->context || !batch->plan || !options || !receipt || !output)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  if (!generativeqc::api::valid_descriptor(options) ||
      !generativeqc::api::valid_descriptor(receipt))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  try {
    if (index >= batch->atom_counts.size() || !batch->atom_counts[index])
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    const auto dimension = size_mul(3, batch->atom_counts[index]);
    if (recycle_width > dimension) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    if (external_batch && (!direction || !direction_count || direction_count % dimension))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    if (!options->expected_atoms || options->expected_atom_count != batch->atom_counts[index] ||
        options->expected_multiplicity < 1 ||
        (!external_batch && (direction ? direction_count != dimension : direction_count != 0)) ||
        output_count != (direction ? direction_count : size_mul(dimension, dimension)) ||
        !options->numeric_bytes || !options->integral_bytes || !options->grid_bytes ||
        !options->response_bytes || !options->ecp_bytes || !options->dispersion_bytes ||
        !options->output_bytes || !options->max_eri_primitives || !options->max_map_terms ||
        !options->max_grid_visits || !options->max_d3_pairs || !options->restart ||
        !options->max_iterations || !options->reorthogonalizations || !options->gmres_bytes ||
        !options->true_residual_every || !options->stagnation_window ||
        !std::isfinite(options->relative_tolerance) || options->relative_tolerance < 0 ||
        options->relative_tolerance >= 1 || !std::isfinite(options->absolute_tolerance) ||
        options->absolute_tolerance < 0 ||
        !(options->relative_tolerance > 0 || options->absolute_tolerance > 0) ||
        !std::isfinite(options->breakdown_tolerance) || options->breakdown_tolerance < 0 ||
        !std::isfinite(options->stagnation_tolerance) || options->stagnation_tolerance < 0 ||
        !std::isfinite(options->symmetry_absolute_tolerance) ||
        options->symmetry_absolute_tolerance < 0 ||
        !std::isfinite(options->symmetry_relative_tolerance) ||
        options->symmetry_relative_tolerance < 0)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    struct Buffer {
      const void* data;
      std::size_t bytes;
    };
    const Buffer buffers[]{{options, sizeof(*options)},
                           {receipt, sizeof(*receipt)},
                           {direction, size_mul(direction_count, sizeof(double))},
                           {output, size_mul(output_count, sizeof(double))},
                           {options->expected_atoms,
                            size_mul(options->expected_atom_count, sizeof(generativeqc_atom))}};
    for (std::size_t first = 0; first < std::size(buffers); ++first)
      for (std::size_t second = first + 1; second < std::size(buffers); ++second)
        if (overlap(buffers[first].data, buffers[first].bytes, buffers[second].data,
                    buffers[second].bytes))
          return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    if (direction && !std::all_of(direction, direction + direction_count,
                                  [](double value) { return std::isfinite(value); }))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
#if GENERATIVEQC_HAS_HOST_HESSIAN
    auto host_workspace =
        batch->host_lease ? batch->host_lease->use_workspace() : std::unique_lock<std::mutex>{};
    const auto result = execute(*batch, index, *options, {direction, direction_count},
                                {output, output_count}, receipt, recycle_width, sequential);
    *receipt = result;
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    (void)sequential;
    batch->context->last_detail =
        "CPU Hessian requires a linked native host AOT package; no fallback";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}
}  // namespace

extern "C" generativeqc_status generativeqc_ks_hessian_cpu_v1(
    generativeqc_batch* batch, std::size_t index,
    const generativeqc_ks_host_hessian_options_v1* options, const double* direction,
    std::size_t direction_count, double* output, std::size_t output_count,
    generativeqc_ks_host_hessian_receipt_v1* receipt) {
  return dispatch_cpu_hessian(batch, index, options, direction, direction_count, output,
                              output_count, receipt, 0);
}

extern "C" generativeqc_status generativeqc_ks_hessian_cpu_recycled_v1(
    generativeqc_batch* batch, std::size_t index,
    const generativeqc_ks_host_hessian_options_v1* options, std::size_t block_size, double* output,
    std::size_t output_count, generativeqc_ks_host_hessian_receipt_v1* receipt) {
  if (block_size < 2 || block_size > 4) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  return dispatch_cpu_hessian(batch, index, options, nullptr, 0, output, output_count, receipt,
                              block_size);
}

extern "C" generativeqc_status generativeqc_ks_hessian_cpu_sequential_v1(
    generativeqc_batch* batch, std::size_t index,
    const generativeqc_ks_host_hessian_options_v1* options, double* output,
    std::size_t output_count, generativeqc_ks_host_hessian_receipt_v1* receipt) {
  return dispatch_cpu_hessian(batch, index, options, nullptr, 0, output, output_count, receipt, 0,
                              true);
}

extern "C" generativeqc_status generativeqc_ks_hvp_cpu_batch_v1(
    generativeqc_batch* batch, std::size_t index,
    const generativeqc_ks_host_hessian_options_v1* options, const double* directions,
    std::size_t direction_count, double* output, std::size_t output_count,
    generativeqc_ks_host_hessian_receipt_v1* receipt) {
  return dispatch_cpu_hessian(batch, index, options, directions, direction_count, output,
                              output_count, receipt, 0, false, true);
}

extern "C" generativeqc_status generativeqc_ks_orbitals_cpu_v1(
    generativeqc_batch* batch, std::size_t index,
    const generativeqc_ks_host_hessian_options_v1* options, double* output,
    std::size_t output_count, generativeqc_ks_host_orbitals_receipt_v1* receipt) {
  if (!batch || !batch->context || !batch->plan || !options || !receipt)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  if (!generativeqc::api::valid_descriptor(options) ||
      !generativeqc::api::valid_descriptor(receipt))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  try {
    if (index >= batch->atom_counts.size() || !options->expected_atoms ||
        options->expected_atom_count != batch->atom_counts[index] ||
        options->expected_multiplicity < 1 || !options->numeric_bytes || (!output && output_count))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    struct Buffer {
      const void* data;
      std::size_t bytes;
    };
    const Buffer buffers[]{{options, sizeof(*options)},
                           {receipt, sizeof(*receipt)},
                           {output, size_mul(output_count, sizeof(double))},
                           {options->expected_atoms,
                            size_mul(options->expected_atom_count, sizeof(generativeqc_atom))}};
    for (std::size_t first = 0; first < std::size(buffers); ++first)
      for (std::size_t second = first + 1; second < std::size(buffers); ++second)
        if (overlap(buffers[first].data, buffers[first].bytes, buffers[second].data,
                    buffers[second].bytes))
          return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
#if GENERATIVEQC_HAS_HOST_HESSIAN
    auto host_workspace = batch->host_lease ? batch->host_lease->use_workspace()
                                             : std::unique_lock<std::mutex>{};
    dft::CudaKsFinalStateToken token;
    std::string detail;
    require(methods::detail::dft_final_state_token(*batch->plan, index, token, detail), detail);
    methods::detail::KsHostDerivativeSources source;
    require(methods::detail::borrow_dft_host_derivative_sources(*batch->plan, index, token, source,
                                                                detail),
            detail);
    for (const auto& buffer : buffers) reject_sources(buffer.data, buffer.bytes, source);
    validate_sources(*batch, index, token, source, *options);
    const auto basis = source.basis->nao;
    const auto spins = token.identity.model.spins;
    const auto& occupied = token.identity.determinant.occupied;
    if ((spins != 1 && spins != 2) || occupied.size() != spins ||
        std::any_of(occupied.begin(), occupied.end(), [&](auto count) { return count > basis; }))
      throw std::invalid_argument("CPU orbital frame has invalid spin occupations");
    generativeqc_ks_host_orbitals_receipt_v1 result;
    result.basis = basis;
    result.spins = spins;
    result.alpha_occupied = occupied[0];
    result.beta_occupied = spins == 2 ? occupied[1] : 0;
    const auto capacity = api::host_orbital_capacity(basis, spins);
    result.output_elements = capacity.output_elements;
    result.numeric_reservation_bytes = capacity.numeric_bytes;
    budget(result.numeric_reservation_bytes, options->numeric_bytes,
           "CPU orbital frame numerical reservation exceeds root overlap budget");
    if (batch->cpu_hessian_workspace_bytes &&
        api::host_orbital_workspace_bytes(basis, spins) > batch->cpu_hessian_workspace_bytes)
      throw methods::MethodError(GENERATIVEQC_STATUS_RESOURCE_LIMIT,
                                 "CPU orbital export exceeds admitted Hessian host workspace");
    if (output) {
      if (output_count != result.output_elements)
        throw std::invalid_argument("CPU orbital frame output shape disagrees with its owner");
      dft::VerifiedKsFinalState frame;
      require(
          methods::detail::read_dft_final_state(*batch->plan, index, token, false, frame, detail),
          detail);
      if (frame.identity != token.identity || frame.orbitals.size() != spins)
        throw std::invalid_argument("CPU orbital frame identity changed");
      std::vector<double> candidate;
      candidate.reserve(result.output_elements);
      for (const auto& orbital : frame.orbitals) {
        if (orbital.vectors.size() != size_mul(basis, basis) ||
            !std::all_of(orbital.vectors.begin(), orbital.vectors.end(),
                         [](double value) { return std::isfinite(value); }))
          throw std::invalid_argument("CPU orbital frame is incomplete or nonfinite");
        candidate.insert(candidate.end(), orbital.vectors.begin(), orbital.vectors.end());
      }
      validate_sources(*batch, index, token, source, *options);
      std::copy(candidate.begin(), candidate.end(), output);
    }
    *receipt = result;
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    batch->context->last_detail = "CPU orbitals require a linked native host AOT package";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}
