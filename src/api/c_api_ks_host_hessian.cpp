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
#include "runtime/bounded_workspace.hpp"
#include "runtime/df_progress_trace.hpp"

#if GENERATIVEQC_HAS_HOST_HESSIAN
#include "dft/host_ks_nuclear_hessian.hpp"
#include "methods/dft_method.hpp"
#include "methods/named_ks_catalog.hpp"
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

/** Compare generated programs, not display names or a duplicated B3LYP model. */
bool linked_program(std::string_view identity) {
  if (dft::host_hessian_aot::find_method(identity)) return true;
  const auto programs = methods::named_ks::methods();
  const auto found = std::find_if(programs.begin(), programs.end(), [&](const auto& program) {
    return program.identity == identity;
  });
  if (found == programs.end() || !found->cpu || found->bound_basis || found->nonlocal.variant ||
      found->range_omega)
    return false;
  const auto& correction = found->dispersion;
  if (correction.kind &&
      (correction.kind != 1 || correction.s9 || correction.cn_cutoff || correction.pair_cutoff ||
       correction.pair_switch_width || correction.composite_profile))
    return false;
  for (const auto& binding : dft::host_hessian_aot::method_bindings()) {
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
                   }))
      return true;
  }
  return false;
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

generativeqc_ks_host_hessian_receipt_v1 execute(
    generativeqc_batch& batch, std::size_t index,
    const generativeqc_ks_host_hessian_options_v1& options, std::span<const double> external,
    std::span<double> output, const generativeqc_ks_host_hessian_receipt_v1* receipt) {
  const auto started = std::chrono::steady_clock::now();
  const bool matrix = external.empty();
  const auto dimension = size_mul(3, batch.atom_counts[index]);
  const auto columns = matrix ? dimension : 1;
  runtime::df_progress::Scope progress(matrix ? "cpu_hessian" : "cpu_hessian_vector");
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
  reject_sources(&options, sizeof(options), source);
  reject_sources(external.data(), external.size_bytes(), source);
  reject_sources(options.expected_atoms,
                 size_mul(options.expected_atom_count, sizeof(generativeqc_atom)), source);
  const dft::host_hessian_aot::MethodBinding* binding{};
  for (const auto& candidate : dft::host_hessian_aot::method_bindings())
    if (candidate.functional == token.identity.model.functional &&
        candidate.scf_domain == token.identity.model.scf_domain_version &&
        candidate.spins == token.identity.model.spins)
      binding = &candidate;
  if (!binding)
    throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                               "CPU Hessian has no exact linked electronic MethodIR");
  const auto atoms = source.system->atoms.size(), basis = source.basis->nao;
  const auto spins = token.identity.model.spins;
  runtime::df_progress::number("atoms", atoms);
  runtime::df_progress::number("basis", basis);
  if (size_mul(atoms, 3) != dimension)
    throw std::invalid_argument("CPU Hessian batch and native geometry disagree");
  const auto validate = [&] { validate_sources(batch, index, token, source, options); };
  validate();
  std::optional<dft::dispersion::D3Parameters> bj;
  const auto& correction = *source.dispersion;
  if (correction.kind) {
    if (correction.kind != 1 || correction.s9 || correction.cn_cutoff || correction.pair_cutoff ||
        correction.pair_switch_width || correction.composite_profile)
      throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                                 "CPU Hessian requires uncut two-body D3(BJ), ATM off");
    bj = dft::dispersion::D3Parameters{correction.s6, correction.s8, correction.a1, correction.a2,
                                       correction.s9};
  }
  const auto first = dft::HostKsNuclearIntegrals::plan(*source.basis, spins);
  const auto curvature = dft::HostKsNuclearIntegralCurvature::plan(
      *source.basis, spins, dft::HostNuclearIntegralCurvatureSchedule::Column);
  const auto curvature_matrix =
      matrix ? dft::HostKsNuclearIntegralCurvature::plan(
                   *source.basis, spins, dft::HostNuclearIntegralCurvatureSchedule::Matrix)
             : dft::HostNuclearIntegralPlan{};
  const bool projected_basis =
      matrix && dft::HostKsNuclearIntegralCurvature::projected_matrix_supported(*source.basis, {});
  const auto projected_matrix =
      projected_basis
          ? dft::HostKsNuclearIntegralCurvature::plan(
                *source.basis, spins, dft::HostNuclearIntegralCurvatureSchedule::ProjectedMatrix)
          : curvature_matrix;
  budget(size_add(first.numeric_bytes, curvature.numeric_bytes), options.integral_bytes,
         "CPU Hessian integral storage cap exceeded");
  const auto curvature_lower =
      matrix ? std::min(size_mul(columns, curvature.eri_primitives),
                        std::min(curvature_matrix.eri_primitives, projected_matrix.eri_primitives))
             : size_mul(columns, curvature.eri_primitives);
  budget(curvature_lower, options.max_eri_primitives,
         "CPU Hessian complete nuclear primitive-work cap exceeded");
  budget(matrix ? std::min(size_mul(columns, curvature.map_terms), curvature_matrix.map_terms)
                : size_mul(columns, curvature.map_terms),
         options.max_map_terms, "CPU Hessian complete integral-map work cap exceeded");
  const auto first_ecp = integrals::ecp_direction_plan(*source.system).limits;
  const auto second_ecp = integrals::ecp_curvature_plan(*source.system).limits;
  budget(std::max(first_ecp.workspace_bytes, second_ecp.workspace_bytes), options.ecp_bytes,
         "CPU Hessian ECP storage cap exceeded");
  const auto d3_plan = dft::dispersion::HostD3Column::plan(atoms);
  if (bj) {
    budget(d3_plan.numeric_bytes, options.dispersion_bytes, "CPU Hessian D3 storage cap exceeded");
    budget(size_mul(columns, d3_plan.pair_visits), options.max_d3_pairs,
           "CPU Hessian complete D3 pair-work cap exceeded");
  }
  const auto fields = size_add(2, size_mul(2, dimension));
  const auto panel_bytes =
      size_mul(size_add(size_mul(2, output.size()), size_add(dimension, fields)), sizeof(double));
  budget(panel_bytes, options.output_bytes, "CPU Hessian output/staging cap exceeded");
  const auto frame_bytes = size_mul(
      size_mul(spins, size_add(size_mul(4, size_mul(basis, basis)), basis)), sizeof(double));
  const auto assembly_bytes = dft::HostKsNuclearHessian::assembly_bytes(atoms, basis, spins);
  auto reservation = size_add(size_add(frame_bytes, panel_bytes), assembly_bytes);
  reservation = size_add(reservation, options.response_bytes);
  reservation = size_add(reservation, options.grid_bytes);
  reservation = size_add(reservation, curvature.numeric_bytes);
  reservation = size_add(reservation, second_ecp.workspace_bytes);
  if (bj) reservation = size_add(reservation, d3_plan.numeric_bytes);
  budget(reservation, options.numeric_bytes,
         "CPU Hessian aggregate numerical reservation exceeded");
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
  budget(size_mul(columns, size_add(xc_first.map_terms, xc_second.map_terms)),
         options.max_map_terms, "CPU Hessian complete XC map-work cap exceeded");
  budget(size_mul(columns, size_add(xc_first.ao_point_visits, xc_second.ao_point_visits)),
         options.max_grid_visits, "CPU Hessian complete moving-grid work cap exceeded");
  dft::VerifiedKsFinalState state;
  require(methods::detail::read_dft_final_state(*batch.plan, index, token, true, state, detail),
          detail);
  dft::HostKsHessianLimits limits;
  limits.max_numeric_bytes = options.numeric_bytes - frame_bytes - panel_bytes - geometry_bytes;
  // Optional coordinate batching uses only already available output/aggregate
  // headroom. An exact original budget retains the scalar column schedule.
  limits.max_assembly_numeric_bytes = size_add(
      assembly_bytes,
      matrix ? std::min(options.output_bytes - panel_bytes, options.numeric_bytes - reservation)
             : 0);
  limits.max_nuclear_pairs =
      atoms % 2 ? size_mul(atoms, (atoms - 1) / 2) : size_mul(atoms / 2, atoms - 1);
  limits.response.max_numeric_bytes = options.response_bytes;
  limits.response.perturbation_map_terms = limits.response.assembly_map_terms =
      options.max_map_terms;
  limits.response.integrals.limits = first;
  if (matrix)
    limits.response.coordinate_source_bytes =
        options.integral_bytes - first.numeric_bytes - curvature.numeric_bytes;
  limits.response.xc = xc_first;
  limits.response.ecp = first_ecp;
  limits.integrals.limits = curvature;
  limits.xc = xc_second;
  limits.ecp = second_ecp;
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
  const bool coordinate_sources = matrix && owner.admit_coordinate_sources(solver);
  const auto base_reserved = owner.resources(solver).reserved_numeric_bytes;
  std::size_t remaining_integral{};
  if (coordinate_sources) {
    const auto panel = dft::HostKsNuclearIntegrals::plan(
        *source.basis, spins, dft::HostNuclearEriSchedule::SymmetryUnique,
        dft::HostNuclearFirstSchedule::CoordinatePanel);
    remaining_integral = options.integral_bytes - first.numeric_bytes - curvature.numeric_bytes -
                         panel.numeric_bytes;
  }
  const dft::HostNuclearIntegralPlan matrix_allowance{
      std::min(remaining_integral, options.numeric_bytes - reservation),
      curvature_matrix.one_electron_primitives, options.max_eri_primitives, options.max_map_terms};
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
  budget(size_add(size_mul(first_passes, first.eri_primitives), curvature_visits),
         options.max_eri_primitives, "CPU Hessian complete nuclear primitive-work cap exceeded");
  const auto curvature_terms =
      matrix_integrals
          ? size_add(selected_matrix.map_terms,
                     size_mul(columns, dft::HostKsNuclearIntegrals::density_response_terms(
                                           *source.basis, spins)))
          : size_mul(columns, curvature.map_terms);
  budget(size_add(size_mul(first_passes, first.map_terms), curvature_terms), options.max_map_terms,
         "CPU Hessian complete integral-map work cap exceeded");
  const auto response_width = matrix ? owner.admit_response_batch(solver) : 1;
  owner.admit_response_cache(solver);
  const auto resources = owner.resources(solver);
  if (!resources.admitted)
    throw std::length_error("CPU Hessian response/GMRES workspace admission failed");
  runtime::df_progress::number("admitted_first_eri_primitives",
                               size_mul(first_passes, first.eri_primitives));
  runtime::df_progress::number("admitted_curvature_eri_primitives", curvature_visits);
  runtime::df_progress::number(
      "admitted_xc_map_terms",
      size_mul(columns, size_add(xc_first.map_terms, xc_second.map_terms)));
  const char* integral_schedule = "column";
  if (matrix_integrals) integral_schedule = projected_integrals ? "projected_matrix" : "matrix";
  runtime::df_progress::label("frozen_integral_schedule", integral_schedule);
  runtime::df_progress::number("independent_response_width", response_width);
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
  reservation = size_add(reservation, resources.reserved_numeric_bytes - base_reserved);
  std::vector<double> right(dimension), column(fields), candidate(output.size());
  if (!matrix) std::copy(external.begin(), external.end(), right.begin());
  generativeqc_ks_host_hessian_receipt_v1 result;
  result.numeric_reservation_bytes = reservation;
  result.frame_bytes = frame_bytes;
  result.geometry_bytes = geometry_bytes;
  result.output_bytes = panel_bytes;
  result.fixed_bound_bytes = resources.fixed_bound_bytes;
  result.gmres_workspace_bytes = resources.gmres_bytes;
  const auto require_response = [](const dft::HostKsHessianDiagnostic& diagnostic,
                                   std::size_t coordinate) {
    if (!diagnostic.converged()) {
      std::ostringstream message;
      message.precision(17);
      message << "CPU Hessian nuclear response failed at column " << coordinate
              << "; perturbation status=" << static_cast<int>(diagnostic.response.status)
              << "; GMRES status=" << static_cast<int>(diagnostic.response.response.status)
              << "; true physical residual=" << diagnostic.response.physical_residual_norm
              << "; relative physical residual=" << diagnostic.response.relative_physical_residual;
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
        group = owner.apply_coordinates(coordinate, std::min(response_width, columns - coordinate),
                                        solver);
        for (std::size_t index = 0; index < group.size(); ++index)
          require_response(group[index], coordinate + index);
      }
      diagnostic = group[local];
      values = owner.coordinate_column(local);
    } else {
      if (matrix) {
        std::fill(right.begin(), right.end(), 0);
        right[coordinate] = 1;
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
      candidate[matrix ? row * dimension + coordinate : row] = values[2 + dimension + row];
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
    for (std::size_t row = 0; row < dimension; ++row)
      for (std::size_t coordinate = 0; coordinate < row; ++coordinate)
        result.raw_symmetry_error =
            std::max(result.raw_symmetry_error, std::abs(candidate[row * dimension + coordinate] -
                                                         candidate[coordinate * dimension + row]));
    if (result.raw_symmetry_error >
        options.symmetry_absolute_tolerance + options.symmetry_relative_tolerance * maximum)
      throw methods::MethodError(
          GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
          "CPU Hessian raw symmetry gate failed; no matrix repair is applied");
  }
  validate();
  if (!matrix && !std::equal(right.begin(), right.end(), external.begin()))
    throw std::invalid_argument("CPU Hessian external direction changed before publication");
  result.execution_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  std::copy(candidate.begin(), candidate.end(), output.begin());
  return result;
}
#endif
}  // namespace

extern "C" std::uint32_t generativeqc_ks_hessian_cpu_version_v1() {
  return GENERATIVEQC_HAS_HOST_HESSIAN ? 1 : 0;
}

extern "C" std::uint32_t generativeqc_ks_hessian_cpu_method_v1(const char* identity) {
#if GENERATIVEQC_HAS_HOST_HESSIAN
  return identity && linked_program(identity) ? 1 : 0;
#else
  (void)identity;
  return 0;
#endif
}

extern "C" generativeqc_status generativeqc_ks_hessian_cpu_v1(
    generativeqc_batch* batch, std::size_t index,
    const generativeqc_ks_host_hessian_options_v1* options, const double* direction,
    std::size_t direction_count, double* output, std::size_t output_count,
    generativeqc_ks_host_hessian_receipt_v1* receipt) {
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
    if (!options->expected_atoms || options->expected_atom_count != batch->atom_counts[index] ||
        options->expected_multiplicity < 1 ||
        (direction ? direction_count != dimension : direction_count != 0) ||
        output_count != (direction ? dimension : size_mul(dimension, dimension)) ||
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
    const auto result = execute(*batch, index, *options, {direction, direction_count},
                                {output, output_count}, receipt);
    *receipt = result;
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    batch->context->last_detail =
        "CPU Hessian requires a linked native host AOT package; no fallback";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
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
    result.output_elements = size_mul(spins, size_mul(basis, basis));
    result.numeric_reservation_bytes = size_mul(
        size_add(size_mul(5, result.output_elements), size_mul(spins, basis)), sizeof(double));
    budget(result.numeric_reservation_bytes, options->numeric_bytes,
           "CPU orbital frame numerical reservation exceeds root overlap budget");
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
