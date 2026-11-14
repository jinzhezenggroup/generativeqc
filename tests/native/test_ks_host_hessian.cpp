#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "api/handles.hpp"
#include "api/ks_host_hessian.hpp"
#include "dft/host_gga_fields.hpp"
#include "dft/host_nuclear_integrals.hpp"
#include "methods/dft_method.hpp"
#include "methods/named_ks_catalog.hpp"

namespace {
using namespace generativeqc;

void require(bool condition, const std::string& detail) {
  if (!condition) throw std::runtime_error(detail);
}

/** Every derivative is requested through the bridge on a public prepared batch.
 * Compact H2/H3 controls exercise ownership and composition, not heavy chemistry.
 */
class NativeBatch {
 public:
  generativeqc_context* context{};
  generativeqc_system* system{};
  generativeqc_batch* batch{};
  std::size_t atoms{}, dimension{};
  std::vector<generativeqc_atom> centers;
  generativeqc_ks_host_hessian_options_v1 options;

  NativeBatch(unsigned spins, bool bj, bool polarization = false)
      : atoms(spins + 1), dimension(3 * atoms) {
    generativeqc_context_descriptor context_options{};
    context_options.struct_size = sizeof(context_options);
    context_options.abi_version = GENERATIVEQC_ABI_VERSION;
    context_options.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
    check(generativeqc_context_create(&context_options, &context));
    std::vector<generativeqc_shell> shells;
    std::vector<generativeqc_primitive> primitives;
    for (std::uint32_t atom = 0; atom < atoms; ++atom) {
      centers.push_back({1, .17 * atom * atom, -.08 * atom, 1.6 * atom});
      for (const auto exponent : {1.2, .35}) {
        shells.push_back({atom, 0, static_cast<std::uint32_t>(primitives.size()), 1});
        primitives.push_back({exponent, 1});
      }
      if (polarization && atom == 0) {
        shells.push_back({atom, 2, static_cast<std::uint32_t>(primitives.size()), 1});
        primitives.push_back({.7, 1});
      }
    }
    generativeqc_system_descriptor geometry{};
    geometry.struct_size = sizeof(geometry);
    geometry.abi_version = GENERATIVEQC_ABI_VERSION;
    geometry.atoms = centers.data();
    geometry.atom_count = centers.size();
    geometry.shells = shells.data();
    geometry.shell_count = shells.size();
    geometry.primitives = primitives.data();
    geometry.primitive_count = primitives.size();
    geometry.multiplicity = spins;
    geometry.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
    check(generativeqc_system_create(context, &geometry, &system));
    options.expected_atoms = centers.data();
    options.expected_atom_count = centers.size();
    options.expected_multiplicity = spins;
    const auto name = std::string(bj ? "b3lyp-d3(bj)" : "b3lyp") + (spins == 1 ? "-rks" : "-uks");
    const auto* named = methods::named_ks::find_method(name);
    require(named && named->cpu, "missing exact native total-method program");
    require(generativeqc_ks_hessian_cpu_method_v1(named->identity.data()),
            "CPU query omitted an exact native total-method projection");
    generativeqc_ks_options grid{};
    grid.struct_size = sizeof(grid);
    grid.abi_version = GENERATIVEQC_ABI_VERSION;
    grid.grid_version = 1;
    grid.radial_points = 12;
    grid.angular_polar = 4;
    grid.angular_azimuth = 8;
    grid.partition_iterations = 3;
    grid.coincident_tolerance = 1e-12;
    grid.tile_points = 64;
    generativeqc_method_descriptor method{};
    method.struct_size = sizeof(method);
    method.abi_version = GENERATIVEQC_ABI_VERSION;
    method.method = named->carrier;
    method.max_iterations = 200;
    method.diis_history = 8;
    method.energy_tolerance = 1e-12;
    method.density_tolerance = 1e-10;
    method.screening_tolerance = 1e-14;
    method.ks_options = &grid;
    generativeqc_named_ks_descriptor selection{sizeof(selection), GENERATIVEQC_ABI_VERSION,
                                               name.c_str(), named->identity.data(), 16U << 20};
    const generativeqc_system* systems[]{system};
    check(generativeqc_batch_prepare_named_ks_v1(context, systems, 1, &method, &selection,
                                                 GENERATIVEQC_BATCH_ENABLE_WARM_STARTS, &batch));
  }

  ~NativeBatch() {
    if (batch) generativeqc_batch_destroy(batch);
    if (system) generativeqc_system_destroy(system);
    if (context) generativeqc_context_destroy(context);
  }

  void check(generativeqc_status status) const {
    require(status == GENERATIVEQC_STATUS_SUCCESS,
            context ? context->last_detail : "native context creation failed");
  }

  void run() const {
    generativeqc_batch_item_result_descriptor result{};
    result.struct_size = sizeof(result);
    result.abi_version = GENERATIVEQC_ABI_VERSION;
    check(generativeqc_batch_execute(batch, nullptr, 0, &result, 1));
    check(result.status);
    require(result.converged, "actual native batch did not converge");
  }

  generativeqc_status apply(const double* direction, std::vector<double>& output,
                            generativeqc_ks_host_hessian_receipt_v1& receipt) {
    return generativeqc_ks_hessian_cpu_v1(batch, 0, &options, direction, direction ? dimension : 0,
                                          output.data(), output.size(), &receipt);
  }
};

void orbital_case(NativeBatch& fixture, unsigned spins) {
  generativeqc_ks_host_orbitals_receipt_v1 receipt;
  fixture.check(
      generativeqc_ks_orbitals_cpu_v1(fixture.batch, 0, &fixture.options, nullptr, 0, &receipt));
  const auto basis = receipt.basis;
  require(receipt.spins == spins && basis == 2 * fixture.atoms &&
              receipt.output_elements == spins * basis * basis &&
              receipt.alpha_occupied == (spins == 1 ? 1 : 2) &&
              receipt.beta_occupied == (spins == 1 ? 0 : 1),
          "native orbital query has wrong basis/spin/occupation semantics");
  std::vector<double> orbitals(receipt.output_elements, -613.25);
  const auto original = fixture.options;
  const auto refusal = [&] {
    const auto before = orbitals;
    const auto before_receipt = receipt;
    require(
        generativeqc_ks_orbitals_cpu_v1(fixture.batch, 0, &fixture.options, orbitals.data(),
                                        orbitals.size(), &receipt) != GENERATIVEQC_STATUS_SUCCESS &&
            orbitals == before && std::memcmp(&receipt, &before_receipt, sizeof(receipt)) == 0,
        "failed native orbital export published a frame/receipt");
  };
  fixture.options.numeric_bytes = receipt.numeric_reservation_bytes - 1;
  refusal();
  fixture.options = original;
  fixture.centers[0].x += .01;
  refusal();
  fixture.centers[0].x -= .01;
  fixture.options.expected_multiplicity += 1;
  refusal();
  fixture.options = original;
  fixture.check(generativeqc_ks_orbitals_cpu_v1(fixture.batch, 0, &fixture.options, orbitals.data(),
                                                orbitals.size(), &receipt));
  dft::CudaKsFinalStateToken token;
  std::string detail;
  fixture.check(methods::detail::dft_final_state_token(*fixture.batch->plan, 0, token, detail));
  methods::detail::KsHostDerivativeSources sources;
  fixture.check(methods::detail::borrow_dft_host_derivative_sources(*fixture.batch->plan, 0, token,
                                                                    sources, detail));
  const auto& metric = sources.fock->one_electron().overlap;
  for (std::size_t spin = 0; spin < spins; ++spin)
    for (std::size_t first = 0; first < basis; ++first)
      for (std::size_t second = 0; second < basis; ++second) {
        double product{};
        for (std::size_t row = 0; row < basis; ++row)
          for (std::size_t column = 0; column < basis; ++column)
            product += orbitals[spin * basis * basis + row * basis + first] *
                       metric[row * basis + column] *
                       orbitals[spin * basis * basis + column * basis + second];
        require(std::abs(product - (first == second ? 1 : 0)) < 1e-10,
                "native orbital output is not row-major metric-orthonormal C");
      }
  require(generativeqc_ks_orbitals_cpu_v1(
              fixture.batch, 0, &fixture.options,
              const_cast<double*>(sources.fock->one_electron().overlap.data()), orbitals.size(),
              &receipt) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "native orbital export accepted a borrowed source alias");
}

void endpoint_case(unsigned spins, bool bj) {
  NativeBatch fixture(spins, bj);
  const auto dimension = fixture.dimension;
  std::vector<double> matrix(dimension * dimension, -781.25);
  generativeqc_ks_host_hessian_receipt_v1 receipt;
  receipt.total_energy = -781.25;
  const auto refusal = [&] {
    const auto before = matrix;
    const auto before_receipt = receipt;
    require(fixture.apply(nullptr, matrix, receipt) != GENERATIVEQC_STATUS_SUCCESS,
            "invalid endpoint unexpectedly succeeded");
    require(matrix == before && std::memcmp(&receipt, &before_receipt, sizeof(receipt)) == 0,
            "refused endpoint published a partial result/receipt");
  };
  refusal();
  fixture.run();
  orbital_case(fixture, spins);
  const auto original = fixture.options;
  fixture.options.max_eri_primitives = 1;
  refusal();
  fixture.options = original;
  fixture.centers[0].x += .01;
  refusal();
  fixture.centers[0].x -= .01;
  fixture.options.expected_charge += 1;
  refusal();
  fixture.options = original;
  fixture.options.numeric_bytes = 1;
  refusal();
  fixture.options = original;
  fixture.options.output_bytes = 1;
  refusal();
  fixture.options = original;
  fixture.check(fixture.apply(nullptr, matrix, receipt));
  require(receipt.completed_directions == dimension && receipt.induced_fock_actions &&
              receipt.first_eri_primitives && receipt.curvature_eri_primitives &&
              receipt.source_xc_point_visits && receipt.response_xc_point_visits &&
              receipt.raw_symmetry_error < 1e-7,
          "native endpoint has incomplete electronic work provenance");
  require((receipt.d3_pair_visits > 0) == bj && (receipt.d3_energy < 0) == bj,
          "native endpoint omitted or invented its total-method correction");
  std::vector<double> direction(dimension), product(dimension);
  for (std::size_t coordinate = 0; coordinate < dimension; ++coordinate)
    direction[coordinate] = .02 * std::sin(1.0 + coordinate);
  generativeqc_ks_host_hessian_receipt_v1 hvp;
  fixture.check(fixture.apply(direction.data(), product, hvp));
  require(hvp.completed_directions == 1, "HVP replayed a full matrix");
  require(receipt.d3_pair_visits == dimension * hvp.d3_pair_visits &&
              receipt.d3_reference_terms == dimension * hvp.d3_reference_terms,
          "full-request D3 work receipt reports only its final column");
  dft::CudaKsFinalStateToken reuse_token;
  std::string reuse_detail;
  fixture.check(
      methods::detail::dft_final_state_token(*fixture.batch->plan, 0, reuse_token, reuse_detail));
  methods::detail::KsHostDerivativeSources reuse_sources;
  fixture.check(methods::detail::borrow_dft_host_derivative_sources(
      *fixture.batch->plan, 0, reuse_token, reuse_sources, reuse_detail));
  const auto first_plan = dft::HostKsNuclearIntegrals::plan(*reuse_sources.basis, spins);
  const auto second_plan = dft::HostKsNuclearIntegralCurvature::plan(
      *reuse_sources.basis, spins, dft::HostNuclearIntegralCurvatureSchedule::Column);
  const auto matrix_plan = dft::HostKsNuclearIntegralCurvature::plan(
      *reuse_sources.basis, spins, dft::HostNuclearIntegralCurvatureSchedule::Matrix);
  const auto panel_plan = dft::HostKsNuclearIntegrals::plan(
      *reuse_sources.basis, spins, dft::HostNuclearEriSchedule::SymmetryUnique,
      dft::HostNuclearFirstSchedule::CoordinatePanel);
  const auto selected_cap = first_plan.eri_primitives + matrix_plan.eri_primitives;
  fixture.options.max_eri_primitives = selected_cap;
  std::vector<double> reused(matrix.size()), streamed(matrix.size());
  generativeqc_ks_host_hessian_receipt_v1 reused_receipt, streamed_receipt;
  fixture.check(fixture.apply(nullptr, reused, reused_receipt));
  require(reused_receipt.first_eri_primitives <= first_plan.eri_primitives,
          "matrix endpoint repeated the first primitive sweep for each coordinate");
  require(reused_receipt.curvature_eri_primitives <= matrix_plan.eri_primitives,
          "matrix endpoint repeated global-coordinate integral curvature sweeps");
  fixture.options.integral_bytes =
      first_plan.numeric_bytes + second_plan.numeric_bytes + panel_plan.numeric_bytes;
  refusal();
  fixture.options.max_eri_primitives =
      first_plan.eri_primitives + dimension * second_plan.eri_primitives;
  fixture.check(fixture.apply(nullptr, streamed, streamed_receipt));
  require(streamed_receipt.curvature_eri_primitives > reused_receipt.curvature_eri_primitives &&
              streamed_receipt.first_eri_primitives == reused_receipt.first_eri_primitives &&
              streamed_receipt.fixed_bound_bytes < reused_receipt.fixed_bound_bytes,
          "first-panel-only fallback did not preserve its distinct resource/work contract");
  for (std::size_t element = 0; element < matrix.size(); ++element)
    require(std::abs(streamed[element] - reused[element]) < 2e-8,
            "first-only and full integral reuse endpoints disagree");
  fixture.options.max_eri_primitives = selected_cap;
  fixture.options.integral_bytes = first_plan.numeric_bytes + second_plan.numeric_bytes;
  refusal();
  fixture.options.max_eri_primitives = original.max_eri_primitives;
  fixture.check(fixture.apply(nullptr, streamed, streamed_receipt));
  require(streamed_receipt.fixed_bound_bytes < reused_receipt.fixed_bound_bytes &&
              streamed_receipt.first_eri_primitives > reused_receipt.first_eri_primitives,
          "bounded integral fallback did not release the resident first-source panel");
  for (std::size_t element = 0; element < matrix.size(); ++element)
    require(std::abs(streamed[element] - reused[element]) < 2e-8 &&
                std::abs(matrix[element] - reused[element]) < 2e-8,
            "resident and bounded first-source endpoints disagree");
  fixture.options = original;
  for (std::size_t row = 0; row < dimension; ++row) {
    double expected{};
    for (std::size_t coordinate = 0; coordinate < dimension; ++coordinate)
      expected += matrix[row * dimension + coordinate] * direction[coordinate];
    require(std::abs(expected - product[row]) < 2e-8, "raw Hessian/HVP disagreement");
    for (std::size_t axis = 0; axis < 3; ++axis) {
      double translation{};
      for (std::size_t atom = 0; atom < fixture.atoms; ++atom)
        translation += matrix[row * dimension + 3 * atom + axis];
      require(std::abs(translation) < 2e-7, "native Hessian violates translation invariance");
    }
  }
  const auto before = product;
  require(fixture.apply(product.data(), product, hvp) == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              product == before,
          "native bridge accepted overlapping external panels");
  direction[0] = std::numeric_limits<double>::quiet_NaN();
  require(fixture.apply(direction.data(), product, hvp) == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              product == before,
          "native bridge accepted a nonfinite direction");
  dft::CudaKsFinalStateToken token;
  std::string detail;
  fixture.check(methods::detail::dft_final_state_token(*fixture.batch->plan, 0, token, detail));
  methods::detail::KsHostDerivativeSources sources;
  fixture.check(methods::detail::borrow_dft_host_derivative_sources(*fixture.batch->plan, 0, token,
                                                                    sources, detail));
  const auto source_before = sources.system->atoms;
  require(generativeqc_ks_hessian_cpu_v1(
              fixture.batch, 0, &fixture.options, nullptr, 0,
              const_cast<double*>(reinterpret_cast<const double*>(sources.system->atoms.data())),
              matrix.size(), &hvp) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "native bridge accepted an output alias of its borrowed geometry");
  require(std::memcmp(source_before.data(), sources.system->atoms.data(),
                      source_before.size() * sizeof(source_before[0])) == 0,
          "native bridge corrupted a borrowed source");
  fixture.run();
  fixture.check(fixture.apply(nullptr, matrix, receipt));
  std::cout << "native CPU endpoint spins=" << spins << " bj=" << bj
            << " columns=" << receipt.completed_directions
            << " asymmetry=" << receipt.raw_symmetry_error << " nuclear_eri_visits="
            << receipt.first_eri_primitives + receipt.curvature_eri_primitives
            << " seconds=" << receipt.execution_seconds << '\n';
}
/** A real spherical-d batch must price the selected representation, including
 * an optional-storage fallback whose larger work no longer fits the same cap.
 */
void projected_endpoint_case(unsigned spins) {
  NativeBatch fixture(spins, false, true);
  fixture.run();
  dft::CudaKsFinalStateToken token;
  std::string detail;
  fixture.check(methods::detail::dft_final_state_token(*fixture.batch->plan, 0, token, detail));
  methods::detail::KsHostDerivativeSources sources;
  fixture.check(methods::detail::borrow_dft_host_derivative_sources(*fixture.batch->plan, 0, token,
                                                                    sources, detail));
  const auto first = dft::HostKsNuclearIntegrals::plan(*sources.basis, spins);
  const auto column = dft::HostKsNuclearIntegralCurvature::plan(
      *sources.basis, spins, dft::HostNuclearIntegralCurvatureSchedule::Column);
  const auto panel = dft::HostKsNuclearIntegrals::plan(
      *sources.basis, spins, dft::HostNuclearEriSchedule::SymmetryUnique,
      dft::HostNuclearFirstSchedule::CoordinatePanel);
  const auto original = dft::HostKsNuclearIntegralCurvature::plan(
      *sources.basis, spins, dft::HostNuclearIntegralCurvatureSchedule::Matrix);
  const auto projected = dft::HostKsNuclearIntegralCurvature::plan(
      *sources.basis, spins, dft::HostNuclearIntegralCurvatureSchedule::ProjectedMatrix);
  require(projected.eri_primitives < original.eri_primitives,
          "spherical-d endpoint did not admit a lower-work polynomial representation");
  const auto dimension = fixture.dimension;
  std::vector<double> projected_matrix(dimension * dimension),
      public_matrix(projected_matrix.size());
  generativeqc_ks_host_hessian_receipt_v1 projected_receipt, public_receipt;
  fixture.options.max_eri_primitives = first.eri_primitives + projected.eri_primitives;
  fixture.check(fixture.apply(nullptr, projected_matrix, projected_receipt));
  require(projected_receipt.first_eri_primitives + projected_receipt.curvature_eri_primitives <=
                  fixture.options.max_eri_primitives &&
              projected_receipt.curvature_eri_primitives < original.eri_primitives &&
              projected_receipt.completed_directions == dimension,
          "projected endpoint omitted complete primitive work or matrix columns");
  fixture.options.integral_bytes = first.numeric_bytes + column.numeric_bytes +
                                   panel.numeric_bytes + original.numeric_bytes +
                                   (1 + 2 * dimension + dimension * dimension) * sizeof(double);
  std::fill(public_matrix.begin(), public_matrix.end(), -781.25);
  public_receipt.total_energy = -781.25;
  const auto before = public_receipt;
  require(fixture.apply(nullptr, public_matrix, public_receipt) != GENERATIVEQC_STATUS_SUCCESS,
          "larger-work optional-storage fallback bypassed the complete endpoint work cap");
  require(std::all_of(public_matrix.begin(), public_matrix.end(),
                      [](double value) { return value == -781.25; }) &&
              std::memcmp(&before, &public_receipt, sizeof(before)) == 0,
          "over-cap optional fallback published partial output or work receipts");
  fixture.options.max_eri_primitives = first.eri_primitives + original.eri_primitives;
  fixture.check(fixture.apply(nullptr, public_matrix, public_receipt));
  require(
      public_receipt.curvature_eri_primitives > projected_receipt.curvature_eri_primitives &&
          public_receipt.numeric_reservation_bytes < projected_receipt.numeric_reservation_bytes,
      "bounded public-matrix fallback lost its observable work/storage distinction");
  for (std::size_t index = 0; index < public_matrix.size(); ++index)
    require(std::abs(public_matrix[index] - projected_matrix[index]) < 2e-8,
            "projected/public complete electronic Hessians disagree");
  std::cout << "projected CPU endpoint spins=" << spins
            << " projected_eri=" << projected_receipt.curvature_eri_primitives
            << " public_eri=" << public_receipt.curvature_eri_primitives << '\n';
}
}  // namespace

int main() {
  try {
    const auto subnormal = std::numeric_limits<double>::denorm_min();
    require(dft::host_gga_spin_fields_admitted(0, {0, subnormal, -subnormal}),
            "physical density underflow incorrectly rejects raw subnormal gradients");
    require(
        !dft::host_gga_spin_fields_admitted(-subnormal, {0, 0, 0}) &&
            !dft::host_gga_spin_fields_admitted(0, {std::numeric_limits<double>::min(), 0, 0}) &&
            !dft::host_gga_spin_fields_admitted(1, {std::numeric_limits<double>::infinity(), 0, 0}),
        "underflow admission weakened an invalid spin-field guard");
    require(generativeqc_ks_hessian_cpu_version_v1() == 1, "missing linked CPU Hessian endpoint");
    require(!generativeqc_ks_hessian_cpu_method_v1(nullptr) &&
                !generativeqc_ks_hessian_cpu_method_v1("forged"),
            "CPU endpoint falsely advertises an unlinked electronic method");
    for (const auto spins : {1U, 2U})
      for (const auto bj : {false, true}) endpoint_case(spins, bj);
    for (const auto spins : {1U, 2U}) projected_endpoint_case(spins);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
