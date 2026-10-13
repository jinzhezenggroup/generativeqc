#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "dft/cuda_ks.hpp"
#include "methods/dft_method.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda/final_validation_kernels.hpp"

namespace {
using namespace generativeqc;

void require(bool accepted, const std::string& message) {
  if (!accepted) throw std::runtime_error(message);
}

core::System hydrogens(unsigned count, bool restricted) {
  core::System system;
  system.multiplicity = restricted ? 1 : 2;
  for (unsigned atom = 0; atom < count; ++atom) {
    system.atoms.push_back({1, {0.15 * atom * atom, 0.13 * atom, 1.5 * atom}});
    system.shells.push_back(
        {atom,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}});
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  return system;
}

/** A heteronuclear, non-axis-aligned s/p fixture with explicit custom primitives. */
core::System polarized_lithium_hydride(bool restricted) {
  core::System system;
  system.atoms = {{3, {0.0, 0.0, 0.0}}, {1, {0.21, -0.12, 3.0}}};
  system.shells = {
      {0, 0, {{4.0, 1.0}}}, {0, 0, {{0.5, 1.0}}}, {0, 1, {{0.4, 1.0}}}, {1, 0, {{1.0, 1.0}}}};
  system.charge = restricted ? 0 : 1;
  system.multiplicity = restricted ? 1 : 2;
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  return system;
}

/** Keep descriptor storage alive while prepared owners copy the typed model. */
struct StationaryOptions {
  std::array<generativeqc_ks_semilocal_component, 2> components;
  generativeqc_ks_exchange_term exchange;
  generativeqc_ks_options ks{};
  generativeqc_method_descriptor descriptor{};

  StationaryOptions(bool restricted, bool hybrid)
      : components{{{"GGA_X_PBE", hybrid ? 0.75 : 1.0}, {"GGA_C_PBE", 1.0}}},
        exchange{GENERATIVEQC_KS_EXCHANGE_FULL_RANGE, 0.25, 0.0, restricted ? -0.125 : -0.25} {
    ks.struct_size = sizeof(ks);
    ks.abi_version = GENERATIVEQC_ABI_VERSION;
    ks.scf_domain = dft::semilocal_family_scf_domain(dft::SemilocalFamily::Pbe);
    ks.grid_version = 1;
    ks.radial_points = 24;
    ks.angular_polar = 12;
    ks.angular_azimuth = 24;
    ks.partition_iterations = 3;
    ks.coincident_tolerance = 1e-12;
    ks.tile_points = 257;
    ks.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
    ks.spin_channels = restricted ? 1 : 2;
    ks.semilocal_components = components.data();
    ks.semilocal_component_count = components.size();
    ks.exchange_terms = hybrid ? &exchange : nullptr;
    ks.exchange_term_count = hybrid ? 1 : 0;
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
    descriptor.method = restricted ? GENERATIVEQC_METHOD_PBE_RKS : GENERATIVEQC_METHOD_PBE_UKS;
    descriptor.max_iterations = 200;
    descriptor.diis_history = 8;
    descriptor.energy_tolerance = 1e-12;
    descriptor.density_tolerance = 1e-10;
    descriptor.screening_tolerance = 1e-12;
    descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
    descriptor.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
    descriptor.ks_options = &ks;
  }
};

void run_case(unsigned atoms, bool restricted, bool hybrid) {
  const auto system = hydrogens(atoms, restricted);
  const dft::AoBasis basis(system);
  const dft::MolecularGrid grid(system, {1, 24, 12, 24, 3, 1e-12});
  scf::FockBuildSpec spec;
  spec.derivative_order = 0;
  spec.spin = restricted ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted;
  spec.exchange.present = hybrid;
  spec.exchange.coefficient = restricted ? -0.125 : -0.25;
  const auto strategy = scf::resolve_fock_build(spec, scf::FockBackend::Cuda, 1e-12);
  const scf::PreparedFockPlan fock(system, nullptr, strategy, 0);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  options.max_iterations = 200;
  options.semilocal_exchange_scale = hybrid ? 0.75 : 1.0;
  dft::CudaKsPlan plan(fock, basis, grid, options, dft::SemilocalFamily::Pbe, 257);
  const auto solve = [&] {
    const auto result = plan.run(nullptr, true, false);
    require(result.converged && !plan.failed(), "resident KS solve did not converge");
  };
  solve();
  dft::CudaKsFinalStateToken token;
  dft::CudaKsResidentStationaryWeightsBinding weights;
  dft::KsFinalStateDiagnostic diagnostic;
  std::string detail;
  require(plan.final_state_token(token, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  const auto prepare = [&] {
    return plan.prepare_resident_final_stationary_weights(token, weights, diagnostic, detail);
  };
  auto stale = token;
  ++stale.identity.model.owner;
  const auto before = plan.transfers();
  require(plan.prepare_resident_final_stationary_weights(stale, weights, diagnostic, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              !weights,
          "stale resident preparation published D/W");
  require(plan.transfers().synchronizations == before.synchronizations &&
              plan.transfers().final_state_d2h_bytes == before.final_state_d2h_bytes,
          "stale resident preparation submitted work");
  require(prepare() == GENERATIVEQC_STATUS_SUCCESS, detail);
  const auto proof = plan.transfers();
  const auto packet_bytes =
      (restricted ? 1U : 2U) * sizeof(scf::cuda_df::ValidationPartial) + sizeof(int);
  require(weights && proof.matrix_d2h_bytes == before.matrix_d2h_bytes &&
              proof.final_state_reads == before.final_state_reads &&
              proof.final_state_d2h_bytes - before.final_state_d2h_bytes == packet_bytes &&
              proof.scalar_d2h_bytes - before.scalar_d2h_bytes == packet_bytes &&
              proof.resident_final_validations == before.resident_final_validations + 1 &&
              proof.synchronizations == before.synchronizations + 2,
          "resident proof exported dense state or omitted scalar transport");
  const auto retained = weights;
  require(prepare() == GENERATIVEQC_STATUS_SUCCESS && weights.density == retained.density &&
              weights.weighted_density == retained.weighted_density &&
              plan.transfers().synchronizations == proof.synchronizations &&
              plan.transfers().final_state_d2h_bytes == proof.final_state_d2h_bytes,
          "repeated resident preparation changed storage or resubmitted proof");
  dft::CudaKsResidentDensityBinding capturing_density;
  require(
      plan.resident_final_density(token, capturing_density, detail) == GENERATIVEQC_STATUS_SUCCESS,
      detail);
  const auto capturing_stream = reinterpret_cast<cudaStream_t>(capturing_density.stream);
  require(cudaStreamBeginCapture(capturing_stream, cudaStreamCaptureModeGlobal) == cudaSuccess,
          "resident capture test did not begin");
  const auto capture_status = prepare();
  cudaGraph_t captured{};
  require(cudaStreamEndCapture(capturing_stream, &captured) == cudaSuccess &&
              cudaGraphDestroy(captured) == cudaSuccess,
          "resident capture test did not end");
  require(capture_status == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !weights &&
              plan.transfers().final_state_d2h_bytes == proof.final_state_d2h_bytes,
          "resident lease admitted capture or exported state on rejection");
  require(prepare() == GENERATIVEQC_STATUS_SUCCESS, detail);
  std::vector<double> total_density(weights.matrix_elements),
      total_weighted(weights.matrix_elements);
  require(
      cudaMemcpy(total_density.data(), weights.density, total_density.size() * sizeof(double),
                 cudaMemcpyDeviceToHost) == cudaSuccess &&
          cudaMemcpy(total_weighted.data(), weights.weighted_density,
                     total_weighted.size() * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess,
      "resident D/W observation failed");
  dft::VerifiedKsFinalState detached;
  require(plan.read_final_state(token, true, detached, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail);
  require(
      weights.density == retained.density && weights.weighted_density == retained.weighted_density,
      "detached read replaced published resident weights");
  for (std::size_t element = 0; element < weights.matrix_elements; ++element) {
    double density = 0, weighted = 0;
    for (std::size_t spin = 0; spin < detached.density.size(); ++spin) {
      density += detached.density[spin][element];
      weighted += detached.weighted_density[spin][element];
    }
    require(std::abs(total_density[element] - density) < 1e-12 &&
                std::abs(total_weighted[element] - weighted) < 1e-11,
            "resident D/W differs from independently computed detached products");
  }

  // Corrupt only the current determinant, not the separately retained warm
  // seed. Failure must revoke force eligibility and permit a real warm retry.
  solve();
  require(prepare() == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !weights,
          "a later solve preserved the old resident token");
  require(plan.final_state_token(token, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  dft::CudaKsResidentDensityBinding density;
  require(plan.resident_final_density(token, density, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail);
  const auto stream = reinterpret_cast<cudaStream_t>(density.stream);
  require(cudaMemsetAsync(const_cast<double*>(density.alpha), 0,
                          density.matrix_elements * sizeof(double), stream) == cudaSuccess,
          "resident failure injection failed");
  require(prepare() == GENERATIVEQC_STATUS_NUMERICAL_FAILURE && !weights &&
              diagnostic.determinant.eigenframes.empty() && plan.has_warm_start(),
          "invalid resident proof published weights or discarded warm recovery");
  require(plan.final_state_token(token, detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "invalid resident proof retained final eligibility");
  solve();
  require(plan.final_state_token(token, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  require(prepare() == GENERATIVEQC_STATUS_SUCCESS, detail);
  plan.invalidate_final_state();
  require(prepare() == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !weights,
          "explicit revocation preserved resident weights");
  std::cout << "resident KS atoms=" << atoms << " restricted=" << restricted << " hybrid=" << hybrid
            << " packet_bytes=" << packet_bytes << '\n';

  struct RestoreLedger {
    std::shared_ptr<runtime::DeviceResourceLedger> previous;
    ~RestoreLedger() { runtime::active_device_resource_ledger = previous; }
  };
  // A public ledger deliberately disables unplanned optional point panels.
  // Query under that same admission contract, not the unbound owner's panels.
  const auto capacity = [&] {
    auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
    ledger->limit = plan.resources().state_device_bytes + plan.resources().xc_device_bytes;
    ledger->device = 0;
    RestoreLedger restore{runtime::active_device_resource_ledger};
    runtime::active_device_resource_ledger = ledger;
    dft::CudaKsPlan bounded(fock, basis, grid, options, dft::SemilocalFamily::Pbe, 257);
    const auto capacity =
        bounded.resources().state_device_bytes + bounded.resources().xc_device_bytes;
    require(ledger->live == capacity, "bounded resident KS query omitted owned allocations");
    return capacity;
  }();
  std::cout << "resident KS owned_numeric_capacity=" << capacity << '\n';
  for (const auto allowance : {capacity - 1, capacity}) {
    auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
    ledger->limit = allowance;
    ledger->device = 0;
    RestoreLedger restore{runtime::active_device_resource_ledger};
    runtime::active_device_resource_ledger = ledger;
    bool failed = false;
    try {
      dft::CudaKsPlan measured(fock, basis, grid, options, dft::SemilocalFamily::Pbe, 257);
      require(ledger->live == capacity, "resident KS capacity omitted owned allocations");
      require(measured.run(nullptr, false, false).converged,
              "measured resident KS did not converge");
      const auto allocations = ledger->allocations;
      require(measured.final_state_token(token, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
      require(measured.prepare_resident_final_stationary_weights(
                  token, weights, diagnostic, detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail);
      require(ledger->live == capacity && ledger->peak == capacity &&
                  ledger->allocations == allocations,
              "resident proof introduced unplanned device allocations");
    } catch (const std::exception&) {
      if (allowance == capacity || !ledger->rejected) throw;
      failed = true;
    }
    require(failed == (allowance < capacity) && ledger->live == 0,
            "resident KS exact/below capacity gate leaked storage or admitted the wrong limit");
  }
}

/** A published W lease must exclude detached device proofs from its scratch,
 * even when the experimental selector and a real library handle admit them. */
void published_library_lease_case(bool hybrid) {
  auto system = hydrogens(18, true);
  // Closed-shell separated dimers give an 18-AO library-owned, gapped fixture
  // without adding a cuBLAS allowance to any of the tiny-owner tests.
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
    system.atoms[atom].position = {5.0 * (atom / 2), 0.0, atom % 2 ? 0.71 : -0.71};
  const dft::AoBasis basis(system);
  const dft::MolecularGrid grid(system, {1, 24, 12, 24, 3, 1e-12});
  scf::FockBuildSpec spec;
  spec.derivative_order = 0;
  spec.exchange.present = hybrid;
  spec.exchange.coefficient = -0.125;
  const auto strategy = scf::resolve_fock_build(spec, scf::FockBackend::Cuda, 1e-12);
  const scf::PreparedFockPlan fock(system, nullptr, strategy, 0);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  options.max_iterations = 200;
  options.semilocal_exchange_scale = hybrid ? 0.75 : 1.0;
  dft::CudaKsPlan plan(fock, basis, grid, options, dft::SemilocalFamily::Pbe, 257);
  require(plan.run(nullptr, false, false).converged, "library lease fixture did not converge");
  dft::CudaKsFinalStateToken token;
  dft::VerifiedKsFinalState detached;
  dft::CudaKsResidentStationaryWeightsBinding weights;
  dft::KsFinalStateDiagnostic diagnostic;
  std::string detail;
  require(plan.final_state_token(token, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  const auto before = plan.transfers();
  require(plan.read_final_state(token, false, detached, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail);
  require(plan.transfers().resident_final_validations == before.resident_final_validations + 1,
          "library lease fixture did not exercise the detached device proof");
  require(plan.prepare_resident_final_stationary_weights(token, weights, diagnostic, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  const auto leased = weights;
  std::vector<double> expected(weights.matrix_elements), observed(weights.matrix_elements);
  require(cudaMemcpy(expected.data(), weights.weighted_density, expected.size() * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess,
          "library lease observation failed");
  const auto published = plan.transfers();
  for (bool weighted : {false, true}) {
    require(plan.read_final_state(token, weighted, detached, detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail);
    require(plan.transfers().resident_final_validations == published.resident_final_validations,
            "a detached read borrowed the published D/W scratch for a device proof");
    require(cudaMemcpy(observed.data(), weights.weighted_density, observed.size() * sizeof(double),
                       cudaMemcpyDeviceToHost) == cudaSuccess &&
                observed == expected,
            "a detached read overwrote published stationary weights");
  }
  const auto reread = plan.transfers();
  require(plan.prepare_resident_final_stationary_weights(token, weights, diagnostic, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              weights.density == leased.density &&
              weights.weighted_density == leased.weighted_density &&
              plan.transfers().synchronizations == reread.synchronizations &&
              plan.transfers().final_state_d2h_bytes == reread.final_state_d2h_bytes,
          "a published library lease was revalidated or replaced after detached reads");
  std::cout << "resident KS library lease nao=" << basis.nao << " hybrid=" << hybrid
            << " device_proofs=" << published.resident_final_validations
            << " published_read_proofs="
            << reread.resident_final_validations - published.resident_final_validations << '\n';
}

void prepared_integral_case(bool restricted, bool hybrid, std::size_t atoms = 0,
                            bool polarized = false) {
  auto system = polarized ? polarized_lithium_hydride(restricted)
                          : hydrogens(atoms        ? atoms
                                      : restricted ? 2
                                                   : 3,
                                      restricted);
  if (atoms == 18) {
    // One primitive keeps the independent native CPU force oracle inside its
    // existing work gate, while 18 AOs exercise the real matrix-library lease.
    for (std::size_t atom = 0; atom < system.atoms.size(); ++atom) {
      system.atoms[atom].position = {5.0 * (atom / 2), 0.0, atom % 2 ? 0.71 : -0.71};
      system.shells[atom].primitives = {{0.6, 1.0}};
    }
    std::string normalization_detail;
    require(molecule::validate_and_normalize(system, normalization_detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            normalization_detail);
  }
  StationaryOptions fixture_options(restricted, hybrid);
  auto& ks = fixture_options.ks;
  const auto& descriptor = fixture_options.descriptor;
  core::ContextState context;
  context.requested_backend = context.executed_backend = GENERATIVEQC_BACKEND_CUDA;
  auto batch =
      methods::prepare_batch(context, {system}, descriptor, GENERATIVEQC_BATCH_ENABLE_WARM_STARTS);
  const auto results = batch->execute(methods::Coordinates(1), false);
  require(results.size() == 1 && results[0].status == GENERATIVEQC_STATUS_SUCCESS &&
              results[0].calculation.convergence.converged,
          "prepared integral-source KS solve failed");
  dft::CudaKsFinalStateToken token;
  std::string detail;
  require(methods::detail::dft_final_state_token(*batch, 0, token, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  const auto before = batch->ks_transport_diagnostic(0);
  std::vector<double> resident_sources;
  std::array<std::uint64_t, 9> work{};
  require(methods::detail::dft_cuda_integral_gradient(*batch, 0, token, resident_sources, 64U << 20,
                                                      work, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail);
  const auto after = batch->ks_transport_diagnostic(0);
  require(
      before && after && after->matrix_d2h_bytes == before->matrix_d2h_bytes &&
          after->final_state_reads == before->final_state_reads && work[7] == 0 &&
          work[6] == (restricted ? 1U : 2U) * sizeof(scf::cuda_df::ValidationPartial) + sizeof(int),
      "uncached prepared Direct sources exported D/W or hid final-state movement");
  std::vector<double> native_gradient{42.0};
  std::vector<double> retained_gradient;
  dft::CudaStationaryResources native_resources;
  const auto native_profile =
      hybrid ? (restricted ? "pbe0_rks" : "pbe0_uks") : (restricted ? "pbe_rks" : "pbe_uks");
  const bool native_available = dft::cuda_stationary_binding(native_profile) != nullptr;
  if (native_available) {
    auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
    ledger->limit = 128U << 20;
    ledger->device = 0;
    struct RestoreLedger {
      std::shared_ptr<runtime::DeviceResourceLedger> previous;
      ~RestoreLedger() { runtime::active_device_resource_ledger = previous; }
    } restore{runtime::active_device_resource_ledger};
    runtime::active_device_resource_ledger = ledger;
    require(methods::detail::dft_cuda_native_stationary_gradient(
                *batch, 0, token, 1, native_gradient, native_resources, detail) ==
                    GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
                native_gradient == std::vector<double>{42.0},
            "native stationary resource rejection partially published a force");
    require(methods::detail::dft_cuda_native_stationary_gradient(
                *batch, 0, token, 128U << 20, native_gradient, native_resources, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail);
    const auto native_transport = batch->ks_transport_diagnostic(0);
    require(
        native_transport && native_transport->matrix_d2h_bytes == after->matrix_d2h_bytes &&
            native_transport->final_state_reads == after->final_state_reads &&
            native_resources.grid_work.evaluation_points ==
                system.atoms.size() * ks.radial_points * ks.angular_polar * ks.angular_azimuth &&
            native_gradient.size() == 3 * system.atoms.size(),
        "complete native stationary controller exported a frame or omitted grid work");
    const auto native_upper = native_resources.device_upper_bytes;
    require(ledger->live == native_resources.grid_numeric_bytes +
                                native_resources.stationary_numeric_bytes &&
                ledger->provider_reserved == native_resources.grid_provider_allowance_bytes &&
                ledger->admission_peak == native_upper && ledger->allocations == 2 &&
                !ledger->rejected,
            "complete stationary endpoint hid an owned numeric arena or provider charge");
    const auto allocations_before_replay = ledger->allocations;
    const auto requested_before_replay = ledger->requested_bytes;
    retained_gradient = native_gradient;
    require(methods::detail::dft_cuda_native_stationary_gradient(
                *batch, 0, token, native_upper - 1, native_gradient, native_resources, detail) ==
                    GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
                native_gradient == retained_gradient,
            "retained native controller ignored a reduced resource allowance");
    require(methods::detail::dft_cuda_native_stationary_gradient(
                *batch, 0, token, native_upper, native_gradient, native_resources, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail);
    for (std::size_t coordinate = 0; coordinate < native_gradient.size(); ++coordinate)
      require(std::abs(native_gradient[coordinate] - retained_gradient[coordinate]) < 1e-12,
              "warm native stationary replay changed a source");
    require(ledger->allocations == allocations_before_replay &&
                ledger->requested_bytes == requested_before_replay &&
                ledger->live + ledger->provider_reserved == native_upper,
            "warm stationary replay allocated or lost a retained owner charge");
    {
      // Admit only the additional consumer. The independent cold SCF/Fock/grid
      // owner already has its own lifetime, just as in the enclosing fixture.
      RestoreLedger restore_outer{runtime::active_device_resource_ledger};
      runtime::active_device_resource_ledger.reset();
      auto minimum_batch = methods::prepare_batch(context, {system}, descriptor,
                                                  GENERATIVEQC_BATCH_ENABLE_WARM_STARTS);
      const auto minimum_scf = minimum_batch->execute(methods::Coordinates(1), false);
      require(minimum_scf.size() == 1 && minimum_scf[0].status == GENERATIVEQC_STATUS_SUCCESS &&
                  minimum_scf[0].calculation.convergence.converged,
              "minimum-budget cold SCF did not converge");
      dft::CudaKsFinalStateToken minimum_token;
      require(methods::detail::dft_final_state_token(*minimum_batch, 0, minimum_token, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail);
      const auto minimum_bytes = native_resources.minimum_device_bytes;
      auto minimum_ledger = std::make_shared<runtime::DeviceResourceLedger>();
      minimum_ledger->device = 0;
      minimum_ledger->limit = minimum_bytes - 1;
      runtime::active_device_resource_ledger = minimum_ledger;
      std::vector<double> minimum_gradient{42.0};
      dft::CudaStationaryResources minimum_resources;
      require(methods::detail::dft_cuda_native_stationary_gradient(
                  *minimum_batch, 0, minimum_token, minimum_bytes, minimum_gradient,
                  minimum_resources, detail) == GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
                  minimum_gradient == std::vector<double>{42.0} && !minimum_ledger->allocations &&
                  !minimum_ledger->live && !minimum_ledger->provider_reserved,
              "cold below-minimum request performed work or published a gradient");
      runtime::active_device_resource_ledger.reset();
      minimum_ledger = std::make_shared<runtime::DeviceResourceLedger>();
      minimum_ledger->device = 0;
      minimum_ledger->limit = minimum_bytes;
      runtime::active_device_resource_ledger = minimum_ledger;
      const auto before_minimum = minimum_batch->ks_transport_diagnostic(0);
      require(methods::detail::dft_cuda_native_stationary_gradient(
                  *minimum_batch, 0, minimum_token, minimum_bytes, minimum_gradient,
                  minimum_resources, detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail);
      const auto after_minimum = minimum_batch->ks_transport_diagnostic(0);
      require(minimum_gradient.size() == native_gradient.size() &&
                  minimum_resources.device_upper_bytes == minimum_bytes &&
                  minimum_ledger->live + minimum_ledger->provider_reserved == minimum_bytes &&
                  minimum_ledger->peak + minimum_ledger->provider_reserved == minimum_bytes &&
                  minimum_ledger->admission_peak == minimum_bytes &&
                  minimum_ledger->allocations == 2 && !minimum_ledger->rejected && before_minimum &&
                  after_minimum &&
                  before_minimum->matrix_d2h_bytes == after_minimum->matrix_d2h_bytes &&
                  before_minimum->final_state_reads == after_minimum->final_state_reads,
              "cold minimum stationary endpoint hid allocations or exported a dense frame");
      for (std::size_t coordinate = 0; coordinate < minimum_gradient.size(); ++coordinate)
        require(std::abs(minimum_gradient[coordinate] - native_gradient[coordinate]) < 1e-12,
                "minimum geometry fallback changed the independently qualified gradient");
      runtime::active_device_resource_ledger.reset();
      minimum_batch.reset();
      require(!minimum_ledger->live && !minimum_ledger->provider_reserved,
              "cold minimum endpoint retained a charge after close outside observation");
      std::cout << native_profile << " atoms=" << system.atoms.size()
                << " fixture=" << (polarized ? "lih_sp" : "hydrogen")
                << " cold minimum-gradient/resource parity PASS\n";
    }
    core::ContextState cpu_context;
    cpu_context.requested_backend = cpu_context.executed_backend =
        GENERATIVEQC_BACKEND_CPU_REFERENCE;
    ks.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_HOST_UNFUSED;
    auto cpu_batch = methods::prepare_batch(cpu_context, {system}, descriptor,
                                            GENERATIVEQC_BATCH_ENABLE_WARM_STARTS);
    const auto cpu_results = cpu_batch->execute(methods::Coordinates(1), true);
    ks.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
    require(cpu_results.size() == 1 && cpu_results[0].status == GENERATIVEQC_STATUS_SUCCESS &&
                cpu_results[0].calculation.convergence.converged &&
                cpu_results[0].calculation.forces.size() == native_gradient.size(),
            "independently qualified CPU force endpoint failed");
    for (std::size_t coordinate = 0; coordinate < native_gradient.size(); ++coordinate)
      require(std::abs(native_gradient[coordinate] +
                       cpu_results[0].calculation.forces[coordinate]) < 2e-7,
              "complete native CUDA stationary gradient differs from qualified CPU force");
    std::cout << "native CUDA stationary hybrid=" << hybrid << " spins=" << (restricted ? 1 : 2)
              << " atoms=" << system.atoms.size()
              << " points=" << native_resources.grid_work.evaluation_points
              << " device_upper=" << native_upper
              << " native_d2h=" << native_resources.stationary_work[2] << '\n';
  }
  dft::VerifiedKsFinalState detached;
  require(methods::detail::read_dft_final_state(*batch, 0, token, true, detached, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  if (native_available) {
    // Only the test exports matrices, after the production no-export gate above.
    // The offline libcint/libxc oracle compares the actual final spin D and W.
    std::cout << "native_stationary_oracle {\"profile\":\"" << native_profile << "\",\"fixture\":\""
              << (polarized ? "lih_sp" : "hydrogen") << "\",\"atoms\":" << system.atoms.size()
              << ",\"aos\":" << molecule::ao_count(system)
              << ",\"energy\":" << results[0].calculation.energy << ",\"gradient\":[";
    for (std::size_t coordinate = 0; coordinate < native_gradient.size(); ++coordinate)
      std::cout << (coordinate ? "," : "") << native_gradient[coordinate];
    const auto emit_matrices = [&](const auto& matrices) {
      std::cout << '[';
      for (std::size_t spin = 0; spin < matrices.size(); ++spin) {
        std::cout << (spin ? ",[" : "[");
        for (std::size_t element = 0; element < matrices[spin].size(); ++element)
          std::cout << (element ? "," : "") << matrices[spin][element];
        std::cout << ']';
      }
      std::cout << ']';
    };
    std::cout << "],\"density\":";
    emit_matrices(detached.density);
    std::cout << ",\"weighted_density\":";
    emit_matrices(detached.weighted_density);
    std::cout << "}\n";
  }
  std::vector<double> cached_sources;
  require(methods::detail::dft_cuda_integral_gradient_cached(
              *batch, 0, token, detached.density, detached.weighted_density, cached_sources,
              64U << 20, work, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail);
  require(resident_sources.size() == 12 * system.atoms.size() &&
              resident_sources.size() == cached_sources.size(),
          "resident prepared source shape changed");
  for (std::size_t coordinate = 0; coordinate < resident_sources.size(); ++coordinate)
    require(std::abs(resident_sources[coordinate] - cached_sources[coordinate]) < 1e-12,
            "resident admission changed an integral derivative source");

  // Disabling packaged/native profiles must not disable the resident D/W lease
  // and incumbent integral-source checks. Independent oracle tooling requires
  // the complete profile inventory when qualifying the new controller itself.
  if (!native_available) return;

  // A real geometry displacement invalidates the token and independently
  // checks the force sign against the complete reconverged CUDA energy.
  std::vector<double> positions;
  for (const auto& atom : system.atoms)
    positions.insert(positions.end(), atom.position.begin(), atom.position.end());
  auto displaced = positions;
  constexpr double step = 1e-4;
  displaced[2] += step;
  const auto plus = batch->execute(methods::Coordinates{displaced}, false);
  displaced[2] -= 2 * step;
  const auto minus = batch->execute(methods::Coordinates{displaced}, false);
  require(plus[0].status == GENERATIVEQC_STATUS_SUCCESS &&
              minus[0].status == GENERATIVEQC_STATUS_SUCCESS &&
              plus[0].calculation.convergence.converged &&
              minus[0].calculation.convergence.converged,
          "moved native finite-difference solve failed");
  const auto difference = (plus[0].calculation.energy - minus[0].calculation.energy) / (2 * step);
  require(std::abs(difference - retained_gradient[2]) < 2e-6,
          "complete native gradient differs from reconverged CUDA finite difference");
  const auto stale_output = native_gradient;
  const auto stale_status = methods::detail::dft_cuda_native_stationary_gradient(
      *batch, 0, token, 128U << 20, native_gradient, native_resources, detail);
  require(stale_status == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "genuinely moved owner admitted stale native force token: " + detail);
  require(native_gradient == stale_output, "stale native force rejection changed prior output");
  const auto restored = batch->execute(methods::Coordinates{positions}, false);
  require(restored[0].status == GENERATIVEQC_STATUS_SUCCESS &&
              methods::detail::dft_final_state_token(*batch, 0, token, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              methods::detail::dft_cuda_native_stationary_gradient(
                  *batch, 0, token, 128U << 20, native_gradient, native_resources, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
          detail);
  for (std::size_t coordinate = 0; coordinate < native_gradient.size(); ++coordinate)
    require(std::abs(native_gradient[coordinate] - retained_gradient[coordinate]) < 2e-7,
            "restored native geometry force changed after a moved solve");

  // Qualify the complete public prepared consumer, including its first cold
  // SCF/force at the mandatory minimum. The private independently checked
  // gradient above is an energy derivative; publication negates it once.
  auto public_batch =
      methods::prepare_batch(context, {system}, descriptor, GENERATIVEQC_BATCH_ENABLE_WARM_STARTS);
  require(public_batch->supported_properties(0).value_or(0) & GENERATIVEQC_PROPERTY_FORCES,
          "complete native CUDA force consumer did not advertise its prepared capability");
  dft::CudaStationaryCapacity public_capacity;
  require(methods::detail::dft_cuda_stationary_capacity(*public_batch, 0, public_capacity,
                                                        detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail);
  require(methods::detail::dft_cuda_stationary_limit(*public_batch, 0,
                                                     public_capacity.minimum_device_bytes - 1,
                                                     detail) == GENERATIVEQC_STATUS_OUT_OF_MEMORY,
          "public consumer admitted a below-minimum per-item allowance");
  require(methods::detail::dft_cuda_stationary_limit(*public_batch, 0,
                                                     public_capacity.minimum_device_bytes,
                                                     detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail);
  for (unsigned replay = 0; replay < 2; ++replay) {
    const auto public_results = public_batch->execute(methods::Coordinates(1), true);
    require(public_results.size() == 1 && public_results[0].status == GENERATIVEQC_STATUS_SUCCESS &&
                public_results[0].calculation.forces.size() == retained_gradient.size(),
            "public native CUDA energy/force endpoint failed");
    const auto movement = public_batch->ks_transport_diagnostic(0);
    require(movement && !movement->matrix_d2h_bytes && !movement->final_state_reads,
            "public native CUDA force endpoint exported a dense scientific frame");
    for (std::size_t coordinate = 0; coordinate < retained_gradient.size(); ++coordinate)
      require(std::abs(public_results[0].calculation.forces[coordinate] +
                       retained_gradient[coordinate]) < 2e-7,
              "public native CUDA force has an incorrect sign or scientific source");
    require(methods::detail::dft_final_state_token(*public_batch, 0, token, detail) ==
                    GENERATIVEQC_STATUS_SUCCESS &&
                methods::detail::dft_cuda_native_stationary_gradient(
                    *public_batch, 0, token, public_capacity.minimum_device_bytes, native_gradient,
                    native_resources, detail) == GENERATIVEQC_STATUS_SUCCESS &&
                native_resources.device_upper_bytes == public_capacity.minimum_device_bytes,
            "public force consumer ignored the per-item minimum cache policy");
  }
}

/** Reject failed items transactionally without invalidating successful neighbors. */
void prepared_ragged_stationary_case(bool restricted, bool hybrid) {
  const auto profile =
      hybrid ? (restricted ? "pbe0_rks" : "pbe0_uks") : (restricted ? "pbe_rks" : "pbe_uks");
  if (!dft::cuda_stationary_binding(profile)) return;
  auto system = hydrogens(restricted ? 2 : 3, restricted);
  auto atom = hydrogens(1, false);
  if (restricted) {
    atom.atoms[0].atomic_number = 2;
    atom.multiplicity = 1;
    std::string detail;
    require(molecule::validate_and_normalize(atom, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  }
  auto other = system;
  other.atoms.back().position[2] += 0.2;
  const std::vector<core::System> systems{system, atom, other};
  StationaryOptions options(restricted, hybrid);
  core::ContextState context;
  context.requested_backend = context.executed_backend = GENERATIVEQC_BACKEND_CUDA;
  auto batch = methods::prepare_batch(context, systems, options.descriptor,
                                      GENERATIVEQC_BATCH_ENABLE_WARM_STARTS);
  const auto cold = batch->execute(methods::Coordinates(systems.size()), false);
  std::vector<dft::CudaKsFinalStateToken> tokens(systems.size());
  std::vector<std::vector<double>> gradients(systems.size());
  dft::CudaStationaryResources resources;
  std::string detail;
  const auto publish = [&](std::size_t index) {
    require(methods::detail::dft_final_state_token(*batch, index, tokens[index], detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail);
    require(methods::detail::dft_cuda_native_stationary_gradient(
                *batch, index, tokens[index], 128U << 20, gradients[index], resources, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail);
    require(gradients[index].size() == 3 * systems[index].atoms.size(),
            "ragged stationary force extent follows a neighbor instead of its owner");
  };
  for (std::size_t index = 0; index < systems.size(); ++index) {
    require(cold[index].status == GENERATIVEQC_STATUS_SUCCESS &&
                cold[index].calculation.executed_backend == GENERATIVEQC_BACKEND_CUDA &&
                cold[index].calculation.convergence.converged,
            "ragged native stationary cold solve failed");
    publish(index);
  }
  const auto original_gradients = gradients;
  require(methods::detail::dft_cuda_native_stationary_gradient(*batch, 2, tokens[0], 128U << 20,
                                                               gradients[2], resources, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              gradients[2] == original_gradients[2],
          "a same-shape neighbor token authorized native force publication");
  require(methods::detail::dft_cuda_native_stationary_gradient(*batch, 1, tokens[1], 1,
                                                               gradients[1], resources, detail) ==
                  GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
              gradients[1] == original_gradients[1],
          "ragged resource rejection changed the previously published force");

  // Every batch attempt revokes final eligibility, including malformed items.
  // Last-good warm storage survives independently for later recovery.
  methods::Coordinates malformed(systems.size());
  malformed[1] = std::vector<double>{0.0};
  const auto invalid = batch->execute(malformed, false);
  require(invalid[1].status == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "ragged malformed coordinates were admitted");
  dft::CudaKsFinalStateToken observed;
  require(methods::detail::dft_final_state_token(*batch, 1, observed, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              methods::detail::dft_cuda_native_stationary_gradient(
                  *batch, 1, tokens[1], 128U << 20, gradients[1], resources, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              gradients[1] == original_gradients[1],
          "malformed ragged item retained prior force eligibility or changed output");
  for (const auto index : {0U, 2U}) {
    require(invalid[index].status == GENERATIVEQC_STATUS_SUCCESS && invalid[index].warm_start_used,
            "a malformed ragged neighbor changed a valid warm solve");
    const auto retained = gradients[index];
    require(methods::detail::dft_cuda_native_stationary_gradient(
                *batch, index, tokens[index], 128U << 20, gradients[index], resources, detail) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                gradients[index] == retained,
            "ragged successful replay preserved stale force eligibility");
    publish(index);
  }
  // A genuinely submitted singular geometry also replaces the owner. Recovery
  // must still use last-good D, not the rejected geometry's incomplete state.
  methods::Coordinates singular(systems.size());
  std::vector<double> positions;
  for (const auto& center : system.atoms)
    positions.insert(positions.end(), center.position.begin(), center.position.end());
  for (std::size_t coordinate = 0; coordinate < 3; ++coordinate)
    positions.at(3 + coordinate) = positions.at(coordinate);
  singular[0] = positions;
  const auto failed_token = tokens[0];
  const auto retained = gradients[0];
  const auto failed = batch->execute(singular, false);
  require(failed[0].status != GENERATIVEQC_STATUS_SUCCESS &&
              methods::detail::dft_final_state_token(*batch, 0, observed, detail) !=
                  GENERATIVEQC_STATUS_SUCCESS &&
              methods::detail::dft_cuda_native_stationary_gradient(
                  *batch, 0, failed_token, 128U << 20, gradients[0], resources, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              gradients[0] == retained,
          "failed moved geometry retained stationary eligibility or published output");
  for (const auto index : {1U, 2U}) {
    require(failed[index].status == GENERATIVEQC_STATUS_SUCCESS && failed[index].warm_start_used,
            "a failed moved ragged neighbor damaged a successful item");
    publish(index);
  }
  const auto restored = batch->execute(methods::Coordinates(systems.size()), false);
  for (std::size_t index = 0; index < systems.size(); ++index) {
    require(
        restored[index].status == GENERATIVEQC_STATUS_SUCCESS && restored[index].warm_start_used,
        "ragged rollback did not retain last-good warm recovery");
    publish(index);
    for (std::size_t coordinate = 0; coordinate < gradients[index].size(); ++coordinate)
      require(std::abs(gradients[index][coordinate] - original_gradients[index][coordinate]) < 2e-7,
              "ragged recovered native gradient differs from its isolated cold result");
  }

  options.descriptor.max_iterations = 1;
  auto incomplete = methods::prepare_batch(context, {system}, options.descriptor,
                                           GENERATIVEQC_BATCH_ENABLE_WARM_STARTS);
  const auto before_nonconvergence = gradients[0];
  const auto nonconverged = incomplete->execute(methods::Coordinates(1), false);
  require(nonconverged[0].status == GENERATIVEQC_STATUS_NOT_CONVERGED &&
              !nonconverged[0].calculation.convergence.converged &&
              methods::detail::dft_final_state_token(*incomplete, 0, observed, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              methods::detail::dft_cuda_native_stationary_gradient(
                  *incomplete, 0, tokens[0], 128U << 20, gradients[0], resources, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              gradients[0] == before_nonconvergence,
          "nonconverged native stationary owner published a token or force");
  std::cout << profile << " ragged stationary rejection/rollback/nonconvergence PASS\n";
}
}  // namespace

int main() {
  int devices{};
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
  try {
    std::cout << std::setprecision(17);
    for (bool hybrid : {false, true}) {
      run_case(2, true, hybrid);
      run_case(1, false, hybrid);
      run_case(3, false, hybrid);
      prepared_integral_case(true, hybrid);
      prepared_integral_case(false, hybrid);
      prepared_integral_case(false, hybrid, 1);
      prepared_integral_case(true, hybrid, 18);
      prepared_integral_case(true, hybrid, 0, true);
      prepared_integral_case(false, hybrid, 0, true);
      prepared_ragged_stationary_case(true, hybrid);
      prepared_ragged_stationary_case(false, hybrid);
    }
    // This process alone requests the experimental detached device-proof route.
    require(::setenv("GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION", "1", 1) == 0,
            "could not select detached device validation for the library lease fixture");
    published_library_lease_case(false);
    published_library_lease_case(true);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
