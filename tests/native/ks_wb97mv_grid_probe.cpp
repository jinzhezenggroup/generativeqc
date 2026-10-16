#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_one_electron_response.hpp"
#include "dft/cuda_ks_wb97mv_df_nuclear.hpp"
#include "dft/cuda_ks_wb97mv_df_response.hpp"
#include "dft/cuda_ks_wb97mv_ecp_nuclear.hpp"
#include "dft/cuda_ks_wb97mv_nuclear_jk.hpp"
#include "dft/cuda_ks_wb97mv_response.hpp"
#include "dft/semilocal_family.hpp"
#include "molecule/basis.hpp"
#include "response/cuda_ks_cpks_reference.hpp"

namespace {
using namespace generativeqc;

void require(bool accepted, const std::string& detail) {
  if (!accepted) throw std::runtime_error(detail);
}

void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }

struct Storage {
  std::vector<void*> allocations;
  ~Storage() {
    for (auto* pointer : allocations) cudaFree(pointer);
  }
  double* allocate(std::size_t count) {
    double* pointer{};
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(double)));
    allocations.push_back(pointer);
    return pointer;
  }
};

std::vector<double> download(const double* pointer, std::size_t elements) {
  std::vector<double> values(elements);
  check(cudaMemcpy(values.data(), pointer, elements * sizeof(double), cudaMemcpyDeviceToHost));
  return values;
}

void array(const char* name, const std::vector<double>& values) {
  std::cout << ",\"" << name << "\":[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

/** Test-only molecular fixture: the production adapter borrows all panels from
 * this actual batch. Host observations below feed independent oracle tests;
 * they are not part of native preparation or response execution. */
void run(bool unrestricted, const std::string& mode) {
  const bool physical = mode == "physical";
  const bool fock = mode == "fock" || mode == "df-fock";
  const bool nuclear = mode == "nuclear" || mode == "df-nuclear";
  const bool one_electron = mode == "one-electron" || mode == "df-one-electron";
  const bool nuclear_jk = mode == "nuclear-jk" || mode == "df-nuclear-jk";
  const bool df_nuclear = mode == "df-nuclear-source";
  const bool ecp = mode == "ecp-source" || mode == "df-ecp-source" ||
                   mode == "spherical-ecp-source" || mode == "df-spherical-ecp-source";
  const bool df_algebra = mode == "df-response" || mode == "direct-df-response" || df_nuclear;
  const bool geometry = mode == "geometry" || mode == "df-geometry" || nuclear;
  const bool fitted = mode == "df-fock" || mode == "df-geometry" || mode == "df-nuclear" ||
                      mode == "df-one-electron" || mode == "df-nuclear-jk" ||
                      mode == "df-response" || df_nuclear || mode == "df-ecp-source" ||
                      mode == "df-spherical-ecp-source";
  core::System system;
  system.multiplicity = unrestricted ? 2 : 1;
  for (unsigned atom = 0; atom < (unrestricted ? 3U : 2U); ++atom) {
    system.atoms.push_back({1, {0.15 * atom * atom, 0.13 * atom, 1.5 * atom}});
    system.shells.push_back(
        {atom,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}});
  }
  if (ecp) {
    system.atoms[0].atomic_number = 11;
    system.atoms[0].ecp_core = 10;
    system.shells.insert(system.shells.begin() + 1, {0, 1, {{0.45, 1.0}}});
    system.ecp_terms = {{0, -1, 0, 0.9, 0.25}, {0, -1, 2, 0.4, -0.1}, {0, 0, 1, 0.8, 0.3},
                        {0, 1, 2, 0.65, -0.4}, {0, 2, 3, 0.7, 0.12},  {0, 3, 4, 0.55, -0.08}};
    if (mode == "spherical-ecp-source" || mode == "df-spherical-ecp-source")
      system.basis_representation = core::kSphericalBasis;
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  const std::array<generativeqc_ks_semilocal_component, 2> components{
      {{"MGGA_X_WB97M_V", 1.0}, {"MGGA_C_WB97M_V", 1.0}}};
  const double occupation = unrestricted ? 1 : 2;
  const std::array<generativeqc_ks_exchange_term, 2> exchange{{
      {GENERATIVEQC_KS_EXCHANGE_SHORT_RANGE, 0.15, 0.3, -0.15 / occupation},
      {GENERATIVEQC_KS_EXCHANGE_LONG_RANGE, 1.0, 0.3, -1.0 / occupation},
  }};
  generativeqc_ks_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = GENERATIVEQC_ABI_VERSION;
  options.scf_domain = dft::semilocal_family_scf_domain(dft::SemilocalFamily::Wb97mv);
  options.grid_version = 1;
  options.radial_points = 12;
  options.angular_polar = 4;
  options.angular_azimuth = 8;
  options.partition_iterations = 3;
  options.coincident_tolerance = 1e-12;
  options.tile_points = 64;
  options.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
  options.spin_channels = unrestricted ? 2 : 1;
  options.semilocal_components = components.data();
  options.semilocal_component_count = components.size();
  options.semilocal_range_omega = 0.3;
  options.exchange_terms = exchange.data();
  options.exchange_term_count = exchange.size();
  options.has_nonlocal_correlation = 1;
  options.nonlocal_variant = GENERATIVEQC_NONLOCAL_VV10;
  options.nonlocal_b = 6;
  options.nonlocal_c = 0.01;
  options.nonlocal_coefficient = 1;
  options.nonlocal_maximum_bytes = 1 << 24;
  generativeqc_method_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.method = unrestricted ? GENERATIVEQC_METHOD_PBE_UKS : GENERATIVEQC_METHOD_PBE_RKS;
  descriptor.max_iterations = 250;
  descriptor.diis_history = 8;
  descriptor.energy_tolerance = 1e-12;
  descriptor.density_tolerance = 1e-10;
  descriptor.screening_tolerance = 1e-14;
  descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
  descriptor.density_fitting_mode =
      fitted ? GENERATIVEQC_DENSITY_FITTING_CUDA : GENERATIVEQC_DENSITY_FITTING_NONE;
  descriptor.density_fitting_relative_threshold = 1e-10;
  descriptor.ks_options = &options;
  core::ContextState context;
  context.requested_backend = context.executed_backend = GENERATIVEQC_BACKEND_CUDA;
  auto batch =
      methods::prepare_batch(context, {system}, descriptor, GENERATIVEQC_BATCH_ENABLE_WARM_STARTS);
  const auto solve = [&] {
    const auto results = batch->execute(methods::Coordinates(1), false);
    require(results.size() == 1 && results[0].status == GENERATIVEQC_STATUS_SUCCESS &&
                results[0].calculation.convergence.converged,
            "native molecular WB97M-V SCF failed");
  };
  solve();
  dft::CudaKsFinalStateToken token;
  require(methods::detail::dft_final_state_token(*batch, 0, token, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  dft::CudaKsGridReference reference(*batch, 0, token);
  require(reference.prepare(detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  const auto shape = reference.shape();
  const auto lease = reference.reference();
  const auto elements = shape.spins * shape.nao * shape.nao;
  const auto density = download(lease.density, elements);
  std::vector<double> direction(elements), curvature(elements);
  for (unsigned spin = 0; spin < shape.spins; ++spin)
    for (std::size_t row = 0; row < shape.nao; ++row)
      for (std::size_t column = 0; column < shape.nao; ++column)
        direction[(spin * shape.nao + row) * shape.nao + column] =
            0.02 * std::cos((spin + 1) * (row + column + 1.0)) * (row == column ? -1 : 1);
  if (physical) {
    const auto generator = [](unsigned spin, std::size_t row, std::size_t column) {
      return 0.05 * (spin + 1) * std::sin(static_cast<double>(row) - column);
    };
    for (unsigned spin = 0; spin < shape.spins; ++spin)
      for (std::size_t row = 0; row < shape.nao; ++row)
        for (std::size_t column = 0; column < shape.nao; ++column) {
          const auto index = (spin * shape.nao + row) * shape.nao + column;
          direction[index] = 0;
          for (std::size_t inner = 0; inner < shape.nao; ++inner) {
            direction[index] += generator(spin, row, inner) *
                                    density[(spin * shape.nao + inner) * shape.nao + column] +
                                density[(spin * shape.nao + row) * shape.nao + inner] *
                                    generator(spin, column, inner);
            for (std::size_t second = 0; second < shape.nao; ++second)
              curvature[index] += generator(spin, row, inner) *
                                  density[(spin * shape.nao + inner) * shape.nao + second] *
                                  generator(spin, column, second);
          }
        }
  }
  dft::CudaWb97mvGridOptions controls;
  controls.nuclear_response = nuclear;
  controls.max_pair_evaluations = 4'000'000;
  std::size_t bytes{};
  check(dft::cuda_wb97mv_grid_workspace(shape, controls, bytes));
  Storage storage;
  auto* arena = storage.allocate(bytes / sizeof(double));
  auto* input = storage.allocate(elements);
  auto* output = storage.allocate(elements);
  check(cudaMemcpyAsync(input, direction.data(), elements * sizeof(double), cudaMemcpyHostToDevice,
                        lease.stream));
  dft::CudaWb97mvGridResponse response;
  check(response.prepare(shape, controls, lease, arena, bytes));
  check(response.apply(input, output, lease.stream));
  const double* potential{};
  check(response.reference_potential(potential));
  const auto value = download(potential, elements);
  const auto action = download(output, elements);
  const auto basis_elements = 3 * shape.atoms + 2 * shape.primitives + 16 * shape.nao;
  std::cout << std::setprecision(17) << "{\"atoms\":" << shape.atoms
            << ",\"primitives\":" << shape.primitives << ",\"nao\":" << shape.nao
            << ",\"spins\":" << shape.spins << ",\"point_count\":" << shape.points
            << ",\"active\":" << response.work().active_nonlocal_points
            << ",\"reference_pairs\":" << response.work().reference_pairs
            << ",\"response_pairs\":" << response.work().response_pairs;
  array("basis", download(lease.basis, basis_elements));
  array("points", download(lease.points, 3 * shape.points));
  array("weights", download(lease.weights, shape.points));
  array("density", density);
  array("direction", direction);
  array("curvature", curvature);
  array("potential", value);
  array("response", action);
  dft::CudaKsWb97mvEcpNuclear ecp_response;
  integrals::CudaEcpNuclearValues ecp_values;
  integrals::CudaEcpNuclearVectors ecp_vectors;
  double *ecp_motion{}, *ecp_matrices{}, *ecp_vector_storage{};
  if (ecp) {
    for (unsigned bound = 0; bound < 5; ++bound) {
      integrals::CudaEcpNuclearOptions insufficient;
      if (bound == 0) insufficient.max_workspace_bytes = 1;
      if (bound == 1) insufficient.max_terms = 1;
      if (bound == 2) insufficient.max_ao_evaluations = 1;
      if (bound == 3) insufficient.max_affine_terms = 1;
      if (bound == 4) insufficient.max_reduction_terms = 1;
      require(ecp_response.prepare(*batch, 0, token, insufficient, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT,
              "actual ECP source ignored its resource bound");
    }
    require(ecp_response.prepare(*batch, 0, token, {}, detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail);
    const auto matrix = shape.nao * shape.nao, coordinates = 3 * shape.atoms;
    require(ecp_response.shape().nao == shape.nao && ecp_response.shape().atoms == shape.atoms &&
                ecp_response.shape().terms == system.ecp_terms.size() &&
                ecp_response.device_bytes() > ecp_response.work().workspace_bytes,
            "actual ECP source lost canonical basis or staging budget");
    std::vector<double> motion(coordinates);
    for (std::size_t index = 0; index < coordinates; ++index)
      motion[index] = 0.12 * std::cos(index + 0.7);
    ecp_motion = storage.allocate(coordinates);
    check(cudaMemcpy(ecp_motion, motion.data(), coordinates * sizeof(double),
                     cudaMemcpyHostToDevice));
    ecp_matrices = storage.allocate(4 * matrix);
    ecp_vector_storage = storage.allocate(4 * coordinates);
    ecp_values = {ecp_matrices, ecp_matrices + matrix, ecp_matrices + 2 * matrix,
                  ecp_matrices + 3 * matrix};
    ecp_vectors = {ecp_vector_storage, ecp_vector_storage + coordinates,
                   ecp_vector_storage + 2 * coordinates, ecp_vector_storage + 3 * coordinates};
    dft::CudaKsWb97mvResponse ordinary;
    dft::CudaKsWb97mvResponseOptions ordinary_options;
    ordinary_options.grid = controls;
    require(
        ordinary.prepare(*batch, 0, token, ordinary_options, detail) == GENERATIVEQC_STATUS_SUCCESS,
        detail);
    check(ordinary.apply(input, output, lease.stream));
    const auto ordinary_before = download(output, elements);
    check(ecp_response.values(ecp_motion, ecp_values, lease.stream));
    check(ecp_response.contract(ecp_motion, input, ecp_vectors, lease.stream));
    const auto matrices = download(ecp_matrices, 4 * matrix);
    const auto vectors = download(ecp_vector_storage, 4 * coordinates);
    check(ecp_response.values(ecp_motion, ecp_values, lease.stream));
    check(ecp_response.contract(ecp_motion, input, ecp_vectors, lease.stream));
    check(ordinary.apply(input, output, lease.stream));
    require(download(ecp_matrices, matrices.size()) == matrices &&
                download(ecp_vector_storage, vectors.size()) == vectors &&
                download(output, elements) == ordinary_before &&
                download(lease.density, elements) == density,
            "ECP replay changed outputs or ordinary J/K owners");
    auto alias = ecp_values;
    alias.local = const_cast<double*>(lease.density);
    require(ecp_response.values(ecp_motion, alias, lease.stream) == cudaErrorInvalidValue &&
                ecp_response.contract(ecp_motion, nullptr, ecp_vectors, lease.stream) ==
                    cudaErrorInvalidValue,
            "actual ECP source admitted an immutable alias or missing spin motion");
    dft::CudaKsResidentOrbitalFrameBinding protected_frame;
    require(methods::detail::dft_cuda_resident_orbital_frame(*batch, 0, token, protected_frame,
                                                             detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail);
    for (const auto* immutable : {lease.points, lease.weights, protected_frame.coefficients,
                                  protected_frame.orbital_energies, protected_frame.overlap}) {
      alias.local = const_cast<double*>(immutable);
      require(ecp_response.values(ecp_motion, alias, lease.stream) == cudaErrorInvalidValue,
              "actual ECP source admitted an immutable canonical/grid output alias");
    }
    const auto ecp_geometry = reference.geometry_reference();
    require(ecp_geometry.atomic_weights,
            "actual ECP fixture lacks its retained native atomic measure");
    const auto atomic_weights = download(ecp_geometry.atomic_weights, shape.points);
    const auto ecp_publication_preserved = [&] {
      return download(ecp_matrices, matrices.size()) == matrices &&
             download(ecp_vector_storage, vectors.size()) == vectors &&
             download(ecp_geometry.atomic_weights, atomic_weights.size()) == atomic_weights;
    };
    for (const auto member :
         {&integrals::CudaEcpNuclearValues::local, &integrals::CudaEcpNuclearValues::nonlocal,
          &integrals::CudaEcpNuclearValues::direction_local,
          &integrals::CudaEcpNuclearValues::direction_nonlocal}) {
      auto atomic_alias = ecp_values;
      atomic_alias.*member = const_cast<double*>(ecp_geometry.atomic_weights);
      require(
          ecp_response.values(ecp_motion, atomic_alias, lease.stream) == cudaErrorInvalidValue &&
              ecp_publication_preserved(),
          "ECP value alias changed its immutable atomic measure or outputs");
    }
    for (const auto member : {&integrals::CudaEcpNuclearVectors::gradient_local,
                              &integrals::CudaEcpNuclearVectors::gradient_nonlocal,
                              &integrals::CudaEcpNuclearVectors::hvp_local,
                              &integrals::CudaEcpNuclearVectors::hvp_nonlocal}) {
      auto atomic_alias = ecp_vectors;
      atomic_alias.*member = const_cast<double*>(ecp_geometry.atomic_weights);
      require(ecp_response.contract(ecp_motion, input, atomic_alias, lease.stream) ==
                      cudaErrorInvalidValue &&
                  ecp_publication_preserved(),
              "ECP contraction alias changed its immutable atomic measure or outputs");
    }
    auto invalid = direction;
    invalid[0] = std::numeric_limits<double>::quiet_NaN();
    check(cudaMemcpy(input, invalid.data(), elements * sizeof(double), cudaMemcpyHostToDevice));
    require(ecp_response.contract(ecp_motion, input, ecp_vectors, lease.stream) ==
                    cudaErrorInvalidValue &&
                download(ecp_vector_storage, vectors.size()) == vectors,
            "nonfinite actual ECP spin motion published output");
    if (shape.spins == 2) {
      invalid = direction;
      invalid[1] += 1;
      invalid[matrix + 1] -= 1;
      check(cudaMemcpy(input, invalid.data(), elements * sizeof(double), cudaMemcpyHostToDevice));
      require(ecp_response.contract(ecp_motion, input, ecp_vectors, lease.stream) ==
                      cudaErrorInvalidValue &&
                  download(ecp_vector_storage, vectors.size()) == vectors,
              "antisymmetric spins hidden by a symmetric total were admitted");
    }
    check(cudaMemcpy(input, direction.data(), elements * sizeof(double), cudaMemcpyHostToDevice));
    array("ecp_motion", motion);
    array("ecp_matrices", matrices);
    array("ecp_vectors", vectors);
    const auto& work = ecp_response.work();
    std::cout << ",\"ecp_device_bytes\":" << ecp_response.device_bytes()
              << ",\"ecp_workspace_bytes\":" << work.workspace_bytes
              << ",\"ecp_spin_sum_terms\":" << ecp_response.spin_sum_terms_per_contraction()
              << ",\"ecp_value_actions\":" << work.value_actions
              << ",\"ecp_contraction_actions\":" << work.contraction_actions
              << ",\"ecp_layers_per_action\":" << work.radial_layers_per_action
              << ",\"ecp_ao_per_action\":" << work.ao_evaluations_per_action
              << ",\"ecp_ordinary_owner_preserved\":true";
  }
  dft::CudaKsWb97mvDfResponse df_response;
  dft::CudaKsWb97mvDfNuclear df_nuclear_response;
  dft::CudaWb97mvDfOutput df_output;
  integrals::CudaDfNuclearValues df_sources;
  integrals::CudaDfNuclearWeights df_weights;
  integrals::CudaDfNuclearVectors df_vectors;
  double *df_nuclear_motion{}, *df_source_storage{}, *df_vector_storage{};
  std::size_t df_source_count{};
  double *df_delta_raw{}, *df_delta_metric{}, *df_storage{};
  std::size_t df_count{};
  if (df_algebra && !fitted) {
    require(
        df_response.prepare(*batch, 0, token, {}, detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
        "fitted-primary adapter admitted a Direct model");
    require(df_nuclear_response.prepare(*batch, 0, token, {}, detail) ==
                GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "raw DF source admitted a Direct model");
    std::cout << ",\"df_direct_rejected\":true";
  }
  if (df_algebra && fitted) {
    for (unsigned gate = 0; gate < 2; ++gate) {
      dft::CudaWb97mvDfOptions insufficient;
      if (gate)
        insufficient.max_affine_term_visits = 1;
      else
        insufficient.max_workspace_bytes = 1;
      require(df_response.prepare(*batch, 0, token, insufficient, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT,
              "DF adapter ignored resource admission");
    }
    require(df_response.prepare(*batch, 0, token, {}, detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail);
    dft::CudaWb97mvDfOptions exact_capacity;
    exact_capacity.max_workspace_bytes = df_response.work().workspace_bytes - 1;
    require(df_response.prepare(*batch, 0, token, exact_capacity, detail) ==
                GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "DF adapter failed to charge retained raw A with algebra storage");
    ++exact_capacity.max_workspace_bytes;
    require(df_response.prepare(*batch, 0, token, exact_capacity, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail);
    const auto fitted_shape = df_response.shape();
    require(fitted_shape.nao == shape.nao && fitted_shape.spins == shape.spins && fitted_shape.naux,
            "DF adapter shape differs from its actual owners");
    const auto raw_count = shape.nao * shape.nao * fitted_shape.naux;
    const auto metric_count = fitted_shape.naux * fitted_shape.naux;
    std::vector<double> raw_motion(raw_count), metric_motion(metric_count);
    for (std::size_t row = 0; row < shape.nao; ++row)
      for (std::size_t column = 0; column < shape.nao; ++column)
        for (std::size_t auxiliary = 0; auxiliary < fitted_shape.naux; ++auxiliary)
          raw_motion[(row * shape.nao + column) * fitted_shape.naux + auxiliary] =
              0.03 * std::cos(row + column + auxiliary + 1.0);
    for (std::size_t row = 0; row < fitted_shape.naux; ++row)
      for (std::size_t column = 0; column < fitted_shape.naux; ++column)
        metric_motion[row * fitted_shape.naux + column] = 0.02 * std::sin(row + column + 1.0);
    df_delta_raw = storage.allocate(raw_count);
    df_delta_metric = storage.allocate(metric_count);
    check(cudaMemcpyAsync(df_delta_raw, raw_motion.data(), raw_count * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    check(cudaMemcpyAsync(df_delta_metric, metric_motion.data(), metric_count * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    if (df_nuclear) {
      integrals::CudaDfNuclearOptions insufficient;
      insufficient.max_workspace_bytes = 1;
      require(df_nuclear_response.prepare(*batch, 0, token, insufficient, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT,
              "raw DF source ignored its storage budget");
      require(
          df_nuclear_response.prepare(*batch, 0, token, {}, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail);
      const auto source_shape = df_nuclear_response.shape();
      require(source_shape.nao == shape.nao && source_shape.naux == fitted_shape.naux &&
                  source_shape.atoms == shape.atoms,
              "raw DF source differs from its accepted fitted-primary owner");
      std::vector<double> motion(3 * shape.atoms);
      for (std::size_t index = 0; index < motion.size(); ++index)
        motion[index] = 0.15 * std::sin(index + 2.0);
      df_nuclear_motion = storage.allocate(motion.size());
      check(cudaMemcpyAsync(df_nuclear_motion, motion.data(), motion.size() * sizeof(double),
                            cudaMemcpyHostToDevice, lease.stream));
      df_source_count = 2 * (raw_count + metric_count);
      df_source_storage = storage.allocate(df_source_count);
      df_sources = {df_source_storage, df_source_storage + raw_count,
                    df_source_storage + raw_count + metric_count,
                    df_source_storage + 2 * raw_count + metric_count};
      df_delta_raw = df_sources.direction_three_center;
      df_delta_metric = df_sources.direction_metric;
      check(df_nuclear_response.values(df_nuclear_motion, df_sources, lease.stream));
      const auto raw_first = download(df_source_storage, raw_count + metric_count);
      check(df_nuclear_response.values(df_nuclear_motion, df_sources, lease.stream));
      require(download(df_source_storage, raw_count + metric_count) == raw_first,
              "raw DF source replay changed values");
      raw_motion = download(df_delta_raw, raw_count);
      metric_motion = download(df_delta_metric, metric_count);
      array("df_raw", download(df_sources.three_center, raw_count));
      array("df_metric", download(df_sources.metric, metric_count));
      array("df_nuclear_motion", motion);
    }
    df_count = 1 + 2 * elements + 2 * raw_count + 2 * metric_count;
    df_storage = storage.allocate(df_count);
    df_output = {df_storage,
                 df_storage + 1,
                 df_storage + 1 + elements,
                 df_storage + 1 + 2 * elements,
                 df_storage + 1 + 2 * elements + raw_count,
                 df_storage + 1 + 2 * elements + raw_count + metric_count,
                 df_storage + 1 + 2 * elements + 2 * raw_count + metric_count};
    check(df_response.apply(input, df_delta_raw, df_delta_metric, df_output, lease.stream));
    const auto first = download(df_storage, df_count);
    check(df_response.apply(input, df_delta_raw, df_delta_metric, df_output, lease.stream));
    require(download(df_storage, df_count) == first, "DF adapter replay changed values");
    array("df_values", first);
    array("df_delta_raw", raw_motion);
    array("df_delta_metric", metric_motion);
    std::cout << ",\"df_naux\":" << fitted_shape.naux
              << ",\"df_device_bytes\":" << df_response.work().workspace_bytes;
    if (df_nuclear) {
      df_weights = {df_output.weight_three_center, df_output.weight_metric,
                    df_output.weight_direction_three_center, df_output.weight_direction_metric};
      df_vector_storage = storage.allocate(6 * shape.atoms);
      df_vectors = {df_vector_storage, df_vector_storage + 3 * shape.atoms};
      check(df_nuclear_response.contract(df_nuclear_motion, df_weights, df_vectors, lease.stream));
      const auto vectors = download(df_vector_storage, 6 * shape.atoms);
      check(df_nuclear_response.contract(df_nuclear_motion, df_weights, df_vectors, lease.stream));
      require(download(df_vector_storage, 6 * shape.atoms) == vectors,
              "raw DF source contraction replay changed values");
      array("df_nuclear_vectors", vectors);
      const auto& work = df_nuclear_response.work();
      std::cout << ",\"df_nuclear_bytes\":" << work.workspace_bytes
                << ",\"df_nuclear_components_per_pass\":" << work.component_evaluations_per_pass
                << ",\"df_nuclear_value_passes\":" << work.value_passes
                << ",\"df_nuclear_contraction_passes\":" << work.contraction_passes;
    }
  }
  dft::CudaKsOneElectronResponse one_electron_response;
  double *integral_motion{}, *integral_weights{}, *integral_matrices{}, *integral_vectors{};
  integrals::CudaOneElectronWeights integral_source;
  integrals::CudaOneElectronOutput integral_output;
  if (one_electron) {
    const auto matrix = shape.nao * shape.nao;
    integral_motion = storage.allocate(3 * shape.atoms);
    integral_weights = storage.allocate(4 * matrix);
    integral_matrices = storage.allocate(4 * matrix);
    integral_vectors = storage.allocate(6 * shape.atoms);
    std::vector<double> motion(3 * shape.atoms), weights(4 * matrix);
    for (std::size_t index = 0; index < motion.size(); ++index)
      motion[index] = 0.15 * std::sin(index + 2.0);
    for (unsigned panel = 0; panel < 4; ++panel)
      for (std::size_t row = 0; row < shape.nao; ++row)
        for (std::size_t column = 0; column < shape.nao; ++column)
          weights[panel * matrix + row * shape.nao + column] =
              0.1 * std::cos((panel + 1.0) * (row + column + 1.0));
    check(cudaMemcpyAsync(integral_motion, motion.data(), motion.size() * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    check(cudaMemcpyAsync(integral_weights, weights.data(), weights.size() * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    integral_source = {integral_weights, integral_weights + matrix, integral_weights + 2 * matrix,
                       integral_weights + 3 * matrix};
    integral_output = {integral_matrices,
                       integral_matrices + matrix,
                       integral_matrices + 2 * matrix,
                       integral_matrices + 3 * matrix,
                       integral_vectors,
                       integral_vectors + 3 * shape.atoms};
    auto insufficient = integrals::CudaOneElectronOptions{};
    insufficient.max_workspace_bytes = 1;
    require(one_electron_response.prepare(*batch, 0, token, insufficient, detail) ==
                GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "native one-electron adapter ignored its capacity gate");
    require(
        one_electron_response.prepare(*batch, 0, token, {}, detail) == GENERATIVEQC_STATUS_SUCCESS,
        detail);
    check(one_electron_response.apply(integral_motion, integral_source, integral_output,
                                      lease.stream));
    dft::CudaKsResidentFockBinding source;
    dft::CudaKsResidentOrbitalFrameBinding frame;
    require(methods::detail::dft_cuda_resident_fock_sources(*batch, 0, token, source, detail) ==
                    GENERATIVEQC_STATUS_SUCCESS &&
                methods::detail::dft_cuda_resident_orbital_frame(*batch, 0, token, frame, detail) ==
                    GENERATIVEQC_STATUS_SUCCESS,
            detail);
    const auto matrices = download(integral_matrices, 4 * matrix);
    const auto core = download(source.hcore, matrix), overlap = download(frame.overlap, matrix);
    for (std::size_t index = 0; index < matrix; ++index)
      require(std::abs(matrices[index] - core[index]) < 2e-11 &&
                  std::abs(matrices[matrix + index] - overlap[index]) < 2e-11,
              "native compiler-AD H/S source differs from the actual accepted KS frame");
    array("integral_matrices", matrices);
    array("integral_vectors", download(integral_vectors, 6 * shape.atoms));
    array("integral_motion", motion);
    array("integral_weights", weights);
    std::cout << ",\"integral_device_bytes\":" << one_electron_response.work().device_bytes;
  }
  dft::CudaKsWb97mvNuclearJk nuclear_jk_response;
  double *jk_motion{}, *jk_storage{};
  dft::CudaWb97mvJkOutput jk_output;
  const auto jk_count = 1 + 2 * elements + 6 * shape.atoms;
  if (nuclear_jk) {
    jk_motion = storage.allocate(3 * shape.atoms);
    jk_storage = storage.allocate(jk_count);
    jk_output = {jk_storage, jk_storage + 1, jk_storage + 1 + elements,
                 jk_storage + 1 + 2 * elements, jk_storage + 1 + 2 * elements + 3 * shape.atoms};
    std::vector<double> motion(3 * shape.atoms);
    for (std::size_t index = 0; index < motion.size(); ++index)
      motion[index] = 0.15 * std::sin(index + 2.0);
    check(cudaMemcpyAsync(jk_motion, motion.data(), motion.size() * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    auto options = dft::CudaWb97mvJkOptions{};
    if (fitted) {
      require(nuclear_jk_response.prepare(*batch, 0, token, options, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT,
              "DF source silently replaced fitted primary with exact J/K");
      options.part = dft::CudaWb97mvJkPart::correction;
    }
    auto insufficient = options;
    insufficient.max_workspace_bytes = 1;
    require(nuclear_jk_response.prepare(*batch, 0, token, insufficient, detail) ==
                GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "native J/K adapter ignored arena admission");
    require(nuclear_jk_response.prepare(*batch, 0, token, options, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail);
    check(nuclear_jk_response.apply(jk_motion, input, jk_output, lease.stream));
    if (!fitted) {
      dft::CudaKsResidentFockBinding source;
      require(methods::detail::dft_cuda_resident_fock_sources(*batch, 0, token, source, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail);
      const auto accepted = download(source.fock, elements),
                 core = download(source.hcore, shape.nao * shape.nao);
      const auto calculated = download(jk_output.fock, elements);
      for (std::size_t index = 0; index < elements; ++index)
        require(std::abs(accepted[index] - core[index % core.size()] - value[index] -
                         calculated[index]) < 3e-8,
                "native J/K source differs from actual accepted WB97M-V Fock");
    }
    array("jk_energy", download(jk_output.energy, 1));
    array("jk_matrices", download(jk_output.fock, 2 * elements));
    array("jk_vectors", download(jk_output.gradient, 6 * shape.atoms));
    array("jk_motion", motion);
    std::cout << ",\"jk_device_bytes\":" << nuclear_jk_response.work().workspace_bytes;
  }
  dft::CudaGridGeometryResponse geometry_response;
  double *geometry_left{}, *geometry_right{}, *geometry_ao{}, *geometry_weights{};
  if (geometry) {
    const auto geometry_shape = reference.geometry_shape();
    const auto geometry_controls = reference.geometry_options();
    const auto geometry_lease = reference.geometry_reference();
    require(geometry_shape.atoms == shape.atoms && geometry_shape.points == shape.points &&
                geometry_shape.points_per_atom * shape.atoms == shape.points &&
                geometry_controls.partition_iterations == options.partition_iterations &&
                geometry_controls.coincident_tolerance == options.coincident_tolerance &&
                geometry_lease.basis == lease.basis && geometry_lease.points == lease.points &&
                geometry_lease.weights == lease.weights && geometry_lease.atomic_weights &&
                geometry_lease.stream == lease.stream,
            "moving-grid source does not borrow the prepared native quadrature");
    std::size_t geometry_bytes{};
    check(dft::cuda_grid_geometry_workspace(geometry_shape, geometry_controls, geometry_bytes));
    auto* geometry_arena = storage.allocate(geometry_bytes / sizeof(double));
    geometry_left = storage.allocate(3 * shape.atoms);
    geometry_right = storage.allocate(3 * shape.atoms);
    geometry_ao = storage.allocate(12 * shape.points * shape.nao);
    geometry_weights = storage.allocate(3 * shape.points);
    std::vector<double> left(3 * shape.atoms), right(3 * shape.atoms);
    for (std::size_t index = 0; index < left.size(); ++index) {
      left[index] = 0.2 * std::cos(index + 1.0);
      right[index] = 0.15 * std::sin(index + 2.0);
    }
    check(cudaMemcpyAsync(geometry_left, left.data(), left.size() * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    check(cudaMemcpyAsync(geometry_right, right.data(), right.size() * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    array("atomic", download(geometry_lease.atomic_weights, shape.points));
    array("left", left);
    array("right", right);
    const auto prepared = geometry_response.prepare(geometry_shape, geometry_controls,
                                                    geometry_lease, geometry_arena, geometry_bytes);
    require(prepared == cudaSuccess,
            std::string("moving-grid preparation: ") + cudaGetErrorString(prepared));
    const auto acted = geometry_response.mixed(geometry_left, geometry_right, geometry_ao,
                                               geometry_weights, lease.stream);
    require(acted == cudaSuccess, std::string("moving-grid action: ") + cudaGetErrorString(acted));
    const double* spatial{};
    check(geometry_response.spatial_jets(spatial));
    std::cout << ",\"points_per_atom\":" << geometry_shape.points_per_atom
              << ",\"partition_iterations\":" << geometry_controls.partition_iterations
              << ",\"geometry_workspace_bytes\":" << geometry_response.work().workspace_bytes
              << ",\"geometry_action_pairs\":" << geometry_response.work().action_partition_pairs;
    array("spatial", download(spatial, 20 * shape.points * shape.nao));
    array("geometry_ao", download(geometry_ao, 12 * shape.points * shape.nao));
    array("geometry_weights", download(geometry_weights, 3 * shape.points));
    const double* point_directions{};
    check(geometry_response.point_directions(point_directions));
    array("point_directions", download(point_directions, 6 * shape.points));
    if (nuclear) {
      auto* scalar = storage.allocate(1);
      const auto panel = 4 * shape.points * shape.nao;
      check(response.prepare_nuclear_direction(geometry_ao + panel, geometry_weights + shape.points,
                                               point_directions + 3 * shape.points, input, output,
                                               lease.stream));
      check(response.nuclear_mixed(geometry_ao, geometry_ao + 2 * panel, geometry_weights,
                                   geometry_weights + 2 * shape.points, point_directions, scalar,
                                   lease.stream));
      const double* energy{};
      check(response.reference_energy(energy));
      array("nuclear_energy", download(energy, 1));
      array("nuclear_fock", download(output, elements));
      array("nuclear_mixed", download(scalar, 1));
      std::cout << ",\"nuclear_reference_pairs\":" << response.work().nuclear_reference_pairs
                << ",\"nuclear_response_pairs\":" << response.work().nuclear_response_pairs
                << ",\"nuclear_affine_terms\":" << response.work().nuclear_affine_terms;
    }
  }
  dft::CudaKsWb97mvResponse full_response;
  if (fock) {
    dft::CudaKsWb97mvResponseOptions full_controls;
    full_controls.grid = controls;
    for (unsigned bound = 0; bound < 3; ++bound) {
      auto insufficient = full_controls;
      if (bound == 0) insufficient.max_device_bytes = 1;
      if (bound == 1) insufficient.max_affine_term_visits = 1;
      if (bound == 2) insufficient.max_direct_eri_evaluations = 1;
      require(full_response.prepare(*batch, 0, token, insufficient, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT,
              "physical response ignored its explicit capacity/work bound");
    }
    require(full_response.prepare(*batch, 0, token, full_controls, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail);
    check(full_response.apply(input, output, lease.stream));
    array("fock_response", download(output, elements));
    dft::CudaKsResidentFockBinding fock_source;
    dft::CudaKsResidentOrbitalFrameBinding frame;
    require(methods::detail::dft_cuda_resident_fock_sources(
                *batch, 0, token, fock_source, detail) == GENERATIVEQC_STATUS_SUCCESS &&
                methods::detail::dft_cuda_resident_orbital_frame(*batch, 0, token, frame, detail) ==
                    GENERATIVEQC_STATUS_SUCCESS,
            detail);
    array("hcore", download(fock_source.hcore, shape.nao * shape.nao));
    array("fock", download(fock_source.fock, elements));
    array("overlap", download(frame.overlap, shape.nao * shape.nao));
    require(full_response.apply(input, input, lease.stream) == cudaErrorInvalidValue &&
                full_response.apply(input, const_cast<double*>(frame.density), lease.stream) ==
                    cudaErrorInvalidValue &&
                download(lease.density, elements) == density,
            "physical response accepted aliased output or changed its reference");
    std::vector<double> bad_input = direction;
    bad_input[0] = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> unchanged(elements, 27);
    check(cudaMemcpyAsync(input, bad_input.data(), elements * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    check(cudaMemcpyAsync(output, unchanged.data(), elements * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    require(full_response.apply(input, output, lease.stream) == cudaErrorInvalidValue &&
                download(output, elements) == unchanged,
            "nonfinite physical response published an output");
    check(cudaMemcpyAsync(input, direction.data(), elements * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    response::CudaKsCpksReference cpks_reference(*batch, 0, token);
    require(cpks_reference.prepare(detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
    response::CudaCpksOptions cpks_controls;
    cpks_controls.gmres.restart = 12;
    cpks_controls.gmres.max_iterations = 120;
    cpks_controls.gmres.absolute_tolerance = 1e-12;
    cpks_controls.gmres.relative_tolerance = 1e-10;
    const auto cpks_shape = cpks_reference.shape();
    std::size_t cpks_bytes{};
    check(response::cuda_cpks_workspace(cpks_shape, cpks_controls, cpks_bytes));
    auto* cpks_arena = storage.allocate((cpks_bytes + sizeof(double) - 1) / sizeof(double));
    auto* overlap_direction = storage.allocate(shape.nao * shape.nao);
    auto* weighted_direction = storage.allocate(elements);
    check(cudaMemsetAsync(overlap_direction, 0, shape.nao * shape.nao * sizeof(double),
                          lease.stream));
    response::CudaCpksResult cpks_result;
    check(response::solve_cuda_cpks(cpks_shape, cpks_controls, cpks_reference.reference(), input,
                                    overlap_direction, dft::CudaKsWb97mvResponse::induced_fock,
                                    &full_response, output, weighted_direction, cpks_arena,
                                    cpks_bytes, lease.stream, cpks_result));
    require(cpks_result.converged(), "physical molecular CPKS did not converge");
    array("cpks_density", download(output, elements));
    array("cpks_weighted", download(weighted_direction, elements));
    const auto work = full_response.work();
    std::cout << ",\"capacity_alias_finite_gates\":true,\"cpks_actions\":"
              << cpks_result.induced_fock_actions << ",\"fock_actions\":" << work.actions
              << ",\"fock_device_bytes\":" << work.device_bytes
              << ",\"affine_reference_terms\":" << work.affine_reference_terms
              << ",\"affine_action_terms\":" << work.affine_action_terms
              << ",\"reference_eris\":" << work.reference_eris
              << ",\"response_eris\":" << work.response_eris;
  }
  solve();
  if (ecp) {
    const std::vector<double> matrix_sentinel(4 * shape.nao * shape.nao, 27);
    const std::vector<double> vector_sentinel(12 * shape.atoms, 27);
    check(cudaMemcpy(ecp_matrices, matrix_sentinel.data(), matrix_sentinel.size() * sizeof(double),
                     cudaMemcpyHostToDevice));
    check(cudaMemcpy(ecp_vector_storage, vector_sentinel.data(),
                     vector_sentinel.size() * sizeof(double), cudaMemcpyHostToDevice));
    require(ecp_response.values(ecp_motion, ecp_values, lease.stream) ==
                    cudaErrorInvalidResourceHandle &&
                ecp_response.contract(ecp_motion, input, ecp_vectors, lease.stream) ==
                    cudaErrorInvalidResourceHandle &&
                download(ecp_matrices, matrix_sentinel.size()) == matrix_sentinel &&
                download(ecp_vector_storage, vector_sentinel.size()) == vector_sentinel,
            "revoked actual ECP source published partial output");
  }
  if (df_nuclear) {
    const std::vector<double> source_sentinel(df_source_count, 27);
    const std::vector<double> vector_sentinel(6 * shape.atoms, 27);
    check(cudaMemcpy(df_source_storage, source_sentinel.data(), df_source_count * sizeof(double),
                     cudaMemcpyHostToDevice));
    check(cudaMemcpy(df_vector_storage, vector_sentinel.data(),
                     vector_sentinel.size() * sizeof(double), cudaMemcpyHostToDevice));
    require(df_nuclear_response.values(df_nuclear_motion, df_sources, lease.stream) ==
                    cudaErrorInvalidResourceHandle &&
                download(df_source_storage, df_source_count) == source_sentinel,
            "revoked raw DF source published values");
    require(df_nuclear_response.contract(df_nuclear_motion, df_weights, df_vectors, lease.stream) ==
                    cudaErrorInvalidResourceHandle &&
                download(df_vector_storage, vector_sentinel.size()) == vector_sentinel,
            "revoked raw DF source published vectors");
  }
  if (df_algebra && fitted) {
    const std::vector<double> df_sentinel(df_count, 27);
    check(cudaMemcpy(df_storage, df_sentinel.data(), df_count * sizeof(double),
                     cudaMemcpyHostToDevice));
    require(df_response.apply(input, df_delta_raw, df_delta_metric, df_output, lease.stream) ==
                    cudaErrorInvalidResourceHandle &&
                download(df_storage, df_count) == df_sentinel,
            "revoked native DF adapter published partial output");
  }
  if (one_electron) {
    const std::vector<double> matrix_sentinel(4 * shape.nao * shape.nao, 27);
    const std::vector<double> vector_sentinel(6 * shape.atoms, 27);
    check(cudaMemcpy(integral_matrices, matrix_sentinel.data(),
                     matrix_sentinel.size() * sizeof(double), cudaMemcpyHostToDevice));
    check(cudaMemcpy(integral_vectors, vector_sentinel.data(),
                     vector_sentinel.size() * sizeof(double), cudaMemcpyHostToDevice));
    require(one_electron_response.apply(integral_motion, integral_source, integral_output,
                                        lease.stream) == cudaErrorInvalidResourceHandle &&
                download(integral_matrices, matrix_sentinel.size()) == matrix_sentinel &&
                download(integral_vectors, vector_sentinel.size()) == vector_sentinel,
            "revoked native one-electron adapter published partial output");
  }
  if (nuclear_jk) {
    const std::vector<double> jk_sentinel(jk_count, 27);
    check(cudaMemcpy(jk_storage, jk_sentinel.data(), jk_count * sizeof(double),
                     cudaMemcpyHostToDevice));
    require(nuclear_jk_response.apply(jk_motion, input, jk_output, lease.stream) ==
                    cudaErrorInvalidResourceHandle &&
                download(jk_storage, jk_count) == jk_sentinel,
            "revoked native J/K adapter published partial output");
  }
  std::vector<double> sentinel(elements, 27);
  check(cudaMemcpyAsync(output, sentinel.data(), elements * sizeof(double), cudaMemcpyHostToDevice,
                        lease.stream));
  check(cudaStreamSynchronize(lease.stream));
  require(response.apply(input, output, lease.stream) == cudaErrorInvalidResourceHandle &&
              reference.shape().nao == 0,
          "native grid response accepted a revoked molecular owner");
  require(download(output, elements) == sentinel, "revoked molecular response published output");
  if (fock) {
    require(full_response.apply(input, output, lease.stream) == cudaErrorInvalidResourceHandle,
            "physical induced Fock accepted a revoked molecular owner");
    require(download(output, elements) == sentinel, "revoked physical Fock published output");
  }
  if (geometry) {
    const std::vector<double> ao_sentinel(12 * shape.points * shape.nao, 27);
    const std::vector<double> weight_sentinel(3 * shape.points, 27);
    check(cudaMemcpyAsync(geometry_ao, ao_sentinel.data(), ao_sentinel.size() * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    check(cudaMemcpyAsync(geometry_weights, weight_sentinel.data(),
                          weight_sentinel.size() * sizeof(double), cudaMemcpyHostToDevice,
                          lease.stream));
    check(cudaStreamSynchronize(lease.stream));
    const double* spatial{};
    require(geometry_response.mixed(geometry_left, geometry_right, geometry_ao, geometry_weights,
                                    lease.stream) == cudaErrorInvalidResourceHandle &&
                geometry_response.spatial_jets(spatial) == cudaErrorInvalidResourceHandle &&
                !spatial && download(geometry_ao, ao_sentinel.size()) == ao_sentinel &&
                download(geometry_weights, weight_sentinel.size()) == weight_sentinel,
            "revoked molecular geometry published partial output");
  }
  std::cout << ",\"revoked_without_publication\":true}\n";
}
}  // namespace

int main(int argc, char** argv) {
  int devices{};
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
  try {
    const std::string mode = argc == 3 ? argv[2] : "grid";
    require((argc == 2 ||
             (argc == 3 &&
              (mode == "physical" || mode == "fock" || mode == "df-fock" || mode == "geometry" ||
               mode == "df-geometry" || mode == "nuclear" || mode == "df-nuclear" ||
               mode == "one-electron" || mode == "df-one-electron" || mode == "nuclear-jk" ||
               mode == "df-nuclear-jk" || mode == "df-response" || mode == "direct-df-response" ||
               mode == "df-nuclear-source" || mode == "ecp-source" || mode == "df-ecp-source" ||
               mode == "spherical-ecp-source" || mode == "df-spherical-ecp-source"))) &&
                (std::string(argv[1]) == "rks" || std::string(argv[1]) == "uks"),
            "usage: ks_wb97mv_grid_probe rks|uks "
            "[physical|fock|df-fock|geometry|df-geometry|nuclear|df-nuclear|one-electron|df-one-"
            "electron|nuclear-jk|df-nuclear-jk|df-response|direct-df-response|df-nuclear-source|"
            "ecp-source|df-ecp-source|spherical-ecp-source|df-spherical-ecp-source]");
    run(std::string(argv[1]) == "uks", mode);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
