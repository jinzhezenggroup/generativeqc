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
  const bool geometry = mode == "geometry" || mode == "df-geometry";
  const bool fitted = mode == "df-fock" || mode == "df-geometry";
  core::System system;
  system.multiplicity = unrestricted ? 2 : 1;
  for (unsigned atom = 0; atom < (unrestricted ? 3U : 2U); ++atom) {
    system.atoms.push_back({1, {0.15 * atom * atom, 0.13 * atom, 1.5 * atom}});
    system.shells.push_back(
        {atom,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}});
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
  std::vector<double> sentinel(elements, 27);
  check(cudaMemcpyAsync(output, sentinel.data(), elements * sizeof(double), cudaMemcpyHostToDevice,
                        lease.stream));
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
    require(
        (argc == 2 || (argc == 3 && (mode == "physical" || mode == "fock" || mode == "df-fock" ||
                                     mode == "geometry" || mode == "df-geometry"))) &&
            (std::string(argv[1]) == "rks" || std::string(argv[1]) == "uks"),
        "usage: ks_wb97mv_grid_probe rks|uks [physical|fock|df-fock|geometry|df-geometry]");
    run(std::string(argv[1]) == "uks", mode);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
