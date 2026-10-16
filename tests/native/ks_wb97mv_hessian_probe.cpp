#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_hessian.hpp"
#include "dft/semilocal_family.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda_density_fitting_device.hpp"

namespace {
using namespace generativeqc;
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }
struct Storage {
  std::vector<double*> allocations;
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
std::vector<double> download(const double* pointer, std::size_t count) {
  std::vector<double> result(count);
  check(cudaMemcpy(result.data(), pointer, count * sizeof(double), cudaMemcpyDeviceToHost));
  return result;
}
void array(const char* name, const std::vector<double>& values) {
  std::cout << ",\"" << name << "\":[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}
void run(bool unrestricted, const std::string& mode, bool full) {
  const bool fitted = mode.find("df") != std::string::npos,
             ecp = mode.find("ecp") != std::string::npos,
             spherical = mode.find("spherical") != std::string::npos,
             auxiliary_d = mode.find("auxd") != std::string::npos,
             auxiliary_g = mode.find("auxg") != std::string::npos,
             orbital_d = mode.find("orbd") != std::string::npos,
             orbital_f = mode.find("orbf") != std::string::npos,
             benchmark_h2 = mode.find("benchmark-h2") != std::string::npos,
             benchmark_water = mode.find("benchmark-water") != std::string::npos;
  const unsigned orbital_angular = orbital_f ? 3U : orbital_d ? 2U : 0U;
  const bool separate_auxiliary = auxiliary_d || auxiliary_g;
  const bool compact_fixture = separate_auxiliary || orbital_angular;
  require(!auxiliary_d || !auxiliary_g, "select only one independent auxiliary fixture");
  require(!orbital_d || !orbital_f, "select only one high-angular orbital fixture");
  require(!separate_auxiliary || fitted, "the independent auxiliary fixture requires DF");
  require(!benchmark_h2 || !benchmark_water, "select only one external-comparison geometry");
  require(!(benchmark_h2 || benchmark_water) ||
              (!unrestricted && !fitted && !ecp && !separate_auxiliary && !orbital_angular),
          "external-comparison fixtures require all-electron direct RKS");
  core::System system;
  system.multiplicity = unrestricted ? 2 : 1;
  if (spherical) system.basis_representation = core::kSphericalBasis;
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
  }
  if (compact_fixture) {
    // Single radial primitives keep the complete high-angular molecular response
    // gate bounded; they do not replace the independent auxiliary topology.
    const std::array<std::array<double, 3>, 3> centers{
        {{{0.13, -0.21, 0.17}}, {{0.43, 0.19, 2.1}}, {{-0.3, 0.4, -1.6}}}};
    system.shells.clear();
    for (unsigned atom = 0; atom < system.atoms.size(); ++atom) {
      system.atoms[atom].position = centers[atom];
      system.shells.push_back({atom, 0, {{atom == 0 ? 0.7 : 0.9, 1.0}}});
    }
    if (ecp) system.ecp_terms = {{0, -1, 2, 0.8, -0.4}, {0, 0, 2, 0.8, 0.3}};
  }
  if (orbital_angular)
    system.shells.insert(system.shells.begin() + 1, {0, orbital_angular, {{0.48, 1.0}}});
  if (benchmark_h2) system.atoms[0].position = {0.03, -0.02, 0.01};
  if (benchmark_water) {
    // Use the CLI's canonical STO-3G decimals, not a rounded vendor basis alias.
    system.atoms = {{8, {0.0, 0.0, 0.0}},
                    {1, {1.43233673, 0.0, 1.10715266}},
                    {1, {-1.43233673, 0.0, 1.10715266}}};
    system.shells = {
        {0,
         0,
         {{130.7093214, 0.1543289673}, {23.80886605, 0.5353281423}, {6.443608313, 0.4446345422}}},
        {0,
         0,
         {{5.033151319, -0.09996722919}, {1.169596125, 0.3995128261}, {0.38038896, 0.7001154689}}},
        {0,
         1,
         {{5.033151319, 0.155916275}, {1.169596125, 0.6076837186}, {0.38038896, 0.3919573931}}},
        {1,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
        {2,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}}};
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  std::optional<core::System> auxiliary;
  if (separate_auxiliary) {
    auxiliary = system;
    auxiliary->ecp_terms.clear();
    for (auto& atom : auxiliary->atoms) atom.ecp_core = 0;
    auxiliary->shells.clear();
    for (unsigned atom = 0; atom < system.atoms.size(); ++atom)
      auxiliary->shells.push_back({atom, 0, {{atom == 0 ? 0.8 : 0.65, 1.0}}});
    auxiliary->shells.insert(auxiliary->shells.begin() + 1,
                             {0, auxiliary_g ? 4U : 2U, {{0.6, 1.0}}});
    require(molecule::validate_and_normalize(*auxiliary, detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail);
  }
  const std::array<generativeqc_ks_semilocal_component, 2> components{
      {{"MGGA_X_WB97M_V", 1.0}, {"MGGA_C_WB97M_V", 1.0}}};
  const double occupation = unrestricted ? 1 : 2;
  const std::array<generativeqc_ks_exchange_term, 2> exchange{
      {{GENERATIVEQC_KS_EXCHANGE_SHORT_RANGE, 0.15, 0.3, -0.15 / occupation},
       {GENERATIVEQC_KS_EXCHANGE_LONG_RANGE, 1.0, 0.3, -1.0 / occupation}}};
  generativeqc_ks_options ks{};
  ks.struct_size = sizeof(ks);
  ks.abi_version = GENERATIVEQC_ABI_VERSION;
  ks.scf_domain = dft::semilocal_family_scf_domain(dft::SemilocalFamily::Wb97mv);
  ks.grid_version = 1;
  ks.radial_points = compact_fixture || benchmark_h2 || benchmark_water ? 8 : 12;
  ks.angular_polar = 4;
  ks.angular_azimuth = 8;
  ks.partition_iterations = 3;
  ks.coincident_tolerance = 1e-12;
  ks.tile_points = 64;
  ks.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
  ks.spin_channels = unrestricted ? 2 : 1;
  ks.semilocal_components = components.data();
  ks.semilocal_component_count = components.size();
  ks.semilocal_range_omega = 0.3;
  ks.exchange_terms = exchange.data();
  ks.exchange_term_count = exchange.size();
  ks.has_nonlocal_correlation = 1;
  ks.nonlocal_variant = GENERATIVEQC_NONLOCAL_VV10;
  ks.nonlocal_b = 6;
  ks.nonlocal_c = 0.01;
  ks.nonlocal_coefficient = 1;
  ks.nonlocal_maximum_bytes = 1 << 24;
  generativeqc_method_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.method = unrestricted ? GENERATIVEQC_METHOD_PBE_UKS : GENERATIVEQC_METHOD_PBE_RKS;
  descriptor.max_iterations = 250;
  descriptor.diis_history = 8;
  descriptor.energy_tolerance = 1e-13;
  descriptor.density_tolerance = 1e-11;
  descriptor.screening_tolerance = 1e-14;
  descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
  descriptor.density_fitting_mode =
      fitted ? GENERATIVEQC_DENSITY_FITTING_CUDA : GENERATIVEQC_DENSITY_FITTING_NONE;
  descriptor.density_fitting_relative_threshold = 1e-10;
  descriptor.ks_options = &ks;
  core::ContextState context;
  context.requested_backend = context.executed_backend = GENERATIVEQC_BACKEND_CUDA;
  auto batch =
      methods::prepare_batch(context, {system}, descriptor, GENERATIVEQC_BATCH_ENABLE_WARM_STARTS,
                             auxiliary ? &*auxiliary : nullptr);
  const auto solve = [&] {
    const auto results = batch->execute(methods::Coordinates(1), false);
    require(results.size() == 1 && results[0].status == GENERATIVEQC_STATUS_SUCCESS &&
                results[0].calculation.convergence.converged,
            "native molecular WB97M-V SCF failed");
    return results[0].calculation.energy;
  };
  const auto energy = solve();
  dft::CudaKsFinalStateToken token;
  require(methods::detail::dft_final_state_token(*batch, 0, token, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  dft::CudaKsGridReference reference(*batch, 0, token);
  require(reference.prepare(detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  const auto shape = reference.shape();
  const auto lease = reference.reference();
  const auto coordinates = 3 * shape.atoms, panels = shape.spins * shape.nao * shape.nao;
  dft::CudaKsResidentFockBinding fock;
  dft::CudaKsResidentOrbitalFrameBinding frame;
  require(methods::detail::dft_cuda_resident_fock_sources(*batch, 0, token, fock, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  require(methods::detail::dft_cuda_resident_orbital_frame(*batch, 0, token, frame, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  Storage storage;
  auto* direction = storage.allocate(coordinates);
  auto* output = storage.allocate(coordinates);
  auto* matrix = storage.allocate(coordinates * coordinates);
  std::vector<double> motion(coordinates), sentinel(coordinates, 27.0);
  for (std::size_t index = 0; index < coordinates; ++index)
    motion[index] = 0.12 * std::cos(index + 0.7);
  if (separate_auxiliary && unrestricted)
    motion = {0.02, -0.03, 0.1, -0.04, 0.01, -0.02, 0.02, 0.02, -0.08};
  check(cudaMemcpyAsync(direction, motion.data(), coordinates * sizeof(double),
                        cudaMemcpyHostToDevice, lease.stream));
  check(cudaMemcpyAsync(output, sentinel.data(), coordinates * sizeof(double),
                        cudaMemcpyHostToDevice, lease.stream));
  dft::CudaKsWb97mvHessian endpoint;
  dft::CudaKsWb97mvHessianOptions options;
  if (mode.find("streamed") != std::string::npos) options.exact.max_resident_integral_bytes = 0;
  options.cpks.gmres.absolute_tolerance = 2e-12;
  options.cpks.gmres.relative_tolerance = 2e-11;
  auto insufficient = options;
  insufficient.max_device_bytes = 1;
  require(endpoint.prepare(*batch, 0, token, insufficient, detail) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "stationary endpoint ignored aggregate device bound");
  const auto prepared = endpoint.prepare(*batch, 0, token, options, detail);
  require(prepared == GENERATIVEQC_STATUS_SUCCESS, detail);
  dft::CudaKsWb97mvHessianResult result;
  require(endpoint.apply(direction, direction, lease.stream, result) != cudaSuccess,
          "stationary endpoint admitted input/output alias");
  require(endpoint.apply(direction, const_cast<double*>(lease.density), lease.stream, result) !=
              cudaSuccess,
          "stationary endpoint admitted immutable density output");
  const double nonfinite = std::numeric_limits<double>::infinity();
  check(
      cudaMemcpyAsync(direction, &nonfinite, sizeof(double), cudaMemcpyHostToDevice, lease.stream));
  require(endpoint.apply(direction, output, lease.stream, result) != cudaSuccess &&
              download(output, coordinates) == sentinel,
          "nonfinite direction changed output");
  check(cudaMemcpyAsync(direction, motion.data(), coordinates * sizeof(double),
                        cudaMemcpyHostToDevice, lease.stream));
  check(endpoint.apply(direction, output, lease.stream, result));
  require(result.converged() && result.response.converged(), "stationary CPKS did not converge");
  const auto hvp = download(output, coordinates);
  std::cout << std::setprecision(17) << "{\"atoms\":" << shape.atoms
            << ",\"primitives\":" << shape.primitives << ",\"nao\":" << shape.nao
            << ",\"spins\":" << shape.spins << ",\"point_count\":" << shape.points
            << ",\"orbital_angular\":" << orbital_angular
            << ",\"benchmark_water\":" << (benchmark_water ? "true" : "false")
            << ",\"auxiliary_d\":" << (auxiliary_d ? "true" : "false")
            << ",\"auxiliary_g\":" << (auxiliary_g ? "true" : "false") << ",\"energy\":" << energy
            << ",\"response_residual\":" << result.response.gmres.residual_norm;
  array("basis", download(lease.basis, 3 * shape.atoms + 2 * shape.primitives + 16 * shape.nao));
  array("points", download(lease.points, 3 * shape.points));
  array("weights", download(lease.weights, shape.points));
  array("atomic", download(reference.geometry_reference().atomic_weights, shape.points));
  array("density", download(lease.density, panels));
  array("hcore", download(fock.hcore, shape.nao * shape.nao));
  array("fock", download(fock.fock, panels));
  array("overlap", download(frame.overlap, shape.nao * shape.nao));
  if (fitted) {
    const auto source =
        scf::cuda_density_fitting_readonly_source(fock.primary->cuda_fitted_source());
    require(static_cast<bool>(source), "accepted DF source is not readable");
    require(source.naux == molecule::ao_count(auxiliary ? *auxiliary : system),
            "explicit native auxiliary basis was ignored");
    auto* raw = storage.allocate(shape.nao * shape.nao * source.naux);
    require(scf::enqueue_cuda_density_fitting_raw_values(source.integrals, 0, shape.nao * shape.nao,
                                                         lease.stream, raw,
                                                         detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail);
    check(cudaStreamSynchronize(lease.stream));
    std::cout << ",\"naux\":" << source.naux;
    array("raw", download(raw, shape.nao * shape.nao * source.naux));
    array("whitening", download(source.whitening, source.naux * source.naux));
  }
  array("motion", motion);
  array("hvp", hvp);
  std::cout << ",\"resident_integral_bytes\":" << endpoint.work().exact.resident_integral_bytes;
  if (full) {
    check(endpoint.hessian(matrix, lease.stream, result));
    require(result.converged() && result.completed_directions == coordinates,
            "complete Hessian was not accepted");
    array("hessian", download(matrix, coordinates * coordinates));
  }
  if (unrestricted) {
    // CUDA success is deliberately distinct from numerical convergence. A
    // bounded failed response must not publish even the first Hessian column.
    dft::CudaKsWb97mvHessian limited;
    auto short_solve = options;
    short_solve.cpks.gmres.max_iterations = short_solve.cpks.gmres.restart = 1;
    short_solve.cpks.gmres.absolute_tolerance = 1e-30;
    short_solve.cpks.gmres.relative_tolerance = 1e-30;
    require(limited.prepare(*batch, 0, token, short_solve, detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail);
    const std::vector<double> untouched(coordinates * coordinates, 33.0);
    check(cudaMemcpyAsync(matrix, untouched.data(), untouched.size() * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    check(limited.hessian(matrix, lease.stream, result));
    require(!result.converged() &&
                result.status == dft::CudaKsWb97mvHessianStatus::response_not_converged &&
                download(matrix, untouched.size()) == untouched,
            "nonconverged response published a partial Hessian");
  }
  const auto work = endpoint.work();
  std::cout << ",\"device_bytes\":" << work.device_bytes
            << ",\"directions\":" << work.completed_directions
            << ",\"induced_actions\":" << work.induced.actions
            << ",\"raw_df_value_passes\":" << work.raw_df.value_passes
            << ",\"raw_df_contraction_passes\":" << work.raw_df.contraction_passes
            << ",\"grid_right_directions\":" << work.grid.nuclear_directions
            << ",\"grid_mixed_contractions\":" << work.grid.nuclear_contractions;
  cudaStream_t other{};
  check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
  require(endpoint.apply(direction, output, other, result) != cudaSuccess,
          "foreign stream admitted");
  check(cudaStreamDestroy(other));
  check(cudaStreamBeginCapture(lease.stream, cudaStreamCaptureModeThreadLocal));
  require(
      endpoint.apply(direction, output, lease.stream, result) == cudaErrorStreamCaptureUnsupported,
      "stationary endpoint admitted capture");
  cudaGraph_t graph{};
  check(cudaStreamEndCapture(lease.stream, &graph));
  check(cudaGraphDestroy(graph));
  solve();
  require(
      endpoint.apply(direction, output, lease.stream, result) == cudaErrorInvalidResourceHandle &&
          download(output, coordinates) == hvp,
      "revoked endpoint changed published HVP");
  std::cout << ",\"lifecycle\":true}\n";
}
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 3 || argc == 4, "usage: probe rks|uks direct|df[-spherical][-ecp] [full]");
    require(std::string(argv[1]) == "rks" || std::string(argv[1]) == "uks", "invalid spin mode");
    run(std::string(argv[1]) == "uks", argv[2], argc == 4 && std::string(argv[3]) == "full");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
