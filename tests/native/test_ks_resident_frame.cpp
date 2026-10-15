#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dft/cuda_ks.hpp"
#include "dft/cuda_ks_grid_reference.hpp"
#include "methods/dft_method.hpp"
#include "molecule/basis.hpp"
#include "response/cuda_ks_cpks_reference.hpp"
#include "scf/cuda/final_validation_kernels.hpp"

namespace {
using namespace generativeqc;

void require(bool accepted, const std::string& detail) {
  if (!accepted) throw std::runtime_error(detail);
}

void test_batch_port(const core::System& system, bool restricted, bool hybrid) {
  const std::array<generativeqc_ks_semilocal_component, 2> components{
      {{"GGA_X_PBE", hybrid ? 0.75 : 1.0}, {"GGA_C_PBE", 1.0}}};
  const generativeqc_ks_exchange_term exchange{GENERATIVEQC_KS_EXCHANGE_FULL_RANGE, 0.25, 0,
                                               restricted ? -0.125 : -0.25};
  generativeqc_ks_options ks{};
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
  generativeqc_method_descriptor descriptor{};
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
  core::ContextState context;
  context.requested_backend = context.executed_backend = GENERATIVEQC_BACKEND_CUDA;
  auto batch =
      methods::prepare_batch(context, {system}, descriptor, GENERATIVEQC_BATCH_ENABLE_WARM_STARTS);
  const auto results = batch->execute(methods::Coordinates(1), false);
  require(results.size() == 1 && results[0].status == GENERATIVEQC_STATUS_SUCCESS &&
              results[0].calculation.convergence.converged,
          "prepared-batch SCF failed");
  dft::CudaKsFinalStateToken token;
  dft::CudaKsResidentOrbitalFrameBinding frame;
  std::string detail;
  require(methods::detail::dft_final_state_token(*batch, 0, token, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  const auto borrow = [&](std::size_t index, const auto& expected, bool prepare) {
    return methods::detail::dft_cuda_resident_orbital_frame(*batch, index, expected, frame, detail,
                                                            prepare);
  };
  require(borrow(0, token, false) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !frame,
          "prepared-batch port admitted an unvalidated frame");
  dft::CudaKsResidentGridBinding grid_binding;
  const auto borrow_grid = [&](std::size_t index, const auto& expected) {
    return methods::detail::dft_cuda_resident_grid_reference(*batch, index, expected, grid_binding,
                                                             detail);
  };
  require(borrow_grid(0, token) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !grid_binding,
          "prepared-batch port admitted an unvalidated grid reference");
  dft::CudaKsResidentFockBinding fock_binding;
  const auto borrow_fock = [&](std::size_t index, const auto& expected) {
    return methods::detail::dft_cuda_resident_fock_sources(*batch, index, expected, fock_binding,
                                                           detail);
  };
  require(borrow_fock(0, token) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !fock_binding,
          "prepared-batch port admitted unvalidated Fock sources");
  dft::CudaKsGridReference grid_reference(*batch, 0, token);
  require(grid_reference.shape().nao == 0 && !grid_reference.reference().authenticate,
          "unprepared grid adapter published a reference");
  response::CudaKsCpksReference response_reference(*batch, 0, token);
  require(response_reference.shape().nao == 0 && !response_reference.reference().authenticate,
          "unprepared CPKS adapter published a reference");
  require(response_reference.prepare(detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  const auto reference = response_reference.reference();
  require(reference.authenticate && reference.authenticate(reference.owner) == cudaSuccess &&
              response_reference.shape().nao == system.shells.size() &&
              response_reference.shape().spins == (restricted ? 1U : 2U),
          "native CPKS adapter did not authenticate the actual prepared owner");
  require(borrow(0, token, true) == GENERATIVEQC_STATUS_SUCCESS && frame, detail);
  const auto leased = frame;
  require(borrow_fock(0, token) == GENERATIVEQC_STATUS_SUCCESS && fock_binding &&
              fock_binding.fock == leased.fock && fock_binding.nao == leased.nao &&
              fock_binding.spins == leased.spins && fock_binding.stream == leased.stream &&
              fock_binding.primary->strategy() == token.identity.determinant.model &&
              fock_binding.correction == nullptr,
          "prepared Fock source does not match the actual canonical owner");
  const auto immutable_fock = fock_binding;
  require(borrow_fock(0, token) == GENERATIVEQC_STATUS_SUCCESS && fock_binding == immutable_fock,
          "Fock source reauthentication replaced its native owner");
  require(borrow(0, token, false) == GENERATIVEQC_STATUS_SUCCESS &&
              frame.coefficients == leased.coefficients &&
              frame.orbital_energies == leased.orbital_energies,
          "prepared-batch reauthentication replaced the frame");
  require(grid_reference.prepare(detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  const auto grid_lease = grid_reference.reference();
  require(grid_lease.authenticate && grid_lease.authenticate(grid_lease.owner) == cudaSuccess &&
              grid_lease.density == leased.density && grid_lease.device_id == leased.device_id &&
              reinterpret_cast<void*>(grid_lease.stream) == leased.stream &&
              grid_reference.shape().nao == leased.nao &&
              grid_reference.shape().spins == leased.spins,
          "grid adapter does not authenticate the actual prepared frame");
  require(borrow_grid(0, token) == GENERATIVEQC_STATUS_SUCCESS && grid_binding, detail);
  const auto immutable_grid = grid_binding;
  require(borrow_grid(0, token) == GENERATIVEQC_STATUS_SUCCESS && grid_binding == immutable_grid,
          "grid reauthentication replaced retained storage");
  const dft::AoBasis expected_basis(system);
  require(grid_binding.atoms == expected_basis.natom &&
              grid_binding.primitives == expected_basis.nprimitive &&
              grid_binding.nao == expected_basis.nao &&
              grid_binding.point_count == expected_basis.natom * 24 * 12 * 24,
          "grid reference has incorrect basis/grid extents");
  std::vector<double> packed(expected_basis.packed.size());
  require(cudaMemcpy(packed.data(), grid_binding.basis, packed.size() * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess,
          "test-only grid reference observations failed");
  require(packed == expected_basis.packed, "grid reference has incorrect AoBasis records");
  int grid_device = -1;
  const double *grid_points{}, *grid_weights{}, *atomic_weights{};
  std::size_t grid_count{};
  require(methods::detail::dft_cuda_resident_grid(*batch, 0, token, grid_device, grid_points,
                                                  grid_weights, atomic_weights, grid_count,
                                                  detail) == GENERATIVEQC_STATUS_SUCCESS &&
              grid_device == grid_binding.device_id && grid_points == grid_binding.points &&
              grid_weights == grid_binding.weights && grid_count == grid_binding.point_count,
          "grid reference does not borrow the actual immutable molecular quadrature");
  require(borrow_grid(1, token) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !grid_binding,
          "missing prepared-batch item retained a grid reference");
  require(borrow_fock(1, token) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !fock_binding,
          "missing prepared-batch item retained Fock sources");
  require(borrow(1, token, true) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !frame,
          "missing prepared-batch item retained a frame");
  auto stale = token;
  ++stale.identity.model.owner;
  require(borrow(0, stale, true) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !frame,
          "prepared-batch port accepted another owner's token");
  require(borrow_grid(0, stale) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !grid_binding,
          "prepared-batch grid accepted another owner's token");
  require(borrow_fock(0, stale) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !fock_binding,
          "prepared-batch Fock accepted another owner's token");
  const auto warmed = batch->execute(methods::Coordinates(1), false);
  require(warmed[0].status == GENERATIVEQC_STATUS_SUCCESS &&
              warmed[0].calculation.convergence.converged,
          "prepared-batch warm replay failed");
  require(borrow(0, token, false) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !frame,
          "prepared-batch warm replay retained a stale lease");
  require(borrow_fock(0, token) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !fock_binding,
          "prepared-batch warm replay retained stale Fock sources");
  require(reference.authenticate(reference.owner) == cudaErrorInvalidResourceHandle &&
              response_reference.shape().nao == 0 && !response_reference.reference().authenticate,
          "native CPKS adapter retained a stale prepared owner");
  require(grid_lease.authenticate(grid_lease.owner) == cudaErrorInvalidResourceHandle &&
              grid_reference.shape().nao == 0 && !grid_reference.reference().authenticate &&
              borrow_grid(0, token) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !grid_binding,
          "native grid adapter retained a stale prepared owner");
}

void run(unsigned atoms, bool restricted, bool hybrid) {
  core::System system;
  system.multiplicity = restricted ? 1 : 2;
  for (unsigned atom = 0; atom < atoms; ++atom) {
    system.atoms.push_back({1, {0.15 * atom * atom, 0.13 * atom, 1.5 * atom}});
    system.shells.push_back(
        {atom,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}});
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  const dft::AoBasis basis(system);
  const dft::MolecularGrid grid(system, {1, 24, 12, 24, 3, 1e-12});
  scf::FockBuildSpec spec;
  spec.derivative_order = 0;
  spec.spin = restricted ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted;
  spec.exchange.present = hybrid;
  spec.exchange.coefficient = restricted ? -0.125 : -0.25;
  const scf::PreparedFockPlan fock(system, nullptr,
                                   scf::resolve_fock_build(spec, scf::FockBackend::Cuda, 1e-12), 0);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  options.max_iterations = 200;
  options.semilocal_exchange_scale = hybrid ? 0.75 : 1;
  dft::CudaKsPlan plan(fock, basis, grid, options, dft::SemilocalFamily::Pbe, 257);
  const auto solve = [&] {
    require(plan.run(nullptr, true, false).converged && !plan.failed(), "SCF did not converge");
  };
  solve();
  dft::CudaKsFinalStateToken token;
  require(plan.final_state_token(token, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  dft::CudaKsResidentOrbitalFrameBinding frame;
  dft::KsFinalStateDiagnostic diagnostic;
  dft::CudaKsResidentGridBinding native_grid;
  const auto prepare = [&](const auto& expected) {
    return plan.prepare_resident_final_orbital_frame(expected, frame, diagnostic, detail);
  };
  require(plan.resident_final_orbital_frame(token, frame, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              !frame,
          "unvalidated frame was borrowable");
  require(plan.resident_final_grid(token, native_grid, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              !native_grid,
          "unvalidated native grid was borrowable");
  auto stale = token;
  ++stale.identity.model.owner;
  const auto before = plan.transfers();
  require(prepare(stale) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !frame,
          "stale preparation published a frame");
  require(plan.resident_final_grid(stale, native_grid, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              !native_grid,
          "stale native grid was borrowable");
  require(plan.transfers().synchronizations == before.synchronizations &&
              plan.transfers().final_state_d2h_bytes == before.final_state_d2h_bytes,
          "stale preparation submitted GPU work");
  require(prepare(token) == GENERATIVEQC_STATUS_SUCCESS && frame, detail);
  const auto proof = plan.transfers();
  const auto packet_bytes =
      (restricted ? 1U : 2U) * sizeof(scf::cuda_df::ValidationPartial) + sizeof(int);
  require(proof.matrix_d2h_bytes == before.matrix_d2h_bytes &&
              proof.final_state_reads == before.final_state_reads &&
              proof.final_state_d2h_bytes - before.final_state_d2h_bytes == packet_bytes &&
              proof.scalar_d2h_bytes - before.scalar_d2h_bytes == packet_bytes,
          "frame proof exported dense matrices instead of scalar evidence");
  const auto retained = frame;
  require(plan.resident_final_grid(token, native_grid, detail) == GENERATIVEQC_STATUS_SUCCESS &&
              native_grid && native_grid.density == frame.density && native_grid.nao == frame.nao &&
              native_grid.spins == frame.spins && native_grid.stream == frame.stream,
          "native grid is not bound to its canonical frame");
  const auto retained_grid = native_grid;
  require(plan.resident_final_grid(token, native_grid, detail) == GENERATIVEQC_STATUS_SUCCESS &&
              native_grid == retained_grid &&
              plan.transfers().synchronizations == proof.synchronizations &&
              plan.transfers().scalar_d2h_bytes == proof.scalar_d2h_bytes &&
              plan.transfers().matrix_d2h_bytes == proof.matrix_d2h_bytes &&
              plan.transfers().final_state_d2h_bytes == proof.final_state_d2h_bytes,
          "native grid reborrowing submitted work, transferred data or replaced storage");
  require(prepare(token) == GENERATIVEQC_STATUS_SUCCESS &&
              frame.coefficients == retained.coefficients &&
              frame.orbital_energies == retained.orbital_energies &&
              plan.transfers().synchronizations == proof.synchronizations &&
              plan.transfers().final_state_d2h_bytes == proof.final_state_d2h_bytes,
          "cached frame preparation repeated GPU work or replaced storage");
  const auto stream = reinterpret_cast<cudaStream_t>(frame.stream);
  require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess,
          "capture begin failed");
  const auto capture_status = prepare(token);
  cudaGraph_t captured{};
  require(cudaStreamEndCapture(stream, &captured) == cudaSuccess &&
              cudaGraphDestroy(captured) == cudaSuccess,
          "capture end failed");
  require(capture_status == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !frame,
          "capturing frame was published");
  require(prepare(token) == GENERATIVEQC_STATUS_SUCCESS, detail);
  dft::VerifiedKsFinalState detached;
  require(plan.read_final_state(token, true, detached, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail);
  require(plan.resident_final_orbital_frame(token, frame, detail) == GENERATIVEQC_STATUS_SUCCESS &&
              frame.coefficients == retained.coefficients &&
              frame.orbital_energies == retained.orbital_energies,
          "detached oracle replaced resident frame storage");
  const auto nao = frame.nao;
  require(frame.spins == (restricted ? 1U : 2U) && nao == atoms, "frame shape mismatch");
  require(frame.device_id == token.identity.model.device, "frame device differs from its token");
  std::vector<double> overlap(nao * nao);
  require(cudaMemcpy(overlap.data(), frame.overlap, overlap.size() * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess,
          "resident overlap observation failed");
  for (std::size_t element = 0; element < overlap.size(); ++element)
    require(std::abs(overlap[element] - fock.one_electron().overlap[element]) < 1e-12,
            "resident overlap differs from the independent prepared integral source");
  for (unsigned spin = 0; spin < frame.spins; ++spin) {
    require(frame.occupied[spin] == token.identity.determinant.occupied[spin],
            "frame spin occupation differs from its authenticated token");
    std::vector<double> coefficients(nao * nao), energies(nao), density(nao * nao),
        physical(nao * nao);
    require(
        cudaMemcpy(coefficients.data(), frame.coefficients + spin * nao * nao,
                   coefficients.size() * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(energies.data(), frame.orbital_energies + spin * nao,
                       energies.size() * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(density.data(), frame.density + spin * nao * nao,
                       density.size() * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(physical.data(), frame.fock + spin * nao * nao,
                       physical.size() * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess,
        "resident-frame oracle observations failed");
    for (std::size_t row = 0; row < nao; ++row) {
      require(std::abs(energies[row] - detached.orbitals[spin].values[row]) < 1e-12,
              "orbital energies differ");
      for (std::size_t column = 0; column < nao; ++column) {
        const auto position = row * nao + column;
        require(std::abs(coefficients[column * nao + row] -
                         detached.orbitals[spin].vectors[position]) < 1e-12,
                "resident frame has the wrong column-major layout or numerical values");
        require(std::abs(density[position] - detached.density[spin][position]) < 1e-12 &&
                    std::abs(physical[position] - detached.fock[spin][position]) < 1e-12,
                "resident D/F differs from the independent detached proof");
      }
    }
  }
  solve();
  require(prepare(token) == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !frame,
          "warm solve preserved stale frame");
  require(plan.final_state_token(token, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  dft::CudaKsResidentDensityBinding density;
  require(plan.resident_final_density(token, density, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail);
  require(cudaMemsetAsync(const_cast<double*>(density.alpha), 0,
                          density.matrix_elements * sizeof(double),
                          reinterpret_cast<cudaStream_t>(density.stream)) == cudaSuccess,
          "proof corruption injection failed");
  require(
      prepare(token) == GENERATIVEQC_STATUS_NUMERICAL_FAILURE && !frame && plan.has_warm_start(),
      "corrupt determinant published a frame or lost warm recovery");
  require(plan.final_state_token(token, detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "failed proof retained final eligibility");
  solve();
  require(plan.final_state_token(token, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  require(prepare(token) == GENERATIVEQC_STATUS_SUCCESS, detail);
  plan.invalidate_final_state();
  require(plan.resident_final_orbital_frame(token, frame, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              !frame,
          "explicit invalidation preserved the frame lease");
  require(plan.resident_final_grid(token, native_grid, detail) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              !native_grid,
          "explicit invalidation preserved the grid lease");
  test_batch_port(system, restricted, hybrid);
  std::cout << "frame atoms=" << atoms << " restricted=" << restricted << " hybrid=" << hybrid
            << " PASS\n";
}
}  // namespace

int main() {
  int devices{};
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
  try {
    for (const bool hybrid : {false, true}) {
      run(2, true, hybrid);
      run(1, false, hybrid);
      run(3, false, hybrid);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
