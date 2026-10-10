#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "runtime/cuda_resources.cuh"
#include "scf/cuda/resident_final_validation.hpp"

namespace {
using namespace generativeqc;
using namespace scf::cuda_execution;

void require(bool result, const char* message) {
  if (!result) throw std::runtime_error(message);
}

template <class Operation>
void rejects(Operation operation) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "invalid resident validation request was not rejected");
}

void analytic_frames(cudaStream_t stream, MatrixLibraryResources library, std::size_t dimension,
                     unsigned spin_count) {
  const auto check = runtime::cuda_resource_check;
  const auto matrix = dimension * dimension;
  const auto partial_count = resident_final_validation_partial_count(dimension);
  runtime::OwnedCudaBuffer<double> input(0, 6 * matrix, stream), values(0, dimension, stream),
      scratch(0, 4 * (matrix + 2), stream);
  runtime::OwnedCudaBuffer<int> info(0, 1, stream);
  runtime::OwnedCudaBuffer<std::uint64_t> generation(0, 1, stream);
  runtime::OwnedCudaBuffer<scf::cuda_df::ValidationPartial> partial(0, partial_count + 2, stream);
  std::vector<double> host(6 * matrix), energies(dimension);
  auto* metric = host.data();
  auto* core = metric + matrix;
  auto* fock = core + matrix;
  auto* coefficients = host.data() + 5 * matrix;
  for (std::size_t orbital = 0; orbital < dimension; ++orbital) {
    const double overlap = orbital == 1 ? 4 : 2;
    energies[orbital] = orbital == 0 ? -1.0 : static_cast<double>(orbital) + 2;
    metric[orbital * (dimension + 1)] = overlap;
    core[orbital * (dimension + 1)] = 1;
    fock[orbital * (dimension + 1)] = overlap * energies[orbital];
    coefficients[orbital * (dimension + 1)] = 1 / std::sqrt(overlap);
  }
  // Closed-form rotation in a nonidentity metric catches transposed C and
  // column/row-major confusion without either production or oracle eigensolves.
  coefficients[0] = .6 / std::sqrt(2.0);
  coefficients[1] = .4;
  coefficients[dimension] = -.8 / std::sqrt(2.0);
  coefficients[dimension + 1] = .3;
  fock[0] = 2 * (-.36 + 3 * .64);
  fock[1] = fock[dimension] = -1.92 * std::sqrt(8.0);
  fock[dimension + 1] = 4 * (-.64 + 3 * .36);
  double expected_energy = .3;
  for (unsigned spin = 0; spin < spin_count; ++spin) {
    auto* density = host.data() + (3 + spin) * matrix;
    const double weight = spin_count == 1 ? 2 : 1;
    if (spin == 0)
      for (std::size_t column = 0; column < 2; ++column)
        for (std::size_t row = 0; row < 2; ++row)
          density[column * dimension + row] = weight * coefficients[row] * coefficients[column];
    for (std::size_t element = 0; element < matrix; ++element)
      expected_energy += .5 * density[element] * (core[element] + fock[element]);
  }
  check(cudaMemcpy(input.get(), host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
  check(cudaMemcpy(values.get(), energies.data(), energies.size() * sizeof(double),
                   cudaMemcpyHostToDevice));
  check(cudaMemset(info.get(), 0, sizeof(int)));
  const std::uint64_t current_generation = 17;
  check(cudaMemcpy(generation.get(), &current_generation, sizeof(current_generation),
                   cudaMemcpyHostToDevice));
  constexpr double canary = 43.25;
  std::vector<double> scratch_host(4 * (matrix + 2), canary);
  check(cudaMemcpy(scratch.get(), scratch_host.data(), scratch_host.size() * sizeof(double),
                   cudaMemcpyHostToDevice));
  std::vector<scf::cuda_df::ValidationPartial> packet_host(partial_count + 2);
  packet_host.front().invalid = packet_host.back().invalid = 12345;
  check(cudaMemcpy(partial.get(), packet_host.data(), packet_host.size() * sizeof(packet_host[0]),
                   cudaMemcpyHostToDevice));
  ResidentFinalValidationWorkspace workspace;
  for (std::size_t index = 0; index < workspace.matrices.size(); ++index)
    workspace.matrices[index] = scratch.get() + index * (matrix + 2) + 1;
  workspace.partial = partial.get() + 1;
  workspace.partial_count = partial_count;
  std::array<scf::cuda_df::ValidationInputs, 2> channels{};
  for (unsigned spin = 0; spin < spin_count; ++spin) {
    auto& channel = channels[spin];
    channel.n = dimension;
    channel.occupied = spin == 0 ? 1 : 0;
    channel.weight = spin_count == 1 ? 2 : 1;
    channel.s = input.get();
    channel.h = input.get() + matrix;
    channel.f = input.get() + 2 * matrix;
    channel.d = input.get() + (3 + spin) * matrix;
    channel.c = input.get() + 5 * matrix;
    channel.values = values.get();
    channel.info = info.get();
    channel.generation = generation.get();
    channel.expected_generation = current_generation;
    channel.physical_fock = true;
  }
  scf::solver::FinalStateDiagnostic diagnostic;
  std::string detail;
  const auto evaluate = [&] {
    return resident_final_state_products(library, std::span(channels.data(), spin_count), workspace,
                                         true, .3, diagnostic, detail);
  };
  if (!evaluate()) throw std::runtime_error("analytic resident products failed: " + detail);
  require(std::abs(diagnostic.energy - expected_energy) < 1e-12,
          "resident energy differs from independent analytic trace");
  for (double error : {diagnostic.maximum_density_error, diagnostic.maximum_canonical_error,
                       diagnostic.maximum_idempotency_error, diagnostic.maximum_commutator,
                       diagnostic.density_rms, diagnostic.maximum_trace_error})
    require(std::isfinite(error) && error < 1e-12, "analytic resident product error");
  require(diagnostic.eigenframes.size() == spin_count, "resident products lost a spin");
  for (const auto& frame : diagnostic.eigenframes)
    require(scf::solver::accept_eigen_frame(frame, detail), "analytic resident eigen gate failed");

  diagnostic.maximum_commutator = 99;
  require(evaluate() && diagnostic.maximum_commutator < 1e-12,
          "resident evidence retained stale diagnostics");
  // A deliberately nonsymmetric F must still produce faithful evidence; the
  // shared physical gate, not this algebra adapter, rejects that candidate.
  // Its two orientations have independent closed-form FC residual maxima.
  const double perturbed_fock = fock[1] + 1e-4;
  check(cudaMemcpy(input.get() + 2 * matrix + 1, &perturbed_fock, sizeof(double),
                   cudaMemcpyHostToDevice));
  for (unsigned spin = 0; spin < spin_count; ++spin) channels[spin].physical_fock = false;
  require(evaluate() && std::abs(diagnostic.eigenframes[0].maximum_eigen_residual -
                                 1e-4 * .8 / std::sqrt(2.0)) < 1e-12,
          "column-major operator residual lost its asymmetric orientation");
  for (unsigned spin = 0; spin < spin_count; ++spin) channels[spin].transposed_operators = true;
  require(
      evaluate() && std::abs(diagnostic.eigenframes[0].maximum_eigen_residual - 1e-4 * .4) < 1e-12,
      "row-major operator residual lost its asymmetric orientation");
  for (unsigned spin = 0; spin < spin_count; ++spin) {
    channels[spin].transposed_operators = false;
    channels[spin].physical_fock = true;
  }
  check(cudaMemcpy(input.get() + 2 * matrix + 1, fock + 1, sizeof(double), cudaMemcpyHostToDevice));
  const auto saved_workspace = workspace;
  workspace.matrices[1] = workspace.matrices[0] + 1;
  rejects(evaluate);
  workspace = saved_workspace;
  workspace.matrices[0] = const_cast<double*>(channels[0].s);
  rejects(evaluate);
  workspace = saved_workspace;
  --workspace.partial_count;
  rejects(evaluate);
  workspace = saved_workspace;
  ++channels[0].expected_generation;
  rejects(evaluate);
  --channels[0].expected_generation;
  const int failed_solver = 1;
  check(cudaMemcpy(info.get(), &failed_solver, sizeof(int), cudaMemcpyHostToDevice));
  rejects(evaluate);
  check(cudaMemset(info.get(), 0, sizeof(int)));
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  rejects(evaluate);
  cudaGraph_t graph{};
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphDestroy(graph));
  std::swap(energies[0], energies[1]);
  check(cudaMemcpy(values.get(), energies.data(), energies.size() * sizeof(double),
                   cudaMemcpyHostToDevice));
  require(!evaluate(), "unsorted resident orbital energies passed");
  std::swap(energies[0], energies[1]);
  check(cudaMemcpy(values.get(), energies.data(), energies.size() * sizeof(double),
                   cudaMemcpyHostToDevice));
  const double overflow = std::numeric_limits<double>::max();
  check(cudaMemcpy(input.get() + 5 * matrix, &overflow, sizeof(double), cudaMemcpyHostToDevice));
  require(!evaluate(), "nonfinite resident products passed");
  check(cudaMemcpy(scratch_host.data(), scratch.get(), scratch_host.size() * sizeof(double),
                   cudaMemcpyDeviceToHost));
  for (std::size_t index = 0; index < 4; ++index)
    require(scratch_host[index * (matrix + 2)] == canary &&
                scratch_host[index * (matrix + 2) + matrix + 1] == canary,
            "resident matrix scratch canary changed");
  std::vector<scf::cuda_df::ValidationPartial> observed_packets(partial_count + 2);
  check(cudaMemcpy(observed_packets.data(), partial.get(),
                   observed_packets.size() * sizeof(observed_packets[0]), cudaMemcpyDeviceToHost));
  require(
      std::memcmp(&observed_packets.front(), &packet_host.front(), sizeof(packet_host[0])) == 0 &&
          std::memcmp(&observed_packets.back(), &packet_host.back(), sizeof(packet_host[0])) == 0,
      "resident reduction scratch canary changed");
}
}  // namespace

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
  try {
    runtime::OwnedCudaStream stream(0);
    MatrixLibraryOwner library;
    require(library.prepare(stream.get(), 37) == GENERATIVEQC_STATUS_SUCCESS &&
                library.library_enabled(),
            "resident validation requires the matrix library");
    for (const auto dimension : {2U, 37U})
      for (const auto spins : {1U, 2U})
        analytic_frames(stream.get(), library.view(), dimension, spins);
    std::cout << "resident final-state validation tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
