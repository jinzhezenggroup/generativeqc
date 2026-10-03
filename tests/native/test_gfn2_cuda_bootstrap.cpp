// Exercise the real molecular CUDA transaction, including first-call failures.
// Oracle values are supplied from the independent tblite fixture by the runner.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/gfn2_cuda_execution.hpp"

using namespace generativeqc::xtb::detail;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class T>
generativeqc_xtb_const_buffer_t input(const std::vector<T>& values) {
  return {values.data(), values.size() * sizeof(T), GENERATIVEQC_XTB_MEMORY_HOST, 0};
}
template <class T>
generativeqc_xtb_buffer_t output(std::vector<T>& values) {
  return {values.data(), values.size() * sizeof(T), GENERATIVEQC_XTB_MEMORY_HOST, 0};
}
struct Molecule {
  std::string name;
  std::vector<std::int32_t> numbers;
  std::vector<double> positions, forces;
  double energy;
};
struct Result {
  std::vector<double> energy{123.0}, forces;
  std::vector<std::int32_t> iterations{-42}, statuses{-43};
  std::vector<std::uint8_t> converged{0xa5};
  std::uint32_t flags = 0;
  generativeqc_xtb_status_t status = -1;
};

Result run(Gfn2CudaExecutionCache& cache, const Molecule& molecule, bool device_input,
           bool forces = true, int maximum_iterations = 300) {
  const auto count = static_cast<std::int64_t>(molecule.numbers.size());
  std::vector<std::int64_t> offsets{0, count};
  std::vector<double> charge{0};
  std::vector<std::int32_t> unpaired{0}, spin{1};
  generativeqc_xtb_batch_t batch{};
  batch.struct_size = sizeof(batch);
  batch.api_version = GENERATIVEQC_XTB_API_VERSION;
  batch.batch_size = 1;
  batch.total_atoms = count;
  batch.atom_offsets = input(offsets);
  batch.atomic_numbers = input(molecule.numbers);
  batch.positions = input(molecule.positions);
  batch.molecular_charges = input(charge);
  batch.unpaired_electrons = input(unpaired);
  batch.spin_channels = input(spin);
  struct DevicePositions {
    double* pointer = nullptr;
    ~DevicePositions() {
      if (pointer) cudaFree(pointer);
    }
  } device;
  if (device_input) {
    check(cudaMalloc(reinterpret_cast<void**>(&device.pointer),
                     molecule.positions.size() * sizeof(double)));
    check(cudaMemcpy(device.pointer, molecule.positions.data(),
                     molecule.positions.size() * sizeof(double), cudaMemcpyHostToDevice));
    batch.positions.data = device.pointer;
    batch.positions.memory_space = GENERATIVEQC_XTB_MEMORY_CUDA_DEVICE;
  }
  generativeqc_xtb_compute_options_t options{};
  options.struct_size = sizeof(options);
  options.api_version = GENERATIVEQC_XTB_API_VERSION;
  options.model = GENERATIVEQC_XTB_MODEL_GFN2_XTB;
  options.flags = GENERATIVEQC_XTB_COMPUTE_ENERGY | (forces ? GENERATIVEQC_XTB_COMPUTE_FORCES : 0);
  options.max_scc_iterations = maximum_iterations;
  options.energy_tolerance = 1e-12;
  options.charge_tolerance = 1e-10;
  options.electronic_temperature = GENERATIVEQC_XTB_DEFAULT_ELECTRONIC_TEMPERATURE;
  options.scc_start_mode = GENERATIVEQC_XTB_SCC_START_FRESH;
  options.scc_mixer = GENERATIVEQC_XTB_SCC_MIXER_MODIFIED_BROYDEN;
  options.scc_mixer_history = 8;
  options.scc_mixer_damping = 0.4;
  Result result;
  if (forces) result.forces.assign(3 * count, -321.0);
  generativeqc_xtb_batch_result_t destination{};
  destination.struct_size = sizeof(destination);
  destination.api_version = GENERATIVEQC_XTB_API_VERSION;
  destination.energies = output(result.energy);
  if (forces) destination.forces = output(result.forces);
  destination.scc_iterations = output(result.iterations);
  destination.scc_converged = output(result.converged);
  destination.per_system_status = output(result.statuses);
  std::string error;
  result.status = execute_restricted_gfn2_cuda(cache, batch, options, destination, error);
  result.flags = destination.flags;
  return result;
}

void oracle(const Result& result, const Molecule& molecule) {
  require(result.status == GENERATIVEQC_XTB_STATUS_SUCCESS &&
              result.statuses[0] == GENERATIVEQC_XTB_STATUS_SUCCESS && result.converged[0] == 1,
          "healthy request did not converge");
  require(std::isfinite(result.energy[0]) && std::abs(result.energy[0] - molecule.energy) < 5e-7,
          "first or recovered energy differs from tblite oracle");
  for (std::size_t i = 0; i < result.forces.size(); ++i)
    require(
        std::isfinite(result.forces[i]) && std::abs(result.forces[i] - molecule.forces[i]) < 5e-7,
        "first or recovered force differs from tblite oracle");
}

void failed(const Result& result) {
  if (result.status != GENERATIVEQC_XTB_STATUS_SUCCESS) {
    // Synchronous admission/refresh rejection must not commit caller bytes.
    require(result.energy[0] == 123.0 && result.iterations[0] == -42 && result.statuses[0] == -43 &&
                result.converged[0] == 0xa5 && result.flags == 0,
            "rejected request changed output or diagnostic sentinels");
    for (double value : result.forces) require(value == -321.0, "rejected force was published");
  } else {
    // A documented terminal SCC failure commits diagnostics and NaN science.
    require(result.statuses[0] != GENERATIVEQC_XTB_STATUS_SUCCESS && result.converged[0] == 0,
            "invalid first request published success");
    require(std::isnan(result.energy[0]), "failed SCC published a finite energy");
    for (double value : result.forces) require(std::isnan(value), "failed SCC published a force");
  }
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected independent fixture path");
    std::ifstream stream(argv[1]);
    int cases = 0;
    stream >> cases;
    require(cases > 0, "empty oracle fixture");
    std::vector<Molecule> molecules;
    for (int i = 0; i < cases; ++i) {
      Molecule molecule;
      int atoms = 0;
      stream >> molecule.name >> atoms >> molecule.energy;
      require(atoms > 0, "invalid oracle atom count");
      molecule.numbers.resize(atoms);
      molecule.positions.resize(3 * atoms);
      molecule.forces.resize(3 * atoms);
      for (int atom = 0; atom < atoms; ++atom) {
        stream >> molecule.numbers[atom];
        for (int k = 0; k < 3; ++k) stream >> molecule.positions[3 * atom + k];
        for (int k = 0; k < 3; ++k) stream >> molecule.forces[3 * atom + k];
      }
      require(bool(stream), "truncated oracle fixture");
      molecules.push_back(std::move(molecule));
    }
    check(cudaSetDevice(0));
    for (bool device_input : {false, true}) {
      Gfn2CudaExecutionCache reused(0, nullptr);
      for (const auto& molecule : molecules) {
        Gfn2CudaExecutionCache cold(0, nullptr);
        const auto reference = run(cold, molecule, device_input);
        oracle(reference, molecule);
        const auto first = run(reused, molecule, device_input);
        oracle(first, molecule);
        require(first.iterations == reference.iterations, "rebuilt cache changed fresh SCC work");
        const auto repeated = run(reused, molecule, device_input);
        oracle(repeated, molecule);
        require(repeated.iterations == reference.iterations, "repeated call reused SCC state");
        oracle(run(reused, molecule, device_input, false), molecule);
        oracle(run(reused, molecule, device_input), molecule);
      }
      const auto found_water = std::find_if(molecules.begin(), molecules.end(),
                                            [](const Molecule& m) { return m.name == "h2o"; });
      require(found_water != molecules.end(), "missing water failure/recovery fixture");
      const auto& water = *found_water;
      for (int fault = 0; fault < 4; ++fault) {
        auto bad = water;
        if (fault == 0) bad.positions[0] = std::numeric_limits<double>::quiet_NaN();
        if (fault == 1) bad.positions[0] = 1e308;
        if (fault == 2)
          std::copy(bad.positions.begin(), bad.positions.begin() + 3, bad.positions.begin() + 3);
        if (fault == 3) bad.numbers[0] = 0;
        Gfn2CudaExecutionCache initially_bad(0, nullptr);
        failed(run(initially_bad, bad, device_input));
        oracle(run(initially_bad, water, device_input), water);
        // A failed replacement with another topology must also leave a prior
        // committed calculation usable. This is a different rollback path.
        oracle(run(reused, molecules.front(), device_input), molecules.front());
        failed(run(reused, bad, device_input));
        oracle(run(reused, molecules.front(), device_input), molecules.front());
        oracle(run(reused, water, device_input), water);
      }
      Gfn2CudaExecutionCache unconverged(0, nullptr);
      const auto terminal = run(unconverged, water, device_input, true, 1);
      require(terminal.status == GENERATIVEQC_XTB_STATUS_SUCCESS &&
                  terminal.statuses[0] == GENERATIVEQC_XTB_STATUS_SCC_NOT_CONVERGED,
              "first nonconverged call lost its terminal diagnostic");
      failed(terminal);
      oracle(run(unconverged, water, device_input), water);
    }
    std::puts(
        "CUDA bootstrap: first-call tblite, host/device ingress, failures and recovery passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "CUDA bootstrap qualification failed: %s\n", error.what());
    return 1;
  }
}
