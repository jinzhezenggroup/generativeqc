#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <mutex>

#include "api/handles.hpp"
#include "dft/cuda_stationary_binding.hpp"
#include "dft/grid_task_view.cuh"
#include "methods/dft_method.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda_fock_execution.hpp"

/** Test-only bridge to the complete gradient at an authenticated FP64 state.
 * It neither reruns SCF nor substitutes a fitted Hamiltonian. The ordinary
 * batch/context lifetime and lock remain authoritative throughout the call. */
extern "C" int mixed_force_endpoint_v2(generativeqc_batch* batch, int mixed, int collect,
                                       double* forces, std::size_t count, std::uint64_t* work,
                                       std::size_t work_count, char* error,
                                       std::size_t error_size) noexcept {
  using namespace generativeqc;
  try {
    if (!batch || !batch->context || !batch->plan || !forces || !work || work_count != 12 ||
        batch->atom_counts.size() != 1 || count != 3 * batch->atom_counts[0])
      throw std::invalid_argument("invalid mixed-force qualification buffers");
    std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
    dft::CudaKsFinalStateToken token;
    std::string detail;
    auto status = methods::detail::dft_final_state_token(*batch->plan, 0, token, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    if (mixed < 0 || mixed > 7) throw std::invalid_argument("invalid precision mask");
    scf::DirectForcePrecisionRequest precision{(mixed & 1) != 0, collect != 0};
    precision.mixed_geometry = (mixed & 2) != 0;
    precision.mixed_grid = (mixed & 4) != 0;
    dft::CudaStationaryResources resources;
    std::vector<double> gradient;
    status = methods::detail::dft_cuda_native_stationary_gradient(
        *batch->plan, 0, token, std::size_t{2} << 30, gradient, resources, detail, &precision);
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    if (gradient.size() != count) throw std::runtime_error("incomplete gradient source result");
    std::transform(gradient.begin(), gradient.end(), forces, [](double value) { return -value; });
    std::copy(precision.work.begin(), precision.work.end(), work);
    work[3] = precision.geometry_points;
    work[4] = precision.grid_points;
    work[5] = precision.geometry_retries;
    work[6] = resources.stationary_work[5];
    work[7] = resources.stationary_work[6];
    work[8] = resources.stationary_work[23];
    work[9] = resources.grid_work.ao_jet_values;
    work[10] = resources.grid_work.projection_fma_pairs;
    work[11] = resources.grid_work.projection_output_values;
    return 0;
  } catch (const std::exception& failure) {
    if (error && error_size) std::snprintf(error, error_size, "%s", failure.what());
    return 1;
  }
}

/** Deliberately overflow the float density cotangent without overflowing FP64.
 * The complete task must be recomputed before scatter; partial FP32 atom
 * updates would contaminate this comparison. No SCF state is forged here. */
extern "C" int mixed_force_retry_probe(double* difference, std::uint64_t* work, char* error,
                                       std::size_t error_size) noexcept {
  using namespace generativeqc;
  try {
    core::System system;
    system.atoms = {{1, {0.0, 0.0, 0.0}}, {1, {0.3, -0.2, 1.4}}};
    system.shells = {{0, 0, {{0.7, 1.0}}}, {1, 0, {{0.8, 1.0}}}};
    std::string detail;
    if (molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail);
    scf::FockBuildSpec spec;
    spec.derivative_order = 0;
    spec.exchange.present = true;
    spec.exchange.coefficient = -0.125;
    const auto strategy = scf::resolve_fock_build(spec, scf::FockBackend::Cuda, 1e-12);
    const scf::PreparedFockPlan fock(system, nullptr, strategy, 0, std::size_t{512} << 20, 1U);
    const std::array<double, 4> density{1e20, 0.2e20, 0.2e20, 0.9e20};
    double* device_density = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_density), sizeof(density)) != cudaSuccess)
      throw std::runtime_error("retry probe allocation failed");
    struct Release {
      double* pointer;
      ~Release() { cudaFree(pointer); }
    } release{device_density};
    if (cudaMemcpy(device_density, density.data(), sizeof(density), cudaMemcpyHostToDevice) !=
        cudaSuccess)
      throw std::runtime_error("retry probe upload failed");
    std::vector<double> strict, mixed;
    auto status = scf::execute_prepared_cuda_direct_shell_full_range_derivatives_device(
        fock, device_density, nullptr, density.size(), strict, detail, false, true);
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    scf::DirectForcePrecisionRequest precision{true, true};
    status = scf::execute_prepared_cuda_direct_shell_full_range_derivatives_device(
        fock, device_density, nullptr, density.size(), mixed, detail, false, true, &precision);
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    double maximum = 0, scale = 0;
    for (std::size_t index = 0; index < strict.size(); ++index) {
      maximum = std::max(maximum, std::abs(strict[index] - mixed[index]));
      scale = std::max(scale, std::abs(strict[index]));
    }
    *difference = maximum / scale;
    std::copy(precision.work.begin(), precision.work.end(), work);
    return 0;
  } catch (const std::exception& failure) {
    if (error && error_size) std::snprintf(error, error_size, "%s", failure.what());
    return 1;
  }
}

/** Exercise a real pending geometry failure before the ordinary geometry drain.
 * An early source failure can leave both the numerical flag and a borrowed
 * stream live. Reset must discard the former, fence the latter, and allow a
 * clean strict replay. This tests transport/recovery, not a molecular oracle. */
extern "C" int mixed_geometry_reset_probe(char* error, std::size_t error_size) noexcept {
  using namespace generativeqc;
  try {
    const auto require = [](bool condition, const char* message) {
      if (!condition) throw std::runtime_error(message);
    };
    const auto* binding = dft::cuda_stationary_binding("pbe0_rks");
    require(binding && binding->geometry_fp32, "PBE0 FP32 geometry binding is unavailable");
    cudaDeviceProp property{};
    require(cudaGetDeviceProperties(&property, 0) == cudaSuccess, "device properties failed");
    struct Buffers {
      cudaStream_t stream{};
      double* values{};
      ~Buffers() {
        if (stream) (void)cudaStreamSynchronize(stream);
        if (values) (void)cudaFree(values);
        if (stream) (void)cudaStreamDestroy(stream);
      }
    } buffers;
    struct Owner {
      const dft::CudaStationaryBinding* binding{};
      void* value{};
      ~Owner() { binding->destroy(value); }
    } owner{binding};
    std::array<char, 512> detail{};
    require(binding->create(0, property.major, property.minor, 2, 2, 2, 64, 1, 1, 1, 8U << 20, 64,
                            16, &owner.value, detail.data(), detail.size()) == 0,
            detail.data());
    const std::array<double, 4> primitives{0.8, 0.7, 0.6, 1.0};
    const std::array<std::int64_t, 4> ranges{0, 1, 1, 1};
    const std::array<double, 2> norms{1, 1}, charges{1, 1};
    const std::array<std::int64_t, 2> ao_atoms{0, 1};
    const std::array<double, 6> centers{0, 0, -0.7, 0, 0, 0.7};
    require(binding->topology(owner.value, primitives.data(), ranges.data(), norms.data(),
                              ao_atoms.data(), detail.data(), detail.size()) == 0,
            detail.data());
    constexpr std::size_t feature_offset = 3, ao_offset = 16, work_offset = 36, weight_offset = 52,
                          raw_offset = 53;
    std::vector<double> inputs(raw_offset + 1, 0.0);
    inputs[0] = 0.3;
    inputs[2] = 0.2;
    inputs[feature_offset] = inputs[feature_offset + 5] = 0.3;
    std::fill(inputs.begin() + ao_offset, inputs.begin() + work_offset, 0.1);
    inputs[weight_offset] = inputs[raw_offset] = 0.5;
    require(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking) == cudaSuccess &&
                cudaMalloc(reinterpret_cast<void**>(&buffers.values),
                           inputs.size() * sizeof(double)) == cudaSuccess,
            "geometry test buffers failed");
    dft::GridTaskView view{};
    view.version = 1;
    view.npoint = 1;
    view.nao = view.nactive = 2;
    view.jets = 10;
    view.points = buffers.values;
    view.features = buffers.values + feature_offset;
    view.ao = buffers.values + ao_offset;
    view.stream = buffers.stream;
    const auto reset = [&] {
      require(binding->reset_geometry(owner.value, centers.data(), 1e-12, detail.data(),
                                      detail.size()) == 0,
              detail.data());
    };
    const auto submit = [&](bool mixed) {
      require(cudaMemcpyAsync(buffers.values, inputs.data(), inputs.size() * sizeof(double),
                              cudaMemcpyHostToDevice, buffers.stream) == cudaSuccess,
              "geometry test upload failed");
      const auto geometry = mixed ? binding->geometry_fp32 : binding->geometry;
      require(geometry(owner.value, &view, buffers.values + work_offset, 0, 1,
                       buffers.values + weight_offset, buffers.values + raw_offset, view.generation,
                       0, detail.data(), detail.size()) == 0,
              detail.data());
    };
    const auto read = [&] {
      std::vector<double> result(binding->source_count * 6, 123.0);
      require(binding->source_span(owner.value, 0, binding->source_count, result.data(),
                                   result.size(), detail.data(), detail.size()) == 0,
              detail.data());
      require(std::all_of(result.begin(), result.end(),
                          [](double value) { return std::isfinite(value); }),
              "recovered geometry is nonfinite");
      return result;
    };
    reset();
    submit(false);
    const auto expected = read();
    for (bool inspect_early : {false, true}) {
      reset();
      std::fill(inputs.begin() + work_offset, inputs.begin() + weight_offset, 1e300);
      submit(true);
      require(cudaStreamSynchronize(buffers.stream) == cudaSuccess,
              "numerical overflow became a CUDA driver failure");
      if (inspect_early)
        require(binding->nuclear(owner.value, charges.data(), charges.size(), detail.data(),
                                 detail.size()) == 1,
                "pending FP32 overflow was not detected by the early source");
      reset();
      std::fill(inputs.begin() + work_offset, inputs.begin() + weight_offset, 0.0);
      submit(false);
      const auto recovered = read();
      for (std::size_t index = 0; index < expected.size(); ++index)
        require(std::abs(recovered[index] - expected[index]) < 1e-12,
                "discarded FP32 source contaminated strict replay");
    }
    return 0;
  } catch (const std::exception& failure) {
    if (error && error_size) std::snprintf(error, error_size, "%s", failure.what());
    return 1;
  }
}
