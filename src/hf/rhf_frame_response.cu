#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#include "generated_rhf_frame_response_cuda.cuh"
#include "hf/rhf_frame_response.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/allocation_measurement.hpp"
#include "runtime/cuda_resources.cuh"
#include "scf/cuda_direct_jk_device.hpp"
#include "scf/cuda_one_electron_gradient.hpp"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::hf {
namespace {
namespace maps = scf::generated::rhf_frame;
using generativeqc_tensor::blas_check;
using posthf::checked_add;
using posthf::checked_mul;
using runtime::cuda_resource_check;
constexpr std::size_t kProviderAllowance = 96ULL << 20;
constexpr double kReferenceTolerance = 1e-8;
constexpr double kStationarityTolerance = 1e-8;
constexpr double kResidualTolerance = 1e-10;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
std::size_t bytes(std::size_t n) { return checked_mul(n, sizeof(double)); }
bool finite(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); });
}
double maximum(std::span<const double> values) {
  double out = 0;
  for (double x : values) out = std::max(out, std::abs(x));
  return out;
}
void require(bool ok, const char* reason) {
  if (!ok) throw std::invalid_argument(reason);
}
void status(generativeqc_status code, const std::string& detail) {
  if (code == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (code != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
struct DirectDelete {
  void operator()(scf::CudaDirectJkPlan* plan) const noexcept {
    scf::destroy_cuda_direct_jk_plan(plan);
  }
};
struct Blas {
  cublasHandle_t handle{};
  cudaStream_t stream{};
  ~Blas() {
    if (stream) (void)cudaStreamSynchronize(stream);
    if (handle) (void)cublasDestroy(handle);
  }
};
struct FailureDrain {
  cudaStream_t stream;
  ~FailureDrain() {
    if (stream) (void)cudaStreamSynchronize(stream);
  }
};
// Every BLAS result is audited before arena reuse; later zeros cannot mask
// nonfinite intermediates. The flag is sticky across both AD stages and J/K.
__global__ void audit(const double* data, std::size_t count, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(gridDim.x) * blockDim.x)
    if (!isfinite(data[i])) atomicExch(error, 1);
}
__global__ void combine_jk(const double* j, const double* k, double* out, std::size_t count) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(gridDim.x) * blockDim.x)
    out[i] = j[i] - 0.5 * k[i];
}
unsigned blocks(std::size_t count) {
  return static_cast<unsigned>(std::min<std::size_t>((count + 255) / 256, 65535));
}

/** A single exact-provider stream owns matrices, AD arenas and all J/K work.
 * Host buffers crossing async boundaries are drained within their own scope.
 * The enclosing direct plan is declared first and dies after every borrower.
 */
class Owner {
 public:
  Owner(const core::System& system, const PhysicalReference& ref, int device,
        const RHFFrameResponseOptions& options, RHFFrameResponseResult& diagnostic,
        std::size_t direct_budget, std::size_t arena_elements)
      : n(ref.nbf),
        o(ref.nocc),
        v(n - o),
        nn(checked_mul(n, n)),
        device(device),
        stats(diagnostic),
        profile(options.profile_jk) {
    std::string detail;
    scf::CudaDirectJkPlan* raw{};
    scf::CudaDirectJkDiagnostic direct_stats;
    // Immutable unscreened source permits both fixed-mask Krylov actions and
    // zero-screening reference, final residual and nuclear derivative audits.
    status(scf::create_cuda_direct_jk_plan(device, {system}, 1, 0.0, direct_budget, &raw,
                                           direct_stats, detail),
           detail);
    direct.reset(raw);
    stream = scf::cuda_direct_jk_stream(direct.get());
    require(direct_stats.device_bytes <= direct_budget &&
                direct_stats.host_preparation_bytes + direct_stats.host_bytes <=
                    4 * direct_budget + 4 * posthf::source_capacity(system),
            "RHF exact provider exceeded admitted setup inventory");
    stats.direct_device_bytes = direct_stats.device_bytes;
    stats.linear_screening_available = scf::cuda_direct_jk_linear_available(direct.get());
    stats.applied_screening =
        stats.linear_screening_available ? options.orbital_screening_tolerance : 0.0;
    stats.requested_screening = options.orbital_screening_tolerance;
    stats.jk_timing_measured = profile;
    if (profile) census.allocate(device, 2, stream);
    storage.allocate(device, checked_add(checked_mul(14, nn), checked_add(arena_elements, 1)),
                     stream);
    error.allocate(device, 2, stream);
    stats.owned_device_bytes =
        bytes(storage.size()) + 2 * sizeof(int) + census.size() * sizeof(std::uint64_t);
    state.o = o;
    state.v = v;
    state.stream = stream;
    state.error = error.get();
    double* cursor = storage.get();
    auto take = [&] {
      auto* ptr = cursor;
      cursor += nn;
      return ptr;
    };
    c = take();
    h = take();
    f = take();
    rotation = take();
    direction = take();
    fseed = take();
    cseed = take();
    density_seed = take();
    df = take();
    j = take();
    k = take();
    density = take();
    scratch = take();
    reference_overlap = take();
    one = cursor++;
    state.response_arena = cursor;
    state.coefficients = c;
    state.hcore = h;
    state.fock_ao = f;
    state.rotation = rotation;
    state.d_rotation = direction;
    state.bar_fock_mo = fseed;
    state.bar_frame = cseed;
    state.bar_density = density_seed;
    state.d_fock_ao = df;
    state.bar_reference_electronic_energy = one;
    upload(c, ref.coefficients);
    upload(h, ref.hcore);
    upload(f, ref.fock);
    upload(density, ref.density);
    upload(reference_overlap, ref.overlap);
    std::vector<double> identity(nn, 0.0);
    for (std::size_t i = 0; i < n; ++i) identity[i * n + i] = 1.0;
    const double reference_seed = 1.0;
    FailureDrain upload_fence{stream};
    upload(rotation, identity);
    upload(one, {&reference_seed, 1});
    // Protect constructor-local upload sources even when a later setup fails.
    drain();
    upload_fence.stream = nullptr;
    if (options.matrix_blas) {
      std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
      std::size_t before = 0, after = 0, total = 0;
      cuda_resource_check(cudaMemGetInfo(&before, &total));
      blas.stream = stream;
      blas_check(cublasCreate(&blas.handle));
      cuda_resource_check(cudaMemGetInfo(&after, &total));
      require(before <= after || before - after <= kProviderAllowance,
              "RHF BLAS provider exceeded admitted allowance");
      blas_check(cublasSetStream(blas.handle, stream));
      blas_check(cublasSetPointerMode(blas.handle, CUBLAS_POINTER_MODE_HOST));
      blas_check(cublasSetWorkspace(blas.handle, nullptr, 0));
      state.gemm = [&](char ta, char tb, std::size_t m, std::size_t cols, std::size_t inner,
                       double alpha, const double* a, const double* b, double* output) {
        const double beta = 0.0;
        // Row-major C=A*B equals column-major C^T=B^T*A^T.
        blas_check(cublasDgemm(blas.handle, tb == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T,
                               ta == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T, int(cols), int(m), int(inner),
                               &alpha, b, int(tb == 'N' ? cols : inner), a,
                               int(ta == 'N' ? inner : m), &beta, output, int(cols)));
        audit<<<blocks(m * cols), 256, 0, stream>>>(output, m * cols, state.error);
        cuda_resource_check(cudaGetLastError());
        ++stats.gemms;
      };
    }
  }
  void upload(double* target, std::span<const double> source) {
    cuda_resource_check(cudaMemcpyAsync(target, source.data(), source.size_bytes(),
                                        cudaMemcpyHostToDevice, stream));
    stats.h2d_bytes += source.size_bytes();
  }
  void begin() { cuda_resource_check(cudaMemsetAsync(error.get(), 0, 2 * sizeof(int), stream)); }
  void drain() {
    cuda_resource_check(cudaStreamSynchronize(stream));
    ++stats.synchronizations;
  }
  void finish() {
    std::array<int, 2> flags{};
    // All host destinations survive a failed enqueue as well as a failed drain.
    try {
      cuda_resource_check(cudaMemcpyAsync(flags.data(), error.get(), sizeof(flags),
                                          cudaMemcpyDeviceToHost, stream));
      stats.d2h_bytes += sizeof(flags);
      drain();
    } catch (...) {
      (void)cudaStreamSynchronize(stream);
      throw;
    }
    if (flags[0] || flags[1]) throw std::runtime_error("nonfinite RHF frame response arithmetic");
  }
  std::vector<double> download(const double* source, std::size_t count) {
    std::vector<double> result(count);
    try {
      cuda_resource_check(
          cudaMemcpyAsync(result.data(), source, bytes(count), cudaMemcpyDeviceToHost, stream));
      stats.d2h_bytes += bytes(count);
      drain();
    } catch (...) {
      (void)cudaStreamSynchronize(stream);
      throw;
    }
    require(finite(result), "nonfinite RHF frame output");
    return result;
  }
  void potential(const double* d, double* out, double threshold = 0.0) {
    // Profiling drains matrix-map work before measuring the integral consumer.
    // It is opt-in; ordinary response retains its asynchronous stream contract.
    if (profile) drain();
    const auto started = Clock::now();
    const bool linear = stats.linear_screening_available && (profile || threshold > 0);
    auto spec = scf::make_hf_fock_spec(scf::FockSpin::Restricted);
    spec.derivative_order = 0;
    std::string detail;
    if (linear)
      status(scf::enqueue_cuda_direct_jk_linear_device(direct.get(), spec, d, nn, j, k,
                                                       error.get() + 1, threshold,
                                                       profile ? census.get() : nullptr, detail),
             detail);
    else
      status(scf::enqueue_cuda_direct_jk_device(direct.get(), spec, d, nullptr, nn, j, k, nullptr,
                                                error.get() + 1, detail),
             detail);
    combine_jk<<<blocks(nn), 256, 0, stream>>>(j, k, out, nn);
    audit<<<blocks(nn), 256, 0, stream>>>(out, nn, error.get());
    cuda_resource_check(cudaGetLastError());
    ++stats.jk_actions;
    if (threshold > 0) ++stats.screened_jk_actions;
    if (profile) {
      std::array<std::uint64_t, 2> counts{};
      FailureDrain fence{stream};
      if (linear) {
        cuda_resource_check(cudaMemcpyAsync(counts.data(), census.get(), sizeof(counts),
                                            cudaMemcpyDeviceToHost, stream));
        stats.d2h_bytes += sizeof(counts);
      }
      drain();
      fence.stream = nullptr;
      const auto duration = seconds(started);
      stats.jk_seconds += duration;
      if (threshold > 0) stats.screened_jk_seconds += duration;
      if (linear) {
        ++stats.jk_census_actions;
        stats.jk_quartet_visits = checked_add(stats.jk_quartet_visits, counts[0]);
        stats.jk_eri_evaluations = checked_add(stats.jk_eri_evaluations, counts[1]);
      }
    }
  }
  void reference_audit(const PhysicalReference& ref) {
    begin();
    auto primal = maps::run_primal_cuda(state);
    auto d = download(primal.density, nn), fm = download(primal.fock_mo, nn);
    finish();
    double residual = 0;
    for (std::size_t i = 0; i < nn; ++i) {
      residual = std::max(residual, std::abs(d[i] - ref.density[i]));
      residual = std::max(residual,
                          std::abs(fm[i] - (i / n == i % n ? ref.orbital_energies[i / n] : 0.0)));
    }
    state.fock_ao = reference_overlap;
    begin();
    auto overlap = download(maps::run_primal_cuda(state).fock_mo, nn);
    finish();
    state.fock_ao = f;
    for (std::size_t i = 0; i < nn; ++i)
      residual = std::max(residual, std::abs(overlap[i] - (i / n == i % n ? 1.0 : 0.0)));
    begin();
    potential(density, df);
    auto g = download(df, nn);
    finish();
    for (std::size_t i = 0; i < nn; ++i)
      residual = std::max(residual, std::abs(ref.fock[i] - ref.hcore[i] - g[i]));
    stats.reference_residual = residual;
    require(residual <= kReferenceTolerance,
            "RHF response reference is not a consistent canonical exact RHF state");
  }
  void weights(std::span<const double> bar_f, std::span<const double> bar_c) {
    FailureDrain fence{stream};
    begin();
    upload(fseed, bar_f);
    upload(cseed, bar_c);
    auto p = maps::run_potential_seed_cuda(state);
    potential(p.fock_ao_weights, density_seed);
    // The density cotangent lives outside the shared AD arena, which can now
    // be reused by the complete weight map without preserving bar_Fao twice.
    auto weights = maps::run_weights_cuda(state);
    stats.hcore_weights = download(weights.hcore, nn);
    stats.overlap_weights = download(weights.overlap, nn);
    stats.fock_ao_weights = download(weights.fock_ao_weights, nn);
    stats.stationarity = download(weights.stationarity, nn);
    stats.orbital_rhs = download(weights.orbital_rhs, o * v);
    finish();
    fence.stream = nullptr;
    stats.contraction_terms +=
        maps::potential_seed_contraction_terms(o, v) + maps::weights_contraction_terms(o, v);
  }
  void apply(std::span<const double> x, std::span<double> output, bool scalar = false,
             double threshold = 0.0) {
    require(x.size() == o * v && output.size() == o * v, "RHF response action dimension mismatch");
    std::vector<double> d_rotation(nn, 0.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a) {
        d_rotation[i * n + o + a] = x[i * v + a];
        d_rotation[(o + a) * n + i] = -x[i * v + a];
      }
    auto blas_callback = state.gemm;
    if (scalar) state.gemm = {};
    try {
      begin();
      upload(direction, d_rotation);
      auto dd = maps::run_density_direction_cuda(state);
      potential(dd.density_direction, df, threshold);
      auto y = download(maps::run_orbital_action_cuda(state).orbital_action, o * v);
      finish();
      std::copy(y.begin(), y.end(), output.begin());
    } catch (...) {
      (void)cudaStreamSynchronize(stream);
      state.gemm = std::move(blas_callback);
      throw;
    }
    state.gemm = std::move(blas_callback);
    ++stats.orbital_actions;
    stats.contraction_terms += maps::density_direction_contraction_terms(o, v) +
                               maps::orbital_action_contraction_terms(o, v);
  }
  std::vector<double> quadratic_derivative(const std::vector<double>& d) {
    FailureDrain fence{stream};
    upload(scratch, d);
    drain();
    std::vector<double> result;
    std::string detail;
    auto code = scf::execute_cuda_direct_shell_full_range_derivatives_device(
        direct.get(), scf::FockSpin::Restricted, 1.0, -0.5, scratch, nullptr, nn, result, detail);
    if (code == GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
      status(scf::execute_cuda_direct_energy_derivative(
                 direct.get(), scf::make_hf_fock_spec(scf::FockSpin::Restricted), d, {}, result,
                 detail),
             detail);
    } else {
      status(code, detail);
      require(result.size() % 2 == 0, "invalid shell derivative source count");
      const auto coordinates = result.size() / 2;
      for (std::size_t i = 0; i < coordinates; ++i) result[i] += result[coordinates + i];
      result.resize(coordinates);
    }
    ++stats.derivative_passes;
    fence.stream = nullptr;
    return result;
  }
  std::vector<double> bilinear_derivative(std::span<const double> seed) {
    FailureDrain fence{stream};
    upload(scratch, seed);
    std::vector<double> result;
    std::string detail;
    status(
        scf::execute_cuda_direct_bilinear_derivative_device(
            direct.get(), density, scratch, nn, result, profile ? census.get() : nullptr, detail),
        detail);
    if (profile) {
      std::array<std::uint64_t, 2> counts{};
      // Keep the host destination alive if either enqueue or synchronization fails.
      FailureDrain download_fence{stream};
      cuda_resource_check(cudaMemcpyAsync(counts.data(), census.get(), sizeof(counts),
                                          cudaMemcpyDeviceToHost, stream));
      drain();
      download_fence.stream = nullptr;
      stats.derivative_quartet_visits = counts[0];
      stats.derivative_jet_evaluations = counts[1];
      stats.derivative_census_measured = true;
    }
    ++stats.derivative_passes;
    stats.bilinear_derivative_used = true;
    fence.stream = nullptr;
    return result;
  }
  std::size_t n, o, v, nn;
  int device;
  RHFFrameResponseResult& stats;
  bool profile;
  std::unique_ptr<scf::CudaDirectJkPlan, DirectDelete> direct;
  Blas blas;
  runtime::OwnedCudaBuffer<double> storage;
  runtime::OwnedCudaBuffer<int> error;
  runtime::OwnedCudaBuffer<std::uint64_t> census;
  cudaStream_t stream{};
  maps::CudaState state;
  double *c{}, *h{}, *f{}, *rotation{}, *direction{}, *fseed{}, *cseed{}, *density_seed{}, *df{},
      *j{}, *k{}, *density{}, *scratch{}, *reference_overlap{}, *one{};
};
}  // namespace

RHFFrameResponseResult rhf_frame_response_cuda(const core::System& system,
                                               const PhysicalReference& ref,
                                               std::span<const double> bar_f,
                                               std::span<const double> bar_c, int device,
                                               const RHFFrameResponseOptions& options) {
  const auto started = Clock::now();
  require(std::isfinite(options.orbital_screening_tolerance) &&
              options.orbital_screening_tolerance >= 0,
          "invalid orbital Schwarz screening tolerance");
  const auto n = ref.nbf, o = ref.nocc;
  require(device >= 0 && o && o < n && n == molecule::ao_count(system) &&
              system.ecp_terms.empty() && system.electron_count == int(2 * o) &&
              system.multiplicity == 1,
          "invalid all-electron RHF frame response domain");
  const auto v = n - o, nn = checked_mul(n, n);
  require(n <= std::size_t(std::numeric_limits<int>::max()) && bar_f.size() == nn &&
              bar_c.size() == nn && finite(bar_f) && finite(bar_c),
          "invalid RHF frame response seeds");
  std::size_t reference_values = 0;
  for (const auto* matrix :
       {&ref.coefficients, &ref.hcore, &ref.fock, &ref.overlap, &ref.density}) {
    require(matrix->size() == nn && finite(*matrix), "invalid RHF reference matrices");
    reference_values = checked_add(reference_values, matrix->capacity());
  }
  require(ref.orbital_energies.size() == n && finite(ref.orbital_energies),
          "invalid RHF orbital energies");
  reference_values = checked_add(reference_values, checked_add(ref.orbital_energies.capacity(),
                                                               ref.weighted_density.capacity()));
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = o; a < n; ++a)
      require(ref.orbital_energies[a] - ref.orbital_energies[i] > 1e-8,
              "RHF occupied-virtual gap is unresolved");
  auto gmres_options = options.gmres;
  gmres_options.absolute_tolerance = std::min(
      gmres_options.absolute_tolerance > 0 ? gmres_options.absolute_tolerance : 1e-12, 1e-12);
  gmres_options.relative_tolerance = 0.0;
  const auto zplan = response::prepare_gmres(o * v, gmres_options);
  const auto arena =
      std::max({maps::primal_arena_elements(o, v), maps::potential_seed_arena_elements(o, v),
                maps::weights_arena_elements(o, v), maps::density_direction_arena_elements(o, v),
                maps::orbital_action_arena_elements(o, v)});
  std::size_t primitives = 0;
  for (const auto& shell : system.shells)
    primitives = checked_add(primitives, shell.primitives.size());
  const auto direct_bound = scf::cuda_direct_coulomb_device_bytes(
      1, n, system.atoms.size(), system.shells.size(), primitives, 1);
  const auto source = posthf::source_capacity(system);
  // Includes all simultaneously live response/reference copies, scalar staging,
  // three polarization vectors, public outputs and provider host packing. This
  // conservative matrix bound does not grow as (ov)^2 or N^4.
  auto total =
      checked_add(options.caller_bytes, bytes(checked_add(reference_values, checked_mul(66, nn))));
  total = checked_add(total, checked_add(bytes(checked_add(arena, 1)), 8 * sizeof(int)));
  total = checked_add(total, checked_add(checked_mul(5, direct_bound), checked_mul(6, source)));
  const auto derivative_budget =
      checked_add(checked_mul(4, source), checked_add(bytes(checked_mul(16, nn)), 1ULL << 20));
  total = checked_add(total, checked_mul(2, derivative_budget));
  total = checked_add(total, checked_add(zplan.workspace_bytes, 2 * sizeof(std::uint64_t)));
  // An exact corrective solve owns fresh Krylov storage while its warm-start
  // span still borrows the previous solution. Admit that extra live vector.
  if (options.orbital_screening_tolerance > 0) total = checked_add(total, bytes(checked_mul(o, v)));
  if (options.matrix_blas) total = checked_add(total, kProviderAllowance);
  if (total > options.maximum_bytes)
    throw std::length_error("RHF frame response exceeds complete numeric budget");
  RHFFrameResponseResult result;
  result.numeric_capacity_bytes = total;
  result.matrix_blas = options.matrix_blas;
  result.operator_hash = maps::orbital_action_hash;
  runtime::CudaDeviceScope device_scope(device);
  Owner owner(system, ref, device, options, result, direct_bound, arena);
  result.setup_seconds = seconds(started);
  auto phase = Clock::now();
  owner.reference_audit(ref);
  result.reference_audit_seconds = seconds(phase);
  std::vector<double> seed(bar_f.begin(), bar_f.end());
  phase = Clock::now();
  owner.weights(seed, bar_c);
  result.weights_seconds = seconds(phase);
  if (options.relax_orbitals) {
    double same_space = 0;
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        if ((p < o) == (q < o))
          same_space = std::max(same_space, std::abs(result.stationarity[p * n + q]));
    require(same_space <= kStationarityTolerance, "RHF same-space source stationarity failed");
    std::vector<double> diagonal(o * v);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a)
        diagonal[i * v + a] = ref.orbital_energies[o + a] - ref.orbital_energies[i];
    auto rhs = result.orbital_rhs;
    phase = Clock::now();
    auto z = response::solve_gmres(
        zplan, [&](auto x, auto y) { owner.apply(x, y, false, result.applied_screening); }, rhs, {},
        diagonal);
    result.solve_seconds += seconds(phase);
    const auto audit_solution = [&]() {
      const auto audit_start = Clock::now();
      std::vector<double> residual(o * v);
      // Independently lower matrix maps and use the exact physical operator.
      owner.apply(z.solution, residual, true, 0.0);
      for (std::size_t i = 0; i < o * v; ++i) residual[i] -= rhs[i];
      result.orbital_residual = response::stable_norm(residual);
      result.independent_audit_seconds += seconds(audit_start);
    };
    if (result.applied_screening > 0) {
      result.screened_converged = z.converged();
      result.screened_iterations = z.iterations;
      result.screened_operator_actions = z.operator_actions;
      if (z.converged()) {
        audit_solution();
        result.screened_residual = result.orbital_residual;
      }
      if (!z.converged() || result.orbital_residual > kResidualTolerance) {
        // Screening is a provisional solver schedule, never a different force
        // Hamiltonian. Refine against the original operator and unchanged RHS.
        // The old solution remains live and is explicitly admitted above.
        phase = Clock::now();
        z = response::solve_gmres(
            zplan, [&](auto x, auto y) { owner.apply(x, y); }, rhs,
            z.converged() ? std::span<const double>(z.solution) : std::span<const double>{},
            diagonal);
        result.solve_seconds += seconds(phase);
        ++result.exact_refinements;
      }
    }
    if (!z.converged()) throw std::runtime_error("RHF matrix-free Z response did not converge");
    // Always repeat the unscreened audit after a corrective solve. The initial
    // exact solve also requires this gate; a screened GMRES residual is insufficient.
    if (result.applied_screening == 0 || result.exact_refinements) audit_solution();
    if (result.orbital_residual > kResidualTolerance)
      throw std::runtime_error("RHF independent Z residual failed");
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a) seed[i * n + o + a] -= z.solution[i * v + a];
    result.orbital_response = std::move(z);
    phase = Clock::now();
    owner.weights(seed, bar_c);
    result.weights_seconds += seconds(phase);
  }
  result.maximum_stationarity = maximum(result.stationarity);
  if (options.relax_orbitals && result.maximum_stationarity > kStationarityTolerance)
    throw std::runtime_error("RHF complete frame stationarity failed");
  std::string detail;
  scf::OneElectronGradientResources onee;
  phase = Clock::now();
  status(scf::execute_cuda_one_electron_gradient(device, system, result.overlap_weights,
                                                 result.hcore_weights, result.hcore_weights, 0,
                                                 derivative_budget, result.gradient, detail, &onee),
         detail);
  require(onee.device_bytes <= derivative_budget && onee.host_numeric_bytes <= derivative_budget,
          "RHF one-electron derivative exceeded admitted inventory");
  result.one_electron_seconds = seconds(phase);
  phase = Clock::now();
  if (options.bilinear_derivative && scf::cuda_direct_jk_bilinear_preferred(owner.direct.get())) {
    auto cross = owner.bilinear_derivative(result.fock_ao_weights);
    require(cross.size() == result.gradient.size(), "RHF bilinear derivative dimensions differ");
    for (std::size_t i = 0; i < cross.size(); ++i) result.gradient[i] += cross[i];
  } else {
    // Polarization of E2(D)=D:G(D)/2 supplies P:G'(D) in three bounded passes.
    // Retain the fast specialized SPD consumer and the capacity fallback.
    auto combined = ref.density;
    for (std::size_t i = 0; i < nn; ++i) combined[i] += result.fock_ao_weights[i];
    auto cross = owner.quadratic_derivative(combined);
    auto density_part = owner.quadratic_derivative(ref.density);
    auto seed_part = owner.quadratic_derivative(result.fock_ao_weights);
    require(cross.size() == result.gradient.size() && density_part.size() == cross.size() &&
                seed_part.size() == cross.size(),
            "RHF nuclear derivative dimensions differ");
    for (std::size_t i = 0; i < cross.size(); ++i)
      result.gradient[i] += cross[i] - density_part[i] - seed_part[i];
  }
  result.two_electron_seconds = seconds(phase);
  require(finite(result.gradient), "nonfinite RHF electronic gradient");
  return result;
}
}  // namespace generativeqc::hf
