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
#include "response/low_rank_preconditioner.hpp"
#include "runtime/cuda_resources.cuh"
#include "scf/cuda_direct_jk_device.hpp"
#include "scf/cuda_one_electron_gradient.hpp"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::hf {
namespace {
namespace maps = scf::generated::rhf_frame;
using posthf::checked_add;
using posthf::checked_mul;
using runtime::cuda_resource_check;
constexpr auto kProviderAllowance = tensor::CudaContractionContext::kProviderAllowance;
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
struct FailureDrain {
  cudaStream_t stream;
  ~FailureDrain() {
    if (stream) (void)cudaStreamSynchronize(stream);
  }
};
// Audit the composed exact J/K result. Shared contraction execution audits
// its own intermediates before reuse; both paths retain the same sticky flag.
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
        const RHFFrameResponseOptions& options, bool prepare, RHFFrameResponseResult& diagnostic,
        std::size_t direct_budget, std::size_t arena_elements, std::size_t resident_budget,
        bool& resident_values_prepared)
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
    if (resident_budget) {
      const auto cache_started = Clock::now();
      const auto cache_status =
          scf::prepare_cuda_direct_jk_resident_values(direct.get(), resident_budget, detail);
      stats.resident_jk_setup_seconds = seconds(cache_started);
      if (cache_status != GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
          cache_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED)
        status(cache_status, detail);
      const auto cache_info = scf::cuda_direct_jk_plan_diagnostic(direct.get());
      resident_values_prepared = cache_info.resident_value_bytes != 0;
      stats.resident_jk_reason = detail;
      stats.resident_jk_bytes = cache_info.resident_value_bytes;
      stats.resident_jk_values = cache_info.resident_value_count;
      stats.direct_device_bytes = cache_info.device_bytes;
    }
    stats.numeric_capacity_bytes -= resident_budget - stats.resident_jk_bytes;
    stats.linear_screening_available = scf::cuda_direct_jk_linear_available(direct.get());
    stats.applied_screening =
        stats.linear_screening_available ? options.orbital_screening_tolerance : 0.0;
    stats.requested_screening = options.orbital_screening_tolerance;
    stats.jk_timing_measured = profile;
    if (profile) {
      census.allocate(device, 2, stream);
      jk_start.create(device);
      jk_stop.create(device);
    }
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
    if (prepare && contractions.prepare(stream)) {
      state.contractions = contraction_tables.data();
      maps::prepare_contractions(state, contractions, stats.gemms,
                                 stats.prepared_contraction_summands);
      stats.prepared_contractions = true;
      stats.owned_device_bytes = checked_add(stats.owned_device_bytes, kProviderAllowance);
    } else {
      // Budget or optional setup rejection retains the complete original CUDA
      // traversal. Driver/arithmetic failures still propagate from preparation.
      stats.numeric_capacity_bytes -= stats.contraction_binding_bytes;
      stats.contraction_binding_bytes = 0;
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
  void potential(const double* d, double* out, double threshold = 0.0, bool uncached = false) {
    // Profiling drains matrix-map work before measuring the integral consumer.
    // It is opt-in; ordinary response retains its asynchronous stream contract.
    if (profile) drain();
    const auto started = Clock::now();
    if (profile) jk_start.record(stream);
    const bool linear = stats.linear_screening_available && (uncached || threshold > 0);
    auto spec = scf::make_hf_fock_spec(scf::FockSpin::Restricted);
    spec.derivative_order = 0;
    const bool census_available =
        linear || scf::cuda_direct_jk_value_census_available(direct.get(), spec);
    std::string detail;
    if (linear)
      status(scf::enqueue_cuda_direct_jk_linear_device(direct.get(), spec, d, nn, j, k,
                                                       error.get() + 1, threshold,
                                                       profile ? census.get() : nullptr, detail),
             detail);
    else
      status(scf::enqueue_cuda_direct_jk_device(
                 direct.get(), spec, d, nullptr, nn, j, k, nullptr, error.get() + 1, detail,
                 profile && census_available ? census.get() : nullptr),
             detail);
    combine_jk<<<blocks(nn), 256, 0, stream>>>(j, k, out, nn);
    audit<<<blocks(nn), 256, 0, stream>>>(out, nn, error.get());
    cuda_resource_check(cudaGetLastError());
    if (profile) jk_stop.record(stream);
    ++stats.jk_actions;
    if (stats.resident_jk_bytes && !linear) {
      ++stats.resident_jk_actions;
      stats.resident_jk_value_reads =
          checked_add(stats.resident_jk_value_reads, stats.resident_jk_values);
    }
    if (threshold > 0) ++stats.screened_jk_actions;
    if (profile) {
      std::array<std::uint64_t, 2> counts{};
      FailureDrain fence{stream};
      if (census_available) {
        cuda_resource_check(cudaMemcpyAsync(counts.data(), census.get(), sizeof(counts),
                                            cudaMemcpyDeviceToHost, stream));
        stats.d2h_bytes += sizeof(counts);
      }
      drain();
      fence.stream = nullptr;
      const auto duration = seconds(started);
      const auto device_duration = 1e-3 * jk_stop.elapsed_since(jk_start);
      stats.jk_seconds += duration;
      stats.jk_device_seconds += device_duration;
      if (stats.resident_jk_bytes && !linear) {
        stats.resident_jk_seconds += duration;
        stats.resident_jk_device_seconds += device_duration;
      }
      if (threshold > 0) stats.screened_jk_seconds += duration;
      if (census_available) {
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
    stats.contraction_terms =
        checked_add(stats.contraction_terms, maps::primal_contraction_terms(o, v));
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
    stats.contraction_terms =
        checked_add(stats.contraction_terms, maps::primal_contraction_terms(o, v));
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
    auto* bindings = state.contractions;
    if (scalar) state.contractions = nullptr;
    try {
      begin();
      upload(direction, d_rotation);
      auto dd = maps::run_density_direction_cuda(state);
      potential(dd.density_direction, df, threshold, scalar);
      auto y = download(maps::run_orbital_action_cuda(state).orbital_action, o * v);
      finish();
      std::copy(y.begin(), y.end(), output.begin());
    } catch (...) {
      (void)cudaStreamSynchronize(stream);
      state.contractions = bindings;
      throw;
    }
    state.contractions = bindings;
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
      ++stats.generic_derivative_passes;
    } else {
      status(code, detail);
      ++stats.shell_derivative_passes;
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
  tensor::CudaContractionContext contractions;
  std::array<tensor::PreparedContractions, maps::prepared_stages> contraction_tables;
  runtime::OwnedCudaBuffer<double> storage;
  runtime::OwnedCudaBuffer<int> error;
  runtime::OwnedCudaBuffer<std::uint64_t> census;
  runtime::OwnedCudaEvent jk_start, jk_stop;
  cudaStream_t stream{};
  maps::CudaState state;
  double *c{}, *h{}, *f{}, *rotation{}, *direction{}, *fseed{}, *cseed{}, *density_seed{}, *df{},
      *j{}, *k{}, *density{}, *scratch{}, *reference_overlap{}, *one{};
};
}  // namespace

RHFFrameDFPreconditionerPreparation prepare_rhf_frame_df_preconditioner(
    const core::System& system, const PhysicalReference& ref, std::size_t q,
    std::span<const double> boo, std::span<const double> bov, std::span<const double> bvv,
    std::uint64_t source_identity, std::size_t maximum_bytes) {
  const auto started = std::chrono::steady_clock::now();
  RHFFrameDFPreconditionerPreparation result;
  const auto o = ref.nocc, v = ref.nbf - o;
  require(o && ref.nbf > o && q && source_identity && ref.orbital_energies.size() == ref.nbf,
          "invalid RHF DF preconditioner provenance");
  const auto ov = checked_mul(o, v);
  require(boo.size() == checked_mul(q, checked_mul(o, o)) && bov.size() == checked_mul(q, ov) &&
              bvv.size() == checked_mul(q, checked_mul(v, v)),
          "RHF DF preconditioner factor dimensions differ");
  const auto arena_elements = maps::preconditioner_arena_elements(o, v, q);
  const auto bound = checked_add(RHFFrameIdentity::required_storage_bytes(system, ref),
                                 bytes(checked_add(arena_elements, checked_add(ov, bov.size()))));
  if (bound > maximum_bytes) {
    result.reason = "DF preconditioner preparation budget";
  } else {
    result.numeric_capacity_bytes = bound;
    maps::PreconditionerInputs inputs;
    inputs.eps_o = ref.orbital_energies.data();
    inputs.eps_v = inputs.eps_o + o;
    inputs.boo = boo.data();
    inputs.bov = bov.data();
    inputs.bvv = bvv.data();
    try {
      std::vector<double> arena(arena_elements);
      // This is the complete-map work bound, including a numerically rejected
      // attempt. It is not a completed-work counter for an early failed map.
      result.contraction_terms = maps::preconditioner_contraction_terms(o, v, q);
      const auto output = maps::run_preconditioner_cpu(o, v, q, inputs, arena.data(), arena.size());
      std::vector<double> diagonal(output.diagonal, output.diagonal + ov);
      std::vector<double> low_rank(output.low_rank, output.low_rank + bov.size());
      result.data = std::make_unique<RHFFrameDFPreconditioner>(
          system, ref, q, source_identity, std::move(diagonal), std::move(low_rank));
      result.numeric_capacity_bytes =
          checked_add(result.data->storage_bytes(), bytes(arena.capacity()));
      if (result.numeric_capacity_bytes > maximum_bytes) {
        result.data.reset();
        result.reason = "DF preconditioner actual preparation capacity";
      }
    } catch (const std::bad_alloc&) {
      result.reason = "DF preconditioner preparation allocation";
    } catch (const std::runtime_error&) {
      // Only the optional numerical expression is evaluated here. No exact
      // physical response, provider action or CUDA failure can be swallowed.
      result.reason = "nonfinite DF preconditioner expression";
    }
  }
  result.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return result;
}

static RHFFrameResponseResult rhf_frame_response_cuda_attempt(
    const core::System& system, const PhysicalReference& ref, std::span<const double> bar_f,
    std::span<const double> bar_c, int device, const RHFFrameResponseOptions& options,
    std::unique_ptr<RHFFrameDFPreconditioner> preconditioner, bool& resident_values_prepared,
    std::size_t& attempted_capacity) {
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
  if (total > options.maximum_bytes)
    throw std::length_error("RHF frame response exceeds complete numeric budget");
  RHFFrameResponseResult result;
  const auto binding_bytes = checked_add(kProviderAllowance, maps::prepared_host_bytes());
  const bool prepare =
      maps::prepared_dimensions_fit(o, v) && binding_bytes <= options.maximum_bytes - total;
  result.contraction_binding_bytes = prepare ? binding_bytes : 0;
  total = checked_add(total, result.contraction_binding_bytes);
  // A caller-owned active cache is already live while optional inverse setup
  // copies its inputs. Retire stale/over-budget storage first, or reserve it
  // until the normal cache+image admission below replaces this temporary charge.
  auto* recycling = options.relax_orbitals ? options.recycling : nullptr;
  std::size_t retained_recycle_bytes = 0;
  if (recycling) {
    if (!recycling->matches(system, ref, device, prepare, maps::orbital_action_hash))
      recycling->clear();
    const auto retained = recycling->storage_bytes();
    if (retained > options.maximum_bytes - total) {
      recycling->clear();
    } else {
      retained_recycle_bytes = retained;
      total = checked_add(total, retained);
    }
  }
  std::optional<response::LowRankPreconditioner> inverse;
  const auto setup_started = std::chrono::steady_clock::now();
  if (options.df_preconditioning && options.relax_orbitals && preconditioner) {
    if (!preconditioner->matches(system, ref)) {
      result.preconditioner_reason = "DF preconditioner reference/source identity mismatch";
    } else
      try {
        const auto extra = checked_add(
            preconditioner->storage_bytes(),
            response::LowRankPreconditioner::capacity_bytes(o * v, preconditioner->rank()));
        if (extra <= options.maximum_bytes - total) {
          result.preconditioner_capacity_bytes = extra;
          total = checked_add(total, extra);
          try {
            inverse = response::LowRankPreconditioner::prepare(
                preconditioner->diagonal(), preconditioner->low_rank(), preconditioner->rank(),
                extra - preconditioner->storage_bytes());
            if (!inverse) result.preconditioner_reason = "unsafe DF diagonal or Cholesky";
          } catch (const std::bad_alloc&) {
            result.preconditioner_reason = "DF inverse allocation";
          }
        } else {
          result.preconditioner_reason = "DF inverse response budget";
        }
      } catch (const std::overflow_error&) {
        result.preconditioner_reason = "DF inverse capacity overflow";
      }
  } else if (options.df_preconditioning && options.relax_orbitals) {
    result.preconditioner_reason = "DF preconditioner data unavailable";
  }
  // This owner was transferred, rather than borrowed, so rejection really
  // releases optional storage before the exact diagonal fallback allocates.
  preconditioner.reset();
  result.preconditioner_setup_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_started).count();
  result.df_preconditioned = inverse.has_value();
  result.preconditioner_fallback = options.df_preconditioning && !inverse;
  // Fixed-frame requests do not consume or rebind a retained solved direction,
  // but caller ownership keeps its payload live. Charge it or release it before
  // allocating the physical owner; an unused cache cannot exceed this budget.
  if (!options.relax_orbitals && options.recycling) {
    const auto retained = options.recycling->storage_bytes();
    if (retained > options.maximum_bytes - total) {
      options.recycling->clear();
    } else {
      result.recycle_capacity_bytes = retained;
      total = checked_add(total, retained);
    }
  }
  total -= retained_recycle_bytes;
  if (recycling) {
    // The extra vector holds the independent scalar audit image until all
    // derivative gates pass; it is not reconstructed as RHS + residual.
    const auto image_bytes = bytes(o * v);
    const auto allowance = options.maximum_bytes - total;
    if (recycling->prepare(system, ref, device, prepare, maps::orbital_action_hash,
                           allowance > image_bytes ? allowance - image_bytes : 0)) {
      result.recycle_capacity_bytes = checked_add(recycling->storage_bytes(), image_bytes);
      total = checked_add(total, result.recycle_capacity_bytes);
    } else {
      recycling = nullptr;
    }
  }
  const auto resident_budget = options.resident_jk_allowance(n, options.maximum_bytes - total);
  result.numeric_capacity_bytes = checked_add(total, resident_budget);
  attempted_capacity = result.numeric_capacity_bytes;
  result.operator_hash = maps::orbital_action_hash;
  runtime::CudaDeviceScope device_scope(device);
  Owner owner(system, ref, device, options, prepare, result, direct_bound, arena, resident_budget,
              resident_values_prepared);
  // Cache identity includes the admitted execution policy. Optional provider
  // setup can reject prepared execution after cache admission; do not reuse or
  // publish an image under that now-stale policy. The scalar solve remains valid.
  if (recycling && result.prepared_contractions != prepare) {
    recycling->clear();
    recycling = nullptr;
    result.numeric_capacity_bytes -= result.recycle_capacity_bytes;
    result.recycle_capacity_bytes = 0;
  }
  result.setup_seconds = seconds(started);
  auto phase = Clock::now();
  owner.reference_audit(ref);
  result.reference_audit_seconds = seconds(phase);
  std::vector<double> seed(bar_f.begin(), bar_f.end());
  std::vector<double> exact_image;
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
    auto physical = [&](auto x, auto y) { owner.apply(x, y); };
    auto provisional = [&](auto x, auto y) { owner.apply(x, y, false, result.applied_screening); };
    const auto initial = recycling ? recycling->initial_guess(rhs) : std::span<const double>{};
    result.recycled_guess = !initial.empty();
    phase = Clock::now();
    response::GmresResult z;
    if (inverse) {
      z = response::solve_gmres(zplan, provisional, rhs, initial, {}, [&](auto x, auto y) {
        try {
          inverse->apply(x, y);
        } catch (const std::runtime_error&) {
          // Refuse only this optional numerical callback. Physical actions propagate.
          std::fill(y.begin(), y.end(), std::numeric_limits<double>::quiet_NaN());
        }
      });
    } else {
      z = response::solve_gmres(zplan, provisional, rhs, initial, diagonal);
    }
    result.solve_seconds += seconds(phase);
    const auto audit_solution = [&]() {
      const auto audit_start = Clock::now();
      std::vector<double> residual(o * v);
      // The independent scalar map always uses the exact physical operator.
      owner.apply(z.solution, residual, true, 0.0);
      if (recycling) exact_image = residual;
      for (std::size_t i = 0; i < o * v; ++i) residual[i] -= rhs[i];
      result.orbital_residual = response::stable_norm(residual);
      result.independent_audit_seconds += seconds(audit_start);
    };
    bool exact_retry = false;
    if (result.applied_screening > 0) {
      result.screened_converged = z.converged();
      result.screened_iterations = z.iterations;
      result.screened_operator_actions = z.operator_actions;
      if (z.converged()) {
        audit_solution();
        result.screened_residual = result.orbital_residual;
      }
      exact_retry = !z.converged() || result.orbital_residual > kResidualTolerance;
    } else {
      exact_retry = !z.converged() && (inverse || !initial.empty());
    }
    if (exact_retry) {
      const auto actions = z.operator_actions, iterations = z.iterations,
                 preconditioner_actions = z.preconditioner_actions;
      phase = Clock::now();
      if (result.applied_screening > 0 && z.converged()) {
        // Its warm-start vector is separately admitted. The replacement Krylov
        // workspace is the only live workspace because GMRES owns none on return.
        z = response::solve_gmres(zplan, physical, rhs, z.solution, diagonal);
      } else {
        // Release a refused accelerator's result before the exact diagonal retry.
        z = {};
        z = response::solve_gmres(zplan, physical, rhs, {}, diagonal);
      }
      result.solve_seconds += seconds(phase);
      z.operator_actions += actions;
      z.iterations += iterations;
      z.preconditioner_actions += preconditioner_actions;
      if (result.applied_screening > 0) ++result.exact_refinements;
      if (inverse) {
        result.preconditioner_fallback = true;
        result.preconditioner_reason = "optional accelerator refused; exact diagonal retry";
      }
    }
    if (!z.converged()) throw std::runtime_error("RHF matrix-free Z response did not converge");
    if (result.applied_screening == 0 || exact_retry) audit_solution();
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
    // G is the fixed unscreened linear provider, so E2 is homogeneous quadratic.
    // Symmetric polarization removes one source traversal while retaining the
    // admitted shell consumer's primitive/component reuse. Keep the old three
    // passes for independent schedule comparisons and bounded compatibility.
    auto combined = ref.density;
    for (std::size_t i = 0; i < nn; ++i) combined[i] += result.fock_ao_weights[i];
    auto cross = owner.quadratic_derivative(combined);
    require(cross.size() == result.gradient.size(), "RHF nuclear derivative dimensions differ");
    if (options.symmetric_polarization) {
      // The previous consumer drained before reusing this host input buffer.
      for (std::size_t i = 0; i < nn; ++i) combined[i] = ref.density[i] - result.fock_ao_weights[i];
      auto difference = owner.quadratic_derivative(combined);
      require(difference.size() == cross.size(), "RHF polarization dimensions differ");
      for (std::size_t i = 0; i < cross.size(); ++i)
        result.gradient[i] += 0.5 * (cross[i] - difference[i]);
      result.symmetric_polarization_used = true;
    } else {
      auto density_part = owner.quadratic_derivative(ref.density);
      auto seed_part = owner.quadratic_derivative(result.fock_ao_weights);
      require(density_part.size() == cross.size() && seed_part.size() == cross.size(),
              "RHF nuclear derivative dimensions differ");
      for (std::size_t i = 0; i < cross.size(); ++i)
        result.gradient[i] += cross[i] - density_part[i] - seed_part[i];
    }
  }
  result.two_electron_seconds = seconds(phase);
  require(finite(result.gradient), "nonfinite RHF electronic gradient");
  if (recycling)
    result.recycle_published = recycling->capture(result.orbital_response.solution, exact_image);
  return result;
}
RHFFrameResponseResult rhf_frame_response_cuda(
    const core::System& system, const PhysicalReference& ref, std::span<const double> bar_f,
    std::span<const double> bar_c, int device, const RHFFrameResponseOptions& options,
    std::unique_ptr<RHFFrameDFPreconditioner> preconditioner) {
  bool resident_values_prepared = false;
  std::size_t attempted_capacity = 0;
  const auto started = Clock::now();
  try {
    return rhf_frame_response_cuda_attempt(system, ref, bar_f, bar_c, device, options,
                                           std::move(preconditioner), resident_values_prepared,
                                           attempted_capacity);
  } catch (const std::bad_alloc&) {
    if (!resident_values_prepared) throw;
  }
  // Numeric admission is not a reservation of available device memory. A
  // successful optional lease can crowd out mandatory owner storage or later
  // contraction/derivative temporaries. Unwind every owner before one complete
  // exact retry, rather than moving that allocation failure to a later phase.
  // Only allocation failure permits retry; unrelated CUDA/numerical faults
  // propagate, including a pending execution fault behind an allocation OOM.
  const auto pending = cudaGetLastError();
  if (pending != cudaErrorMemoryAllocation) cuda_resource_check(pending);
  auto recompute = options;
  recompute.resident_jk_maximum_bytes = 0;
  resident_values_prepared = false;
  const auto discarded_seconds = seconds(started);
  const auto discarded_capacity = attempted_capacity;
  std::string retry_reason =
      "resident ERI storage retired after allocation failure; exact recomputation";
  // The transferred optional inverse died with the failed attempt. The exact
  // retry uses the supported diagonal fallback instead of rebuilding it.
  auto result =
      rhf_frame_response_cuda_attempt(system, ref, bar_f, bar_c, device, recompute, nullptr,
                                      resident_values_prepared, attempted_capacity);
  result.numeric_capacity_bytes = std::max(result.numeric_capacity_bytes, discarded_capacity);
  result.resident_jk_discarded_attempt = true;
  result.resident_jk_retry_seconds = discarded_seconds;
  result.resident_jk_reason.swap(retry_reason);
  return result;
}
}  // namespace generativeqc::hf
