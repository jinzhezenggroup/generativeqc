#include "cc/iteration_driver.hpp"
#include "cc/solver.hpp"

#if GENERATIVEQC_HAS_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cc/cuda_solver_support.cuh"
#include "cc/cuda_state.cuh"
#include "cc/df_pair_projection.cuh"
#include "cc/df_plan.hpp"
#include "generated_df_ccsd_core_cpu.hpp"
#include "generated_df_ccsd_core_cuda.cuh"
#include "generated_df_ccsd_cuda.cuh"
#include "generated_df_ccsd_hoisted_cuda.cuh"
#include "generated_df_ccsd_spectator_pairs_cuda.cuh"
#include "generated_rccsd_cpu.hpp"
#include "runtime/allocation_measurement.hpp"
#include "solver/diis_ring.hpp"
#include "tensor/cuda_error.hpp"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::cc {
namespace {

using generativeqc_tensor::cuda_check;
// Shared provider storage is charged separately from the compiler's IR arena.
constexpr auto kContractionProviderAllowance = tensor::CudaContractionContext::kProviderAllowance;

std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("RCCSD CUDA size overflow");
  return a * b;
}
std::size_t checked_add(std::size_t a, std::size_t b) { return generated::checked_add(a, b); }
std::size_t align256(std::size_t x) {
  const auto rem = x % 256;
  return rem ? checked_add(x, 256 - rem) : x;
}

struct DeviceScope {
  int previous{};
  explicit DeviceScope(int device) {
    cuda_check(cudaGetDevice(&previous));
    cuda_check(cudaSetDevice(device));
  }
  ~DeviceScope() { cudaSetDevice(previous); }
};

__global__ void damped_advance(const double* current, const double* undamped, std::size_t count,
                               double factor, double* output, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(blockDim.x) * gridDim.x)
    output[i] = generativeqc_tensor::finite(
        __dadd_rn(current[i], __dmul_rn(factor, __dsub_rn(undamped[i], current[i]))), error, 0);
}

// A Q slice is consumed before its borrowed action arena is reused. Preserve
// the first arithmetic failure across every slice and the subsequent core.
__global__ void accumulate_df(const double* values, std::size_t count, double* sum, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(blockDim.x) * gridDim.x)
    sum[i] = generativeqc_tensor::finite(__dadd_rn(sum[i], values[i]), error, 0);
}

struct Layout {
  std::array<std::size_t, 15> inputs{};
  std::size_t iteration{}, replay{}, last_t1{}, last_t2{}, vectors{}, errors{};
  std::size_t gram{}, system{}, coefficients{}, r1_partials{}, r2_partials{}, scalars{};
  std::size_t status{}, generated_error{}, arithmetic{}, total{};
  std::size_t history_bytes{}, metric_weights{};
  std::size_t df_bov{}, df_bvv{}, df_arena{}, df_sum{}, df_prepare{};
  std::size_t df_paired_tau{}, df_pair_maxima{};
};

std::size_t reserve(Layout& layout, std::size_t& cursor, std::size_t bytes) {
  cursor = align256(cursor);
  const auto result = cursor;
  cursor = checked_add(cursor, bytes);
  return result;
}

struct Owner {
  DeviceScope scope;
  int device{};
  cudaStream_t stream{};
  cudaEvent_t trial_begin{}, trial_end{};
  tensor::CudaContractionContext contractions;
  tensor::PreparedContractions conventional_contractions;
  bool conventional_prepared{};
  unsigned char* base{};
  unsigned char* history_base{};
  unsigned char* metric_weights{};
  Layout layout;
  generated::dfcore::CudaState state;
  generated::df::CudaState df_state;
  generated::df::ReplayCudaState replay_state;
  bool replay_matrix{};
  generated::dfhoist::CudaState hoisted_state;
  generated::dfpairs::CudaState paired_state;
  bool pairs_enabled{};
  double* paired_tau{};
  pair_bound::ProjectionMaxima* pair_maxima{};
  std::array<pair_bound::Power, 2> pair_geometry{};
  std::array<pair_bound::Power, 2> pair_factor_geometry{};
  DFIterationPlan plan;
  double *df_bov{}, *df_bvv{}, *df_sum{};
  std::size_t naux{};
  double *last_t1{}, *last_t2{}, *vectors{}, *errors{}, *gram{}, *system{}, *coefficients{};
  double *r1_partials{}, *r2_partials{}, *scalars{};
  int *status{}, *arithmetic{};
  std::size_t n1{}, n2{}, elements{}, partial1{}, partial2{};
  std::size_t history_elements{}, non_history_capacity{};
  bool packed{};
  int* packing_refused{};
  double* pair_asymmetry{};
  generated::RestrictedPairCoordinates coordinates;
  solver::DiisRing history;
  unsigned restarts{};
  SolverDiagnostic diagnostic;

  Owner(const Problem& p, const SolverOptions& options, int ordinal)
      : scope(ordinal),
        device(ordinal),
        n1(checked_mul(p.nocc, p.nvir)),
        n2(checked_mul(checked_mul(p.nocc, p.nocc), checked_mul(p.nvir, p.nvir))),
        elements(checked_add(n1, n2)),
        coordinates{p.nocc, p.nvir},
        history(options.diis_size) {
    packed = options.packed_diis && options.diis_size;
    // A single occupied block has no missing spectator partner to eliminate.
    bool pair_candidate =
        p.nocc > 1 && p.naux && options.df_occupied_pairs && pair_bound::projection_supported;
    // Supplied amplitudes are a contract, not a projected guess. Require
    // exact symmetry here; the bounded rounding policy applies only later.
    if (packed || pair_candidate) {
      for (std::size_t k = 0; k < n2; ++k) {
        const auto mate = coordinates(k).partner;
        if (std::bit_cast<std::uint64_t>(p.initial_t2[k]) !=
            std::bit_cast<std::uint64_t>(p.initial_t2[mate])) {
          diagnostic.packed_diis_refused = packed;
          diagnostic.df_pair_initial_symmetry_refused = pair_candidate;
          packed = pair_candidate = false;
          break;
        }
      }
    }
    history_elements = packed ? checked_add(n1, checked_add(n2, n1) / 2) : elements;
    naux = p.naux;
    conventional_prepared = !naux && generated::iteration_prepared_dimensions_fit(p.nocc, p.nvir);
    // The compiler derives each flattened dimension from contraction labels;
    // neither tensor rank nor o*v alone bounds the provider's signed extents.
    const bool matrix_dimensions_fit =
        !naux || (generated::dfhoist::prepare_packed_dimensions_fit(p.nocc, p.nvir) &&
                  generated::dfhoist::auxiliary_packed_dimensions_fit(p.nocc, p.nvir) &&
                  generated::dfhoist::iteration_packed_dimensions_fit(p.nocc, p.nvir));
    if (naux)
      plan = df_iteration_plan(p.nocc, p.nvir, naux, true, options.df_auxiliary_reduction,
                               options.df_matrix_gemm && matrix_dimensions_fit);
    const std::array<const std::vector<double>*, 15> host = {
        &p.foo,  &p.fov,  &p.fvv, &p.ovov, &p.ovvo,       &p.oovv,       &p.ovvv,         &p.ovoo,
        &p.oooo, &p.vvvv, &p.d1,  &p.d2,   &p.initial_t1, &p.initial_t2, &p.canonical_eps};
    auto build_layout = [&]() {
      layout = {};
      std::size_t cursor = 0;
      for (std::size_t i = 0; i < host.size(); ++i)
        layout.inputs[i] = reserve(layout, cursor, checked_mul(host[i]->size(), sizeof(double)));
      layout.iteration = reserve(
          layout, cursor,
          checked_mul(naux ? plan.iteration : generated::iteration_arena_elements(p.nocc, p.nvir),
                      sizeof(double)));
      layout.replay =
          reserve(layout, cursor,
                  checked_mul(naux ? generated::dfcore::replay_arena_elements(p.nocc, p.nvir)
                                   : generated::replay_arena_elements(p.nocc, p.nvir),
                              sizeof(double)));
      if (naux) {
        layout.df_bov = reserve(layout, cursor, checked_mul(p.df_bov.size(), sizeof(double)));
        layout.df_bvv = reserve(layout, cursor, checked_mul(p.df_bvv.size(), sizeof(double)));
        auto auxiliary =
            replay_matrix ? std::max(plan.auxiliary,
                                     generated::df::virtual_replay_arena_elements(p.nocc, p.nvir))
                          : plan.auxiliary;
        if (pairs_enabled) {
          auxiliary = std::max(auxiliary,
                               generated::dfpairs::auxiliary_packed_arena_elements(p.nocc, p.nvir));
          if (plan.auxiliary_batch_size > 1)
            auxiliary = std::max(auxiliary, generated::dfpairs::auxiliary_batched_arena_elements(
                                                p.nocc, p.nvir, plan.auxiliary_batch_size));
          layout.df_paired_tau =
              reserve(layout, cursor,
                      checked_mul(generated::dfpairs::occupied_tau_elements(p.nocc, p.nvir),
                                  sizeof(double)));
          layout.df_pair_maxima = reserve(layout, cursor, sizeof(pair_bound::ProjectionMaxima));
        }
        layout.df_arena = reserve(layout, cursor, checked_mul(auxiliary, sizeof(double)));
        layout.df_sum = reserve(layout, cursor, checked_mul(plan.accumulation, sizeof(double)));
        if (plan.hoisted)
          layout.df_prepare =
              reserve(layout, cursor, checked_mul(plan.preparation, sizeof(double)));
      }
      layout.last_t1 = reserve(layout, cursor, checked_mul(n1, sizeof(double)));
      layout.last_t2 = reserve(layout, cursor, checked_mul(n2, sizeof(double)));
      // Histories have a separate allocation so refusal can really release
      // the packed payload before admitting a full-layout replacement.
      std::size_t history_cursor = 0;
      layout.vectors =
          reserve(layout, history_cursor,
                  checked_mul(checked_mul(options.diis_size, history_elements), sizeof(double)));
      layout.errors =
          reserve(layout, history_cursor,
                  checked_mul(checked_mul(options.diis_size, history_elements), sizeof(double)));
      layout.history_bytes = align256(history_cursor);
      layout.metric_weights = reserve(layout, cursor, packed ? history_elements : 0);
      layout.gram =
          reserve(layout, cursor,
                  checked_mul(checked_mul(options.diis_size, options.diis_size), sizeof(double)));
      layout.system = reserve(
          layout, cursor,
          checked_mul(checked_mul(options.diis_size + 1, options.diis_size + 1), sizeof(double)));
      layout.coefficients =
          reserve(layout, cursor, checked_mul(options.diis_size + 1, sizeof(double)));
      partial1 = std::min<std::size_t>((n1 + 255) / 256, 65535);
      partial2 = std::min<std::size_t>((n2 + 255) / 256, 65535);
      layout.r1_partials = reserve(layout, cursor, checked_mul(partial1, sizeof(double)));
      layout.r2_partials = reserve(layout, cursor, checked_mul(partial2, sizeof(double)));
      layout.scalars = reserve(layout, cursor, 3 * sizeof(double));
      // Pack the generated-tensor error beside DIIS status so separating
      // generated and DIIS arithmetic state does not increase the aligned arena.
      layout.status = reserve(layout, cursor, 3 * sizeof(int));
      layout.generated_error = checked_add(layout.status, sizeof(int));
      layout.arithmetic = reserve(layout, cursor, sizeof(int));
      layout.total = align256(cursor);

      // Final detached host amplitudes coexist with this resident device arena.
      auto total = checked_add(
          checked_add(
              p.reference_retained_bytes,
              checked_add(problem_host_bytes(p), checked_add(layout.total, layout.history_bytes))),
          checked_mul(elements, sizeof(double)));
      if (conventional_prepared)
        return checked_add(checked_add(total, kContractionProviderAllowance),
                           tensor::PreparedContractions::storage_bytes(
                               generated::iteration_prepared_contractions));
      if (replay_matrix) total = checked_add(total, generated::df::replay_binding_host_bytes());
      if (pairs_enabled)
        total = checked_add(
            total,
            generated::dfpairs::contraction_host_bytes(
                plan.auxiliary_batch_size > 1 ? 1 + (naux % plan.auxiliary_batch_size > 1) : 0));
      return plan.matrix_gemm ? checked_add(checked_add(total, kContractionProviderAllowance),
                                            generated::dfhoist::contraction_host_bytes(
                                                plan.auxiliary_batch_size > 1
                                                    ? 1 + (naux % plan.auxiliary_batch_size > 1)
                                                    : 0))
                              : total;
    };
    auto combined = build_layout();
    if (plan.matrix_gemm) {
      const auto one_q = plan;
      // Trial plans are pure capacity queries. Failed/overflowing optional
      // tiles cannot consume the one-Q arena or weaken the complete budget.
      for (auto batch = std::min(naux, options.df_auxiliary_batch_limit); batch > 1; batch /= 2) {
        try {
          if (!generated::dfhoist::auxiliary_batched_dimensions_fit(p.nocc, p.nvir, batch))
            continue;
          plan = df_iteration_plan(p.nocc, p.nvir, naux, true, options.df_auxiliary_reduction, true,
                                   batch);
          combined = build_layout();
          if (combined <= options.max_bytes) break;
        } catch (const std::length_error&) {
          // Only optional extent/capacity overflow rejects a larger tile.
        }
        plan = one_q;
        combined = build_layout();
      }
    }
    const auto scalar_plan = [&]() {
      if (pairs_enabled) diagnostic.df_pair_resource_refused = true;
      pairs_enabled = false;
      replay_matrix = false;
      if (!naux)
        conventional_prepared = false;
      else
        plan = df_iteration_plan(p.nocc, p.nvir, naux, true, options.df_auxiliary_reduction);
      combined = build_layout();
      if (plan.hoisted && combined > options.max_bytes) {
        plan = df_iteration_plan(p.nocc, p.nvir, naux, true, false);
        combined = build_layout();
      }
      if (combined > options.max_bytes)
        throw std::length_error("RCCSD CUDA scalar fallback exceeds correlation memory budget");
    };
    if ((conventional_prepared || plan.matrix_gemm) && combined > options.max_bytes) scalar_plan();
    if (plan.hoisted && combined > options.max_bytes) {
      plan = df_iteration_plan(p.nocc, p.nvir, naux, true, false);
      combined = build_layout();
    }
    if (combined > options.max_bytes)
      throw std::length_error("RCCSD CUDA resident state exceeds correlation memory budget");
    // Optional audit storage cannot sacrifice the already admitted primal tile.
    // The scalar expanded replay remains available under the original budget.
    if (plan.matrix_gemm && options.df_replay_matrix_gemm) {
      try {
        if (generated::df::virtual_replay_dimensions_fit(p.nocc, p.nvir)) {
          replay_matrix = true;
          const auto candidate = build_layout();
          if (candidate <= options.max_bytes)
            combined = candidate;
          else
            replay_matrix = false;
        }
      } catch (const std::length_error&) {
        replay_matrix = false;
      }
      if (!replay_matrix) combined = build_layout();
    }
    // Pair storage is last in admission priority: it may not displace the
    // already selected original primal tile or independent replay provider.
    if (pair_candidate && plan.hoisted && plan.matrix_gemm) {
      try {
        if (generated::dfpairs::auxiliary_packed_dimensions_fit(p.nocc, p.nvir) &&
            (plan.auxiliary_batch_size == 1 || generated::dfpairs::auxiliary_batched_dimensions_fit(
                                                   p.nocc, p.nvir, plan.auxiliary_batch_size))) {
          pairs_enabled = true;
          const auto candidate = build_layout();
          pairs_enabled = candidate <= options.max_bytes;
          if (pairs_enabled) combined = candidate;
        }
      } catch (const std::length_error&) {
        pairs_enabled = false;
      }
      if (pairs_enabled) {
        for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
          std::array<pair_bound::Power, generated::dfpairs::geometry_norms.size()> norms;
          for (std::size_t index = 0; index < norms.size(); ++index) {
            const auto binding = generated::dfpairs::geometry_norms[index];
            const bool bov = binding.source == generated::dfpairs::NormSource::bov;
            const auto rows = bov ? p.nocc : p.nvir;
            const auto extent = checked_mul(rows, p.nvir);
            const auto* factor = (bov ? p.df_bov.data() : p.df_bvv.data()) + auxiliary * extent;
            norms[index] = pair_bound::row_l1_upper(factor, rows, p.nvir, binding.summed_axes,
                                                    diagnostic.df_pair_geometry_elements);
            // Reuse the existing geometry audit, without another factor scan.
            auto& magnitude = pair_factor_geometry[bov ? 0 : 1];
            magnitude = pair_bound::maximum(magnitude, norms[index]);
          }
          const auto coefficients = generated::dfpairs::ladder_coefficient_bounds(norms);
          for (std::size_t index = 0; index < pair_geometry.size(); ++index)
            pair_geometry[index] = pair_bound::maximum(pair_geometry[index], coefficients[index]);
        }
        for (auto& coefficient : pair_geometry) {
          coefficient = pair_bound::product({coefficient, pair_bound::extent_upper(naux)});
          if (coefficient.state == pair_bound::Power::State::refused) pairs_enabled = false;
        }
      }
      if (!pairs_enabled) {
        diagnostic.df_pair_resource_refused = true;
        combined = build_layout();
      }
    }
    try {
      cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      if (options.diis_size) {
        cuda_check(cudaEventCreate(&trial_begin));
        cuda_check(cudaEventCreate(&trial_end));
      }
      if ((conventional_prepared || plan.matrix_gemm) && !contractions.prepare(stream))
        scalar_plan();
      auto allocate_numeric = [&]() {
        auto code = cudaMalloc(reinterpret_cast<void**>(&base), layout.total);
        if (code) return code;
        // A later history refusal does not erase this successful allocation's
        // peak: the earlier, larger base really coexisted with retained state.
        diagnostic.owned_device_bytes =
            std::max(diagnostic.owned_device_bytes,
                     checked_add(layout.total, (conventional_prepared || plan.matrix_gemm)
                                                   ? kContractionProviderAllowance
                                                   : 0));
        diagnostic.numeric_capacity_bytes =
            std::max(diagnostic.numeric_capacity_bytes, combined - layout.history_bytes);
        if (!layout.history_bytes) return code;
        code = cudaMalloc(reinterpret_cast<void**>(&history_base), layout.history_bytes);
        if (code == cudaErrorMemoryAllocation) {
          (void)cudaGetLastError();
          // The complete numeric pair owns admission. Release the partial
          // arena before retrying a smaller Q tile or the scalar schedule.
          std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
          cuda_check(cudaStreamSynchronize(stream));
          ++diagnostic.synchronizations;
          cuda_check(cudaFree(base));
          base = nullptr;
        }
        return code;
      };
      auto allocation = allocate_numeric();
      if (allocation == cudaErrorMemoryAllocation && pairs_enabled) {
        (void)cudaGetLastError();
        pairs_enabled = false;
        diagnostic.df_pair_resource_refused = true;
        combined = build_layout();
        allocation = allocate_numeric();
      }
      if (allocation == cudaErrorMemoryAllocation && replay_matrix) {
        (void)cudaGetLastError();
        replay_matrix = false;
        combined = build_layout();
        allocation = allocate_numeric();
      }
      if (allocation == cudaErrorMemoryAllocation && plan.auxiliary_batch_size > 1) {
        (void)cudaGetLastError();
        // The same admitted provider can execute one-Q work without the
        // optional batched outputs. Retry before abandoning matrix execution.
        plan = df_iteration_plan(p.nocc, p.nvir, naux, true, options.df_auxiliary_reduction, true);
        combined = build_layout();
        allocation = allocate_numeric();
      }
      if (allocation == cudaErrorMemoryAllocation && (conventional_prepared || plan.matrix_gemm)) {
        // Only optional-resource failure permits retry. Arithmetic and driver
        // failures are propagated, and a retry never changes the equations.
        (void)cudaGetLastError();
        {
          // A release can hide another owner's measured provider growth.
          std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
          contractions.release_locked();
        }
        scalar_plan();
        allocation = allocate_numeric();
      }
      cuda_check(allocation);

      std::array<double**, 15> fields = {&state.foo,  &state.fov,  &state.fvv,          &state.ovov,
                                         &state.ovvo, &state.oovv, &state.ovvv,         &state.ovoo,
                                         &state.oooo, &state.vvvv, &state.d1,           &state.d2,
                                         &state.t1,   &state.t2,   &state.canonical_eps};
      for (std::size_t i = 0; i < host.size(); ++i) {
        *fields[i] = reinterpret_cast<double*>(base + layout.inputs[i]);
        const auto amount = host[i]->size() * sizeof(double);
        if (amount)
          cuda_check(
              cudaMemcpyAsync(*fields[i], host[i]->data(), amount, cudaMemcpyHostToDevice, stream));
        diagnostic.setup_h2d_bytes += amount;
      }
      if (p.canonical_eps.empty()) state.canonical_eps = nullptr;
      if (p.d2.empty()) state.d2 = nullptr;
      state.canonical_level_shift = p.canonical_level_shift;
      diagnostic.denominator_identity = denominator_identity(p);
      state.o = p.nocc;
      state.v = p.nvir;
      state.stream = stream;
      state.iteration_arena = reinterpret_cast<double*>(base + layout.iteration);
      state.replay_arena = reinterpret_cast<double*>(base + layout.replay);
      state.error = reinterpret_cast<int*>(base + layout.generated_error);
      if (conventional_prepared) {
        state.conventional_contractions = &conventional_contractions;
        generated::prepare_iteration_contractions(state, contractions,
                                                  diagnostic.conventional_contraction_calls,
                                                  diagnostic.conventional_contraction_summands);
      }
      if (naux) {
        df_bov = reinterpret_cast<double*>(base + layout.df_bov);
        df_bvv = reinterpret_cast<double*>(base + layout.df_bvv);
        df_sum = reinterpret_cast<double*>(base + layout.df_sum);
        cuda_check(cudaMemcpyAsync(df_bov, p.df_bov.data(), p.df_bov.size() * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        cuda_check(cudaMemcpyAsync(df_bvv, p.df_bvv.data(), p.df_bvv.size() * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        diagnostic.setup_h2d_bytes += (p.df_bov.size() + p.df_bvv.size()) * sizeof(double);
        state.df_virtual_singles = df_sum;
        state.df_virtual_doubles = df_sum + n1;
        df_state.o = p.nocc;
        df_state.v = p.nvir;
        df_state.stream = stream;
        df_state.error = state.error;
        df_state.response_arena = reinterpret_cast<double*>(base + layout.df_arena);
        hoisted_state.prepare_arena = reinterpret_cast<double*>(base + layout.df_prepare);
        hoisted_state.auxiliary_arena = df_state.response_arena;
        if (plan.matrix_gemm) {
          // Symbolic TensorIR requests are resolved/prepared once at owner
          // construction. Every iteration reuses these immutable bindings.
          hoisted_state.o = p.nocc;
          hoisted_state.v = p.nvir;
          generated::dfhoist::prepare_contractions(
              hoisted_state, contractions, plan.auxiliary_batch_size,
              naux % plan.auxiliary_batch_size, diagnostic.df_gemm_calls,
              diagnostic.df_gemm_summands);
        }
        if (replay_matrix) {
          static_cast<generated::df::CudaState&>(replay_state) = df_state;
          try {
            generated::df::prepare_virtual_replay(
                replay_state, contractions, diagnostic.df_gemm_calls, diagnostic.df_gemm_summands);
          } catch (const std::bad_alloc&) {
            replay_state.contractions.release();
            replay_matrix = false;
          }
        }
        if (pairs_enabled) {
          paired_tau = reinterpret_cast<double*>(base + layout.df_paired_tau);
          pair_maxima =
              reinterpret_cast<pair_bound::ProjectionMaxima*>(base + layout.df_pair_maxima);
          paired_state.o = p.nocc;
          paired_state.v = p.nvir;
          try {
            generated::dfpairs::prepare_contractions(
                paired_state, contractions, plan.auxiliary_batch_size,
                naux % plan.auxiliary_batch_size, diagnostic.df_gemm_calls,
                diagnostic.df_gemm_summands);
          } catch (const std::bad_alloc&) {
            paired_state.paired_contractions.release();
            paired_state.paired_batched_contractions.release();
            pairs_enabled = false;
            diagnostic.df_pair_resource_refused = true;
          }
        }
      }
      last_t1 = reinterpret_cast<double*>(base + layout.last_t1);
      last_t2 = reinterpret_cast<double*>(base + layout.last_t2);
      vectors = history_base ? reinterpret_cast<double*>(history_base + layout.vectors) : nullptr;
      errors = history_base ? reinterpret_cast<double*>(history_base + layout.errors) : nullptr;
      metric_weights = packed ? base + layout.metric_weights : nullptr;
      gram = reinterpret_cast<double*>(base + layout.gram);
      system = reinterpret_cast<double*>(base + layout.system);
      coefficients = reinterpret_cast<double*>(base + layout.coefficients);
      r1_partials = reinterpret_cast<double*>(base + layout.r1_partials);
      r2_partials = reinterpret_cast<double*>(base + layout.r2_partials);
      scalars = reinterpret_cast<double*>(base + layout.scalars);
      status = reinterpret_cast<int*>(base + layout.status);
      packing_refused = status + 2;
      pair_asymmetry = scalars + 2;
      cuda_check(cudaMemsetAsync(packing_refused, 0, sizeof(int), stream));
      cuda_check(cudaMemsetAsync(pair_asymmetry, 0, sizeof(double), stream));
      arithmetic = reinterpret_cast<int*>(base + layout.arithmetic);
      cuda_check(cudaMemcpyAsync(last_t1, state.t1, n1 * sizeof(double), cudaMemcpyDeviceToDevice,
                                 stream));
      cuda_check(cudaMemcpyAsync(last_t2, state.t2, n2 * sizeof(double), cudaMemcpyDeviceToDevice,
                                 stream));
      cuda_check(cudaStreamSynchronize(stream));
      ++diagnostic.synchronizations;
      diagnostic.df_matrix_gemm = plan.matrix_gemm;
      diagnostic.df_replay_matrix_gemm = replay_matrix;
      diagnostic.conventional_prepared_contractions = conventional_prepared;
      diagnostic.conventional_provider_capacity_bytes =
          conventional_prepared ? kContractionProviderAllowance : 0;
      diagnostic.conventional_binding_host_bytes =
          conventional_prepared ? tensor::PreparedContractions::storage_bytes(
                                      generated::iteration_prepared_contractions)
                                : 0;
      diagnostic.df_provider_capacity_bytes = plan.matrix_gemm ? kContractionProviderAllowance : 0;
      diagnostic.df_auxiliary_batch_size = plan.auxiliary_batch_size;
      diagnostic.df_occupied_pairs = pairs_enabled;
      if (layout.df_paired_tau) {
        diagnostic.df_pair_capacity_bytes = checked_add(
            checked_mul(generated::dfpairs::occupied_tau_elements(p.nocc, p.nvir), sizeof(double)),
            sizeof(pair_bound::ProjectionMaxima));
        diagnostic.df_pair_binding_host_bytes = generated::dfpairs::contraction_host_bytes(
            plan.auxiliary_batch_size > 1 ? 1 + (naux % plan.auxiliary_batch_size > 1) : 0);
      }
      diagnostic.owned_device_bytes =
          std::max(diagnostic.owned_device_bytes,
                   checked_add(checked_add(layout.total, layout.history_bytes),
                               checked_add(diagnostic.df_provider_capacity_bytes,
                                           diagnostic.conventional_provider_capacity_bytes)));
      diagnostic.numeric_capacity_bytes =
          std::max(diagnostic.numeric_capacity_bytes, std::max(p.provider_peak_bytes, combined));
      diagnostic.packed_diis = packed;
      diagnostic.diis_history_capacity_bytes = layout.history_bytes;
      non_history_capacity = combined - layout.history_bytes;
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~Owner() { cleanup(); }

  void refuse_packed_history(const SolverOptions& options) {
    // Called only after a drained packing audit, before any extrapolation.
    // Reset histories, retaining the current full physical amplitudes/residual.
    std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
    cuda_check(cudaFree(history_base));
    history_base = nullptr;
    vectors = errors = nullptr;
    history.clear();
    packed = false;
    metric_weights = nullptr;
    diagnostic.packed_diis = false;
    diagnostic.packed_diis_refused = true;
    ++restarts;
    history_elements = elements;
    std::size_t cursor = 0;
    layout.vectors = reserve(layout, cursor,
                             checked_mul(checked_mul(options.diis_size, elements), sizeof(double)));
    layout.errors = reserve(layout, cursor,
                            checked_mul(checked_mul(options.diis_size, elements), sizeof(double)));
    const auto amount = align256(cursor);
    bool admitted = amount <= options.max_bytes - non_history_capacity;
    if (admitted) {
      const auto code = cudaMalloc(reinterpret_cast<void**>(&history_base), amount);
      if (code == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();
        admitted = false;
      } else
        cuda_check(code);
    }
    if (!admitted) {
      // A bounded Jacobi continuation uses no optional history. It must still
      // converge under the original physical replay gates before publication.
      history = solver::DiisRing(0);
      layout.history_bytes = 0;
      diagnostic.diis_disabled_after_packing_refusal = true;
      return;
    }
    layout.history_bytes = amount;
    vectors = reinterpret_cast<double*>(history_base + layout.vectors);
    errors = reinterpret_cast<double*>(history_base + layout.errors);
    diagnostic.diis_history_capacity_bytes =
        std::max(diagnostic.diis_history_capacity_bytes, amount);
    diagnostic.owned_device_bytes =
        std::max(diagnostic.owned_device_bytes,
                 checked_add(checked_add(layout.total, amount),
                             checked_add(diagnostic.df_provider_capacity_bytes,
                                         diagnostic.conventional_provider_capacity_bytes)));
    diagnostic.numeric_capacity_bytes =
        std::max(diagnostic.numeric_capacity_bytes, checked_add(non_history_capacity, amount));
  }

  void virtual_corrections() {
    cuda_check(cudaMemsetAsync(state.error, 0, sizeof(int), stream));
    cuda_check(cudaMemsetAsync(df_sum, 0, elements * sizeof(double), stream));
    df_state.t1 = state.t1;
    df_state.t2 = state.t2;
    for (std::size_t q = 0; q < naux; ++q) {
      df_state.bov = df_bov + q * n1;
      df_state.bvv = df_bvv + q * state.v * state.v;
      generated::df::VirtualOutputs out{};
      if (replay_matrix) {
        static_cast<generated::df::CudaState&>(replay_state) = df_state;
        out = generated::df::run_virtual_replay_cuda(replay_state);
        diagnostic.df_packing_bytes = checked_add(
            diagnostic.df_packing_bytes,
            checked_mul(generated::df::virtual_replay_packing_elements(state.o, state.v),
                        2 * sizeof(double)));
      } else {
        out = generated::df::run_virtual_accumulate_cuda(df_state);
      }
      accumulate_df<<<generativeqc_tensor::blocks(static_cast<generativeqc_tensor::I>(n1), 256),
                      256, 0, stream>>>(out.singles, n1, df_sum, state.error);
      accumulate_df<<<generativeqc_tensor::blocks(static_cast<generativeqc_tensor::I>(n2), 256),
                      256, 0, stream>>>(out.doubles, n2, df_sum + n1, state.error);
      ++diagnostic.df_auxiliary_slices;
      ++diagnostic.df_auxiliary_tiles;
      diagnostic.df_virtual_operations += replay_matrix
                                              ? generated::df::virtual_replay_operations
                                              : generated::df::virtual_cuda_operation_count;
      diagnostic.df_accumulation_calls += 2;
      diagnostic.df_accumulation_bytes =
          checked_add(diagnostic.df_accumulation_bytes, checked_mul(elements, 3 * sizeof(double)));
      diagnostic.df_contraction_terms = checked_add(
          diagnostic.df_contraction_terms,
          replay_matrix
              ? generated::df::virtual_replay_contraction_terms(state.o, state.v)
              : generated::dfhoist::fallback_virtual_cuda_contraction_terms(state.o, state.v));
    }
    cuda_check(cudaGetLastError());
  }

  // Geometry coefficients are immutable admission metadata. The original tau
  // and current singles are audited on-device at every trial/primary state;
  // refusal selects the original complete Q loop without altering amplitudes.
  bool prepare_occupied_pairs(double tolerance) {
    if (!pairs_enabled) return false;
    pair_bound::project_tau(hoisted_state.df_tau, state.t1, state.o, state.v, n2, n1, paired_tau,
                            pair_maxima, stream);
    pair_bound::ProjectionMaxima metadata{};
    cuda_check(
        cudaMemcpyAsync(&metadata, pair_maxima, sizeof(metadata), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    ++diagnostic.synchronizations;
    diagnostic.scalar_d2h_bytes += sizeof(metadata);
    ++diagnostic.df_pair_projection_calls;
    diagnostic.df_pair_projection_bytes = checked_add(
        diagnostic.df_pair_projection_bytes,
        checked_mul(
            checked_add(checked_mul(3, generated::dfpairs::occupied_tau_elements(state.o, state.v)),
                        n1),
            sizeof(double)));
    const auto amplitude_extent =
        checked_mul(generated::dfpairs::amplitude_norm_axes & 1 ? state.o : 1,
                    generated::dfpairs::amplitude_norm_axes & 2 ? state.v : 1);
    const auto amplitude =
        pair_bound::product({pair_bound::magnitude_upper(metadata.t1_magnitude_bits),
                             pair_bound::extent_upper(amplitude_extent)});
    const auto error = pair_bound::product(
        {pair_bound::magnitude_upper(metadata.tau_error_bits),
         generated::dfpairs::ladder_residual_gain,
         pair_bound::sum({pair_geometry[0], pair_bound::product({pair_geometry[1], amplitude})})});
    if (metadata.refused || !pair_bound::within_residual_margin(error, tolerance) ||
        (generated::dfpairs::ladder_dressing_factored &&
         !pair_bound::within_factorization_range(metadata, pair_factor_geometry[0],
                                                 pair_factor_geometry[1], state.o, state.v))) {
      ++diagnostic.df_pair_refusals;
      return false;
    }
    static_cast<generated::dfcore::CudaState&>(paired_state) = state;
    paired_state.auxiliary_arena = hoisted_state.auxiliary_arena;
    paired_state.df_tau = hoisted_state.df_tau;
    paired_state.df_tau_occupied_pairs = paired_tau;
    ++diagnostic.df_pair_evaluations;
    return true;
  }

  generated::DeviceIterationOutputs iteration(double tolerance) {
    if (!naux)
      return conventional_prepared ? generated::run_iteration_prepared_cuda(state)
                                   : generated::run_iteration_cuda(state);
    if (plan.hoisted) {
      cuda_check(cudaMemsetAsync(state.error, 0, sizeof(int), stream));
      cuda_check(cudaMemsetAsync(df_sum, 0, plan.accumulation * sizeof(double), stream));
      // Copy borrowed core state; the dedicated preparation arena must survive
      // every Q action and cannot alias its frequently overwritten scratch.
      static_cast<generated::dfcore::CudaState&>(hoisted_state) = state;
      hoisted_state.df_singles_residual = df_sum;
      hoisted_state.df_D05_vv_ladder = df_sum + n1;
      hoisted_state.df_Lvv = df_sum + elements;
      hoisted_state.df_Wvoov = hoisted_state.df_Lvv + state.v * state.v;
      hoisted_state.df_Wvovo = hoisted_state.df_Wvoov + n2;
      hoisted_state.df_Xv = hoisted_state.df_Wvovo + n2;
      hoisted_state.df_tau = generated::dfhoist::run_prepare_cuda(hoisted_state).tau;
      ++diagnostic.df_preparation_calls;
      const bool use_pairs = prepare_occupied_pairs(tolerance);
      for (std::size_t q = 0; q < naux; q += plan.auxiliary_batch_size) {
        const auto batch = std::min(plan.auxiliary_batch_size, naux - q);
        hoisted_state.q = batch;
        hoisted_state.bov = df_bov + q * n1;
        hoisted_state.bvv = df_bvv + q * state.v * state.v;
        generated::dfhoist::AuxiliaryOutputs row{};
        if (use_pairs) {
          paired_state.q = batch;
          paired_state.bov = hoisted_state.bov;
          paired_state.bvv = hoisted_state.bvv;
          row = generated::dfpairs::run_auxiliary_cuda(paired_state);
        } else {
          row = generated::dfhoist::run_auxiliary_cuda(hoisted_state);
        }
        // These read-only core views alias mutable, owner-retained cut storage.
        if (use_pairs)
          generated::dfpairs::accumulate_occupied_auxiliary_cuda(
              paired_state, row, const_cast<double*>(hoisted_state.df_Lvv),
              const_cast<double*>(hoisted_state.df_Wvoov),
              const_cast<double*>(hoisted_state.df_Wvovo), const_cast<double*>(hoisted_state.df_Xv),
              df_sum + n1, df_sum);
        else
          generated::dfhoist::accumulate_auxiliary_cuda(
              hoisted_state, row, const_cast<double*>(hoisted_state.df_Lvv),
              const_cast<double*>(hoisted_state.df_Wvoov),
              const_cast<double*>(hoisted_state.df_Wvovo), const_cast<double*>(hoisted_state.df_Xv),
              df_sum + n1, df_sum);
        diagnostic.df_auxiliary_slices += batch;
        ++diagnostic.df_auxiliary_tiles;
        diagnostic.df_virtual_operations +=
            use_pairs          ? (batch > 1 ? generated::dfpairs::auxiliary_batched_cuda_operations
                                            : generated::dfpairs::auxiliary_packed_cuda_operations)
            : batch > 1        ? generated::dfhoist::auxiliary_batched_operations
            : plan.matrix_gemm ? generated::dfhoist::auxiliary_packed_operations
                               : generated::dfhoist::auxiliary_operation_count;
        ++diagnostic.df_accumulation_calls;
        diagnostic.df_accumulation_bytes =
            checked_add(diagnostic.df_accumulation_bytes,
                        checked_mul(checked_mul(batch + 2, plan.accumulation), sizeof(double)));
        diagnostic.df_contraction_terms = checked_add(
            diagnostic.df_contraction_terms,
            use_pairs ? (batch > 1 ? generated::dfpairs::auxiliary_batched_contraction_terms(
                                         state.o, state.v, batch)
                                   : generated::dfpairs::auxiliary_packed_contraction_terms(
                                         state.o, state.v))
            : batch > 1
                ? generated::dfhoist::auxiliary_batched_contraction_terms(state.o, state.v, batch)
                : plan.auxiliary_terms);
        if (plan.matrix_gemm) {
          const auto packing =
              use_pairs
                  ? (batch > 1
                         ? generated::dfpairs::auxiliary_batched_packing_elements(state.o, state.v,
                                                                                  batch)
                         : generated::dfpairs::auxiliary_packed_packing_elements(state.o, state.v))
              : batch > 1
                  ? generated::dfhoist::auxiliary_batched_packing_elements(state.o, state.v, batch)
                  : generated::dfhoist::auxiliary_packing_elements(state.o, state.v);
          diagnostic.df_packing_bytes =
              checked_add(diagnostic.df_packing_bytes, checked_mul(packing, 2 * sizeof(double)));
        }
      }
      cuda_check(cudaGetLastError());
      ++diagnostic.df_hoisted_evaluations;
      if (plan.matrix_gemm) {
        // Every explicit transpose reads and writes its full tensor once.
        const auto packed =
            checked_add(generated::dfhoist::prepare_packing_elements(state.o, state.v),
                        generated::dfhoist::iteration_packing_elements(state.o, state.v));
        diagnostic.df_packing_bytes =
            checked_add(diagnostic.df_packing_bytes, checked_mul(packed, 2 * sizeof(double)));
      }
      diagnostic.df_contraction_terms = checked_add(
          diagnostic.df_contraction_terms, checked_add(plan.preparation_terms, plan.core_terms));
      return generated::dfhoist::run_iteration_cuda(hoisted_state);
    }
    virtual_corrections();
    diagnostic.df_contraction_terms =
        checked_add(diagnostic.df_contraction_terms,
                    generated::dfhoist::fallback_core_contraction_terms(state.o, state.v));
    return generated::dfcore::run_iteration_cuda(state);
  }

  generated::DeviceReplayOutputs replay() {
    if (!naux) return generated::run_replay_cuda(state);
    virtual_corrections();
    diagnostic.df_contraction_terms =
        checked_add(diagnostic.df_contraction_terms,
                    generated::dfhoist::fallback_replay_contraction_terms(state.o, state.v));
    return generated::dfcore::run_replay_cuda(state);
  }

  void cleanup() noexcept {
    // Serialize every owned release against provider/graph allocation deltas.
    std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
    if (stream) cudaStreamSynchronize(stream);
    contractions.reset_locked();
    if (trial_begin) cudaEventDestroy(trial_begin);
    if (trial_end) cudaEventDestroy(trial_end);
    trial_begin = nullptr;
    trial_end = nullptr;
    if (base) cudaFree(base);
    if (history_base) cudaFree(history_base);
    if (stream) cudaStreamDestroy(stream);
    base = nullptr;
    history_base = nullptr;
    stream = nullptr;
  }

  template <class Output>
  std::array<double, 3> read_status(const Output& out) {
    generativeqc::cc::residual_partials<<<generativeqc_tensor::blocks(
                                              static_cast<generativeqc_tensor::I>(n1), 256),
                                          256, 0, stream>>>(
        out.r1, static_cast<generativeqc_tensor::I>(n1), r1_partials, state.error);
    generativeqc::cc::residual_partials<<<generativeqc_tensor::blocks(
                                              static_cast<generativeqc_tensor::I>(n2), 256),
                                          256, 0, stream>>>(
        out.r2, static_cast<generativeqc_tensor::I>(n2), r2_partials, state.error);
    generativeqc::cc::residual_finish<<<1, 1, 0, stream>>>(r1_partials, static_cast<int>(partial1),
                                                           scalars);
    generativeqc::cc::residual_finish<<<1, 1, 0, stream>>>(r2_partials, static_cast<int>(partial2),
                                                           scalars + 1);
    cuda_check(cudaGetLastError());
    std::array<double, 3> host{};
    int error = 0;
    cuda_check(
        cudaMemcpyAsync(host.data(), out.energy, sizeof(double), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(host.data() + 1, scalars, 2 * sizeof(double), cudaMemcpyDeviceToHost,
                               stream));
    cuda_check(cudaMemcpyAsync(&error, state.error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    diagnostic.scalar_d2h_bytes += 3 * sizeof(double) + sizeof(int);
    ++diagnostic.synchronizations;
    if (error)
      throw std::runtime_error("nonfinite RCCSD generated CUDA tensor at node " +
                               std::to_string(std::abs(error)));
    return host;
  }

  void advance(const generated::DeviceIterationOutputs& out, double factor) {
    cuda_check(
        cudaMemcpyAsync(last_t1, state.t1, n1 * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    cuda_check(
        cudaMemcpyAsync(last_t2, state.t2, n2 * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    cuda_check(cudaMemsetAsync(state.error, 0, sizeof(int), stream));
    damped_advance<<<generativeqc_tensor::blocks(static_cast<generativeqc_tensor::I>(n1), 256), 256,
                     0, stream>>>(last_t1, out.next_t1, n1, factor, state.t1, state.error);
    damped_advance<<<generativeqc_tensor::blocks(static_cast<generativeqc_tensor::I>(n2), 256), 256,
                     0, stream>>>(last_t2, out.next_t2, n2, factor, state.t2, state.error);
    cuda_check(cudaGetLastError());
  }

  void check_generated_error() {
    ++diagnostic.generated_error_checks;
    int host_error = 0;
    cuda_check(
        cudaMemcpyAsync(&host_error, state.error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    diagnostic.scalar_d2h_bytes += sizeof(int);
    ++diagnostic.synchronizations;
    if (host_error)
      throw std::runtime_error("nonfinite RCCSD generated CUDA tensor at node " +
                               std::to_string(std::abs(host_error)));
  }
};

bool run_diis(Owner& s, const SolverOptions& options,
              const generated::DeviceIterationOutputs& trial) {
  if (!s.history.capacity()) return false;
  // Appending to a full ring overwrites the oldest physical row. The live
  // chronological view advances without copying either complete history.
  const auto slot = s.history.push();
  int count = static_cast<int>(s.history.size());
  const int capacity = static_cast<int>(s.history.capacity());
  s.diagnostic.diis_history_insert_bytes = checked_add(
      s.diagnostic.diis_history_insert_bytes, checked_mul(2 * sizeof(double), s.history_elements));
  if (s.packed) {
    generativeqc_tensor::history_insert_orbits<<<
        generativeqc_tensor::blocks(static_cast<generativeqc_tensor::I>(s.elements), 256), 256, 0,
        s.stream>>>(
        s.state.t1, s.state.t2, trial.r1, trial.r2, static_cast<generativeqc_tensor::I>(s.n1),
        static_cast<generativeqc_tensor::I>(s.n2), s.vectors + slot * s.history_elements,
        s.errors + slot * s.history_elements, s.metric_weights, s.coordinates,
        generated::pair_rounding_tolerance, s.packing_refused, s.pair_asymmetry, s.state.error);
    ++s.diagnostic.diis_pack_calls;
    s.diagnostic.diis_conversion_bytes = checked_add(
        s.diagnostic.diis_conversion_bytes,
        checked_add(checked_mul(2 * sizeof(double), checked_add(s.elements, s.history_elements)),
                    s.history_elements));
    int refused = 0;
    cuda_check(cudaMemcpyAsync(&refused, s.packing_refused, sizeof(int), cudaMemcpyDeviceToHost,
                               s.stream));
    cuda_check(cudaMemcpyAsync(&s.diagnostic.diis_maximum_pair_asymmetry, s.pair_asymmetry,
                               sizeof(double), cudaMemcpyDeviceToHost, s.stream));
    s.diagnostic.scalar_d2h_bytes += sizeof(int) + sizeof(double);
    s.check_generated_error();
    if (refused) {
      s.refuse_packed_history(options);
      return run_diis(s, options, trial);
    }
  } else {
    cuda_check(cudaMemcpyAsync(s.vectors + std::size_t(slot) * s.history_elements, s.state.t1,
                               s.n1 * sizeof(double), cudaMemcpyDeviceToDevice, s.stream));
    cuda_check(cudaMemcpyAsync(s.vectors + std::size_t(slot) * s.history_elements + s.n1,
                               s.state.t2, s.n2 * sizeof(double), cudaMemcpyDeviceToDevice,
                               s.stream));
    cuda_check(cudaMemcpyAsync(s.errors + std::size_t(slot) * s.history_elements, trial.r1,
                               s.n1 * sizeof(double), cudaMemcpyDeviceToDevice, s.stream));
    cuda_check(cudaMemcpyAsync(s.errors + std::size_t(slot) * s.history_elements + s.n1, trial.r2,
                               s.n2 * sizeof(double), cudaMemcpyDeviceToDevice, s.stream));
  }
  // Compute the first self norm too: a subsequent insertion reuses it. Retries
  // below only retire a logical row; no old-old residual dot is recomputed.
  generativeqc_tensor::history_gram_row<<<count, 256, 0, s.stream>>>(
      s.errors, static_cast<generativeqc_tensor::I>(s.history_elements), capacity,
      static_cast<int>(s.history.first()), count, static_cast<int>(slot), s.gram, s.metric_weights);
  ++s.diagnostic.diis_gram_calls;
  s.diagnostic.diis_residual_dot_terms =
      checked_add(s.diagnostic.diis_residual_dot_terms, checked_mul(count, s.history_elements));
  if (s.packed) s.diagnostic.diis_metric_weight_terms += checked_mul(count, s.history_elements);
  s.diagnostic.diis_gram_updates = checked_add(s.diagnostic.diis_gram_updates, 2 * count - 1);
  if (count == 1) {
    if (!s.packed) s.check_generated_error();
    cuda_check(cudaGetLastError());
    return false;
  }
  bool state_modified = false;
  bool generated_error_checked = s.packed;
  int generated_error = 0;
  while (count > 1) {
    generativeqc::cc::diis_coefficients<<<1, 1, 0, s.stream>>>(s.gram, count, s.system,
                                                               s.coefficients, s.status, capacity,
                                                               static_cast<int>(s.history.first()));
    ++s.diagnostic.diis_coefficient_calls;
    // The combine kernels already guard on the device-side DIIS status. Queue
    // them before publishing control state so a successful extrapolation needs
    // only one host fence instead of one fence for coefficients and another
    // for arithmetic validation.
    cuda_check(cudaMemsetAsync(s.arithmetic, 0, sizeof(int), s.stream));
    generativeqc_tensor::diis_combine_slice<<<generativeqc_tensor::blocks(
                                                  static_cast<generativeqc_tensor::I>(s.n1), 256),
                                              256, 0, s.stream>>>(
        s.vectors, s.coefficients, static_cast<generativeqc_tensor::I>(s.history_elements), 0,
        static_cast<generativeqc_tensor::I>(s.n1), count, s.status, s.state.t1, s.arithmetic,
        capacity, static_cast<int>(s.history.first()));
    if (s.packed) {
      generativeqc_tensor::diis_combine_orbits<<<
          generativeqc_tensor::blocks(static_cast<generativeqc_tensor::I>(s.n2), 256), 256, 0,
          s.stream>>>(s.vectors, s.coefficients,
                      static_cast<generativeqc_tensor::I>(s.history_elements),
                      static_cast<generativeqc_tensor::I>(s.n1),
                      static_cast<generativeqc_tensor::I>(s.n2), count, s.status, s.state.t2,
                      s.arithmetic, capacity, static_cast<int>(s.history.first()), s.coordinates);
    } else {
      generativeqc_tensor::diis_combine_slice<<<generativeqc_tensor::blocks(
                                                    static_cast<generativeqc_tensor::I>(s.n2), 256),
                                                256, 0, s.stream>>>(
          s.vectors, s.coefficients, static_cast<generativeqc_tensor::I>(s.history_elements),
          static_cast<generativeqc_tensor::I>(s.n1), static_cast<generativeqc_tensor::I>(s.n2),
          count, s.status, s.state.t2, s.arithmetic, capacity, static_cast<int>(s.history.first()));
    }
    s.diagnostic.diis_combine_calls += 2;
    int host_status = 1, arithmetic = 0;
    if (!generated_error_checked)
      cuda_check(cudaMemcpyAsync(&generated_error, s.state.error, sizeof(int),
                                 cudaMemcpyDeviceToHost, s.stream));
    cuda_check(
        cudaMemcpyAsync(&host_status, s.status, sizeof(int), cudaMemcpyDeviceToHost, s.stream));
    cuda_check(
        cudaMemcpyAsync(&arithmetic, s.arithmetic, sizeof(int), cudaMemcpyDeviceToHost, s.stream));
    cuda_check(cudaStreamSynchronize(s.stream));
    s.diagnostic.scalar_d2h_bytes += (generated_error_checked ? 2 : 3) * sizeof(int);
    ++s.diagnostic.synchronizations;
    if (!generated_error_checked) {
      generated_error_checked = true;
      if (generated_error)
        throw std::runtime_error("nonfinite RCCSD generated CUDA tensor at node " +
                                 std::to_string(std::abs(generated_error)));
    }
    if (host_status == 2) break;
    if (host_status == 0) {
      if (s.packed) s.diagnostic.diis_conversion_bytes += checked_mul(sizeof(double), s.n2);
      if (arithmetic) throw std::runtime_error("nonfinite RCCSD CUDA DIIS extrapolation");
      s.diagnostic.diis_combine_terms =
          checked_add(s.diagnostic.diis_combine_terms, checked_mul(count, s.elements));
      state_modified = true;
      break;
    }
    s.history.retire_oldest();
    --count;
    ++s.restarts;
  }
  cuda_check(cudaGetLastError());
  return state_modified;
}

}  // namespace

SolverResult solve_cuda(const Problem& p, const SolverOptions& options, int device) {
  validate_problem(p, true);
  validate_options(options);
  Owner owner(p, options, device);
  SolverResult result;
  result.reason = "maximum RCCSD iterations reached";
  result.correlation_energy = std::numeric_limits<double>::quiet_NaN();
  result.total_energy = std::numeric_limits<double>::quiet_NaN();
  bool use_last = false;
  const auto started = std::chrono::steady_clock::now();

  run_cc_iterations<generated::DeviceIterationOutputs>(
      options,
      [&](const std::optional<generated::DeviceIterationOutputs>& carried) {
        const auto iteration_started = std::chrono::steady_clock::now();
        generated::DeviceIterationOutputs output{};
        if (carried) {
          output = *carried;
        } else {
          output = owner.iteration(options.residual_tolerance);
          ++owner.diagnostic.iteration_graph_calls;
          if (owner.state.canonical_eps)
            owner.diagnostic.derived_d2_iteration_evaluations += owner.n2;
        }
        const auto status = owner.read_status(output);
        owner.diagnostic.iteration_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - iteration_started)
                .count();
        return std::pair{output, IterationMetrics{status[0], status[1], status[2]}};
      },
      [&](unsigned observations, IterationMetrics status, double delta) {
        result.correlation_energy = status.energy;
        result.total_energy = p.reference_energy + status.energy;
        owner.diagnostic.iterations = observations;
        owner.diagnostic.energy_change = delta;
        owner.diagnostic.r1_max = status.r1;
        owner.diagnostic.r2_max = status.r2;
      },
      [&]() {
        const auto replay_started = std::chrono::steady_clock::now();
        const auto replay = owner.replay();
        const auto replay_status = owner.read_status(replay);
        owner.diagnostic.replay_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - replay_started)
                .count();
        ++owner.diagnostic.replay_graph_calls;
        owner.diagnostic.replay_r1_max = replay_status[1];
        owner.diagnostic.replay_r2_max = replay_status[2];
        return IterationMetrics{replay_status[0], replay_status[1], replay_status[2]};
      },
      [&](const generated::DeviceIterationOutputs& output)
          -> std::optional<generated::DeviceIterationOutputs> {
        const auto update_started = std::chrono::steady_clock::now();
        owner.advance(output, 1.0 - options.damping);
        owner.check_generated_error();
        owner.diagnostic.update_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - update_started)
                .count();
        ++owner.diagnostic.update_calls;
        if (owner.history.capacity()) {
          const auto trial_diis_started = std::chrono::steady_clock::now();
          cuda_check(cudaEventRecord(owner.trial_begin, owner.stream));
          const auto trial = owner.iteration(options.residual_tolerance);
          cuda_check(cudaEventRecord(owner.trial_end, owner.stream));
          ++owner.diagnostic.iteration_graph_calls;
          if (owner.state.canonical_eps)
            owner.diagnostic.derived_d2_iteration_evaluations += owner.n2;
          const bool diis_modified_state = run_diis(owner, options, trial);
          // Every successful DIIS path, including the first history push, has
          // already drained this stream past both events. Do not add a timing
          // fence: the trial's completed device interval belongs to iteration,
          // not to DIIS merely because DIIS performs the existing host drain.
          const double trial_diis_seconds =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - trial_diis_started)
                  .count();
          float trial_ms = 0.0F;
          cuda_check(cudaEventElapsedTime(&trial_ms, owner.trial_begin, owner.trial_end));
          // The remainder includes host enqueue/history/control overhead. Clamp
          // across clock domains so neither phase is negative or double counted.
          const double trial_seconds =
              std::clamp(static_cast<double>(trial_ms) * 1e-3, 0.0, trial_diis_seconds);
          owner.diagnostic.iteration_seconds += trial_seconds;
          owner.diagnostic.diis_seconds += trial_diis_seconds - trial_seconds;
          if (!diis_modified_state) return trial;
        }
        return std::nullopt;
      },
      [&]() {
        result.status = SolveStatus::Converged;
        result.reason = "energy change and freshly expanded physical R1/R2 passed on GPU";
      },
      [&](const std::runtime_error& error) {
        const std::string message = error.what();
        if (message.find("nonfinite RCCSD") == std::string::npos) throw;
        result.status = SolveStatus::NumericalFailure;
        result.reason = message;
        use_last = true;
      });
  owner.diagnostic.diis_restarts = owner.restarts;
  owner.diagnostic.tensor_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  result.diagnostic = owner.diagnostic;
  result.t1.resize(owner.n1);
  result.t2.resize(owner.n2);
  const double* final_t1 = use_last ? owner.last_t1 : owner.state.t1;
  const double* final_t2 = use_last ? owner.last_t2 : owner.state.t2;
  cuda_check(cudaMemcpyAsync(result.t1.data(), final_t1, owner.n1 * sizeof(double),
                             cudaMemcpyDeviceToHost, owner.stream));
  cuda_check(cudaMemcpyAsync(result.t2.data(), final_t2, owner.n2 * sizeof(double),
                             cudaMemcpyDeviceToHost, owner.stream));
  cuda_check(cudaStreamSynchronize(owner.stream));
  result.diagnostic.amplitude_d2h_bytes = (owner.n1 + owner.n2) * sizeof(double);
  ++result.diagnostic.synchronizations;
  result.diagnostic.tensor_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return result;
}

}  // namespace generativeqc::cc

#endif
