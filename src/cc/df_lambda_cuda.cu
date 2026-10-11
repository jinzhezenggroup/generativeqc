#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "cc/df_lambda.hpp"
#include "generated_df_ccsd_core_cpu.hpp"
#include "generated_df_ccsd_cuda.cuh"
#include "generated_df_lambda_cuda.cuh"
#include "posthf/capacity.hpp"
#include "runtime/allocation_measurement.hpp"
#include "runtime/cuda_resources.cuh"
#include "runtime/df_progress_trace.hpp"

namespace generativeqc::cc::detail {
namespace {
using generativeqc_tensor::cuda_check;
using posthf::checked_add;
using posthf::checked_mul;
namespace generated_core = generated::dfcore;
namespace generated_virtual = generated::df;
namespace generated_response = generated::dflambda;

std::size_t bytes(std::size_t n) { return checked_mul(n, sizeof(double)); }
std::size_t reserve(std::size_t& cursor, std::size_t amount) {
  if (cursor % 256) cursor = checked_add(cursor, 256 - cursor % 256);
  const auto offset = cursor;
  cursor = checked_add(cursor, amount);
  return offset;
}
struct Fence {
  cudaStream_t stream;
  ~Fence() {
    if (stream) (void)cudaStreamSynchronize(stream);
  }
  void complete() noexcept { stream = nullptr; }
};
constexpr auto kProviderAllowance = tensor::CudaContractionContext::kProviderAllowance;
struct Storage {
  tensor::CudaContractionContext contractions;
  cudaStream_t stream{};
  unsigned char* base{};
  ~Storage() {
    // Releasing this owner's buffers must not hide another owner's provider
    // allocation inside its before/after cudaMemGetInfo measurement.
    std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
    if (stream) (void)cudaStreamSynchronize(stream);
    contractions.reset_locked();
    if (base) (void)cudaFree(base);
    if (stream) (void)cudaStreamDestroy(stream);
  }
};

__global__ void accumulate(const double* source, std::size_t count, double* target, int* error) {
  for (std::size_t x = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; x < count;
       x += std::size_t(blockDim.x) * gridDim.x)
    target[x] = generativeqc_tensor::finite(target[x] + source[x], error, 1);
}

using Query = std::size_t (*)(std::size_t, std::size_t);
using MatrixQuery = std::size_t (*)(std::size_t, std::size_t, std::size_t);
using FitQuery = bool (*)(std::size_t, std::size_t, std::size_t);
struct Stage {
  Query arena, work;
  std::size_t operations;
  MatrixQuery matrix_arena, matrix_work, packing;
  FitQuery fit;
  std::size_t matrix_operations;
};
#define GQC_STAGE(name)                                                                          \
  Stage {                                                                                        \
    generated_response::name##_arena_elements, generated_response::name##_contraction_terms,     \
        generated_response::name##_operations, generated_response::name##_matrix_arena_elements, \
        generated_response::name##_matrix_contraction_terms,                                     \
        generated_response::name##_matrix_packing_elements,                                      \
        generated_response::name##_matrix_dimensions_fit,                                        \
        generated_response::name##_matrix_operations                                             \
  }
const Stage primal_prepare = GQC_STAGE(staged_primal_prepare),
            primal_auxiliary = GQC_STAGE(staged_primal_auxiliary),
            core_stage = GQC_STAGE(staged_core), auxiliary_stage = GQC_STAGE(staged_auxiliary),
            prepare_stage = GQC_STAGE(staged_prepare), factor_stage = GQC_STAGE(staged_factors),
            audit_core = GQC_STAGE(audit_core), audit_auxiliary = GQC_STAGE(audit_auxiliary),
            primal_virtual = GQC_STAGE(primal_virtual);
using Runner = generated::DeviceParameterOutput (*)(generated_response::CudaState&);
using StagedRunner = generated::DeviceParameterOutput (*)(generated_response::StagedCudaState&);
struct Parameter {
  std::string_view name;
  Query arena, work;
  std::size_t operations;
  Runner run;
  Query staged_arena, staged_work;
  std::size_t staged_operations;
  StagedRunner staged_run;
  Stage stage;
};
#define GQC_DF_PARAMETER(name)                                                                     \
  Parameter {                                                                                      \
    #name, generated_response::parameter_##name##_arena_elements,                                  \
        generated_response::parameter_##name##_contraction_terms,                                  \
        generated_response::parameter_##name##_operations,                                         \
        generated_response::run_parameter_##name##_cuda,                                           \
        generated_response::staged_parameter_##name##_arena_elements,                              \
        generated_response::staged_parameter_##name##_contraction_terms,                           \
        generated_response::staged_parameter_##name##_operations,                                  \
        generated_response::run_staged_parameter_##name##_cuda, GQC_STAGE(staged_parameter_##name) \
  }
const std::array parameters{GQC_DF_PARAMETER(foo),  GQC_DF_PARAMETER(fov),  GQC_DF_PARAMETER(fvv),
                            GQC_DF_PARAMETER(ovov), GQC_DF_PARAMETER(ovvo), GQC_DF_PARAMETER(oovv),
                            GQC_DF_PARAMETER(ovoo), GQC_DF_PARAMETER(oooo)};
#undef GQC_DF_PARAMETER
#undef GQC_STAGE
}  // namespace

struct DFLambdaActions::Impl {
  runtime::CudaDeviceScope device_scope;
  Storage storage;
  generated_response::CudaState state;
  generated_virtual::CudaState auxiliary;
  generated_response::StagedCudaState staged;
  std::size_t q{}, n1{}, n2{}, vv{};
  const double *bov{}, *bvv{};
  double *sum1{}, *sum2{};
  double *tau{}, *tau_seed{};
  std::array<double*, 6> cuts{}, cut_seeds{};
  std::array<std::size_t, 6> cut_sizes{};
  LambdaDiagnostic metrics;
  bool parameters_admitted{};

  Impl(const Problem& p, const SolverResult& cc, const LambdaOptions& options, int device,
       bool with_source, bool with_parameters)
      : device_scope(device) {
    validate_problem(p, true);
    validate_lambda_options(options);
    if (!p.naux) throw std::invalid_argument("DF Lambda actions require auxiliary factors");
    const auto o = p.nocc, v = p.nvir;
    q = p.naux;
    n1 = checked_mul(o, v);
    n2 = checked_mul(n1, n1);
    vv = checked_mul(v, v);
    if (cc.t1.size() != n1 || cc.t2.size() != n2)
      throw std::invalid_argument("DF Lambda CC amplitude shape mismatch");
    const std::array<const std::vector<double>*, 16> host{
        &p.foo,  &p.fov,  &p.fvv, &p.ovov, &p.ovvo, &p.oovv, &p.ovvv,   &p.ovoo,
        &p.oooo, &p.vvvv, &p.d1,  &p.d2,   &cc.t1,  &cc.t2,  &p.df_bov, &p.df_bvv};
    std::array<std::size_t, 16> offsets{};
    std::size_t cursor = 0;
    for (std::size_t x = 0; x < host.size(); ++x)
      offsets[x] = reserve(cursor, bytes(host[x]->size()));
    std::size_t scratch = std::max({generated_core::replay_arena_elements(o, v),
                                    generated_response::rhs_arena_elements(o, v),
                                    generated_response::transpose_arena_elements(o, v),
                                    generated_response::independent_rhs_arena_elements(o, v),
                                    generated_response::independent_transpose_arena_elements(o, v),
                                    generated_virtual::virtual_cuda_arena_elements(o, v),
                                    generated_virtual::amplitude_vjp_cuda_arena_elements(o, v)});
    if (with_parameters) {
      for (const auto& item : parameters) scratch = std::max(scratch, item.arena(o, v));
      scratch = std::max(scratch, generated_virtual::factor_vjp_cuda_arena_elements(o, v));
    }
    const auto arena = reserve(cursor, bytes(scratch));
    const auto sums1 = reserve(cursor, bytes(n1)), sums2 = reserve(cursor, bytes(n2));
    const auto seed0 = reserve(cursor, sizeof(double)), seed1 = reserve(cursor, bytes(n1)),
               seed2 = reserve(cursor, bytes(n2));
    const auto error = reserve(cursor, sizeof(int));
    metrics.owned_device_bytes = cursor;
    // Conservative simultaneous host bound: both packed layouts (validation
    // and owner), dense seeds/actions/results, packed RHS/audits, GMRES storage,
    // optional external energy seeds and all detached parameter publications.
    const auto pairs = checked_add(n1, n2) / 2, dim = checked_add(n1, pairs),
               dense = checked_add(n1, n2);
    auto host_bytes = checked_add(p.reference_retained_bytes, problem_host_bytes(p));
    host_bytes = checked_add(host_bytes, bytes(checked_add(cc.t1.capacity(), cc.t2.capacity())));
    host_bytes = checked_add(host_bytes, bytes(checked_mul(6, checked_add(dense, dim))));
    host_bytes = checked_add(
        host_bytes, checked_mul(2, checked_add(bytes(dim), checked_mul(checked_mul(2, pairs),
                                                                       sizeof(std::size_t)))));
    host_bytes =
        checked_add(host_bytes, response::prepare_gmres(dim, options.gmres).workspace_bytes);
    if (with_source) host_bytes = checked_add(host_bytes, bytes(checked_add(dim, dense)));
    if (with_parameters) {
      std::size_t outputs = checked_add(p.df_bov.size(), p.df_bvv.size());
      for (const auto* values :
           {&p.foo, &p.fov, &p.fvv, &p.ovov, &p.ovvo, &p.oovv, &p.ovoo, &p.oooo})
        outputs = checked_add(outputs, values->size());
      host_bytes = checked_add(host_bytes, bytes(outputs));
    }
    const auto fallback_cursor = cursor;
    std::size_t available_device_bytes = 0, total_device_bytes = 0;
    {
      std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
      cuda_check(cudaMemGetInfo(&available_device_bytes, &total_device_bytes));
    }
    const auto device_limit = std::min(available_device_bytes, options.df_max_device_bytes);
    metrics.df_available_device_bytes = available_device_bytes;
    metrics.df_device_limit_bytes = device_limit;
    const auto device_fits = [&](std::size_t candidate, std::size_t allowance) {
      return checked_add(candidate, allowance) <= device_limit;
    };
    std::size_t staged_arena = arena, tau_offset = 0, tau_seed_offset = 0;
    std::array<std::size_t, 6> cut_offsets{}, seed_offsets{};
    cut_sizes = {vv, n2, n2, n2, n2, n1};
    if (options.df_auxiliary_reduction) {
      // All cached cuts belong to this immutable Problem/T owner. The original
      // arena remains available for expanded physical replay and Lambda audit.
      const auto before = checked_add(
          generated_response::transpose_contraction_terms(o, v),
          checked_mul(q, generated_response::virtual_amplitude_vjp_contraction_terms(o, v)));
      const auto after = checked_add(
          generated_response::staged_core_contraction_terms(o, v),
          checked_add(checked_mul(q, generated_response::staged_auxiliary_contraction_terms(o, v)),
                      generated_response::staged_prepare_contraction_terms(o, v)));
      auto candidate = cursor;
      auto staged_scratch =
          std::max({generated_response::staged_primal_prepare_arena_elements(o, v),
                    generated_response::staged_primal_auxiliary_arena_elements(o, v),
                    generated_response::staged_core_arena_elements(o, v),
                    generated_response::staged_auxiliary_arena_elements(o, v),
                    generated_response::staged_prepare_arena_elements(o, v)});
      if (with_parameters) {
        staged_scratch =
            std::max(staged_scratch, generated_response::staged_factors_arena_elements(o, v));
        for (const auto& item : parameters)
          staged_scratch = std::max(staged_scratch, item.staged_arena(o, v));
      }
      if (staged_scratch > scratch) staged_arena = reserve(candidate, bytes(staged_scratch));
      tau_offset = reserve(candidate, bytes(n2));
      tau_seed_offset = reserve(candidate, bytes(n2));
      for (std::size_t x = 0; x < cuts.size(); ++x) {
        cut_offsets[x] = reserve(candidate, bytes(cut_sizes[x]));
        seed_offsets[x] = reserve(candidate, bytes(cut_sizes[x]));
      }
      if (after < before && checked_add(host_bytes, candidate) <= options.max_bytes &&
          device_fits(candidate, 0)) {
        metrics.df_auxiliary_reduction = true;
        cursor = candidate;
      }
    }
    // Keep the scalar staged layout as a bounded fallback. The matrix arena
    // is optional and includes explicit IR packing/broadcast intermediates.
    const auto scalar_cursor = cursor, scalar_staged_arena = staged_arena;
    const bool scalar_reduction = metrics.df_auxiliary_reduction;
    std::size_t binding_host_bytes = 0, core_reuse_offset = 0, audit_offset = 0;
    std::size_t matrix_scratch_elements = 0;
    auto capacity = [&] {
      metrics.owned_device_bytes = checked_add(cursor, metrics.df_provider_allowance_bytes);
      metrics.numeric_capacity_bytes =
          checked_add(checked_add(host_bytes, binding_host_bytes), metrics.owned_device_bytes);
    };
    auto scalar_plan = [&] {
      binding_host_bytes = 0;
      matrix_scratch_elements = 0;
      metrics.df_matrix_gemm = false;
      metrics.df_core_reuse = false;
      metrics.df_core_reuse_bytes = 0;
      metrics.df_audit_matrix_gemm = false;
      metrics.df_audit_arena_bytes = 0;
      metrics.df_primal_matrix_gemm = false;
      metrics.df_auxiliary_batch_size = 1;
      metrics.df_provider_allowance_bytes = 0;
      metrics.df_auxiliary_reduction = scalar_reduction;
      staged_arena = scalar_staged_arena;
      cursor = scalar_cursor;
      capacity();
    };
    if (scalar_reduction && options.df_matrix_gemm) {
      for (auto batch = std::min(q, options.df_auxiliary_batch_limit); batch; batch /= 2) {
        std::size_t matrix_scratch = 0;
        bool fits = true;
        auto include = [&](const Stage& stage, std::size_t extent) {
          fits = fits && stage.fit(o, v, extent);
          matrix_scratch = std::max(matrix_scratch, stage.matrix_arena(o, v, extent));
        };
        include(primal_prepare, 1);
        include(primal_auxiliary, batch);
        include(core_stage, 1);
        include(auxiliary_stage, batch);
        include(prepare_stage, 1);
        if (with_parameters) {
          include(factor_stage, batch);
          for (const auto& item : parameters) include(item.stage, 1);
        }
        auto candidate = scalar_cursor;
        const auto matrix_arena = reserve(candidate, bytes(matrix_scratch));
        const auto descriptor_bytes =
            generated_response::contraction_host_bytes(q % batch ? 2 : 1, with_parameters);
        if (fits &&
            checked_add(checked_add(checked_add(host_bytes, descriptor_bytes), candidate),
                        kProviderAllowance) <= options.max_bytes &&
            device_fits(candidate, kProviderAllowance)) {
          binding_host_bytes = descriptor_bytes;
          metrics.df_matrix_gemm = true;
          metrics.df_auxiliary_batch_size = batch;
          metrics.df_provider_allowance_bytes = kProviderAllowance;
          staged_arena = matrix_arena;
          matrix_scratch_elements = matrix_scratch;
          cursor = candidate;
          break;
        }
      }
    }
    const auto operator_cursor = cursor, operator_binding_host_bytes = binding_host_bytes;
    auto drop_audit = [&] {
      metrics.df_audit_matrix_gemm = false;
      metrics.df_audit_arena_bytes = 0;
      metrics.df_primal_matrix_gemm = false;
      cursor = operator_cursor;
      binding_host_bytes = operator_binding_host_bytes;
      capacity();
    };
    if (metrics.df_matrix_gemm && options.df_audit_matrix_gemm) {
      const auto batch = metrics.df_auxiliary_batch_size;
      if (audit_core.fit(o, v, 1) && audit_auxiliary.fit(o, v, batch)) {
        const auto audit_bytes = bytes(
            std::max(audit_core.matrix_arena(o, v, 1), audit_auxiliary.matrix_arena(o, v, batch)));
        const auto descriptors =
            generated_response::audit_contraction_host_bytes(q % batch ? 2 : 1);
        const bool replay_fit = options.df_primal_matrix_gemm && primal_virtual.fit(o, v, batch);
        // Fresh replay and the later independent audit have disjoint lifetimes.
        // Replay storage refusal must not refuse the existing matrix audit.
        for (const bool replay : {replay_fit, false}) {
          auto candidate = cursor;
          const auto arena_bytes =
              replay ? std::max(audit_bytes, bytes(primal_virtual.matrix_arena(o, v, batch)))
                     : audit_bytes;
          const auto offset = reserve(candidate, arena_bytes);
          if (checked_add(checked_add(checked_add(host_bytes, binding_host_bytes), descriptors),
                          checked_add(candidate, metrics.df_provider_allowance_bytes)) <=
                  options.max_bytes &&
              device_fits(candidate, metrics.df_provider_allowance_bytes)) {
            metrics.df_audit_matrix_gemm = true;
            metrics.df_audit_arena_bytes = arena_bytes;
            metrics.df_primal_matrix_gemm = replay;
            cursor = candidate;
            audit_offset = offset;
            binding_host_bytes = checked_add(binding_host_bytes, descriptors);
            break;
          }
        }
      }
    }
    const auto matrix_cursor = cursor, matrix_binding_host_bytes = binding_host_bytes;
    auto drop_core_reuse = [&] {
      metrics.df_core_reuse = false;
      metrics.df_core_reuse_bytes = 0;
      cursor = matrix_cursor;
      binding_host_bytes = matrix_binding_host_bytes;
      capacity();
    };
    // The immutable frontier has its own lease; both preparation and dynamic
    // temporaries borrow the already-budgeted matrix arena between Q consumers.
    // Refuse retention if a future graph needs more scratch, never reduce Q or
    // silently grow the mandatory owner to admit an optional cache.
    if (metrics.df_matrix_gemm && options.df_core_reuse &&
        generated_response::staged_core_reuse_scratch_arena_elements(o, v) <=
            matrix_scratch_elements) {
      auto candidate = cursor;
      const auto retained_bytes = bytes(generated_response::staged_core_reuse_arena_elements(o, v));
      const auto offset = reserve(candidate, retained_bytes);
      const auto descriptors = generated_response::core_reuse_contraction_host_bytes();
      if (checked_add(checked_add(checked_add(host_bytes, binding_host_bytes), descriptors),
                      checked_add(candidate, metrics.df_provider_allowance_bytes)) <=
              options.max_bytes &&
          device_fits(candidate, metrics.df_provider_allowance_bytes)) {
        metrics.df_core_reuse = true;
        metrics.df_core_reuse_bytes = retained_bytes;
        core_reuse_offset = offset;
        cursor = candidate;
        binding_host_bytes = checked_add(binding_host_bytes, descriptors);
      }
    }
    capacity();
    if (metrics.numeric_capacity_bytes > options.max_bytes)
      throw std::length_error("DF Lambda complete numeric storage exceeds budget");
    if (metrics.owned_device_bytes > device_limit)
      throw std::length_error("DF Lambda mandatory device storage exceeds available memory");
    parameters_admitted = with_parameters;
    cuda_check(cudaStreamCreateWithFlags(&storage.stream, cudaStreamNonBlocking));
    if (metrics.df_matrix_gemm && !storage.contractions.prepare(storage.stream)) scalar_plan();
    auto allocation = cudaMalloc(reinterpret_cast<void**>(&storage.base), cursor);
    if (allocation == cudaErrorMemoryAllocation && metrics.df_core_reuse) {
      // Refuse only the optional owner-local retention before dropping GEMM.
      // Numerical, launch and driver failures must propagate, not retry.
      (void)cudaGetLastError();
      drop_core_reuse();
      allocation = cudaMalloc(reinterpret_cast<void**>(&storage.base), cursor);
    }
    if (allocation == cudaErrorMemoryAllocation && metrics.df_audit_matrix_gemm) {
      (void)cudaGetLastError();
      drop_audit();
      allocation = cudaMalloc(reinterpret_cast<void**>(&storage.base), cursor);
    }
    if (allocation == cudaErrorMemoryAllocation && metrics.df_matrix_gemm) {
      (void)cudaGetLastError();
      std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
      storage.contractions.release_locked();
      scalar_plan();
      allocation = cudaMalloc(reinterpret_cast<void**>(&storage.base), cursor);
    }
    if (allocation == cudaErrorMemoryAllocation && metrics.df_auxiliary_reduction) {
      // Only allocation failure may discard optional cuts; arithmetic errors
      // and driver failures never trigger a scientific/backend fallback.
      (void)cudaGetLastError();
      metrics.df_auxiliary_reduction = false;
      cursor = fallback_cursor;
      capacity();
      allocation = cudaMalloc(reinterpret_cast<void**>(&storage.base), cursor);
    }
    cuda_check(allocation);
    auto at = [&](std::size_t offset) { return reinterpret_cast<double*>(storage.base + offset); };
    const std::array<double**, 14> fields{
        &state.foo,  &state.fov,  &state.fvv,  &state.ovov, &state.ovvo, &state.oovv, &state.ovvv,
        &state.ovoo, &state.oooo, &state.vvvv, &state.d1,   &state.d2,   &state.t1,   &state.t2};
    for (std::size_t x = 0; x < host.size(); ++x) {
      if (x < fields.size()) *fields[x] = at(offsets[x]);
      if (host[x]->size()) upload(at(offsets[x]), host[x]->data(), host[x]->size());
    }
    bov = at(offsets[14]);
    bvv = at(offsets[15]);
    state.o = o;
    state.v = v;
    state.stream = storage.stream;
    state.response_arena = state.replay_arena = at(arena);
    state.error = reinterpret_cast<int*>(storage.base + error);
    state.bar_correlation_energy = at(seed0);
    state.bar_singles_residual = at(seed1);
    state.bar_doubles_residual = at(seed2);
    sum1 = at(sums1);
    sum2 = at(sums2);
    state.df_virtual_singles = sum1;
    state.df_virtual_doubles = sum2;
    auxiliary.o = o;
    auxiliary.v = v;
    auxiliary.stream = storage.stream;
    auxiliary.error = state.error;
    auxiliary.response_arena = state.response_arena;
    auxiliary.t1 = state.t1;
    auxiliary.t2 = state.t2;
    auxiliary.bar_df_virtual_singles = state.bar_singles_residual;
    auxiliary.bar_df_virtual_doubles = state.bar_doubles_residual;
    if (metrics.df_auxiliary_reduction) {
      static_cast<generated_response::CudaState&>(staged) = state;
      staged.response_arena = at(staged_arena);
      tau = at(tau_offset);
      tau_seed = at(tau_seed_offset);
      staged.df_tau = tau;
      staged.bar_df_tau = tau_seed;
      const std::array<const double**, 6> values{
          &staged.df_Lvv, &staged.df_Wvoov,         &staged.df_Wvovo,
          &staged.df_Xv,  &staged.df_D05_vv_ladder, &staged.df_singles_residual};
      const std::array<const double**, 6> seeds{
          &staged.bar_df_Lvv, &staged.bar_df_Wvoov,         &staged.bar_df_Wvovo,
          &staged.bar_df_Xv,  &staged.bar_df_D05_vv_ladder, &staged.bar_df_singles_residual};
      for (std::size_t x = 0; x < cuts.size(); ++x) {
        *values[x] = cuts[x] = at(cut_offsets[x]);
        *seeds[x] = cut_seeds[x] = at(seed_offsets[x]);
      }
      if (metrics.df_matrix_gemm) {
        // Pre-bind both complete batches and the final Q tail. A partial batch
        // must not allocate/replan inside the iterative response action.
        generated_response::prepare_contractions(
            staged, storage.contractions, metrics.df_auxiliary_batch_size,
            q % metrics.df_auxiliary_batch_size, with_parameters, metrics.df_gemm_calls,
            metrics.df_gemm_summands);
      }
      prepare_cuts();
      if (metrics.df_audit_matrix_gemm) {
        staged.audit_arena = at(audit_offset);
        generated_response::prepare_audit_contractions(
            staged, storage.contractions, metrics.df_auxiliary_batch_size,
            q % metrics.df_auxiliary_batch_size, metrics.df_primal_matrix_gemm,
            metrics.df_gemm_calls, metrics.df_gemm_summands);
        metrics.audit_schedule_hash = generated_response::audit_matrix_operator_hash;
      }
      if (metrics.df_core_reuse) {
        staged.core_reuse_arena = at(core_reuse_offset);
        generated_response::prepare_core_reuse_contractions(
            staged, storage.contractions, metrics.df_gemm_calls, metrics.df_gemm_summands);
        runtime::df_progress::Scope preparation("df_lambda_core_reuse_preparation");
        clear_error();
        // Q preparation may leave a tail extent; core bindings always use q=1.
        staged.q = 1;
        generated_response::run_staged_core_reuse_prepare_cuda(staged);
        record_core_reuse(true);
        finish();
        ++metrics.df_core_reuse_preparations;
        metrics.core_reuse_plan_hash = generated_response::staged_core_reuse_hash;
      }
    } else {
      cuda_check(cudaStreamSynchronize(storage.stream));
      ++metrics.synchronizations;
    }
    metrics.shared_program_hash =
        metrics.df_matrix_gemm           ? generated_response::staged_matrix_operator_hash
        : metrics.df_auxiliary_reduction ? generated_response::staged_operator_hash
                                         : generated_response::operator_hash;
    metrics.independent_program_hash = generated_response::independent_operator_hash;
    metrics.cuda_actions = true;
  }

  void record_stage(const Stage& stage) {
    if (!metrics.df_matrix_gemm) {
      record(stage.work, stage.operations);
      return;
    }
    metrics.df_contraction_terms =
        checked_add(metrics.df_contraction_terms, stage.matrix_work(state.o, state.v, staged.q));
    // One audit kernel replaces each GEMM contraction kernel in this count;
    // provider-internal kernels are deliberately not guessed or counted here.
    metrics.df_generated_kernels =
        checked_add(metrics.df_generated_kernels, stage.matrix_operations);
    metrics.df_packing_output_bytes = checked_add(metrics.df_packing_output_bytes,
                                                  bytes(stage.packing(state.o, state.v, staged.q)));
  }
  void record_core_reuse(bool preparation) {
    const auto terms = preparation
                           ? generated_response::staged_core_reuse_prepare_contraction_terms
                           : generated_response::staged_core_reuse_dynamic_contraction_terms;
    const auto packing = preparation
                             ? generated_response::staged_core_reuse_prepare_packing_elements
                             : generated_response::staged_core_reuse_dynamic_packing_elements;
    metrics.df_contraction_terms =
        checked_add(metrics.df_contraction_terms, terms(state.o, state.v));
    metrics.df_generated_kernels =
        checked_add(metrics.df_generated_kernels,
                    preparation ? generated_response::staged_core_reuse_prepare_operations
                                : generated_response::staged_core_reuse_dynamic_operations);
    metrics.df_packing_output_bytes =
        checked_add(metrics.df_packing_output_bytes, bytes(packing(state.o, state.v)));
  }
  void upload(double* target, const double* source, std::size_t count) {
    cuda_check(
        cudaMemcpyAsync(target, source, bytes(count), cudaMemcpyHostToDevice, storage.stream));
    metrics.h2d_bytes = checked_add(metrics.h2d_bytes, bytes(count));
  }
  void download(double* target, const double* source, std::size_t count) {
    cuda_check(
        cudaMemcpyAsync(target, source, bytes(count), cudaMemcpyDeviceToHost, storage.stream));
    metrics.d2h_bytes = checked_add(metrics.d2h_bytes, bytes(count));
  }
  void clear_error() { cuda_check(cudaMemsetAsync(state.error, 0, sizeof(int), storage.stream)); }
  void finish() {
    int error = 0;
    Fence fence{storage.stream};
    cuda_check(
        cudaMemcpyAsync(&error, state.error, sizeof(int), cudaMemcpyDeviceToHost, storage.stream));
    cuda_check(cudaStreamSynchronize(storage.stream));
    fence.complete();
    ++metrics.synchronizations;
    metrics.d2h_bytes = checked_add(metrics.d2h_bytes, sizeof(int));
    if (error) throw std::runtime_error("nonfinite native DF Lambda generated action");
  }
  void record(Query work, std::size_t operations) {
    metrics.df_contraction_terms =
        checked_add(metrics.df_contraction_terms, work(state.o, state.v));
    metrics.df_generated_kernels = checked_add(metrics.df_generated_kernels, operations);
  }
  void select(std::size_t index, std::size_t count = 1) {
    auxiliary.bov = bov + index * n1;
    auxiliary.bvv = bvv + index * vv;
    staged.bov = auxiliary.bov;
    staged.bvv = auxiliary.bvv;
    staged.q = count;
    metrics.df_auxiliary_slices = checked_add(metrics.df_auxiliary_slices, count);
    ++metrics.df_auxiliary_batches;
  }
  void add_buffer(const double* source, std::size_t count, double* target) {
    accumulate<<<generativeqc_tensor::blocks(count, 256), 256, 0, storage.stream>>>(
        source, count, target, state.error);
    cuda_check(cudaGetLastError());
    ++metrics.df_generated_kernels;
  }
  void copy_buffer(const double* source, std::size_t count, double* target) {
    cuda_check(
        cudaMemcpyAsync(target, source, bytes(count), cudaMemcpyDeviceToDevice, storage.stream));
  }
  void prepare_cuts() {
    runtime::df_progress::Scope preparation("df_lambda_cuts_preparation");
    clear_error();
    staged.q = 1;
    const auto prepared = generated_response::run_staged_primal_prepare_cuda(staged);
    copy_buffer(prepared.df_tau, n2, tau);
    record_stage(primal_prepare);
    for (std::size_t x = 0; x < cuts.size(); ++x)
      cuda_check(cudaMemsetAsync(cuts[x], 0, bytes(cut_sizes[x]), storage.stream));
    for (std::size_t Q = 0; Q < q; Q += metrics.df_auxiliary_batch_size) {
      select(Q, std::min(metrics.df_auxiliary_batch_size, q - Q));
      const auto out = generated_response::run_staged_primal_auxiliary_cuda(staged);
      generated_response::accumulate_staged_primal_auxiliary_cuda(
          staged, out, cuts[4], cuts[0], cuts[1], cuts[2], cuts[3], cuts[5]);
      ++metrics.df_generated_kernels;
      record_stage(primal_auxiliary);
    }
    ++metrics.df_preparation_calls;
    finish();
  }
  void core_seeds() {
    staged.q = 1;
    const auto out = metrics.df_core_reuse
                         ? generated_response::run_staged_core_reuse_dynamic_cuda(staged)
                         : generated_response::run_staged_core_cuda(staged);
    // Borrowed adjoints must survive reuse of the graph arena by every Q VJP.
    const std::array sources{out.bar_df_Lvv, out.bar_df_Wvoov,         out.bar_df_Wvovo,
                             out.bar_df_Xv,  out.bar_df_D05_vv_ladder, out.bar_df_singles_residual};
    for (std::size_t x = 0; x < cuts.size(); ++x)
      copy_buffer(sources[x], cut_sizes[x], cut_seeds[x]);
    copy_buffer(out.bar_t1, n1, sum1);
    copy_buffer(out.bar_t2, n2, sum2);
    if (metrics.df_core_reuse) {
      record_core_reuse(false);
      ++metrics.df_core_reuse_actions;
    } else {
      record_stage(core_stage);
    }
  }
  void reduced_transpose() {
    core_seeds();
    cuda_check(cudaMemsetAsync(tau_seed, 0, bytes(n2), storage.stream));
    for (std::size_t Q = 0; Q < q; Q += metrics.df_auxiliary_batch_size) {
      select(Q, std::min(metrics.df_auxiliary_batch_size, q - Q));
      const auto out = generated_response::run_staged_auxiliary_cuda(staged);
      // The compiler declaration fixes the output order: inspect the IR rather
      // than assuming the tau cut replaces direct T2 dependence.
      generated_response::accumulate_staged_auxiliary_cuda(staged, out, tau_seed, sum1, sum2);
      ++metrics.df_generated_kernels;
      record_stage(auxiliary_stage);
    }
    staged.q = 1;
    const auto out = generated_response::run_staged_prepare_cuda(staged);
    generated_response::accumulate_staged_prepare_cuda(staged, out, sum1, sum2);
    ++metrics.df_generated_kernels;
    record_stage(prepare_stage);
    ++metrics.df_reduced_actions;
  }
  void add(const double* one, const double* two) {
    accumulate<<<generativeqc_tensor::blocks(n1, 256), 256, 0, storage.stream>>>(one, n1, sum1,
                                                                                 state.error);
    accumulate<<<generativeqc_tensor::blocks(n2, 256), 256, 0, storage.stream>>>(two, n2, sum2,
                                                                                 state.error);
    cuda_check(cudaGetLastError());
    metrics.df_generated_kernels = checked_add(metrics.df_generated_kernels, 2);
  }
  void output(const double* a, const double* b, std::vector<double>& one,
              std::vector<double>& two) {
    one.resize(n1);
    two.resize(n2);
    Fence fence{storage.stream};
    download(one.data(), a, n1);
    download(two.data(), b, n2);
    finish();
    fence.complete();
  }
};

DFLambdaActions::DFLambdaActions(const Problem& p, const SolverResult& cc,
                                 const LambdaOptions& options, int device, bool with_source,
                                 bool with_parameters)
    : impl_(std::make_unique<Impl>(p, cc, options, device, with_source, with_parameters)) {}
DFLambdaActions::~DFLambdaActions() = default;
const LambdaDiagnostic& DFLambdaActions::diagnostic() const noexcept { return impl_->metrics; }

void DFLambdaActions::replay(double& energy, std::vector<double>& r1, std::vector<double>& r2) {
  auto& s = *impl_;
  s.clear_error();
  cuda_check(cudaMemsetAsync(s.sum1, 0, bytes(s.n1), s.storage.stream));
  cuda_check(cudaMemsetAsync(s.sum2, 0, bytes(s.n2), s.storage.stream));
  for (std::size_t Q = 0; Q < s.q;) {
    if (s.metrics.df_primal_matrix_gemm) {
      const auto batch = std::min(s.metrics.df_auxiliary_batch_size, s.q - Q);
      s.select(Q, batch);
      const auto out = generated_response::run_primal_virtual_cuda(s.staged);
      generated_response::accumulate_primal_virtual_cuda(s.staged, out, s.sum2, s.sum1);
      ++s.metrics.df_generated_kernels;
      s.record_stage(primal_virtual);
      Q += batch;
    } else {
      s.select(Q);
      const auto out = generated_virtual::run_virtual_accumulate_cuda(s.auxiliary);
      s.add(out.singles, out.doubles);
      s.record(generated_response::virtual_virtual_contraction_terms,
               generated_virtual::virtual_cuda_operation_count);
      ++Q;
    }
  }
  const auto out = generated_core::run_replay_cuda(s.state);
  s.record(generated_response::replay_contraction_terms, generated_core::replay_operation_count);
  Fence fence{s.storage.stream};
  s.download(&energy, out.energy, 1);
  s.output(out.r1, out.r2, r1, r2);
  fence.complete();
}

void DFLambdaActions::rhs(bool independent, std::vector<double>& one, std::vector<double>& two) {
  auto& s = *impl_;
  const double seed = -1;
  Fence fence{s.storage.stream};
  s.clear_error();
  s.upload(s.state.bar_correlation_energy, &seed, 1);
  const auto out = independent ? generated_response::run_independent_rhs_cuda(s.state)
                               : generated_response::run_rhs_cuda(s.state);
  s.record(independent ? generated_response::independent_rhs_contraction_terms
                       : generated_response::rhs_contraction_terms,
           independent ? generated_response::independent_rhs_operations
                       : generated_response::rhs_operations);
  s.output(out.t1, out.t2, one, two);
  fence.complete();
}

void DFLambdaActions::transpose(bool independent, std::span<const double> one,
                                std::span<const double> two, std::vector<double>& out_one,
                                std::vector<double>& out_two) {
  auto& s = *impl_;
  Fence fence{s.storage.stream};
  if (one.size() != s.n1 || two.size() != s.n2)
    throw std::invalid_argument("DF Lambda transpose seed shape mismatch");
  s.clear_error();
  s.upload(s.state.bar_singles_residual, one.data(), s.n1);
  s.upload(s.state.bar_doubles_residual, two.data(), s.n2);
  if (s.metrics.df_auxiliary_reduction && !independent) {
    s.reduced_transpose();
    s.output(s.sum1, s.sum2, out_one, out_two);
    fence.complete();
    return;
  }
  if (independent && s.metrics.df_audit_matrix_gemm) {
    // Keep the expanded independent core and original per-Q virtual adjoint;
    // only their compiler lowering changes, never the solver's staged cuts.
    s.staged.q = 1;
    const auto core = generated_response::run_audit_core_cuda(s.staged);
    s.copy_buffer(core.bar_t1, s.n1, s.sum1);
    s.copy_buffer(core.bar_t2, s.n2, s.sum2);
    s.record_stage(audit_core);
    for (std::size_t auxiliary_begin = 0; auxiliary_begin < s.q;
         auxiliary_begin += s.metrics.df_auxiliary_batch_size) {
      s.select(auxiliary_begin, std::min(s.metrics.df_auxiliary_batch_size, s.q - auxiliary_begin));
      const auto out = generated_response::run_audit_auxiliary_cuda(s.staged);
      generated_response::accumulate_audit_auxiliary_cuda(s.staged, out, s.sum1, s.sum2);
      ++s.metrics.df_generated_kernels;
      s.record_stage(audit_auxiliary);
    }
    s.output(s.sum1, s.sum2, out_one, out_two);
    fence.complete();
    return;
  }
  const auto core = independent ? generated_response::run_independent_transpose_cuda(s.state)
                                : generated_response::run_transpose_cuda(s.state);
  s.record(independent ? generated_response::independent_transpose_contraction_terms
                       : generated_response::transpose_contraction_terms,
           independent ? generated_response::independent_transpose_operations
                       : generated_response::transpose_operations);
  // Consume borrowed core output before its scratch is reused for a Q action.
  cuda_check(
      cudaMemcpyAsync(s.sum1, core.t1, bytes(s.n1), cudaMemcpyDeviceToDevice, s.storage.stream));
  cuda_check(
      cudaMemcpyAsync(s.sum2, core.t2, bytes(s.n2), cudaMemcpyDeviceToDevice, s.storage.stream));
  for (std::size_t Q = 0; Q < s.q; ++Q) {
    s.select(Q);
    const auto out = generated_virtual::run_amplitude_vjp_accumulate_cuda(s.auxiliary);
    s.add(out.t1, out.t2);
    s.record(generated_response::virtual_amplitude_vjp_contraction_terms,
             generated_virtual::amplitude_vjp_cuda_operation_count);
  }
  s.output(s.sum1, s.sum2, out_one, out_two);
  fence.complete();
}

void DFLambdaActions::seeds(std::span<const double> one, std::span<const double> two) {
  auto& s = *impl_;
  const double energy = 1;
  Fence fence{s.storage.stream};
  if (!s.parameters_admitted || one.size() != s.n1 || two.size() != s.n2)
    throw std::invalid_argument("DF parameter response not admitted or seed shape mismatch");
  s.clear_error();
  s.upload(s.state.bar_correlation_energy, &energy, 1);
  s.upload(s.state.bar_singles_residual, one.data(), s.n1);
  s.upload(s.state.bar_doubles_residual, two.data(), s.n2);
  s.finish();
  fence.complete();
}

std::vector<double> DFLambdaActions::parameter(std::string_view name, std::size_t count) {
  auto& s = *impl_;
  if (!s.parameters_admitted) throw std::logic_error("DF parameter storage not admitted");
  const auto found = std::find_if(parameters.begin(), parameters.end(),
                                  [&](const auto& p) { return p.name == name; });
  if (found == parameters.end())
    throw std::invalid_argument(
        "DF parameter response cannot materialize a virtual integral block");
  const auto oo = checked_mul(s.state.o, s.state.o);
  const auto expected = name == "foo"    ? oo
                        : name == "fov"  ? s.n1
                        : name == "fvv"  ? s.vv
                        : name == "ovoo" ? checked_mul(s.n1, oo)
                        : name == "oooo" ? checked_mul(oo, oo)
                                         : s.n2;
  if (count != expected) throw std::invalid_argument("DF parameter response output shape mismatch");
  std::vector<double> values(count);
  Fence fence{s.storage.stream};
  s.clear_error();
  s.staged.q = 1;
  const auto out =
      s.metrics.df_auxiliary_reduction ? found->staged_run(s.staged) : found->run(s.state);
  if (s.metrics.df_auxiliary_reduction)
    s.record_stage(found->stage);
  else
    s.record(found->work, found->operations);
  s.download(values.data(), out.values, count);
  s.finish();
  fence.complete();
  return values;
}

std::pair<std::vector<double>, std::vector<double>> DFLambdaActions::virtual_factors() {
  auto& s = *impl_;
  if (!s.parameters_admitted) throw std::logic_error("DF factor publication not admitted");
  std::pair<std::vector<double>, std::vector<double>> result{
      std::vector<double>(checked_mul(s.q, s.n1)), std::vector<double>(checked_mul(s.q, s.vv))};
  Fence fence{s.storage.stream};
  s.clear_error();
  if (s.metrics.df_auxiliary_reduction) s.core_seeds();
  for (std::size_t Q = 0; Q < s.q; Q += s.metrics.df_auxiliary_batch_size) {
    s.select(Q, std::min(s.metrics.df_auxiliary_batch_size, s.q - Q));
    if (s.metrics.df_auxiliary_reduction) {
      const auto out = generated_response::run_staged_factors_cuda(s.staged);
      s.record_stage(factor_stage);
      s.download(result.first.data() + Q * s.n1, out.bar_bov, checked_mul(s.staged.q, s.n1));
      s.download(result.second.data() + Q * s.vv, out.bar_bvv, checked_mul(s.staged.q, s.vv));
      continue;
    }
    const auto out = generated_virtual::run_factor_vjp_accumulate_cuda(s.auxiliary);
    s.record(generated_response::virtual_factor_vjp_contraction_terms,
             generated_virtual::factor_vjp_cuda_operation_count);
    // The stream consumes each borrowed output before the next Q reuses it.
    s.download(result.first.data() + Q * s.n1, out.bov, s.n1);
    s.download(result.second.data() + Q * s.vv, out.bvv, s.vv);
  }
  s.finish();
  fence.complete();
  return result;
}
}  // namespace generativeqc::cc::detail
