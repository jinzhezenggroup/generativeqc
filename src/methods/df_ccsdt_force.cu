#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "integrals/electron_interaction_source.hpp"
#include "methods/df_ccsdt_force.hpp"
#include "molecule/nuclear_gradient.hpp"
#include "posthf/capacity.hpp"
#include "runtime/cuda_resources.cuh"
#include "runtime/df_progress_trace.hpp"
#include "runtime/execution_context.hpp"
#include "scf/cuda_df_nuclear_sink.hpp"

namespace generativeqc::methods::detail {
namespace {
using posthf::checked_add;
using posthf::checked_mul;
using Clock = std::chrono::steady_clock;
std::size_t bytes(std::size_t n) { return checked_mul(n, sizeof(double)); }
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
std::size_t capacity(std::initializer_list<const std::vector<double>*> arrays) {
  std::size_t count = 0;
  for (const auto* a : arrays) count = checked_add(count, a->capacity());
  return bytes(count);
}
void add(std::vector<double>& target, const std::vector<double>& source) {
  if (target.size() != source.size()) throw std::invalid_argument("DF force source shape mismatch");
  for (std::size_t i = 0; i < target.size(); ++i) target[i] += source[i];
}
std::size_t difference(std::size_t total, std::size_t included) {
  if (included > total)
    throw std::logic_error("DF force phase accounting overlap exceeds live state");
  return total - included;
}
std::size_t primal_host_bytes(const RccsdNativeState& state) {
  return checked_add(cc::problem_host_bytes(state.problem),
                     capacity({&state.solved.t1, &state.solved.t2, &state.eps_o, &state.eps_v}));
}
std::uint64_t primal_identity(const RccsdNativeState& state) {
  // A diagnostic bit-pattern census, not a cache key or a second scientific map.
  auto identity = std::uint64_t{14695981039346656037ULL};
  for (const auto* values :
       {&state.problem.df_bov, &state.problem.df_bvv, &state.problem.ovoo, &state.problem.ovov,
        &state.problem.fov, &state.solved.t1, &state.solved.t2, &state.eps_o, &state.eps_v}) {
    identity = (identity ^ values->size()) * 1099511628211ULL;
    for (const double value : *values)
      identity = (identity ^ std::bit_cast<std::uint64_t>(value)) * 1099511628211ULL;
  }
  return identity;
}
}  // namespace

static DFCCSDTResult run_df_ccsdt_native_attempt(
    runtime::ExecutionContext& execution, const core::System& system, const core::System& auxiliary,
    const generativeqc_method_descriptor& descriptor, bool forces, bool with_triples,
    bool df_auxiliary_reduction, bool df_matrix_gemm, bool lambda_matrix_gemm,
    std::size_t lambda_batch_limit, std::size_t ccsd_batch_limit,
    const hf::RHFFrameResponseOptions& frame_options, bool derived_denominators, bool packed_diis,
    bool parallel_gap_reduction, bool request_triples_gap_cotangents,
    RccsdNativeState* replay_state = nullptr, std::size_t retained_host_bytes = 0,
    std::size_t retained_df_source_bytes = 0) {
  const auto started = Clock::now();
  runtime::df_progress::Scope trace(replay_state ? "df_ccsdt_response_replay" : "df_ccsdt_native");
  using Trace = runtime::df_progress::Scope;
  if (trace.enabled())
    Trace::label("phase", replay_state ? "same_primal_response" : "cold_rhf_df_ccsd");
  if (!execution.cuda_requested()) throw std::invalid_argument("DF force endpoint requires CUDA");
  // Both sources are normalized views of the same immutable nuclear geometry.
  if (system.atoms.size() != auxiliary.atoms.size())
    throw std::invalid_argument("DF force auxiliary geometry differs from orbital system");
  for (std::size_t a = 0; a < system.atoms.size(); ++a)
    if (system.atoms[a].position != auxiliary.atoms[a].position ||
        system.atoms[a].atomic_number != auxiliary.atoms[a].atomic_number)
      throw std::invalid_argument("DF force auxiliary geometry differs from orbital system");
  const auto recycle_bytes = frame_options.recycling ? frame_options.recycling->storage_bytes() : 0;
  // A caller-owned recycled subspace is live during RHF/CC as well. Reserve it
  // in every phase, then let the response owner rebind/release it explicitly.
  auto state = replay_state ? std::move(*replay_state)
                            : run_rccsd_native_state(
                                  execution, system, descriptor, nullptr, nullptr, nullptr,
                                  recycle_bytes, &auxiliary, forces, df_matrix_gemm, nullptr,
                                  ccsd_batch_limit, derived_denominators, packed_diis);
  if (!state.solved.converged()) throw std::runtime_error("DF force CCSD did not converge");
  DFCCSDTResult result;
  result.reference_energy = state.reference->energy;
  result.reference_energy_change = state.reference_energy_change;
  result.reference_density_rms = state.reference_density_rms;
  result.reference_iterations = state.reference_iterations;
  result.correlation_energy = state.solved.correlation_energy;
  result.energy = state.solved.total_energy;
  result.primal = state.performance;
  result.solver = state.solved.diagnostic;
  result.method_result = state.result;
  result.correlation = state.diagnostic;
  const auto reserved_bytes = state.external_reservation_bytes;
  const auto external_bytes = checked_add(reserved_bytes, retained_host_bytes);
  if (reserved_bytes < recycle_bytes || retained_host_bytes >= state.budget)
    throw std::length_error("DF force retained caller state exhausts complete budget");
  result.numeric_capacity_bytes =
      difference(state.diagnostic.numeric_capacity_bytes, reserved_bytes);
  const auto budget = state.budget - retained_host_bytes;
  const auto device = execution.device_id();
  auto& p = state.problem;
  const auto o = p.nocc, v = p.nvir, n = o + v, q = p.naux, nn = checked_mul(n, n);
  const auto coords = checked_mul(system.atoms.size(), 3);
  auto base = checked_add(
      p.reference_retained_bytes,
      checked_add(cc::problem_host_bytes(p), capacity({&state.solved.t1, &state.solved.t2})));
  if (replay_state) result.numeric_capacity_bytes = base;
  const auto borrowed = bytes(p.df_bov.size() + p.df_bvv.size() + p.ovoo.size() + p.ovov.size() +
                              p.fov.size() + state.solved.t1.size() + state.solved.t2.size() +
                              state.eps_o.size() + state.eps_v.size());
  cc::triples::DFCudaResponseResult t;
  std::size_t tbytes = 0, fbytes = 0;
  auto phase = Clock::now();
  if (trace.enabled()) Trace::label("phase", "triples");
  if (with_triples) {
    if (forces) {
      t = cc::triples::pullback_df_cuda(
          o, v, q, p.df_bov.data(), p.df_bvv.data(), p.ovoo.data(), p.ovov.data(), p.fov.data(),
          state.solved.t1.data(), state.solved.t2.data(), state.eps_o.data(), state.eps_v.data(),
          1e-10, budget, device, difference(base, borrowed), 3, parallel_gap_reduction,
          request_triples_gap_cotangents);
      result.triples = t.diagnostic;
      result.triples_gap = t.gap;
      result.numeric_capacity_bytes =
          std::max(result.numeric_capacity_bytes, t.numeric_capacity_bytes);
      tbytes =
          capacity({&t.bov, &t.bvv, &t.ovoo, &t.ovov, &t.fov, &t.t1, &t.t2, &t.eps_o, &t.eps_v});
      result.triples_fock = cc::triples::fock_response_df_cuda(
          o, v, q, p.df_bov.data(), p.df_bvv.data(), p.ovoo.data(), p.ovov.data(), p.fov.data(),
          state.solved.t1.data(), state.solved.t2.data(), state.eps_o.data(), state.eps_v.data(),
          1e-10, budget, device, checked_add(difference(base, borrowed), tbytes));
      result.numeric_capacity_bytes =
          std::max(result.numeric_capacity_bytes, result.triples_fock.numeric_capacity_bytes);
      fbytes = capacity({&result.triples_fock.foo, &result.triples_fock.fvv});
    } else {
      result.triples = cc::triples::evaluate_df_cuda(
          o, v, q, p.df_bov.data(), p.df_bvv.data(), p.ovoo.data(), p.ovov.data(), p.fov.data(),
          state.solved.t1.data(), state.solved.t2.data(), state.eps_o.data(), state.eps_v.data(),
          1e-10, difference(budget, base), device);
      result.numeric_capacity_bytes = std::max(result.numeric_capacity_bytes,
                                               checked_add(base, result.triples.workspace_bytes));
    }
    result.triples_energy = result.triples.energy;
    result.energy += result.triples_energy;
  }
  result.triples_seconds = elapsed(phase);
  result.method_result.energy = result.energy;
  result.correlation.numeric_capacity_bytes = result.numeric_capacity_bytes;
  result.correlation.ccsd_t_triples_energy = result.triples_energy;
  result.correlation.ccsd_t_virtual_triples = result.triples.virtual_triples;
  result.correlation.ccsd_t_workspace_bytes = result.triples.workspace_bytes;
  if (with_triples)
    result.correlation.minimum_absolute_denominator =
        std::min(result.correlation.minimum_absolute_denominator,
                 result.triples.minimum_absolute_denominator);
  if (!forces) {
    result.numeric_capacity_bytes = checked_add(result.numeric_capacity_bytes, external_bytes);
    // The public diagnostic includes the caller-owned reservation restored at
    // this boundary, just like the complete native endpoint allowance.
    result.correlation.numeric_capacity_bytes = result.numeric_capacity_bytes;
    result.total_seconds = elapsed(started);
    return result;
  }
  if (!state.df_source.response_state || state.df_source.source_identity != p.df_source_identity)
    throw std::logic_error("DF force lost its original molecular source/frame");
  phase = Clock::now();
  if (trace.enabled()) Trace::label("phase", "corrected_lambda");
  cc::LambdaOptions lambda_options;
  lambda_options.df_auxiliary_reduction = df_auxiliary_reduction;
  lambda_options.df_matrix_gemm = lambda_matrix_gemm;
  lambda_options.df_auxiliary_batch_limit = lambda_batch_limit;
  lambda_options.cc_tolerance = 1e-9;
  lambda_options.lambda_tolerance = 1e-9;
  const auto lambda_external = checked_add(tbytes, fbytes);
  lambda_options.max_bytes = difference(budget, lambda_external);
  lambda_options.gmres.max_workspace_bytes = lambda_options.max_bytes;
  lambda_options.gmres.absolute_tolerance = 1e-12;
  auto parameters = with_triples ? cc::solve_lambda_parameter_response_cuda_with_energy_source(
                                       p, state.solved, t.t1, t.t2, device, lambda_options)
                                 : cc::solve_lambda_parameter_response_cuda(p, state.solved, device,
                                                                            lambda_options);
  if (!parameters.lambda.converged())
    throw std::runtime_error("DF force corrected Lambda did not converge");
  result.lambda = parameters.lambda.diagnostic;
  result.numeric_capacity_bytes =
      std::max(result.numeric_capacity_bytes,
               checked_add(lambda_external, result.lambda.numeric_capacity_bytes));
  if (with_triples) {
    add(parameters.foo, result.triples_fock.foo);
    add(parameters.fvv, result.triples_fock.fvv);
    add(parameters.fov, t.fov);
    add(parameters.ovov, t.ovov);
    add(parameters.ovoo, t.ovoo);
    add(parameters.df_bov, t.bov);
    add(parameters.df_bvv, t.bvv);
    // Full Fock matrices replace epsilon sources; never add t.eps_o/v.
    t = {};
    std::vector<double>().swap(result.triples_fock.foo);
    std::vector<double>().swap(result.triples_fock.fvv);
  }
  // Lambda vectors are no longer needed once their parameter VJPs are detached.
  parameters.lambda = {};
  result.lambda_seconds = elapsed(phase);
  phase = Clock::now();
  if (trace.enabled()) Trace::label("phase", "factor_and_nuclear_source");
  const auto parameter_bytes =
      capacity({&parameters.foo, &parameters.fov, &parameters.fvv, &parameters.ovov,
                &parameters.ovvo, &parameters.oovv, &parameters.ovoo, &parameters.oooo,
                &parameters.df_bov, &parameters.df_bvv});
  const auto frame_phase =
      checked_add(base, checked_add(parameter_bytes, bytes(checked_mul(2, nn))));
  if (frame_phase > budget)
    throw std::length_error("DF force frame buffers exceed complete numeric budget");
  std::vector<double> bar_f(nn, 0.0), bar_c(nn);
  for (std::size_t i = 0; i < o; ++i) {
    for (std::size_t j = 0; j < o; ++j) bar_f[i * n + j] = parameters.foo[i * o + j];
    for (std::size_t a = 0; a < v; ++a) bar_f[i * n + o + a] = parameters.fov[i * v + a];
  }
  for (std::size_t a = 0; a < v; ++a)
    for (std::size_t b = 0; b < v; ++b) bar_f[(o + a) * n + o + b] = parameters.fvv[a * v + b];
  auto live = checked_add(base, checked_add(parameter_bytes, capacity({&bar_f, &bar_c})));
  const cc::DFFactorResponseView view{p.df_boo,
                                      p.df_bov,
                                      p.df_bvv,
                                      parameters.ovov,
                                      parameters.ovvo,
                                      parameters.oovv,
                                      parameters.ovoo,
                                      parameters.oooo,
                                      parameters.df_bov,
                                      parameters.df_bvv,
                                      p.df_source_identity};
  std::size_t factor_inputs = 0;
  for (auto span : {view.boo, view.bov, view.bvv, view.bar_ovov, view.bar_ovvo, view.bar_oovv,
                    view.bar_ovoo, view.bar_oooo, view.bar_bov, view.bar_bvv})
    factor_inputs = checked_add(factor_inputs, span.size_bytes());
  auto factors =
      cc::pullback_df_factors_cuda(o, v, q, view, budget, device, difference(live, factor_inputs));
  result.numeric_capacity_bytes =
      std::max(result.numeric_capacity_bytes, factors.numeric_capacity_bytes);
  parameters = {};
  std::vector<double> correlation_gradient;
  {
    const auto outer = checked_add(
        base, checked_add(capacity({&bar_f, &bar_c, &factors.boo, &factors.bov, &factors.bvv}),
                          bytes(coords)));
    scf::CudaDfNuclearSink sink(device, system, auxiliary, difference(budget, outer));
    const auto caller =
        checked_add(difference(base, state.df_source.retained_source_bytes),
                    checked_add(capacity({&bar_f, &bar_c}),
                                checked_add(bytes(coords), sink.numeric_capacity_bytes())));
    const auto reverse = cc::pullback_df_source_cuda(
        state.df_source.response_state, factors,
        [&](std::size_t mu, const double* row, cudaStream_t stream) {
          sink.consume(0, {mu * n * q, 1, 1, 1}, row, n * q, stream);
          result.source_weight_values += n * q;
        },
        [&](const double* dc, const double* dm, cudaStream_t stream) {
          sink.consume(1, {0, 1, 1, 1}, dm, q * q, stream);
          result.metric_weight_values += q * q;
          runtime::cuda_resource_check(
              cudaMemcpyAsync(bar_c.data(), dc, bytes(nn), cudaMemcpyDeviceToHost, stream));
        },
        budget, caller);
    result.numeric_capacity_bytes =
        std::max(result.numeric_capacity_bytes, reverse.numeric_capacity_bytes);
    correlation_gradient = sink.finish();
  }
  if (result.source_weight_values != checked_mul(nn, q) ||
      result.metric_weight_values != checked_mul(q, q))
    throw std::logic_error("DF force physical source traversal was incomplete");
  factors = {};
  result.source_response_seconds = elapsed(phase);
  phase = Clock::now();
  if (trace.enabled()) Trace::label("phase", "exact_orbital_and_nuclear_response");
  auto orbital_options = frame_options;
  hf::RHFFrameDFPreconditionerPreparation preconditioner;
  if (orbital_options.df_preconditioning) {
    // Preparation precedes source release so it consumes the very same frame
    // and factors, without rebuilding either source. Charge all surviving CC
    // payloads while the generated map, copies and identity snapshot coexist.
    const auto outer = checked_add(base, capacity({&bar_f, &bar_c, &correlation_gradient}));
    if (outer < budget) {
      preconditioner =
          hf::prepare_rhf_frame_df_preconditioner(system, *state.reference, q, p.df_boo, p.df_bov,
                                                  p.df_bvv, p.df_source_identity, budget - outer);
      result.numeric_capacity_bytes = std::max(
          result.numeric_capacity_bytes, checked_add(outer, preconditioner.numeric_capacity_bytes));
    } else {
      preconditioner.reason = "DF preconditioner caller budget";
    }
  }
  // The source stream is drained and its sink has died. Release all completed
  // CC/source numeric owners before allocating the exact-reference response.
  state.df_source = {};
  state.problem = {};
  state.solved = {};
  state.result = {};
  std::vector<double>().swap(state.eps_o);
  std::vector<double>().swap(state.eps_v);
  result.numeric_capacity_bytes = checked_add(result.numeric_capacity_bytes, external_bytes);
  orbital_options.maximum_bytes = checked_add(state.budget, reserved_bytes);
  // The original replay source survives CC release. The exact-reference source
  // also stays live in this state while the independent orbital provider runs.
  const auto exact_source_bytes = state.reference_interaction_source
                                      ? state.reference_interaction_source->retained_numeric_bytes()
                                      : 0;
  orbital_options.caller_bytes = checked_add(
      difference(external_bytes, recycle_bytes),
      checked_add(
          retained_df_source_bytes,
          checked_add(exact_source_bytes,
                      checked_add(posthf::source_capacity(auxiliary),
                                  checked_add(capacity({&correlation_gradient}), bytes(coords))))));
  result.orbital = hf::rhf_frame_response_cuda(system, *state.reference, bar_f, bar_c, device,
                                               orbital_options, std::move(preconditioner.data));
  result.orbital.preconditioner_setup_seconds += preconditioner.seconds;
  result.orbital.preconditioner_contraction_terms = preconditioner.contraction_terms;
  if (!preconditioner.reason.empty()) result.orbital.preconditioner_reason = preconditioner.reason;
  result.numeric_capacity_bytes =
      std::max(result.numeric_capacity_bytes, result.orbital.numeric_capacity_bytes);
  result.orbital_seconds = elapsed(phase);
  result.forces = std::move(correlation_gradient);
  add(result.forces, result.orbital.gradient);
  molecule::add_nuclear_repulsion_gradient(system, result.forces);
  for (double& value : result.forces) {
    value = -value;
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite complete DF CCSD(T) force");
  }
  result.method_result.forces = result.forces;
  result.correlation.numeric_capacity_bytes = result.numeric_capacity_bytes;
  result.total_seconds = elapsed(started);
  return result;
}

DFGapForceComparison diagnose_df_ccsdt_gap_schedules(
    runtime::ExecutionContext& execution, const core::System& system, const core::System& auxiliary,
    const generativeqc_method_descriptor& descriptor,
    const hf::RHFFrameResponseOptions& frame_options, std::size_t lambda_batch_limit,
    std::size_t ccsd_batch_limit, bool derived_denominators, bool packed_diis) {
  const auto started = Clock::now();
  runtime::df_progress::Scope trace("df_gap_same_primal_comparison");
  if (!execution.cuda_requested() || !descriptor.correlation_memory_budget_bytes ||
      frame_options.recycling)
    throw std::invalid_argument(
        "DF gap comparison requires CUDA, explicit budget and no recycling");
  if (system.atoms.size() != auxiliary.atoms.size())
    throw std::invalid_argument("DF gap comparison auxiliary geometry differs from orbital system");
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
    if (system.atoms[atom].position != auxiliary.atoms[atom].position ||
        system.atoms[atom].atomic_number != auxiliary.atoms[atom].atomic_number)
      throw std::invalid_argument(
          "DF gap comparison auxiliary geometry differs from orbital system");
  DFGapForceComparison comparison;
  const auto coords = checked_mul(system.atoms.size(), 3);
  comparison.output_bytes = checked_mul(comparison.cases.size(), bytes(coords));
  if (comparison.output_bytes >= descriptor.correlation_memory_budget_bytes)
    throw std::length_error("DF gap comparison output buffers exceed complete budget");
  for (auto& snapshot : comparison.cases) snapshot.forces.resize(coords);
  comparison.output_bytes = 0;
  for (const auto& snapshot : comparison.cases)
    comparison.output_bytes =
        checked_add(comparison.output_bytes, bytes(snapshot.forces.capacity()));
  if (comparison.output_bytes >= descriptor.correlation_memory_budget_bytes)
    throw std::length_error("DF gap comparison output capacities exceed complete budget");
  {
    const auto primal_started = Clock::now();
    const auto original = run_rccsd_native_state(
        execution, system, descriptor, nullptr, nullptr, nullptr, comparison.output_bytes,
        &auxiliary, true, true, nullptr, ccsd_batch_limit, derived_denominators, packed_diis);
    comparison.primal_seconds = elapsed(primal_started);
    if (!original.solved.converged() || !original.df_source.response_state ||
        original.df_source.source_identity != original.problem.df_source_identity)
      throw std::logic_error("DF gap comparison lost its converged native source/frame");
    // These blocks must have moved into Problem; a future additional detached
    // owner needs its own copy/phase charge before this diagnostic can accept it.
    if (capacity({&original.df_source.boo, &original.df_source.bov, &original.df_source.bvv,
                  &original.df_source.ovov, &original.df_source.ovvo, &original.df_source.oovv,
                  &original.df_source.ovoo, &original.df_source.oooo, &original.result.forces}))
      throw std::logic_error("DF gap comparison has an unaccounted detached primal owner");
    comparison.nocc = original.problem.nocc;
    comparison.nvir = original.problem.nvir;
    comparison.naux = original.problem.naux;
    comparison.primal = original.performance;
    comparison.solver = original.solved.diagnostic;
    comparison.reference_energy = original.reference->energy;
    comparison.reference_energy_change = original.reference_energy_change;
    comparison.reference_density_rms = original.reference_density_rms;
    comparison.reference_iterations = original.reference_iterations;
    comparison.source_identity = original.problem.df_source_identity;
    comparison.denominator_identity = cc::denominator_identity(original.problem);
    comparison.primal_identity = primal_identity(original);
    comparison.retained_primal_host_bytes = primal_host_bytes(original);
    comparison.retained_df_source_bytes = original.df_source.retained_source_bytes;
    comparison.retained_exact_source_bytes =
        original.reference_interaction_source
            ? original.reference_interaction_source->retained_numeric_bytes()
            : 0;
    // reference_retained_bytes already charges the clone's split epsilon vectors
    // and each shared reference/source once. The original host copy is additional.
    comparison.clone_admission_bytes = checked_add(
        comparison.output_bytes,
        checked_add(
            comparison.retained_primal_host_bytes,
            checked_add(original.problem.reference_retained_bytes,
                        checked_add(cc::problem_host_bytes(original.problem),
                                    capacity({&original.solved.t1, &original.solved.t2})))));
    const auto complete_budget = checked_add(original.budget, original.external_reservation_bytes);
    if (comparison.clone_admission_bytes > complete_budget)
      throw std::length_error("DF gap comparison native copies exceed complete budget");
    for (std::size_t index = 0; index < comparison.cases.size(); ++index) {
      auto& snapshot = comparison.cases[index];
      const auto clone_started = Clock::now();
      auto state = original;
      if (primal_identity(state) != comparison.primal_identity)
        throw std::logic_error("DF gap comparison did not preserve bitwise native inputs");
      snapshot.clone_seconds = elapsed(clone_started);
      const auto result = run_df_ccsdt_native_attempt(
          execution, system, auxiliary, descriptor, true, true, true, true, true,
          lambda_batch_limit, ccsd_batch_limit, frame_options, derived_denominators, packed_diis,
          index == 1 || index == 2, index != 2, &state, comparison.retained_primal_host_bytes,
          comparison.retained_df_source_bytes);
      if (result.forces.size() != coords || primal_identity(original) != comparison.primal_identity)
        throw std::logic_error("DF gap comparison changed its immutable primal or force shape");
      std::copy(result.forces.begin(), result.forces.end(), snapshot.forces.begin());
      snapshot.energy = result.energy;
      snapshot.triples_energy = result.triples_energy;
      snapshot.orbital_residual = result.orbital.orbital_residual;
      snapshot.maximum_stationarity = result.orbital.maximum_stationarity;
      snapshot.response_seconds = result.total_seconds;
      snapshot.triples_seconds = result.triples_seconds;
      snapshot.lambda_seconds = result.lambda_seconds;
      snapshot.source_response_seconds = result.source_response_seconds;
      snapshot.orbital_seconds = result.orbital_seconds;
      snapshot.numeric_capacity_bytes =
          std::max(result.numeric_capacity_bytes, comparison.clone_admission_bytes);
      snapshot.source_weight_values = result.source_weight_values;
      snapshot.metric_weight_values = result.metric_weight_values;
      snapshot.orbital_iterations = result.orbital.orbital_response.iterations;
      snapshot.orbital_actions = result.orbital.orbital_response.operator_actions;
      snapshot.fock_response_work = result.triples_fock.contraction_summands;
      snapshot.lambda = result.lambda;
      snapshot.triples = result.triples;
      snapshot.gap = result.triples_gap;
    }
  }
  comparison.total_seconds = elapsed(started);
  return comparison;
}

DFCCSDTResult run_df_ccsdt_native(
    runtime::ExecutionContext& execution, const core::System& system, const core::System& auxiliary,
    const generativeqc_method_descriptor& descriptor, bool forces, bool with_triples,
    bool df_auxiliary_reduction, bool df_matrix_gemm, bool lambda_matrix_gemm,
    std::size_t lambda_batch_limit, std::size_t ccsd_batch_limit,
    const hf::RHFFrameResponseOptions& frame_options, bool derived_denominators, bool packed_diis,
    bool parallel_gap_reduction, bool request_triples_gap_cotangents) {
  const auto started = Clock::now();
  auto* const recycling = frame_options.recycling;
  const bool had_retained_cache = recycling && recycling->storage_bytes();
  const auto attempt = [&] {
    return run_df_ccsdt_native_attempt(
        execution, system, auxiliary, descriptor, forces, with_triples, df_auxiliary_reduction,
        df_matrix_gemm, lambda_matrix_gemm, lambda_batch_limit, ccsd_batch_limit, frame_options,
        derived_denominators, packed_diis, parallel_gap_reduction, request_triples_gap_cotangents);
  };
  try {
    return attempt();
  } catch (const MethodError& error) {
    if (!had_retained_cache || !recycling->storage_bytes() ||
        error.status() != GENERATIVEQC_STATUS_OUT_OF_MEMORY)
      throw;
  } catch (const std::length_error&) {
    if (!had_retained_cache || !recycling->storage_bytes()) throw;
  } catch (const std::bad_alloc&) {
    if (!had_retained_cache || !recycling->storage_bytes()) throw;
  }
  // The failed attempt has unwound every phase owner and drained its streams.
  // A retained cache can constrain triples/Lambda/source as well as the primal.
  // Release it before one complete cold retry; other physical/CUDA errors pass
  // through unchanged, and a second resource failure propagates normally.
  recycling->clear();
  auto result = attempt();
  result.recycling_discarded_primal_attempt = true;
  result.total_seconds = elapsed(started);
  return result;
}
}  // namespace generativeqc::methods::detail
