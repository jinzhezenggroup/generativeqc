#pragma once

// Shared production execution templates. Batch tuning instantiates these exact
// kernels in one translation unit per candidate; equations remain generated.
#include <type_traits>

#ifdef GENERATIVEQC_DF_SHELL_MATH_HEADER
#include GENERATIVEQC_DF_SHELL_MATH_HEADER
#else
#include "generated_df_rys_shell.cuh"
#endif
#include "generated_df_screening.cuh"
#include "molecule/basis.hpp"
#include "runtime/homogeneous_task_packet.hpp"
#include "scf/cuda/df_shell_derivatives.cuh"

namespace generativeqc::scf {
namespace generated_df_fp32_shell {
template <unsigned A, unsigned B, unsigned C>
struct Shell;
template <unsigned A, unsigned B, unsigned C>
struct RysShell;
}  // namespace generated_df_fp32_shell
namespace {
namespace generated = generated_df_shell;
namespace scalar = generated_df_derivatives;

/** Publish one aggregated owner count; normal kernels receive a null sink. */
__device__ __forceinline__ void record_work(unsigned long long* work, DfShellWork kind,
                                            unsigned long long value) {
  if (work && value) atomicAdd(work + static_cast<unsigned>(kind), value);
}

/** A cast must not turn nonzero work into zero or overflow an otherwise finite
 * strict input. This is a representability guard, not a force-error certificate. */
__device__ __forceinline__ bool df_shell_float_input(double value) {
  const float narrowed = static_cast<float>(value);
  return isfinite(narrowed) && (value == 0.0 || narrowed != 0.0f);
}

/** One generated subgroup owns a shell triple and contracts its public weights.
 * Sparse normalized AO expansions are applied to weights once, before any
 * primitive work. Partial auxiliary shells contribute only in-panel weights.
 * Every subgroup lane rendezvous uses its original mask, including lanes with
 * zero weights or no component. Independent groups may share a physical warp;
 * no barrier waits on another group's primitive count or panel clipping.
 * No derivative array escapes the subgroup.
 */
template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false,
          bool Screening = false, bool Mixed = false>
__device__ __forceinline__ void contract_shell_task(
    DfShellBasisView first, DfShellBasisView second, DfShellBasisView auxiliary,
    const double* positions, std::size_t panel_begin, std::size_t panel_count,
    const double* weights, double* gradient, unsigned long long* counters, DfDerivativePairs pairs,
    bool triangle, std::size_t task, std::size_t tasks, unsigned long long* work,
    unsigned long long* precision_work = nullptr) {
  using Math = std::conditional_t<Rys, generated::RysShell<A, B, C>, generated::Shell<A, B, C>>;
  using Schedule = generated::Schedule<A, B, C, Variant>;
  constexpr bool prepare_cache = !Rys || Math::shared_root_state;
  constexpr auto lanes = Schedule::lanes, groups = Schedule::groups;
  union CacheStorage {
    double strict[3 * Math::axis_size];
    float mixed[3 * Math::axis_size];
  };
  __shared__ double all_weights[groups][Math::components];
  __shared__ CacheStorage all_cache[groups];
  __shared__ scalar::Geometry all_geometry[groups];
  const unsigned lane = threadIdx.x % lanes, group = threadIdx.x / lanes;
  const unsigned mask = (0xffffffffU >> (32 - lanes)) << ((threadIdx.x % 32) / lanes * lanes);
  if (work) work += (blockIdx.x % DfShellDiagnostics::shards) * DfShellDiagnostics::metrics;
  auto* cart_weights = all_weights[group];
  auto& geometry = all_geometry[group];
  if (task >= tasks) return;
  const auto sc = auxiliary.shell_ids[auxiliary.begin[C] + task % auxiliary.count[C]];
  const auto pair = task / auxiliary.count[C];
  auto ia_shell = pair / second.count[B], ib_shell = pair % second.count[B];
  if (triangle) {
    // Invert the compact triangular index, correcting roundoff at boundaries.
    ia_shell = static_cast<std::size_t>((sqrt(8.0 * pair + 1) - 1) * .5);
    while (ia_shell * (ia_shell + 1) / 2 > pair) --ia_shell;
    while ((ia_shell + 1) * (ia_shell + 2) / 2 <= pair) ++ia_shell;
    ib_shell = pair - ia_shell * (ia_shell + 1) / 2;
  }
  const auto sb = second.shell_ids[second.begin[B] + ib_shell];
  const auto sa = first.shell_ids[first.begin[A] + ia_shell];
  if (lane == 0 && counters) atomicAdd(counters, 1ULL);
  if (lane == 0) record_work(work, DfShellWork::shell_tasks, 1);
  const auto& o = first.basis;
  const auto& x = auxiliary.basis;
  const auto oa = first.ao_offsets[sa], ob = second.ao_offsets[sb];
  const auto oc = auxiliary.ao_offsets[sc];
  const auto na = first.ao_offsets[sa + 1] - oa;
  const auto nb = second.ao_offsets[sb + 1] - ob;
  const auto nc = auxiliary.ao_offsets[sc + 1] - oc;
  if (oc >= panel_begin + panel_count || oc + nc <= panel_begin) return;
  for (unsigned i = lane; i < Math::components; i += lanes) cart_weights[i] = 0;
  __syncwarp(mask);
  unsigned public_work = 0, public_loads = 0;
  unsigned long long expansion_work = 0, convolution_work = 0, recurrence_work = 0;
  for (auto i = std::int64_t{lane}; i < na * nb * nc; i += lanes) {
    const auto ai = oa + i / nb / nc, bi = ob + i / nc % nb, ci = oc + i % nc;
    if (ci < panel_begin || ci - panel_begin >= panel_count) continue;
    double weight;
    if (pairs == DfDerivativePairs::packed) {
      // A diagonal shell shares one physical center; its lower AO triangle
      // carries the complete folded response, including spherical expansions.
      if (sa == sb && ai < bi) continue;
      const auto hi = ai > bi ? ai : bi, lo = ai > bi ? bi : ai;
      weight = weights[(ci - panel_begin) * o.nbf * (o.nbf + 1) / 2 + hi * (hi + 1) / 2 + lo];
      ++public_loads;
    } else {
      const auto offset = (ci - panel_begin) * o.nbf * o.nbf;
      weight = weights[offset + ai * o.nbf + bi];
      ++public_loads;
      if (pairs == DfDerivativePairs::symmetric && sa != sb) {
        // Read both adjoints: exact also for nonsymmetric diagnostic weights.
        // Swapping orbital centers leaves the physical atomic derivative equal.
        weight += weights[offset + bi * o.nbf + ai];
        ++public_loads;
      }
    }
    if (weight == 0) continue;
    ++public_work;
    if (work)
      expansion_work += static_cast<unsigned long long>(o.term_counts[ai]) * o.term_counts[bi] *
                        x.term_counts[ci];
    for (unsigned at = 0; at < o.term_counts[ai]; ++at)
      for (unsigned bt = 0; bt < o.term_counts[bi]; ++bt)
        for (unsigned ct = 0; ct < x.term_counts[ci]; ++ct) {
          const auto ia = ai * first.expansion_terms + at;
          const auto ib = bi * second.expansion_terms + bt;
          const auto ic = ci * auxiliary.expansion_terms + ct;
          const auto ac = generated::cartesian_index(A, o.term_angular + 3 * ia);
          const auto bc = generated::cartesian_index(B, o.term_angular + 3 * ib);
          const auto cc = generated::cartesian_index(C, x.term_angular + 3 * ic);
          atomicAdd(
              cart_weights + (ac * Math::nb + bc) * Math::nc + cc,
              weight * o.term_coefficients[ia] * o.term_coefficients[ib] * x.term_coefficients[ic]);
        }
  }
  __syncwarp(mask);
  unsigned component_work = 0;
  double absolute_weight = 0;
  for (unsigned i = lane; i < Math::components; i += lanes)
    if (cart_weights[i] != 0) {
      ++component_work;
      if (first.force_shell_norms) absolute_weight += fabs(cart_weights[i]);
      if (work) {
        convolution_work += Math::convolution_work(i);
        // Component-local work skips zeros. Cooperative root/axis preparation
        // is fixed work for every active shell and is counted separately below.
        if constexpr (Rys) recurrence_work += Math::recurrence_work(i);
      }
    }
  const bool active = __any_sync(mask, component_work != 0);
  for (unsigned delta = lanes / 2; delta; delta /= 2) {
    public_work += __shfl_down_sync(mask, public_work, delta, lanes);
    public_loads += __shfl_down_sync(mask, public_loads, delta, lanes);
    component_work += __shfl_down_sync(mask, component_work, delta, lanes);
    if (first.force_shell_norms)
      absolute_weight += __shfl_down_sync(mask, absolute_weight, delta, lanes);
    if (work) {
      expansion_work += __shfl_down_sync(mask, expansion_work, delta, lanes);
      convolution_work += __shfl_down_sync(mask, convolution_work, delta, lanes);
      if constexpr (Rys) recurrence_work += __shfl_down_sync(mask, recurrence_work, delta, lanes);
    }
  }
  if (lane == 0 && counters) {
    atomicAdd(counters + 2, static_cast<unsigned long long>(public_work));
    atomicAdd(counters + 5, static_cast<unsigned long long>(public_loads));
  }
  if (lane == 0) {
    record_work(work, DfShellWork::public_weight_loads, public_loads);
    record_work(work, DfShellWork::public_nonzero_weights, public_work);
    record_work(work, DfShellWork::expansion_term_products, expansion_work);
    record_work(work, DfShellWork::folding_shared_atomics, expansion_work);
    record_work(work, DfShellWork::subgroup_rendezvous, 2);
  }
  if (!active) return;
  if (first.force_shell_norms) {
    // The Cartesian weights already contain spherical/component normalization
    // and the physical pair multiplicity. A Coulomb Cauchy bound therefore
    // certifies the whole contracted shell before any primitive recurrence.
    int skip = 0;
    if (lane == 0) {
      const auto low = oc > panel_begin ? oc : panel_begin;
      const auto high = oc + nc < panel_begin + panel_count ? oc + nc : panel_begin + panel_count;
      const double bound = absolute_weight *
                           first.force_shell_norms[sa * first.force_shell_stride + sb] *
                           auxiliary.force_shell_norms[sc];
      const double allowance = first.force_shell_budget * na * nb * (high - low);
      // A positive-product underflow retains work rather than certifying zero.
      skip = isfinite(bound) && bound > 0 && bound <= allowance;
      if (first.force_shell_counts) {
        atomicAdd(first.force_shell_counts, 1ULL);
        if (skip) {
          atomicAdd(first.force_shell_counts + 1, 1ULL);
          const auto primitive_a = o.primitive_offsets[sa + 1] - o.primitive_offsets[sa];
          const auto primitive_b = o.primitive_offsets[sb + 1] - o.primitive_offsets[sb];
          const auto primitive_c = x.primitive_offsets[sc + 1] - x.primitive_offsets[sc];
          atomicAdd(first.force_shell_counts + 2,
                    static_cast<unsigned long long>(primitive_a) * primitive_b * primitive_c);
        }
      }
    }
    skip = __shfl_sync(mask, skip, 0, lanes);
    if (skip) return;
  }
  const auto atom_a = o.shell_atoms[sa], atom_b = o.shell_atoms[sb], atom_c = x.shell_atoms[sc];
  const auto* ra = positions + 3 * atom_a;
  const auto* rb = positions + 3 * atom_b;
  const auto* rc = positions + 3 * atom_c;
  const scalar::Vec3 center_a{ra[0], ra[1], ra[2]}, center_b{rb[0], rb[1], rb[2]},
      center_c{rc[0], rc[1], rc[2]};
  double output[6]{};
  unsigned long long primitive_work = 0, skipped_work = 0;
  unsigned long long series_iterations = 0, series = 0, small_argument = 0, large_argument = 0;
  // Launch-uniform loop lengths keep compact subgroups on the same primitive
  // iteration. Bounds are runtime parameters; the generated math is unchanged.
  const auto pa0 = o.primitive_offsets[sa], pb0 = o.primitive_offsets[sb];
  const auto pc0 = x.primitive_offsets[sc];
  const auto na_prim = first.primitives ? first.primitives : o.primitive_offsets[sa + 1] - pa0;
  const auto nb_prim = second.primitives ? second.primitives : o.primitive_offsets[sb + 1] - pb0;
  const auto nc_prim =
      auxiliary.primitives ? auxiliary.primitives : x.primitive_offsets[sc + 1] - pc0;
  // Reuse the same strict-sized shared workspace on retry. Allocating a second
  // strict cache inside the mixed specialization would exceed the qualified
  // subgroup shared-memory bound for high-angular classes.
  const auto evaluate = [&]<bool Narrow>() -> bool {
    using Value = std::conditional_t<Narrow, float, double>;
    using Candidate = std::conditional_t<Rys, generated_df_fp32_shell::RysShell<A, B, C>,
                                         generated_df_fp32_shell::Shell<A, B, C>>;
    using Arithmetic = std::conditional_t<Narrow, Candidate, Math>;
    auto* cache = [&] {
      if constexpr (Narrow)
        return all_cache[group].mixed;
      else
        return all_cache[group].strict;
    }();
    Value independent[6]{};
    Value compensation[6]{};
    // Name the array member in an assignment to begin its union lifetime,
    // including the float-to-double transition on replay. The first primitive's
    // geometry rendezvous publishes this before any lane fills the cache.
    if constexpr (prepare_cache) {
      if (lane == 0) {
        if constexpr (Narrow)
          all_cache[group].mixed[0] = 0.0f;
        else
          all_cache[group].strict[0] = 0.0;
      }
    }
    primitive_work = skipped_work = 0;
    series_iterations = series = small_argument = large_argument = 0;
    for (std::size_t ia = 0; ia < na_prim; ++ia)
      for (std::size_t ib = 0; ib < nb_prim; ++ib)
        for (std::size_t ic = 0; ic < nc_prim; ++ic) {
          const auto pa = pa0 + ia, pb = pb0 + ib, pc = pc0 + ic;
          const double alpha = o.exponents[pa], beta = o.exponents[pb], gamma = x.exponents[pc];
          if constexpr (Screening) {
            static_assert(A + B + C == 0, "only the SSS derivative bound is qualified");
            int skip = 0;
            if (lane == 0) {
              const double weight =
                  cart_weights[0] * o.coefficients[pa] * o.coefficients[pb] * x.coefficients[pc];
              const double bound =
                  scalar::sss_force_bound(alpha, center_a, beta, center_b, gamma, center_c, weight);
              skip = isfinite(bound) && bound <= first.force_screen_budget;
              skipped_work += skip;
              if (first.force_screen_counts) {
                atomicAdd(first.force_screen_counts, 1ULL);
                if (skip) atomicAdd(first.force_screen_counts + 1, 1ULL);
              }
            }
            skip = __shfl_sync(mask, skip, 0, lanes);
            if (skip) continue;
          }
          if (lane == 0) {
            if constexpr (Rys) {
              scalar::prepare_geometry_rys<Math::nroots>(alpha, center_a, beta, center_b, gamma,
                                                         center_c, A + B + C, geometry);
            } else if (work) {
              scalar::BoysWork observed;
              scalar::prepare_geometry(alpha, center_a, beta, center_b, gamma, center_c, A + B + C,
                                       geometry, &observed);
              series_iterations += observed.series_iterations;
              series += observed.series;
              small_argument += observed.small_argument;
              large_argument += observed.large_argument;
            } else
              scalar::prepare_geometry(alpha, center_a, beta, center_b, gamma, center_c, A + B + C,
                                       geometry);
            ++primitive_work;
          }
          __syncwarp(mask);
          if constexpr (Narrow) {
            bool unsafe = !df_shell_float_input(alpha) || !df_shell_float_input(beta);
            if (lane == 0) {
              unsafe |= !df_shell_float_input(geometry.prefactor) ||
                        !df_shell_float_input(geometry.sx) || !df_shell_float_input(geometry.sy) ||
                        !df_shell_float_input(geometry.ip) || !df_shell_float_input(geometry.iq);
              for (unsigned axis = 0; axis < 3; ++axis)
                unsafe |= !df_shell_float_input(geometry.pa[axis]) ||
                          !df_shell_float_input(geometry.pb[axis]) ||
                          !df_shell_float_input(geometry.dx[axis]);
              constexpr auto moment_count = [] {
                if constexpr (Rys)
                  return 2 * Math::nroots;
                else
                  return A + B + C + 2;
              }();
              for (unsigned moment = 0; moment < moment_count; ++moment)
                unsafe |= !df_shell_float_input(geometry.f[moment]);
            }
            if (__any_sync(mask, unsafe)) return false;
          }
          if constexpr (prepare_cache) {
            Arithmetic::prepare(geometry, cache, lane, lanes);
            __syncwarp(mask);
          }
          bool unsafe = false;
          if constexpr (Narrow && prepare_cache)
            for (unsigned entry = lane; entry < 3 * Arithmetic::axis_size; entry += lanes)
              unsafe |= !isfinite(cache[entry]);
          for (unsigned i = lane; i < Math::components; i += lanes)
            if (cart_weights[i] != 0) {
              const double weight =
                  cart_weights[i] * o.coefficients[pa] * o.coefficients[pb] * x.coefficients[pc];
              if constexpr (Narrow) {
                unsafe |= !df_shell_float_input(weight);
                Value contribution[6]{};
                Arithmetic::accumulate(i, Value(alpha), Value(beta), geometry, cache, Value(weight),
                                       contribution);
                // Signed high-angular response can cancel across hundreds of
                // component/primitive products. Compensate these six FP32 sums
                // instead of accepting error growth with the task's work count.
                for (unsigned coordinate = 0; coordinate < 6; ++coordinate) {
                  const Value adjusted = contribution[coordinate] - compensation[coordinate];
                  const Value updated = independent[coordinate] + adjusted;
                  compensation[coordinate] = (updated - independent[coordinate]) - adjusted;
                  independent[coordinate] = updated;
                }
              } else {
                Arithmetic::accumulate(i, alpha, beta, geometry, cache, weight, independent);
              }
            }
          // The next primitive must not overwrite moments still being consumed.
          __syncwarp(mask);
          if constexpr (Narrow) {
            for (const auto value : independent) unsafe |= !isfinite(value);
            for (const auto value : compensation) unsafe |= !isfinite(value);
            if (__any_sync(mask, unsafe)) return false;
          }
        }
    for (unsigned i = 0; i < 6; ++i)
      for (unsigned delta = lanes / 2; delta; delta /= 2)
        independent[i] += __shfl_down_sync(mask, independent[i], delta, lanes);
    if constexpr (Narrow) {
      bool unsafe = false;
      for (const auto value : independent) unsafe |= !isfinite(value);
      if (__any_sync(mask, unsafe)) return false;
    }
    for (unsigned coordinate = 0; coordinate < 6; ++coordinate)
      output[coordinate] = independent[coordinate];
    return true;
  };
  if constexpr (Mixed) {
    const bool accepted = evaluate.template operator()<true>();
    if (lane == 0 && precision_work) {
      atomicAdd(precision_work, 1ULL);
      atomicAdd(precision_work + 1, primitive_work);
      if (!accepted) atomicAdd(precision_work + 2, 1ULL);
    }
    // Every subgroup lane reaches the rejection together, and nothing has
    // reached the global gradient. Replay the complete task, never add a strict
    // result to partially scattered mixed work.
    if (!accepted) {
      __syncwarp(mask);
      evaluate.template operator()<false>();
    }
  } else {
    evaluate.template operator()<false>();
  }
  if (lane == 0) {
    if constexpr (Screening)
      if (first.force_screen_counts && skipped_work && !primitive_work)
        atomicAdd(first.force_screen_counts + 2, 1ULL);
    const runtime::cuda_gaussian_products::Factor factors[3]{{o, oa}, {o, ob}, {x, oc}};
    runtime::cuda_gaussian_products::scatter(factors, Math::finish(output), 1.0, gradient);
    if (work) {
      record_work(work, DfShellWork::active_shell_tasks, 1);
      record_work(work, DfShellWork::primitive_products, primitive_work);
      record_work(work, DfShellWork::geometry_preparations, primitive_work);
      if constexpr (Rys) {
        record_work(work, DfShellWork::rys_evaluations, primitive_work);
        record_work(work, DfShellWork::rys_roots, Math::nroots * primitive_work);
        record_work(work, DfShellWork::recurrence_states,
                    (Math::shared_recurrence_states + recurrence_work) * primitive_work);
      } else {
        record_work(work, DfShellWork::boys_evaluations, primitive_work);
        record_work(work, DfShellWork::boys_order_sum, primitive_work * (A + B + C + 1));
        record_work(work, DfShellWork::boys_series_iterations, series_iterations);
        record_work(work, DfShellWork::boys_series, series);
        record_work(work, DfShellWork::boys_small_argument, small_argument);
        record_work(work, DfShellWork::boys_large_argument, large_argument);
      }
      record_work(work, DfShellWork::axis_polynomial_calls,
                  primitive_work * Math::polynomial_calls);
      record_work(work, DfShellWork::specialized_prepare_axis_calls,
                  primitive_work * Math::specialized_axis_calls);
      record_work(work, DfShellWork::cache_coefficient_values,
                  primitive_work * Math::cache_coefficient_values);
      record_work(work, DfShellWork::convolution_iterations, primitive_work * convolution_work);
      record_work(work, DfShellWork::active_component_products, primitive_work * component_work);
      record_work(work, DfShellWork::gradient_atomics_a, 3);
      record_work(work, DfShellWork::gradient_atomics_b, 3);
      record_work(work, DfShellWork::gradient_atomics_c, 3);
      const unsigned shared = 3 * (unsigned(atom_a == atom_b || atom_a == atom_c) +
                                   unsigned(atom_b == atom_a || atom_b == atom_c) +
                                   unsigned(atom_c == atom_a || atom_c == atom_b));
      record_work(work, DfShellWork::gradient_atomics_shared_atom, shared);
      record_work(work, DfShellWork::gradient_atomics_distinct_atom, 9 - shared);
      record_work(work, DfShellWork::subgroup_rendezvous, (prepare_cache ? 3 : 2) * primitive_work);
    }
    if (counters) {
      atomicAdd(counters + 1, 1ULL);
      atomicAdd(counters + 3, primitive_work);
      atomicAdd(counters + 4, primitive_work * component_work);
    }
  }
}

/** One bounded descriptor names a homogeneous product, never individual triples. */
struct SignatureSlice {
  const std::int32_t *a_ids{}, *b_ids{}, *c_ids{};
  std::size_t a_begin{}, b_begin{}, c_begin{}, a_count{}, b_count{}, c_count{};
  std::size_t a_primitives{}, b_primitives{}, c_primitives{};
  std::size_t first_block{}, tasks{};
  bool triangle{};
};
static_assert(generated::signature_packet_capacity == DfShellDiagnostics::packet_capacity);
using SignaturePacket =
    runtime::HomogeneousTaskPacket<SignatureSlice, generated::signature_packet_capacity>;
// Stay within the original CUDA 4-KiB kernel-argument limit, including views
// and scalar arguments. Driver-copied parameters need no mutable device queue.
static_assert(sizeof(SignaturePacket) + 2 * sizeof(DfShellBasisView) + 128 <= 4096);

template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false,
          bool Screening = false, bool Mixed = false>
__global__ void shell_panel(DfShellBasisView first, DfShellBasisView second,
                            DfShellBasisView auxiliary, const double* positions, std::size_t begin,
                            std::size_t count, const double* weights, double* gradient,
                            unsigned long long* counters, DfDerivativePairs pairs, bool triangle,
                            std::size_t tasks, unsigned long long* work,
                            unsigned long long* precision_work = nullptr) {
  using Schedule = generated::Schedule<A, B, C, Variant>;
  const auto task = std::size_t{blockIdx.x} * Schedule::groups + threadIdx.x / Schedule::lanes;
  contract_shell_task<A, B, C, Variant, Rys, Screening, Mixed>(
      first, second, auxiliary, positions, begin, count, weights, gradient, counters, pairs,
      triangle, task, tasks, work, precision_work);
}

template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false,
          bool Screening = false, bool Mixed = false>
__global__ void shell_packet(DfShellBasisView orbital, DfShellBasisView auxiliary,
                             const double* positions, std::size_t begin, std::size_t count,
                             const double* weights, double* gradient, unsigned long long* counters,
                             DfDerivativePairs pairs,
                             const __grid_constant__ SignaturePacket packet,
                             unsigned long long* work,
                             unsigned long long* precision_work = nullptr) {
  using Schedule = generated::Schedule<A, B, C, Variant>;
  // A block owns only one signature. Prefixing block counts pools small slices
  // in a single grid without reintroducing mixed primitive lengths in a warp.
  unsigned low = 0, high = packet.count;
  while (low + 1 < high) {
    const auto middle = (low + high) / 2;
    if (packet.slices[middle].first_block <= blockIdx.x)
      low = middle;
    else
      high = middle;
  }
  // grid_constant prevents an address-taken packet from becoming a private
  // multi-KiB local copy in every thread. All fields remain read-only parameters.
  const auto& slice = packet.slices[low];
  auto first = orbital, second = orbital, third = auxiliary;
  first.shell_ids = slice.a_ids;
  second.shell_ids = slice.b_ids;
  third.shell_ids = slice.c_ids;
  first.begin[A] = slice.a_begin;
  second.begin[B] = slice.b_begin;
  third.begin[C] = slice.c_begin;
  first.count[A] = slice.a_count;
  second.count[B] = slice.b_count;
  third.count[C] = slice.c_count;
  first.primitives = slice.a_primitives;
  second.primitives = slice.b_primitives;
  third.primitives = slice.c_primitives;
  const auto task = (std::size_t{blockIdx.x} - slice.first_block) * Schedule::groups +
                    threadIdx.x / Schedule::lanes;
  contract_shell_task<A, B, C, Variant, Rys, Screening, Mixed>(
      first, second, third, positions, begin, count, weights, gradient, counters, pairs,
      slice.triangle, task, slice.tasks,
      work ? work + low * DfShellDiagnostics::row_elements : nullptr, precision_work);
}

}  // namespace
}  // namespace generativeqc::scf
