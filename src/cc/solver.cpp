#include "cc/solver.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

#include "cc/df_plan.hpp"
#include "cc/iteration_driver.hpp"
#include "generated_df_ccsd_core_cpu.hpp"
#include "generated_rccsd_cpu.hpp"
#include "solver/diis.hpp"

namespace generativeqc::cc {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b) { return generated::checked_add(a, b); }
std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("RCCSD size overflow");
  return a * b;
}
std::size_t bytes(std::size_t elements) { return checked_mul(elements, sizeof(double)); }

generated::Inputs inputs(const Problem& p, const double* t1, const double* t2) {
  return {p.foo.data(),
          p.fov.data(),
          p.fvv.data(),
          p.ovov.data(),
          p.ovvo.data(),
          p.oovv.data(),
          p.ovvv.data(),
          p.ovoo.data(),
          p.oooo.data(),
          p.vvvv.data(),
          p.d1.data(),
          p.d2.data(),
          t1,
          t2,
          p.canonical_eps.empty() ? nullptr : p.canonical_eps.data(),
          p.canonical_level_shift};
}

double max_abs(const double* p, std::size_t n) {
  double result = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(p[i])) throw std::runtime_error("nonfinite RCCSD physical residual");
    result = std::max(result, std::abs(p[i]));
  }
  return result;
}

}  // namespace

namespace {

double validate_canonical_spectrum(const Problem& p, bool check_singles) {
  const auto o = p.nocc, v = p.nvir;
  if (!o || !v || p.canonical_eps.size() != checked_add(o, v) ||
      !std::isfinite(p.canonical_level_shift) || p.canonical_level_shift < 0.0 ||
      !std::isfinite(p.canonical_denominator_threshold) ||
      !(p.canonical_denominator_threshold > 0.0) ||
      !std::all_of(
          p.canonical_eps.begin(), p.canonical_eps.end(),
          [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("invalid canonical RCCSD denominator provenance");
  if (check_singles && p.d1.size() != checked_mul(o, v))
    throw std::invalid_argument("invalid canonical RCCSD singles denominator shape");
  double minimum = std::numeric_limits<double>::infinity();
  double most_negative = 0.0;
  std::size_t far_i = 0, far_a = 0;
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a) {
      double physical, shifted;
      generated::canonical_single(p.canonical_eps[i], p.canonical_eps[o + a],
                                  p.canonical_level_shift, physical, shifted);
      if (!std::isfinite(physical) || physical >= 0.0 ||
          std::abs(physical) <= p.canonical_denominator_threshold || !std::isfinite(shifted))
        throw std::invalid_argument("near-zero, nonnegative or nonfinite RCCSD denominator");
      if (check_singles && p.d1[i * v + a] != shifted)
        throw std::invalid_argument("canonical RCCSD singles denominator disagrees with spectrum");
      minimum = std::min(minimum, std::abs(physical));
      if (physical < most_negative) {
        most_negative = physical;
        far_i = i;
        far_a = a;
      }
    }
  // IEEE round-to-nearest addition/subtraction is monotone. Every physical
  // doubles value is a sum of two admitted negative single gaps, so it cannot
  // approach zero more closely than either gap. Its most negative member uses
  // the most negative gap twice; checking that member also bounds all shifted
  // values for a nonnegative shift. Thus no full doubles validation pass is
  // needed. Keep the original (ei-ea)+(ej-eb)-2*shift grouping in this check.
  double physical, shifted;
  generated::canonical_double(p.canonical_eps[far_i], p.canonical_eps[o + far_a],
                              p.canonical_eps[far_i], p.canonical_eps[o + far_a],
                              p.canonical_level_shift, physical, shifted);
  if (!std::isfinite(physical) || !std::isfinite(shifted))
    throw std::invalid_argument("nonfinite canonical RCCSD doubles denominator");
  return minimum;
}

}  // namespace

void initialize_canonical_denominators(Problem& p, std::span<const double> energies,
                                       const SolverOptions& options, bool derived) {
  p.canonical_eps.assign(energies.begin(), energies.end());
  p.canonical_level_shift = options.level_shift;
  p.canonical_denominator_threshold = options.denominator_threshold;
  p.minimum_absolute_denominator = validate_canonical_spectrum(p, false);
  p.d1.resize(checked_mul(p.nocc, p.nvir));
  for (std::size_t i = 0; i < p.nocc; ++i)
    for (std::size_t a = 0; a < p.nvir; ++a) {
      double physical;
      generated::canonical_single(energies[i], energies[p.nocc + a], options.level_shift, physical,
                                  p.d1[i * p.nvir + a]);
    }
  if (derived) {
    p.denominator_representation = DenominatorRepresentation::CanonicalSpectrum;
    std::vector<double>().swap(p.d2);
  } else {
    p.denominator_representation = DenominatorRepresentation::Explicit;
    p.d2.resize(checked_mul(p.d1.size(), p.d1.size()));
    for (std::size_t flat = 0; flat < p.d2.size(); ++flat)
      p.d2[flat] = generated::canonical_d2_at(flat, p.nocc, p.nvir, p.canonical_eps.data(),
                                              p.canonical_level_shift);
    std::vector<double>().swap(p.canonical_eps);
    p.canonical_level_shift = p.canonical_denominator_threshold = 0.0;
  }
}

double doubles_denominator_at(const Problem& p, std::size_t flat) {
  return p.denominator_representation == DenominatorRepresentation::CanonicalSpectrum
             ? generated::canonical_d2_at(flat, p.nocc, p.nvir, p.canonical_eps.data(),
                                          p.canonical_level_shift)
             : p.d2[flat];
}

std::uint64_t denominator_identity(const Problem& p) {
  std::uint64_t hash = 14695981039346656037ULL;
  const auto word = [&](std::uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (8 * byte)) & 255;
      hash *= 1099511628211ULL;
    }
  };
  word(1);  // Provenance schema, including ordered-pair FP64 grouping.
  word(static_cast<std::uint64_t>(p.denominator_representation));
  word(p.nocc);
  word(p.nvir);
  for (const auto* values : {&p.d1, &p.d2, &p.canonical_eps}) {
    word(values->size());
    for (double value : *values) word(std::bit_cast<std::uint64_t>(value));
  }
  word(std::bit_cast<std::uint64_t>(p.canonical_level_shift));
  word(std::bit_cast<std::uint64_t>(p.canonical_denominator_threshold));
  return hash;
}

void validate_problem(const Problem& p, bool allow_df_virtual) {
  if (p.naux && !allow_df_virtual)
    throw std::invalid_argument("DF virtual inputs require a factorized execution owner");
  if (!p.nocc || !p.nvir)
    throw std::invalid_argument("RCCSD requires occupied and virtual orbitals");
  const auto o = p.nocc, v = p.nvir;
  auto expect = [](const std::vector<double>& x, std::size_t n, const char* name) {
    if (x.size() != n ||
        !std::all_of(x.begin(), x.end(), [](double y) { return std::isfinite(y); }))
      throw std::invalid_argument(std::string("invalid RCCSD input ") + name);
  };
  const auto ov = checked_mul(o, v), oo = checked_mul(o, o), vv = checked_mul(v, v);
  const auto oovv = checked_mul(oo, vv);
  expect(p.foo, oo, "foo");
  expect(p.fov, ov, "fov");
  expect(p.fvv, vv, "fvv");
  expect(p.ovov, oovv, "ovov");
  expect(p.ovvo, oovv, "ovvo");
  expect(p.oovv, oovv, "oovv");
  if (p.naux) {
    expect(p.ovvv, 0, "ovvv must be empty for DF");
    expect(p.vvvv, 0, "vvvv must be empty for DF");
    expect(p.df_bov, checked_mul(p.naux, ov), "df_bov");
    expect(p.df_bvv, checked_mul(p.naux, vv), "df_bvv");
    if (!p.df_boo.empty()) {
      expect(p.df_boo, checked_mul(p.naux, oo), "df_boo");
      for (std::size_t q = 0; q < p.naux; ++q)
        for (std::size_t i = 0; i < o; ++i)
          for (std::size_t j = 0; j < i; ++j)
            if (std::abs(p.df_boo[q * oo + i * o + j] - p.df_boo[q * oo + j * o + i]) > 1e-10)
              throw std::invalid_argument("DF B_oo must preserve symmetric spatial-MO pairs");
    }
    for (std::size_t q = 0; q < p.naux; ++q)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < a; ++b)
          if (std::abs(p.df_bvv[q * vv + a * v + b] - p.df_bvv[q * vv + b * v + a]) > 1e-10)
            throw std::invalid_argument("DF B_vv must preserve symmetric spatial-MO pairs");
  } else {
    expect(p.df_bov, 0, "df_bov requires naux");
    expect(p.df_bvv, 0, "df_bvv requires naux");
    expect(p.df_boo, 0, "df_boo requires naux");
    expect(p.ovvv, checked_mul(o, checked_mul(vv, v)), "ovvv");
    expect(p.vvvv, checked_mul(vv, vv), "vvvv");
  }
  expect(p.ovoo, checked_mul(ov, oo), "ovoo");
  expect(p.oooo, checked_mul(oo, oo), "oooo");
  expect(p.d1, ov, "d1");
  if (p.denominator_representation == DenominatorRepresentation::CanonicalSpectrum) {
    expect(p.d2, 0, "canonical d2 must be absent");
    validate_canonical_spectrum(p, true);
  } else if (p.denominator_representation == DenominatorRepresentation::Explicit) {
    expect(p.d2, oovv, "d2");
    expect(p.canonical_eps, 0, "explicit d2 must not carry a canonical spectrum");
    if (p.canonical_level_shift != 0.0 || p.canonical_denominator_threshold != 0.0)
      throw std::invalid_argument("explicit RCCSD denominators carry canonical metadata");
  } else {
    throw std::invalid_argument("unknown RCCSD denominator representation");
  }
  expect(p.initial_t1, ov, "initial_t1");
  expect(p.initial_t2, oovv, "initial_t2");
  if (!std::isfinite(p.reference_energy))
    throw std::invalid_argument("nonfinite RCCSD reference energy");
}

void validate_options(const SolverOptions& o) {
  if (!o.max_iterations || o.diis_size == 1 || o.diis_size > 20 || !o.max_bytes ||
      !o.df_auxiliary_batch_limit)
    throw std::invalid_argument("invalid RCCSD iteration/history/budget option");
  if (!(o.energy_tolerance > 0.0 && o.energy_tolerance <= 1e-8) ||
      !(o.residual_tolerance > 0.0 && o.residual_tolerance <= 1e-9) ||
      !(o.denominator_threshold > 0.0) || !(o.damping >= 0.0 && o.damping < 1.0) ||
      !(o.level_shift >= 0.0) || !std::isfinite(o.energy_tolerance) ||
      !std::isfinite(o.residual_tolerance) || !std::isfinite(o.denominator_threshold) ||
      !std::isfinite(o.damping) || !std::isfinite(o.level_shift))
    throw std::invalid_argument("invalid RCCSD numeric option");
}

std::size_t problem_host_bytes(const Problem& p) {
  std::size_t result = 0;
  const std::vector<double>* values[] = {
      &p.foo,  &p.fov,  &p.fvv, &p.ovov, &p.ovvo,       &p.oovv,       &p.ovvv,   &p.ovoo,
      &p.oooo, &p.vvvv, &p.d1,  &p.d2,   &p.initial_t1, &p.initial_t2, &p.df_bov, &p.df_bvv};
  for (const auto* value : values) result = checked_add(result, bytes(value->capacity()));
  result = checked_add(result, bytes(p.df_boo.capacity()));
  result = checked_add(result, bytes(p.canonical_eps.capacity()));
  return result;
}

SolverResult solve_cpu(const Problem& p, const SolverOptions& options) {
  validate_problem(p, true);
  validate_options(options);
  const auto n1 = checked_mul(p.nocc, p.nvir);
  const auto n2 = checked_mul(checked_mul(p.nocc, p.nocc), checked_mul(p.nvir, p.nvir));
  const auto elements = checked_add(n1, n2);
  auto plan = p.naux
                  ? df_iteration_plan(p.nocc, p.nvir, p.naux, false, options.df_auxiliary_reduction)
                  : DFIterationPlan{};
  const auto replay_elements = p.naux ? generated::dfcore::replay_arena_elements(p.nocc, p.nvir)
                                      : generated::replay_arena_elements(p.nocc, p.nvir);
  const auto uncached_elements =
      p.naux ? std::size_t{0} : generated::iteration_arena_elements(p.nocc, p.nvir);
  auto conventional_elements = uncached_elements;
  auto capacity_for = [&](const DFIterationPlan& choice, std::size_t dense_elements) {
    std::size_t capacity = checked_add(p.reference_retained_bytes, problem_host_bytes(p));
    capacity = checked_add(capacity, bytes(p.naux ? choice.iteration : dense_elements));
    capacity = checked_add(capacity, bytes(replay_elements));
    capacity = checked_add(capacity, bytes(choice.auxiliary));
    capacity = checked_add(capacity, bytes(choice.preparation));
    capacity = checked_add(capacity, bytes(choice.accumulation));
    // Without DIIS only current and the next Jacobi trial coexist. With DIIS,
    // trial/error plus the copied history vector coexist before trimming.
    // DIIS additionally retains Gram/original augmented arrays while solve_linear
    // owns its by-value matrix/RHS copies. These are numeric storage, not overhead.
    const auto host_vectors = options.diis_size ? 4 + 2 * options.diis_size : 2;
    capacity = checked_add(capacity, bytes(checked_mul(host_vectors, elements)));
    if (options.diis_size) {
      const std::size_t h = options.diis_size, n = h + 1;
      const auto scratch = checked_add(
          checked_mul(h, h), checked_add(checked_mul(2, checked_mul(n, n)), checked_mul(2, n)));
      capacity = checked_add(capacity, bytes(scratch));
    }
    return capacity;
  };
  if (plan.hoisted && capacity_for(plan, conventional_elements) > options.max_bytes)
    plan = df_iteration_plan(p.nocc, p.nvir, p.naux, false, false);
  auto capacity = capacity_for(plan, conventional_elements);
  if (capacity > options.max_bytes)
    throw std::length_error("RCCSD CPU solve exceeds correlation memory budget");

  bool reuse_invariants = false;
  if (options.iteration_invariant_reuse && !p.naux &&
      generated::iteration_invariant_operation_count) {
    // Charge the complete endpoint, including pinned reference intermediates,
    // before selecting reuse. Overflow in optional storage is also a fallback.
    try {
      const auto retained = generated::iteration_reuse_arena_elements(p.nocc, p.nvir);
      const auto retained_capacity = capacity_for(plan, retained);
      if (retained_capacity <= options.max_bytes) {
        conventional_elements = retained;
        capacity = retained_capacity;
        reuse_invariants = true;
      }
    } catch (const std::length_error&) {
    }
  }

  std::vector<double> iteration_arena;
  if (reuse_invariants) {
    try {
      iteration_arena.resize(conventional_elements);
    } catch (const std::bad_alloc&) {
      reuse_invariants = false;
    } catch (const std::length_error&) {
      reuse_invariants = false;
    }
    if (!reuse_invariants) {
      // Failed vector growth leaves the empty vector unchanged: the retry
      // never holds both optional and baseline numeric arenas simultaneously.
      conventional_elements = uncached_elements;
      capacity = capacity_for(plan, conventional_elements);
    }
  }
  if (!reuse_invariants) iteration_arena.resize(p.naux ? plan.iteration : conventional_elements);
  std::vector<double> replay_arena(replay_elements);
  std::vector<double> virtual_arena(plan.auxiliary), virtual_sum(plan.accumulation),
      prepare_arena(plan.preparation);
  std::vector<double> current;
  current.reserve(elements);
  current.insert(current.end(), p.initial_t1.begin(), p.initial_t1.end());
  current.insert(current.end(), p.initial_t2.begin(), p.initial_t2.end());
  generativeqc::solver::Diis diis(options.diis_size, elements);
  SolverResult result;
  result.diagnostic.denominator_identity = denominator_identity(p);
  result.diagnostic.numeric_capacity_bytes = std::max(p.provider_peak_bytes, capacity);
  result.diagnostic.iteration_reuse = reuse_invariants;
  result.reason = "maximum RCCSD iterations reached";
  const auto solve_started = std::chrono::steady_clock::now();

  // The same accumulation owner serves current/trial/replay amplitudes. Never
  // reuse a correction after DIIS changes T, or count only a single Q slice.
  auto df_inputs = [&](const generated::Inputs& in) {
    std::fill(virtual_sum.begin(), virtual_sum.end(), 0.0);
    generated::df::Inputs factors{};
    factors.t1 = in.t1;
    factors.t2 = in.t2;
    for (std::size_t q = 0; q < p.naux; ++q) {
      factors.bov = p.df_bov.data() + q * n1;
      factors.bvv = p.df_bvv.data() + q * p.nvir * p.nvir;
      const auto out = generated::df::run_virtual_cpu(p.nocc, p.nvir, factors, virtual_arena.data(),
                                                      virtual_arena.size());
      for (std::size_t k = 0; k < n1; ++k) virtual_sum[k] += out.singles[k];
      for (std::size_t k = 0; k < n2; ++k) virtual_sum[n1 + k] += out.doubles[k];
      ++result.diagnostic.df_auxiliary_slices;
      result.diagnostic.df_virtual_operations += generated::df::virtual_cpu_operation_count;
      result.diagnostic.df_accumulation_calls += 2;
      result.diagnostic.df_contraction_terms =
          checked_add(result.diagnostic.df_contraction_terms,
                      generated::dfhoist::fallback_virtual_cpu_contraction_terms(p.nocc, p.nvir));
    }
    return generated::dfcore::Inputs{in.foo,
                                     in.fov,
                                     in.fvv,
                                     in.ovov,
                                     in.ovvo,
                                     in.oovv,
                                     in.ovoo,
                                     in.oooo,
                                     in.d1,
                                     in.d2,
                                     in.t1,
                                     in.t2,
                                     virtual_sum.data(),
                                     virtual_sum.data() + n1,
                                     in.canonical_eps,
                                     in.canonical_level_shift};
  };
  // This epoch is exactly this synchronous solve on const Problem&. Its owned
  // reference vectors cannot change during the call; current/trial amplitudes
  // are separate dynamic inputs. No state or retained value survives return,
  // so a new reference/geometry/basis/method or solve always prepares anew.
  bool invariants_prepared = false;
  auto run_iteration = [&](const generated::Inputs& in) -> generated::IterationOutputs {
    if (!p.naux && reuse_invariants) {
      const bool had_prepared = invariants_prepared;
      if (!invariants_prepared) {
        generated::run_iteration_reuse_prepare_cpu(p.nocc, p.nvir, in, iteration_arena.data(),
                                                   iteration_arena.size());
        // A throwing/partially written preparation is never published as ready.
        invariants_prepared = true;
        ++result.diagnostic.iteration_invariant_preparations;
        result.diagnostic.iteration_invariant_operations +=
            generated::iteration_invariant_operation_count;
      }
      const auto out = generated::run_iteration_reused_cpu(
          p.nocc, p.nvir, in, iteration_arena.data(), iteration_arena.size());
      ++result.diagnostic.iteration_reused_evaluations;
      result.diagnostic.iteration_dynamic_operations +=
          generated::iteration_dynamic_operation_count;
      if (had_prepared)
        result.diagnostic.iteration_invariant_operations_saved +=
            generated::iteration_invariant_operation_count;
      return out;
    }
    if (!p.naux) {
      const auto out = generated::run_iteration_cpu(p.nocc, p.nvir, in, iteration_arena.data(),
                                                    iteration_arena.size());
      result.diagnostic.iteration_invariant_operations +=
          generated::iteration_invariant_operation_count;
      result.diagnostic.iteration_dynamic_operations +=
          generated::iteration_dynamic_operation_count;
      return out;
    }
    if (plan.hoisted) {
      generated::dfhoist::Inputs fast{};
      // The first two sums preserve the old singles/ladder layout so expanded
      // convergence replay can reuse the same allocated accumulator.
      static_cast<generated::dfcore::Inputs&>(fast) = {in.foo,
                                                       in.fov,
                                                       in.fvv,
                                                       in.ovov,
                                                       in.ovvo,
                                                       in.oovv,
                                                       in.ovoo,
                                                       in.oooo,
                                                       in.d1,
                                                       in.d2,
                                                       in.t1,
                                                       in.t2,
                                                       virtual_sum.data(),
                                                       virtual_sum.data() + n1,
                                                       in.canonical_eps,
                                                       in.canonical_level_shift};
      fast.df_singles_residual = virtual_sum.data();
      fast.df_D05_vv_ladder = virtual_sum.data() + n1;
      fast.df_Lvv = virtual_sum.data() + elements;
      fast.df_Wvoov = fast.df_Lvv + p.nvir * p.nvir;
      fast.df_Wvovo = fast.df_Wvoov + n2;
      fast.df_Xv = fast.df_Wvovo + n2;
      std::fill(virtual_sum.begin(), virtual_sum.end(), 0.0);
      fast.df_tau = generated::dfhoist::run_prepare_cpu(p.nocc, p.nvir, fast, prepare_arena.data(),
                                                        prepare_arena.size())
                        .tau;
      ++result.diagnostic.df_preparation_calls;
      for (std::size_t q = 0; q < p.naux; ++q) {
        fast.bov = p.df_bov.data() + q * n1;
        fast.bvv = p.df_bvv.data() + q * p.nvir * p.nvir;
        const auto row = generated::dfhoist::run_auxiliary_cpu(
            p.nocc, p.nvir, fast, virtual_arena.data(), virtual_arena.size());
        const std::array<const double*, 6> sources{row.singles, row.ladder, row.lvv,
                                                   row.wvoov,   row.wvovo,  row.xv};
        const std::array<std::size_t, 6> counts{n1, n2, p.nvir * p.nvir, n2, n2, n2};
        std::size_t offset = 0;
        for (std::size_t field = 0; field < counts.size(); ++field) {
          for (std::size_t k = 0; k < counts[field]; ++k)
            virtual_sum[offset + k] += sources[field][k];
          offset += counts[field];
        }
        ++result.diagnostic.df_auxiliary_slices;
        result.diagnostic.df_virtual_operations += generated::dfhoist::auxiliary_operation_count;
        result.diagnostic.df_accumulation_calls += counts.size();
      }
      ++result.diagnostic.df_hoisted_evaluations;
      result.diagnostic.df_contraction_terms = checked_add(
          result.diagnostic.df_contraction_terms,
          checked_add(plan.preparation_terms,
                      checked_add(checked_mul(p.naux, plan.auxiliary_terms), plan.core_terms)));
      const auto out = generated::dfhoist::run_iteration_cpu(
          p.nocc, p.nvir, fast, iteration_arena.data(), iteration_arena.size());
      return {out.energy, out.r1, out.r2, out.next_t1, out.next_t2};
    }
    const auto core = df_inputs(in);
    const auto out = generated::dfcore::run_iteration_cpu(
        p.nocc, p.nvir, core, iteration_arena.data(), iteration_arena.size());
    result.diagnostic.df_contraction_terms =
        checked_add(result.diagnostic.df_contraction_terms,
                    generated::dfhoist::fallback_core_contraction_terms(p.nocc, p.nvir));
    return {out.energy, out.r1, out.r2, out.next_t1, out.next_t2};
  };
  auto run_replay = [&](const generated::Inputs& in) -> generated::ReplayOutputs {
    if (!p.naux)
      return generated::run_replay_cpu(p.nocc, p.nvir, in, replay_arena.data(),
                                       replay_arena.size());
    const auto core = df_inputs(in);
    const auto out = generated::dfcore::run_replay_cpu(p.nocc, p.nvir, core, replay_arena.data(),
                                                       replay_arena.size());
    result.diagnostic.df_contraction_terms =
        checked_add(result.diagnostic.df_contraction_terms,
                    generated::dfhoist::fallback_replay_contraction_terms(p.nocc, p.nvir));
    return {out.energy, out.r1, out.r2};
  };

  run_cc_iterations<generated::IterationOutputs>(
      options,
      [&](const std::optional<generated::IterationOutputs>& carried) {
        auto in = inputs(p, current.data(), current.data() + n1);
        const auto iteration_started = std::chrono::steady_clock::now();
        generated::IterationOutputs out{};
        if (carried) {
          out = *carried;
        } else {
          out = run_iteration(in);
          ++result.diagnostic.iteration_graph_calls;
          if (!p.canonical_eps.empty()) result.diagnostic.derived_d2_iteration_evaluations += n2;
        }
        result.diagnostic.iteration_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - iteration_started)
                .count();
        const double r1 = max_abs(out.r1, n1), r2 = max_abs(out.r2, n2);
        return std::pair{out, IterationMetrics{out.energy, r1, r2}};
      },
      [&](unsigned observations, IterationMetrics status, double delta) {
        result.correlation_energy = status.energy;
        result.total_energy = p.reference_energy + status.energy;
        result.diagnostic.iterations = observations;
        result.diagnostic.energy_change = delta;
        result.diagnostic.r1_max = status.r1;
        result.diagnostic.r2_max = status.r2;
      },
      [&]() {
        auto in = inputs(p, current.data(), current.data() + n1);
        const auto replay_started = std::chrono::steady_clock::now();
        const auto replay = run_replay(in);
        result.diagnostic.replay_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - replay_started)
                .count();
        ++result.diagnostic.replay_graph_calls;
        result.diagnostic.replay_r1_max = max_abs(replay.r1, n1);
        result.diagnostic.replay_r2_max = max_abs(replay.r2, n2);
        return IterationMetrics{replay.energy, result.diagnostic.replay_r1_max,
                                result.diagnostic.replay_r2_max};
      },
      [&](const generated::IterationOutputs& out) -> std::optional<generated::IterationOutputs> {
        const auto update_started = std::chrono::steady_clock::now();
        std::vector<double> trial(elements);
        const double jacobi = 1.0 - options.damping;
        for (std::size_t k = 0; k < n1; ++k)
          trial[k] = current[k] + jacobi * (out.next_t1[k] - current[k]);
        for (std::size_t k = 0; k < n2; ++k)
          trial[n1 + k] = current[n1 + k] + jacobi * (out.next_t2[k] - current[n1 + k]);
        result.diagnostic.update_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - update_started)
                .count();
        ++result.diagnostic.update_calls;
        if (!options.diis_size) {
          current = std::move(trial);
          return std::nullopt;
        }
        auto trial_in = inputs(p, trial.data(), trial.data() + n1);
        const auto trial_started = std::chrono::steady_clock::now();
        const auto trial_out = run_iteration(trial_in);
        result.diagnostic.iteration_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - trial_started).count();
        ++result.diagnostic.iteration_graph_calls;
        if (!p.canonical_eps.empty()) result.diagnostic.derived_d2_iteration_evaluations += n2;
        std::vector<double> error;
        error.reserve(elements);
        error.insert(error.end(), trial_out.r1, trial_out.r1 + n1);
        error.insert(error.end(), trial_out.r2, trial_out.r2 + n2);
        const auto diis_started = std::chrono::steady_clock::now();
        auto update = diis.update_with_status(std::move(trial), std::move(error));
        current = std::move(update.vector);
        result.diagnostic.diis_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - diis_started).count();
        if (!update.modified) return trial_out;
        return std::nullopt;
      },
      [&]() {
        result.status = SolveStatus::Converged;
        result.reason = "energy change and expanded physical R1/R2 passed";
      },
      [&](const std::runtime_error& error) {
        result.status = SolveStatus::NumericalFailure;
        result.reason = error.what();
      });
  result.diagnostic.diis_restarts = diis.restarts();
  result.diagnostic.tensor_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_started).count();
  result.t1.assign(current.begin(), current.begin() + static_cast<std::ptrdiff_t>(n1));
  result.t2.assign(current.begin() + static_cast<std::ptrdiff_t>(n1), current.end());
  return result;
}

#if !GENERATIVEQC_HAS_CUDA
SolverResult solve_cuda(const Problem&, const SolverOptions&, int) {
  throw std::runtime_error("CUDA RCCSD is not compiled");
}
#endif

}  // namespace generativeqc::cc
