#include "cc/uccsd_solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "cc/iteration_driver.hpp"
#include "generated_uccsd_cpu.hpp"
#include "posthf/capacity.hpp"
#include "solver/diis.hpp"

namespace generativeqc::cc {
namespace {

namespace graph = ucc_generated;
using Shape = std::array<std::size_t, 4>;
using Clock = std::chrono::steady_clock;
constexpr std::size_t kMaximumWork = 100000000000ULL;
constexpr std::array<const char*, 5> kAmplitudes{"t1a", "t1b", "t2aa", "t2ab", "t2bb"};

double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

Shape populations(const hf::UnrestrictedPhysicalReference& reference) {
  return {reference.nocc[0], reference.nocc[1], reference.nbf - reference.nocc[0],
          reference.nbf - reference.nocc[1]};
}

Shape feed_shape(const graph::FeedSpec& feed, const Shape& sizes) {
  Shape shape{1, 1, 1, 1};
  for (unsigned axis = 0; axis < feed.rank; ++axis) shape[axis] = sizes[feed.axes[axis]];
  return shape;
}

std::size_t elements(const Shape& shape) {
  return graph::checked_product({shape[0], shape[1], shape[2], shape[3]});
}

std::size_t offset(const Shape& indices, const Shape& shape, unsigned rank) {
  std::size_t result = 0;
  for (unsigned axis = 0; axis < rank; ++axis) result = result * shape[axis] + indices[axis];
  return result;
}

Shape coordinates(std::size_t flat, const Shape& shape, unsigned rank) {
  Shape result{};
  for (unsigned axis = rank; axis > 0; --axis) {
    result[axis - 1] = flat % shape[axis - 1];
    flat /= shape[axis - 1];
  }
  return result;
}

std::size_t falling(std::size_t count, unsigned selected) {
  std::size_t result = 1;
  for (unsigned index = 0; index < selected; ++index) {
    if (count <= index) return 0;
    result = posthf::checked_mul(result, count - index);
  }
  return result;
}

std::size_t triple_count(const Shape& sizes, unsigned alpha_count) {
  return graph::checked_product({falling(sizes[0], alpha_count), falling(sizes[1], 3 - alpha_count),
                                 falling(sizes[2], alpha_count),
                                 falling(sizes[3], 3 - alpha_count)});
}

std::size_t triples_required(const Shape& sizes) {
  return std::max({graph::required_aaa(sizes[0], sizes[1], sizes[2], sizes[3], 1),
                   graph::required_aab(sizes[0], sizes[1], sizes[2], sizes[3], 1),
                   graph::required_abb(sizes[0], sizes[1], sizes[2], sizes[3], 1),
                   graph::required_bbb(sizes[0], sizes[1], sizes[2], sizes[3], 1)});
}

struct Feed {
  Shape shape{};
  unsigned rank{};
  std::vector<double> values;
};
using Feeds = std::map<std::string, Feed>;

/** Transform real antisymmetrized physicists' blocks from the native ordered
 * chemists' provider. The spin Kronecker deltas are independent: mixed-spin
 * exchange is neither invented nor discarded by restricted symmetry. */
void integral_feed(const graph::FeedSpec& spec, Feed& feed, const Shape& sizes,
                   posthf::NativeBlockProvider& provider, posthf::ProviderWork& work) {
  if (feed.values.empty()) return;
  Shape spins{};
  posthf::MOSlots slots;
  for (unsigned axis = 0; axis < 4; ++axis) {
    spins[axis] = spec.axes[axis] % 2;
    const auto begin = spec.axes[axis] >= 2 ? sizes[spins[axis]] : 0;
    slots[axis].resize(feed.shape[axis]);
    std::iota(slots[axis].begin(), slots[axis].end(), begin);
  }
  const auto accumulate = [&](const std::array<unsigned, 4>& axes, double sign) {
    posthf::SpinMOSlots request;
    Shape request_shape{};
    for (unsigned axis = 0; axis < 4; ++axis) {
      request.slots[axis] = slots[axes[axis]];
      request.spins[axis] = static_cast<unsigned>(spins[axes[axis]]);
      request_shape[axis] = feed.shape[axes[axis]];
    }
    const auto transformed = provider.get_spin(request, &work);
    if (transformed.size() != feed.values.size())
      throw std::runtime_error("UCCSD MO provider returned an invalid block");
    for (std::size_t flat = 0; flat < feed.values.size(); ++flat) {
      const auto indices = coordinates(flat, feed.shape, 4);
      Shape ordered{};
      for (unsigned axis = 0; axis < 4; ++axis) ordered[axis] = indices[axes[axis]];
      feed.values[flat] += sign * transformed[offset(ordered, request_shape, 4)];
    }
  };
  if (spins[0] == spins[2] && spins[1] == spins[3]) accumulate({0, 2, 1, 3}, 1.0);
  if (spins[0] == spins[3] && spins[1] == spins[2]) accumulate({0, 3, 1, 2}, -1.0);
}

struct Excitation {
  unsigned block{};
  Shape indices{};
  double denominator{};
};

std::vector<Excitation> excitations(const hf::UnrestrictedPhysicalReference& reference,
                                    const Feeds& feeds, const SolverOptions& options,
                                    double& minimum) {
  std::vector<Excitation> result;
  minimum = std::numeric_limits<double>::infinity();
  for (unsigned block = 0; block < kAmplitudes.size(); ++block) {
    const auto& feed = feeds.at(kAmplitudes[block]);
    const unsigned left = block == 1 || block == 4 ? 1 : 0;
    const unsigned right = block == 3 || block == 4 ? 1 : 0;
    for (std::size_t flat = 0; flat < feed.values.size(); ++flat) {
      const auto indices = coordinates(flat, feed.shape, feed.rank);
      if ((block == 2 || block == 4) && (indices[0] >= indices[1] || indices[2] >= indices[3]))
        continue;
      double denominator = reference.orbital_energies[left][indices[0]];
      if (block < 2) {
        denominator -= reference.orbital_energies[left][reference.nocc[left] + indices[1]];
      } else {
        denominator += reference.orbital_energies[right][indices[1]];
        denominator -= reference.orbital_energies[left][reference.nocc[left] + indices[2]];
        denominator -= reference.orbital_energies[right][reference.nocc[right] + indices[3]];
      }
      if (!std::isfinite(denominator) || denominator >= -options.denominator_threshold)
        throw std::invalid_argument("UCCSD requires separated unshifted occupied/virtual energies");
      minimum = std::min(minimum, -denominator);
      denominator -= (block < 2 ? 1 : 2) * options.level_shift;
      if (!std::isfinite(denominator))
        throw std::invalid_argument("nonfinite UCCSD shifted denominator");
      result.push_back({block, indices, denominator});
    }
  }
  return result;
}

void unpack(const std::vector<double>& packed, const std::vector<Excitation>& entries,
            Feeds& feeds) {
  for (const auto* name : kAmplitudes) {
    auto& values = feeds.at(name).values;
    std::fill(values.begin(), values.end(), 0.0);
  }
  for (std::size_t slot = 0; slot < entries.size(); ++slot) {
    const auto& entry = entries[slot];
    auto& feed = feeds.at(kAmplitudes[entry.block]);
    feed.values[offset(entry.indices, feed.shape, feed.rank)] = packed[slot];
    if (entry.block != 2 && entry.block != 4) continue;
    auto indices = entry.indices;
    std::swap(indices[0], indices[1]);
    feed.values[offset(indices, feed.shape, 4)] = -packed[slot];
    std::swap(indices[2], indices[3]);
    feed.values[offset(indices, feed.shape, 4)] = packed[slot];
    std::swap(indices[0], indices[1]);
    feed.values[offset(indices, feed.shape, 4)] = -packed[slot];
  }
}

std::array<const double*, 5> residuals(const graph::Outputs& output) {
  return {output.r1a, output.r1b, output.r2aa, output.r2ab, output.r2bb};
}

IterationMetrics metrics(const graph::Outputs& output, const Feeds& feeds) {
  IterationMetrics result{*output.correlation_energy, 0.0, 0.0};
  const auto blocks = residuals(output);
  if (!std::isfinite(result.energy)) throw std::runtime_error("nonfinite UCCSD energy");
  for (unsigned block = 0; block < blocks.size(); ++block) {
    const auto size = feeds.at(kAmplitudes[block]).values.size();
    auto& maximum = block < 2 ? result.r1 : result.r2;
    for (std::size_t index = 0; index < size; ++index) {
      if (!std::isfinite(blocks[block][index]))
        throw std::runtime_error("nonfinite UCCSD residual");
      maximum = std::max(maximum, std::abs(blocks[block][index]));
    }
  }
  return result;
}

std::vector<double> packed_residual(const graph::Outputs& output,
                                    const std::vector<Excitation>& entries, const Feeds& feeds) {
  const auto blocks = residuals(output);
  std::vector<double> result(entries.size());
  for (std::size_t slot = 0; slot < entries.size(); ++slot) {
    const auto& entry = entries[slot];
    const auto& feed = feeds.at(kAmplitudes[entry.block]);
    result[slot] = blocks[entry.block][offset(entry.indices, feed.shape, feed.rank)];
  }
  return result;
}

/** A singleton tile never materializes a six-dimensional triples tensor.
 * Contracted m/e axes retain their full own-spin population; external slots
 * are gathered through compiler-owned bindings, including mixed-spin signs. */
template <class Inputs, class Bindings, class Evaluate>
double triples_channel(const std::array<unsigned, 3>& spins, const Bindings& bindings,
                       Evaluate evaluate, const hf::UnrestrictedPhysicalReference& reference,
                       const Feeds& feeds, const Shape& sizes, const SolverOptions& options,
                       std::vector<double>& arena, UnrestrictedSolve& result) {
  const unsigned alpha_count = static_cast<unsigned>(std::count(spins.begin(), spins.end(), 0));
  if (!triple_count(sizes, alpha_count)) return 0.0;
  std::array<std::size_t, 6> shape{};
  std::size_t count = 1;
  for (unsigned axis = 0; axis < 6; ++axis) {
    shape[axis] = sizes[spins[axis % 3] + (axis < 3 ? 0 : 2)];
    count = posthf::checked_mul(count, shape[axis]);
  }
  std::vector<std::vector<double>> buffers(bindings.size());
  Inputs inputs;
  double energy = 0.0;
  for (std::size_t flat = 0; flat < count; ++flat) {
    auto remainder = flat;
    std::array<std::size_t, 6> selected{};
    for (unsigned axis = 6; axis > 0; --axis) {
      selected[axis - 1] = remainder % shape[axis - 1];
      remainder /= shape[axis - 1];
    }
    bool repeated = false;
    for (unsigned first = 0; first < 3; ++first)
      for (unsigned second = first + 1; second < 3; ++second)
        if (spins[first] == spins[second] &&
            (selected[first] == selected[second] || selected[first + 3] == selected[second + 3]))
          repeated = true;
    if (repeated) continue;
    double denominator = 0.0;
    for (unsigned axis = 0; axis < 3; ++axis) {
      const auto spin = spins[axis];
      denominator += reference.orbital_energies[spin][selected[axis]];
      denominator -= reference.orbital_energies[spin][reference.nocc[spin] + selected[axis + 3]];
    }
    if (!std::isfinite(denominator) || denominator >= -options.denominator_threshold)
      throw std::invalid_argument("near-zero or nonphysical unrestricted triples denominator");
    result.minimum_denominator = std::min(result.minimum_denominator, -denominator);
    for (std::size_t binding_index = 0; binding_index < bindings.size(); ++binding_index) {
      const auto& binding = bindings[binding_index];
      auto& buffer = buffers[binding_index];
      const std::string name = binding.source;
      if (name.starts_with("eps_")) {
        const auto position = std::string("ijkabc").find(binding.labels[0]);
        const auto spin = spins[position % 3];
        buffer.resize(1);
        buffer[0] = reference.orbital_energies[spin][selected[position] +
                                                     (position < 3 ? 0 : reference.nocc[spin])];
      } else {
        const auto& feed = feeds.at(name);
        Shape tile_shape{1, 1, 1, 1};
        for (unsigned axis = 0; axis < binding.rank; ++axis) {
          const char label = binding.labels[axis];
          if (label == 'm' || label == 'e') tile_shape[axis] = feed.shape[binding.axes[axis]];
        }
        buffer.resize(elements(tile_shape));
        for (std::size_t tile_flat = 0; tile_flat < buffer.size(); ++tile_flat) {
          const auto tile_indices = coordinates(tile_flat, tile_shape, binding.rank);
          Shape source_indices{};
          for (unsigned axis = 0; axis < binding.rank; ++axis) {
            const auto external = std::string("ijkabc").find(binding.labels[axis]);
            source_indices[binding.axes[axis]] =
                external == std::string::npos ? tile_indices[axis] : selected[external];
          }
          buffer[tile_flat] = feed.values[offset(source_indices, feed.shape, feed.rank)];
        }
      }
      inputs.*(binding.field) = buffer.data();
    }
    const auto tile =
        evaluate(sizes[0], sizes[1], sizes[2], sizes[3], 1, inputs, arena.data(), arena.size());
    energy += *tile.triples_energy;
    if (!std::isfinite(energy)) throw std::runtime_error("nonfinite unrestricted triples energy");
    ++result.triples_tiles;
  }
  return energy;
}

}  // namespace

std::size_t unrestricted_capacity(std::size_t nbf, std::size_t alpha, std::size_t beta,
                                  const SolverOptions& options, bool triples,
                                  std::size_t reference_capacity) {
  if (nbf > 12 || !nbf || alpha > nbf || beta > nbf)
    throw std::invalid_argument(
        "native unrestricted CC supports valid references through twelve AOs");
  const Shape sizes{alpha, beta, nbf - alpha, nbf - beta};
  std::size_t feed_elements = 0;
  for (const auto& feed : graph::kFeeds)
    feed_elements = posthf::checked_add(feed_elements, elements(feed_shape(feed, sizes)));
  const auto arena = graph::required(sizes[0], sizes[1], sizes[2], sizes[3]);
  const auto triples_arena = triples ? triples_required(sizes) : 0;
  const auto n4 = graph::checked_product({nbf, nbf, nbf, nbf});
  // Dense feed elements bound packed excitation coordinates, without relying
  // on equality of either spin population or a particular DIIS history rank.
  const auto state =
      posthf::checked_mul(feed_elements, posthf::checked_add(12, 2 * options.diis_size));
  auto retained = posthf::checked_add(feed_elements, std::max(arena, triples_arena));
  retained = posthf::checked_add(retained, state);
  // Provider transform outputs/scratch plus every singleton tile input. The
  // latter is linear in contracted m/e extents, not in six external extents.
  retained = posthf::checked_add(
      retained, posthf::checked_add(posthf::checked_mul(8, n4), posthf::checked_mul(256, nbf)));
  auto capacity =
      posthf::checked_add(reference_capacity, posthf::checked_mul(sizeof(double), retained));
  capacity = posthf::checked_add(
      capacity, posthf::checked_mul(
                    sizeof(double),
                    graph::checked_product({options.diis_size + 1ULL, options.diis_size + 1ULL})));
  if (capacity > options.max_bytes)
    throw std::length_error("complete native UCCSD endpoint exceeds numeric memory budget");
  auto work =
      posthf::checked_mul(graph::work(sizes[0], sizes[1], sizes[2], sizes[3]),
                          posthf::checked_add(posthf::checked_mul(2, options.max_iterations), 3));
  work = posthf::checked_add(work, graph::checked_product({8, n4, nbf, graph::kFeeds.size()}));
  if (triples) {
    const std::array<std::size_t, 4> tile_work{
        graph::work_bbb(sizes[0], sizes[1], sizes[2], sizes[3], 1),
        graph::work_abb(sizes[0], sizes[1], sizes[2], sizes[3], 1),
        graph::work_aab(sizes[0], sizes[1], sizes[2], sizes[3], 1),
        graph::work_aaa(sizes[0], sizes[1], sizes[2], sizes[3], 1)};
    for (unsigned alpha_count = 0; alpha_count <= 3; ++alpha_count)
      work = posthf::checked_add(
          work, posthf::checked_mul(triple_count(sizes, alpha_count), tile_work[alpha_count]));
  }
  if (work > kMaximumWork)
    throw std::length_error("complete native UCCSD endpoint exceeds scalar-work bound");
  return capacity;
}

UnrestrictedSolve solve_unrestricted(const hf::UnrestrictedPhysicalReference& reference,
                                     const integrals::ElectronInteractionSource& source,
                                     const SolverOptions& options, bool triples,
                                     std::size_t reference_capacity) {
  UnrestrictedSolve result;
  result.numeric_capacity_bytes = unrestricted_capacity(
      reference.nbf, reference.nocc[0], reference.nocc[1], options, triples, reference_capacity);
  const auto sizes = populations(reference);
  auto started = Clock::now();
  posthf::NativeBlockProvider provider(source, reference, options.max_bytes);
  Feeds feeds;
  graph::Inputs inputs;
  for (const auto& spec : graph::kFeeds) {
    const std::string name = spec.name;
    auto& feed = feeds[name];
    feed.shape = feed_shape(spec, sizes);
    feed.rank = spec.rank;
    feed.values.resize(elements(feed.shape));
    if (name.starts_with('g')) integral_feed(spec, feed, sizes, provider, result.provider_work);
    if (name.starts_with('f')) {
      const auto spin = spec.axes[0] % 2;
      const auto& coefficients = reference.coefficients[spin];
      const auto& fock = reference.fock[spin];
      for (std::size_t row = 0; row < feed.shape[0]; ++row)
        for (std::size_t column = 0; column < feed.shape[1]; ++column) {
          const auto orbital_row = row + (spec.axes[0] >= 2 ? sizes[spin] : 0);
          const auto orbital_column = column + (spec.axes[1] >= 2 ? sizes[spin] : 0);
          double value = 0.0;
          for (std::size_t mu = 0; mu < reference.nbf; ++mu)
            for (std::size_t nu = 0; nu < reference.nbf; ++nu)
              value += coefficients[mu * reference.nbf + orbital_row] *
                       fock[mu * reference.nbf + nu] *
                       coefficients[nu * reference.nbf + orbital_column];
          feed.values[row * feed.shape[1] + column] = value;
        }
    }
    inputs.*(spec.field) = feed.values.data();
  }
  const auto entries = excitations(reference, feeds, options, result.minimum_denominator);
  if (entries.empty()) result.minimum_denominator = 0.0;
  std::vector<double> current(entries.size());
  std::vector<double> arena(graph::required(sizes[0], sizes[1], sizes[2], sizes[3]));
  solver::Diis diis(options.diis_size, current.size());
  result.problem_seconds = seconds(started);
  started = Clock::now();
  const auto evaluate = [&](bool replay = false) {
    const auto phase_started = Clock::now();
    unpack(current, entries, feeds);
    ++result.graph_calls;
    const auto output =
        graph::evaluate(sizes[0], sizes[1], sizes[2], sizes[3], inputs, arena.data(), arena.size());
    if (replay)
      result.replay_seconds += seconds(phase_started);
    else
      result.iteration_seconds += seconds(phase_started);
    return output;
  };
  run_cc_iterations<graph::Outputs>(
      options,
      [&](const auto&) {
        const auto output = evaluate();
        return std::pair{output, metrics(output, feeds)};
      },
      [&](unsigned iterations, const IterationMetrics& status, double delta) {
        result.iterations = iterations;
        result.correlation_energy = status.energy;
        result.energy_change = delta;
        result.singles_residual = status.r1;
        result.doubles_residual = status.r2;
      },
      [&] {
        const auto fresh = metrics(evaluate(true), feeds);
        ++result.replay_calls;
        result.replay_singles_residual = fresh.r1;
        result.replay_doubles_residual = fresh.r2;
        return fresh;
      },
      [&](const graph::Outputs& output) -> std::optional<graph::Outputs> {
        ++result.update_calls;
        auto phase_started = Clock::now();
        auto error = packed_residual(output, entries, feeds);
        for (std::size_t slot = 0; slot < current.size(); ++slot) {
          current[slot] += (1 - options.damping) * error[slot] / entries[slot].denominator;
          if (!std::isfinite(current[slot]))
            throw std::runtime_error("nonfinite UCCSD amplitude update");
        }
        result.update_seconds += seconds(phase_started);
        const auto trial = evaluate();
        phase_started = Clock::now();
        error = packed_residual(trial, entries, feeds);
        current = diis.update_with_status(current, std::move(error)).vector;
        result.diis_seconds += seconds(phase_started);
        return std::nullopt;
      },
      [&] { result.converged = true; },
      [](const std::runtime_error& error) { throw std::runtime_error(error.what()); });
  result.solver_seconds = seconds(started);
  if (!result.converged || !triples) return result;
  // The converged replay leaves feeds bound to exactly the admitted parent
  // excitation state. Retire DIIS and CC arena before allocating triples work.
  diis.clear();
  std::vector<double>().swap(current);
  std::vector<double>().swap(arena);
  arena.resize(triples_required(sizes));
  result.triples_workspace_bytes = posthf::checked_mul(
      sizeof(double), posthf::checked_add(arena.size(), posthf::checked_mul(256, reference.nbf)));
  started = Clock::now();
  result.triples_energy +=
      triples_channel<graph::Inputs_aaa>({0, 0, 0}, graph::kBindings_aaa, graph::evaluate_aaa,
                                         reference, feeds, sizes, options, arena, result);
  result.triples_energy +=
      triples_channel<graph::Inputs_aab>({0, 0, 1}, graph::kBindings_aab, graph::evaluate_aab,
                                         reference, feeds, sizes, options, arena, result);
  result.triples_energy +=
      triples_channel<graph::Inputs_abb>({0, 1, 1}, graph::kBindings_abb, graph::evaluate_abb,
                                         reference, feeds, sizes, options, arena, result);
  result.triples_energy +=
      triples_channel<graph::Inputs_bbb>({1, 1, 1}, graph::kBindings_bbb, graph::evaluate_bbb,
                                         reference, feeds, sizes, options, arena, result);
  result.triples_seconds = seconds(started);
  return result;
}

const char* unrestricted_equation_hash() noexcept { return graph::kEquationHash; }
const char* unrestricted_triples_hash() noexcept { return graph::kTriplesHash; }

}  // namespace generativeqc::cc
