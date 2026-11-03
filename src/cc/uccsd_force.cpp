#include "cc/uccsd_force.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>

#include "cc/uccsd_executor.hpp"
#include "generated_uccsd_response.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "molecule/nuclear_gradient.hpp"
#include "posthf/capacity.hpp"
#include "posthf/cuda_derivative.hpp"
#include "posthf/df_interaction_source.hpp"
#include "posthf/native_provider.hpp"
#include "posthf/ump2_energy.hpp"
#include "response/solve.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::cc {
namespace {
namespace graph = ucc_generated;
using Values = graph::ResponseFeeds;
using Shape = std::array<std::size_t, 4>;
constexpr std::array<const char*, 5> amplitudes{"t1a", "t1b", "t2aa", "t2ab", "t2bb"};
constexpr std::array<const char*, 5> residuals{"r1a", "r1b", "r2aa", "r2ab", "r2bb"};
constexpr std::size_t maximum_work = 100000000000ULL;

std::size_t add(std::size_t left, std::size_t right) { return posthf::checked_add(left, right); }
std::size_t mul(std::size_t left, std::size_t right) { return posthf::checked_mul(left, right); }
bool finite(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}
Shape coordinates(std::size_t flat, const Shape& shape, unsigned rank) {
  Shape result{};
  for (unsigned axis = rank; axis > 0; --axis) {
    result[axis - 1] = flat % shape[axis - 1];
    flat /= shape[axis - 1];
  }
  return result;
}
std::size_t offset(const Shape& indices, const Shape& shape, unsigned rank) {
  std::size_t result = 0;
  for (unsigned axis = 0; axis < rank; ++axis) result = result * shape[axis] + indices[axis];
  return result;
}
Shape shape(const graph::FeedSpec& spec, const Shape& sizes) {
  Shape result{1, 1, 1, 1};
  for (unsigned axis = 0; axis < spec.rank; ++axis) result[axis] = sizes[spec.axes[axis]];
  return result;
}

/** One bounded arena, with detached outputs before its next use. Input shape
 * checks and semantic work admission precede every generated program action. */
class Runtime {
 public:
  Runtime(std::size_t nbf, std::size_t alpha, std::size_t beta)
      : sizes{alpha, beta, nbf - alpha, nbf - beta}, nbf(nbf) {}

  graph::ResponseSizes input_sizes(const std::string& program) const {
    return graph::response_inputs(program, sizes[0], sizes[1], sizes[2], sizes[3], nbf, nbf, 1);
  }
  graph::ResponseSizes output_sizes(const std::string& program) const {
    return graph::response_outputs(program, sizes[0], sizes[1], sizes[2], sizes[3], nbf, nbf, 1);
  }
  std::size_t required(const std::string& program) const {
    return graph::response_required(program, sizes[0], sizes[1], sizes[2], sizes[3], nbf, nbf, 1);
  }
  std::size_t work(const std::string& program) const {
    return graph::response_work(program, sizes[0], sizes[1], sizes[2], sizes[3], nbf, nbf, 1);
  }
  Values zero_inputs(const std::string& program) const {
    Values result;
    for (const auto& [name, count] : input_sizes(program)) result[name].resize(count);
    return result;
  }
  Values run(const std::string& program, const Values& inputs) {
    executed_work = add(executed_work, work(program));
    if (executed_work > maximum_work)
      throw std::length_error("native unrestricted response exceeds scalar-work bound");
    for (const auto& [name, count] : input_sizes(program)) {
      const auto found = inputs.find(name);
      if (found == inputs.end() || found->second.size() != count || !finite(found->second))
        throw std::invalid_argument("invalid native unrestricted response input: " + name);
    }
    if (executor) return executor->evaluate(program, inputs);
    arena.resize(required(program));
    const auto views = graph::response_evaluate(program, sizes[0], sizes[1], sizes[2], sizes[3],
                                                nbf, nbf, 1, inputs, arena.data(), arena.size());
    Values result;
    for (const auto& [name, count] : output_sizes(program)) {
      auto& values = result[name];
      if (count) values.assign(views.at(name), views.at(name) + count);
      if (!finite(values)) throw std::runtime_error("nonfinite unrestricted response output");
    }
    return result;
  }

  Shape sizes;
  std::size_t nbf, executed_work{};
  std::vector<double> arena;
  bool cuda{};
  int device{};
  std::unique_ptr<UnrestrictedCudaExecutor> executor;
};

struct Excitation {
  unsigned block{}, rank{};
  Shape indices{}, shape{};
};

std::vector<Excitation> excitations(const Runtime& runtime) {
  std::vector<Excitation> result;
  for (unsigned block = 0; block < amplitudes.size(); ++block) {
    const auto spec =
        std::find_if(graph::kFeeds.begin(), graph::kFeeds.end(),
                     [&](const auto& item) { return std::string(item.name) == amplitudes[block]; });
    const auto dimensions = shape(*spec, runtime.sizes);
    const auto count =
        graph::checked_product({dimensions[0], dimensions[1], dimensions[2], dimensions[3]});
    for (std::size_t flat = 0; flat < count; ++flat) {
      const auto indices = coordinates(flat, dimensions, spec->rank);
      if ((block == 2 || block == 4) && (indices[0] >= indices[1] || indices[2] >= indices[3]))
        continue;
      result.push_back({block, spec->rank, indices, dimensions});
    }
  }
  return result;
}

/** Pull back with U*, not by choosing one same-spin tensor entry. The dual
 * residual lift below includes its distinct 1/4 Frobenius multiplicity. */
std::vector<double> pullback(const Values& bars, const std::vector<Excitation>& entries) {
  std::vector<double> result(entries.size());
  for (std::size_t slot = 0; slot < entries.size(); ++slot) {
    const auto& entry = entries[slot];
    const auto& values = bars.at("bar_" + std::string(amplitudes[entry.block]));
    auto indices = entry.indices;
    result[slot] = values[offset(indices, entry.shape, entry.rank)];
    if (entry.block != 2 && entry.block != 4) continue;
    std::swap(indices[0], indices[1]);
    result[slot] -= values[offset(indices, entry.shape, 4)];
    std::swap(indices[2], indices[3]);
    result[slot] += values[offset(indices, entry.shape, 4)];
    std::swap(indices[0], indices[1]);
    result[slot] -= values[offset(indices, entry.shape, 4)];
  }
  return result;
}
void lift(Values& inputs, std::span<const double> packed, const std::vector<Excitation>& entries) {
  for (const auto* name : residuals) {
    auto& values = inputs.at("bar_" + std::string(name));
    std::fill(values.begin(), values.end(), 0.0);
  }
  for (std::size_t slot = 0; slot < entries.size(); ++slot) {
    const auto& entry = entries[slot];
    auto& values = inputs.at("bar_" + std::string(residuals[entry.block]));
    auto indices = entry.indices;
    const double value = packed[slot] / ((entry.block == 2 || entry.block == 4) ? 4.0 : 1.0);
    values[offset(indices, entry.shape, entry.rank)] = value;
    if (entry.block != 2 && entry.block != 4) continue;
    std::swap(indices[0], indices[1]);
    values[offset(indices, entry.shape, 4)] = -value;
    std::swap(indices[2], indices[3]);
    values[offset(indices, entry.shape, 4)] = value;
    std::swap(indices[0], indices[1]);
    values[offset(indices, entry.shape, 4)] = -value;
  }
}
void accumulate(Values& destination, const Values& source) {
  for (const auto& [name, values] : source) {
    auto& target = destination.at(name);
    if (target.size() != values.size())
      throw std::invalid_argument("unrestricted cotangent shape mismatch");
    for (std::size_t item = 0; item < values.size(); ++item) target[item] += values[item];
    if (!finite(target)) throw std::runtime_error("nonfinite unrestricted accumulated cotangent");
  }
}

Values raw_inputs(Runtime& runtime, const hf::UnrestrictedPhysicalReference& reference,
                  const integrals::ElectronInteractionSource& source, std::size_t budget,
                  const integrals::ElectronInteractionSource* independent_reference) {
  auto inputs = runtime.zero_inputs(independent_reference ? "raw_separate" : "raw");
  std::vector<double> scratch(mul(runtime.nbf, runtime.nbf));
  for (unsigned spin = 0; spin < 2; ++spin) {
    const std::string suffix = spin ? "b" : "a";
    tensor::cpu_congruence('T', runtime.nbf, reference.coefficients[spin].data(),
                           reference.hcore.data(), inputs.at("h_" + suffix).data(), scratch.data());
    auto& rotation = inputs.at("rotation_" + suffix);
    for (std::size_t orbital = 0; orbital < runtime.nbf; ++orbital)
      rotation[orbital * runtime.nbf + orbital] = 1.0;
  }
  posthf::NativeBlockProvider provider(
      source, reference, budget, runtime.cuda ? 8 : 2,
      runtime.cuda ? posthf::AOTileDomain::Basis : posthf::AOTileDomain::Shell);
  std::vector<std::size_t> all(runtime.nbf);
  std::iota(all.begin(), all.end(), 0);
  for (const auto& pair : std::array<std::string, 3>{"aa", "ab", "bb"}) {
    const auto left = pair[0] == 'a' ? 0U : 1U, right = pair[1] == 'a' ? 0U : 1U;
    inputs.at("g_" + pair) = provider.get_spin({{all, all, all, all}, {left, left, right, right}},
                                               nullptr, runtime.cuda, runtime.device);
  }
  if (independent_reference) {
    posthf::NativeBlockProvider reference_provider(
        *independent_reference, reference, budget, runtime.cuda ? 8 : 2,
        runtime.cuda ? posthf::AOTileDomain::Basis : posthf::AOTileDomain::Shell);
    for (const auto& pair : std::array<std::string, 3>{"aa", "ab", "bb"}) {
      const auto left = pair[0] == 'a' ? 0U : 1U, right = pair[1] == 'a' ? 0U : 1U;
      inputs.at("reference_g_" + pair) =
          reference_provider.get_spin({{all, all, all, all}, {left, left, right, right}}, nullptr,
                                      runtime.cuda, runtime.device);
    }
  }
  return inputs;
}

template <class Bindings, class Consume>
Values tile_inputs(const Bindings& bindings, const std::array<std::size_t, 6>& selected,
                   const std::array<unsigned, 3>& spins, const Runtime& runtime,
                   const hf::UnrestrictedPhysicalReference& reference, const Values& feeds,
                   Consume consume) {
  Values result;
  for (const auto& binding : bindings) {
    auto& buffer = result[binding.name];
    if (std::string(binding.source).starts_with("eps_")) {
      const auto position = std::string("ijkabc").find(binding.labels[0]);
      const auto spin = spins[position % 3];
      buffer = {reference.orbital_energies[spin][selected[position] +
                                                 (position < 3 ? 0 : reference.nocc[spin])]};
      continue;
    }
    const auto spec =
        std::find_if(graph::kFeeds.begin(), graph::kFeeds.end(),
                     [&](const auto& item) { return std::string(item.name) == binding.source; });
    if (spec == graph::kFeeds.end())
      throw std::invalid_argument("unknown unrestricted triples source");
    const auto dimensions = shape(*spec, runtime.sizes);
    Shape tile_shape{1, 1, 1, 1};
    for (unsigned axis = 0; axis < binding.rank; ++axis)
      if (binding.labels[axis] == 'm' || binding.labels[axis] == 'e')
        tile_shape[axis] = dimensions[binding.axes[axis]];
    buffer.resize(
        graph::checked_product({tile_shape[0], tile_shape[1], tile_shape[2], tile_shape[3]}));
    for (std::size_t flat = 0; flat < buffer.size(); ++flat) {
      const auto indices = coordinates(flat, tile_shape, binding.rank);
      Shape source_indices{};
      for (unsigned axis = 0; axis < binding.rank; ++axis) {
        const auto external = std::string("ijkabc").find(binding.labels[axis]);
        source_indices[binding.axes[axis]] =
            external == std::string::npos ? indices[axis] : selected[external];
      }
      const auto source_offset = offset(source_indices, dimensions, binding.rank);
      buffer[flat] = feeds.at(binding.source)[source_offset];
      consume(binding, flat, source_offset);
    }
  }
  return result;
}

bool repeated(const std::array<std::size_t, 6>& selected, const std::array<unsigned, 3>& spins) {
  for (unsigned first = 0; first < 3; ++first)
    for (unsigned second = first + 1; second < 3; ++second)
      if (spins[first] == spins[second] &&
          (selected[first] == selected[second] || selected[first + 3] == selected[second + 3]))
        return true;
  return false;
}

/** Bounded singleton reverse tiles plus paired X/Y pages. Off-diagonal Fock
 * bars are generated at exactly the same parent amplitudes as diagonal bars. */
template <class Bindings>
double triples_response(const std::string& channel, const std::array<unsigned, 3>& spins,
                        const Bindings& bindings, Runtime& runtime,
                        const hf::UnrestrictedPhysicalReference& reference, const Values& feeds,
                        Values& parameter_seeds, Values& amplitude_bars) {
  std::array<std::size_t, 6> dimensions{};
  std::size_t count = 1;
  for (unsigned axis = 0; axis < 6; ++axis) {
    dimensions[axis] = runtime.sizes[spins[axis % 3] + (axis < 3 ? 0 : 2)];
    count = mul(count, dimensions[axis]);
  }
  const auto alpha = static_cast<unsigned>(std::count(spins.begin(), spins.end(), 0));
  if (runtime.sizes[0] < alpha || runtime.sizes[1] < 3 - alpha || runtime.sizes[2] < alpha ||
      runtime.sizes[3] < 3 - alpha)
    return 0.0;
  const auto select = [&](std::size_t flat, const auto& extents) {
    std::array<std::size_t, 6> selected{};
    for (unsigned axis = 6; axis > 0; --axis) {
      selected[axis - 1] = flat % extents[axis - 1];
      flat /= extents[axis - 1];
    }
    return selected;
  };
  const auto gather = [&](const auto& selected) {
    return tile_inputs(bindings, selected, spins, runtime, reference, feeds,
                       [](const auto&, auto, auto) {});
  };
  double energy = 0.0;
  for (std::size_t flat = 0; flat < count; ++flat) {
    const auto selected = select(flat, dimensions);
    if (repeated(selected, spins)) continue;
    auto inputs = gather(selected);
    energy += runtime.run("vectors_" + channel, inputs).at("triples_energy")[0];
    inputs["bar_triples_energy"] = {1.0};
    const auto bars = runtime.run("bars_" + channel, inputs);
    tile_inputs(bindings, selected, spins, runtime, reference, feeds,
                [&](const auto& binding, auto tile_offset, auto source_offset) {
                  const auto& source = std::string(binding.source);
                  const auto value = bars.at("bar_" + std::string(binding.name))[tile_offset];
                  auto& target = source.starts_with('t') ? amplitude_bars.at("bar_" + source)
                                                         : parameter_seeds.at("bar_" + source);
                  target[source_offset] += value;
                });
  }
  for (unsigned axis = 0; axis < 6; ++axis) {
    auto extents = dimensions;
    extents[axis] = 1;
    const auto others = count / dimensions[axis];
    std::vector<double> numerators(dimensions[axis]), resolvents(dimensions[axis]);
    const auto program = "moment_" + channel + "_" + std::to_string(axis);
    const auto spin = spins[axis % 3];
    const std::string name = std::string(axis < 3 ? "foo_" : "fvv_") + (spin ? "bb" : "aa");
    auto& target = parameter_seeds.at("bar_" + name);
    for (std::size_t flat = 0; flat < others; ++flat) {
      auto selected = select(flat, extents);
      for (std::size_t orbital = 0; orbital < dimensions[axis]; ++orbital) {
        selected[axis] = orbital;
        numerators[orbital] = resolvents[orbital] = 0.0;
        if (repeated(selected, spins)) continue;
        const auto vectors = runtime.run("vectors_" + channel, gather(selected));
        numerators[orbital] = vectors.at("x")[0];
        resolvents[orbital] = vectors.at("y")[0];
      }
      for (std::size_t left = 0; left < dimensions[axis]; ++left)
        for (std::size_t right = left; right < dimensions[axis]; ++right) {
          const auto moment = runtime.run(program, {{"x_left", {numerators[left]}},
                                                    {"y_left", {resolvents[left]}},
                                                    {"x_right", {numerators[right]}},
                                                    {"y_right", {resolvents[right]}}});
          const auto value = moment.at(name)[0];
          target[left * dimensions[axis] + right] += value;
          if (left != right) target[right * dimensions[axis] + left] += value;
        }
    }
  }
  return energy;
}

/** Four O(N^5) staged transforms keep each spin's physical frame on its own
 * chemists' slots. The AO weight is contracted shell-locally, not differentiated
 * by materializing coordinate-major ERI derivative tensors. */
std::vector<double> derivative(const hf::UnrestrictedPhysicalReference& reference,
                               const core::System& system, const Values& weights, bool cuda,
                               int device, std::size_t budget,
                               const posthf::DensityFittedInteractionSource* fitted,
                               bool separate_reference,
                               const integrals::CorrelationIntegralBackend& backend) {
  const auto nbf = reference.nbf, n2 = mul(nbf, nbf), n4 = mul(n2, n2);
  std::vector<double> one(n2), overlap(n2), temporary(n2), scratch(n2);
  for (unsigned spin = 0; spin < 2; ++spin) {
    const std::string suffix = spin ? "b" : "a";
    for (const auto& item : std::array<std::string, 2>{"h_", "overlap_"}) {
      tensor::cpu_congruence('N', nbf, reference.coefficients[spin].data(),
                             weights.at(item + suffix).data(), temporary.data(), scratch.data());
      auto& target = item == "h_" ? one : overlap;
      for (std::size_t flat = 0; flat < n2; ++flat) target[flat] += temporary[flat];
    }
  }
  std::vector<double> gradient;
  if (cuda) {
#if GENERATIVEQC_HAS_CUDA
    if (!backend.one_electron_gradient)
      throw std::invalid_argument("unrestricted CUDA forces require an explicit integral backend");
    gradient = backend.one_electron_gradient(device, system, overlap, one, budget);
    molecule::add_nuclear_repulsion_gradient(system, gradient);
#else
    throw std::invalid_argument("unrestricted CUDA force requires a CUDA build");
#endif
  } else
    gradient = integrals::contract_weighted_one_electron_derivative(system, overlap, one, true);
#if GENERATIVEQC_HAS_CUDA
  std::unique_ptr<posthf::CudaEriDerivativeBatch> cuda_derivative;
  if (cuda)
    cuda_derivative = std::make_unique<posthf::CudaEriDerivativeBatch>(device, system, budget);
#endif
  std::vector<double> ao(n4), first(n4), second(n4);
  const auto transform = [&](const std::string& prefix) {
    std::fill(ao.begin(), ao.end(), 0.0);
    for (const auto& pair : std::array<std::string, 3>{"aa", "ab", "bb"}) {
      first = weights.at(prefix + pair);
      for (unsigned axis = 0; axis < 4; ++axis) {
        const auto spin = pair[axis / 2] == 'a' ? 0U : 1U;
        const auto& coefficients = reference.coefficients[spin];
        const auto rest = n4 / nbf;
        for (std::size_t flat = 0; flat < rest; ++flat)
          for (std::size_t row = 0; row < nbf; ++row) {
            double value = 0.0;
            for (std::size_t orbital = 0; orbital < nbf; ++orbital)
              value += coefficients[row * nbf + orbital] * first[orbital * rest + flat];
            second[flat * nbf + row] = value;
          }
        first.swap(second);
      }
      for (std::size_t flat = 0; flat < n4; ++flat) ao[flat] += first[flat];
    }
  };
  transform("g_");
  if (fitted) {
    const auto fitted_gradient = fitted->nuclear_derivative(ao);
    for (std::size_t coordinate = 0; coordinate < gradient.size(); ++coordinate)
      gradient[coordinate] += fitted_gradient.at(coordinate);
    if (separate_reference)
      transform("reference_g_");
    else
      std::fill(ao.begin(), ao.end(), 0.0);
  }
  std::vector<std::size_t> offsets{0};
  for (const auto& shell : system.shells)
    offsets.push_back(
        add(offsets.back(), system.basis_representation == GENERATIVEQC_BASIS_SPHERICAL
                                ? 2 * shell.angular_momentum + 1
                                : molecule::cartesian_count(shell.angular_momentum)));
  if (offsets.back() != nbf)
    throw std::invalid_argument("unrestricted derivative shell/frame mismatch");
  const auto shells = fitted && !separate_reference ? 0 : system.shells.size();
  for (std::size_t first_shell = 0; first_shell < shells; ++first_shell)
    for (std::size_t second_shell = 0; second_shell < shells; ++second_shell)
      for (std::size_t third_shell = 0; third_shell < shells; ++third_shell)
        for (std::size_t fourth_shell = 0; fourth_shell < shells; ++fourth_shell) {
          const std::array<std::size_t, 4> selected{first_shell, second_shell, third_shell,
                                                    fourth_shell};
          Shape extents{};
          for (unsigned axis = 0; axis < 4; ++axis)
            extents[axis] = offsets[selected[axis] + 1] - offsets[selected[axis]];
          std::vector<double> tile(
              graph::checked_product({extents[0], extents[1], extents[2], extents[3]}));
          for (std::size_t flat = 0; flat < tile.size(); ++flat) {
            auto indices = coordinates(flat, extents, 4);
            for (unsigned axis = 0; axis < 4; ++axis) indices[axis] += offsets[selected[axis]];
            tile[flat] =
                ao[((indices[0] * nbf + indices[1]) * nbf + indices[2]) * nbf + indices[3]];
          }
          if (cuda) {
#if GENERATIVEQC_HAS_CUDA
            cuda_derivative->append(selected, tile, gradient);
            continue;
#endif
          }
          const auto values =
              integrals::contract_weighted_eri_shell_derivative(system, selected, tile);
          for (unsigned slot = 0; slot < 4; ++slot)
            for (unsigned axis = 0; axis < 3; ++axis)
              gradient[3 * system.shells[selected[slot]].atom_index + axis] +=
                  values[3 * slot + axis];
        }
#if GENERATIVEQC_HAS_CUDA
  if (cuda_derivative) cuda_derivative->finish(gradient);
#endif
  for (auto& value : gradient) value = -value;
  if (!finite(gradient)) throw std::runtime_error("nonfinite unrestricted analytic force");
  return gradient;
}

}  // namespace

std::size_t unrestricted_force_capacity(std::size_t nbf, std::size_t alpha, std::size_t beta,
                                        const SolverOptions& options, bool triples,
                                        std::size_t reference_capacity, bool separate_reference,
                                        std::size_t derivative_capacity) {
  const auto primal = unrestricted_capacity(nbf, alpha, beta, options, triples, reference_capacity);
  Runtime runtime(nbf, alpha, beta);
  const std::string suffix = separate_reference ? "_separate" : "";
  std::vector<std::string> programs{"lambda", "raw" + suffix, "physical" + suffix,
                                    "correction" + suffix};
  if (triples)
    for (const auto& channel : std::array<std::string, 4>{"aaa", "aab", "abb", "bbb"}) {
      programs.push_back("vectors_" + channel);
      programs.push_back("bars_" + channel);
      for (unsigned axis = 0; axis < 6; ++axis)
        programs.push_back("moment_" + channel + "_" + std::to_string(axis));
    }
  std::size_t arena = 0, retained = 0;
  for (const auto& program : programs) {
    arena = std::max(arena, runtime.required(program));
    for (const auto& [name, count] : runtime.input_sizes(program)) retained = add(retained, count);
    for (const auto& [name, count] : runtime.output_sizes(program)) retained = add(retained, count);
  }
  // Bound all detached frames, canonical seeds, map copies, derivative buffers
  // and callback results simultaneously, including overallocated arena capacity.
  auto count =
      add(mul(4, retained), add(mul(2, arena), graph::checked_product({16, nbf, nbf, nbf, nbf})));
  const auto pairs = [](std::size_t count) { return count < 2 ? 0 : mul(count, count - 1) / 2; };
  const auto dimension =
      add(add(mul(alpha, nbf - alpha), mul(beta, nbf - beta)),
          add(add(mul(pairs(alpha), pairs(nbf - alpha)), mul(pairs(beta), pairs(nbf - beta))),
              graph::checked_product({alpha, beta, nbf - alpha, nbf - beta})));
  const auto orbital_dimension = add(mul(alpha, nbf - alpha), mul(beta, nbf - beta));
  response::GmresOptions controls;
  controls.absolute_tolerance = 1e-11;
  auto peak = add(primal, add(mul(count, sizeof(double)),
                              std::max(derivative_capacity, posthf::source_scratch_bytes)));
  if (dimension) peak = add(peak, response::prepare_gmres(dimension, controls).workspace_bytes);
  if (orbital_dimension)
    peak = add(peak, response::prepare_gmres(orbital_dimension, controls).workspace_bytes);
  if (peak > options.max_bytes)
    throw std::length_error(
        "complete native unrestricted force endpoint exceeds numeric memory budget");
  return peak;
}

UnrestrictedForce unrestricted_force(
    const hf::UnrestrictedPhysicalReference& reference,
    const integrals::ElectronInteractionSource& source, const UnrestrictedSolve& parent,
    const SolverOptions& options, bool triples, std::size_t reference_capacity,
    const response::LinearResponseProblem* orbital_problem, bool cuda, int device,
    const posthf::DensityFittedInteractionSource* fitted,
    const integrals::ElectronInteractionSource* independent_reference,
    const integrals::CorrelationIntegralBackend& backend) {
  const std::string suffix = independent_reference ? "_separate" : "";
  const auto derivative_capacity = cuda ? posthf::one_electron_response_capacity(source.orbital())
                                        : posthf::source_scratch_bytes;
  UnrestrictedForce result;
  result.numeric_capacity_bytes = unrestricted_force_capacity(
      reference.nbf, reference.nocc[0], reference.nocc[1], options, triples, reference_capacity,
      independent_reference != nullptr, derivative_capacity);
  mp2::validate_unrestricted_energy_input(reference, source, options.denominator_threshold);
  if (!parent.converged || reference.commutator_residual > 1e-8 ||
      std::max(parent.replay_singles_residual, parent.replay_doubles_residual) >
          options.residual_tolerance)
    throw std::invalid_argument("unrestricted force requires a freshly qualified physical parent");
  Runtime runtime(reference.nbf, reference.nocc[0], reference.nocc[1]);
  runtime.cuda = cuda;
  runtime.device = device;
  if (cuda) {
#if GENERATIVEQC_HAS_CUDA
    runtime.executor =
        make_unrestricted_cuda_executor(reference.nbf, reference.nocc[0], reference.nocc[1], device,
                                        options.max_bytes - result.numeric_capacity_bytes, true);
    posthf::NativeBlockProvider provider(source, reference, options.max_bytes, 8,
                                         posthf::AOTileDomain::Basis);
    result.numeric_capacity_bytes =
        add(result.numeric_capacity_bytes,
            add(runtime.executor->numeric_capacity_bytes(),
                provider.batch_bytes({reference.nbf, reference.nbf, reference.nbf, reference.nbf},
                                     1, true)));
    if (result.numeric_capacity_bytes > options.max_bytes)
      throw std::length_error("complete unrestricted CUDA force exceeds numeric memory budget");
#else
    throw std::invalid_argument("unrestricted CUDA force requires a CUDA build");
#endif
  }
  auto raw = raw_inputs(runtime, reference, source, options.max_bytes, independent_reference);
  auto fields = runtime.run("raw" + suffix, raw);
  const auto n2 = mul(reference.nbf, reference.nbf);
  double electronic = 0.0;
  for (unsigned spin = 0; spin < 2; ++spin)
    for (std::size_t flat = 0; flat < n2; ++flat)
      electronic += 0.5 * reference.density[spin][flat] *
                    (reference.hcore[flat] + reference.fock[spin][flat]);
  if (std::abs(fields.at("reference_electronic_energy")[0] - electronic) > 1e-8)
    throw std::runtime_error("unrestricted response raw-Hamiltonian/reference mismatch");
  for (unsigned block = 0; block < amplitudes.size(); ++block)
    fields[amplitudes[block]] = parent.amplitudes[block];
  const auto replay = runtime.run("cc", fields);
  double residual = 0;
  for (const auto* name : residuals)
    for (double value : replay.at(name)) residual = std::max(residual, std::abs(value));
  if (std::abs(replay.at("correlation_energy")[0] - parent.correlation_energy) > 1e-10 ||
      residual > options.residual_tolerance)
    throw std::runtime_error("unrestricted response failed fresh physical CC replay");
  auto lambda_inputs = runtime.zero_inputs("lambda");
  for (auto& [name, values] : lambda_inputs)
    if (const auto found = fields.find(name); found != fields.end()) values = found->second;
  lambda_inputs.at("bar_correlation_energy")[0] = 1.0;
  auto parameter_seeds = runtime.zero_inputs("correction" + suffix);
  for (auto& [name, values] : parameter_seeds)
    if (const auto found = raw.find(name); found != raw.end()) values = found->second;
  Values amplitude_bars;
  for (unsigned block = 0; block < amplitudes.size(); ++block)
    amplitude_bars["bar_" + std::string(amplitudes[block])].resize(parent.amplitudes[block].size());
  if (triples) {
    double energy = 0;
    energy += triples_response("aaa", {0, 0, 0}, graph::kBindings_aaa, runtime, reference, fields,
                               parameter_seeds, amplitude_bars);
    energy += triples_response("aab", {0, 0, 1}, graph::kBindings_aab, runtime, reference, fields,
                               parameter_seeds, amplitude_bars);
    energy += triples_response("abb", {0, 1, 1}, graph::kBindings_abb, runtime, reference, fields,
                               parameter_seeds, amplitude_bars);
    energy += triples_response("bbb", {1, 1, 1}, graph::kBindings_bbb, runtime, reference, fields,
                               parameter_seeds, amplitude_bars);
    if (!std::isfinite(energy) || std::abs(energy - parent.triples_energy) > 1e-10)
      throw std::runtime_error("unrestricted triples response/energy replay mismatch");
  }
  auto bars = runtime.run("lambda", lambda_inputs);
  accumulate(bars, amplitude_bars);
  const auto entries = excitations(runtime);
  auto rhs = pullback(bars, entries);
  for (auto& value : rhs) value = -value;
  response::GmresOptions controls;
  controls.absolute_tolerance = 1e-11;
  controls.relative_tolerance = 1e-11;
  controls.max_iterations = 300;
  const auto action = [&](std::span<const double> input, std::span<double> output) {
    lambda_inputs.at("bar_correlation_energy")[0] = 0.0;
    lift(lambda_inputs, input, entries);
    const auto values = pullback(runtime.run("lambda", lambda_inputs), entries);
    std::copy(values.begin(), values.end(), output.begin());
  };
  result.lambda.status = response::GmresStatus::initial_residual;
  if (!entries.empty()) {
    result.lambda =
        response::solve_gmres(response::prepare_gmres(entries.size(), controls), action, rhs);
    if (!result.lambda.converged())
      throw std::runtime_error("unrestricted corrected Lambda did not converge");
  }
  lambda_inputs.at("bar_correlation_energy")[0] = 1.0;
  lift(lambda_inputs, result.lambda.solution, entries);
  bars = runtime.run("lambda", lambda_inputs);
  accumulate(bars, amplitude_bars);
  if (response::stable_norm(pullback(bars, entries)) > 1e-9)
    throw std::runtime_error("fresh unrestricted Lagrangian amplitude stationarity failed");
  auto physical = runtime.zero_inputs("physical" + suffix);
  for (auto& [name, values] : physical) {
    if (const auto found = raw.find(name); found != raw.end()) values = found->second;
    if (const auto found = lambda_inputs.find(name); found != lambda_inputs.end())
      values = found->second;
  }
  const auto correction = runtime.run("correction" + suffix, parameter_seeds);
  auto weights = runtime.run("physical" + suffix, physical);
  accumulate(weights, correction);
  std::vector<double> orbital_rhs, diagonal;
  for (unsigned spin = 0; spin < 2; ++spin) {
    const auto& values = weights.at(spin ? "orbital_rhs_b" : "orbital_rhs_a");
    orbital_rhs.insert(orbital_rhs.end(), values.begin(), values.end());
    for (std::size_t occupied = 0; occupied < reference.nocc[spin]; ++occupied)
      for (std::size_t virtual_orbital = reference.nocc[spin]; virtual_orbital < reference.nbf;
           ++virtual_orbital)
        diagonal.push_back(reference.orbital_energies[spin][virtual_orbital] -
                           reference.orbital_energies[spin][occupied]);
  }
  result.orbital.status = response::GmresStatus::initial_residual;
  if (!orbital_rhs.empty()) {
    if (!orbital_problem || orbital_problem->dimension() != orbital_rhs.size())
      throw std::invalid_argument("unrestricted force orbital owner/dimension mismatch");
    const auto plan = response::prepare_response(*orbital_problem, controls);
    result.orbital = response::solve_response(plan, *orbital_problem, orbital_rhs, {}, diagonal);
    if (!result.orbital.converged())
      throw std::runtime_error("unrestricted UHF adjoint did not converge");
  }
  physical.at("bar_reference_electronic_energy")[0] = 1.0;
  std::size_t position = 0;
  for (const auto* name : {"bar_fov_aa", "bar_fov_bb"})
    for (auto& value : physical.at(name)) value = result.orbital.solution[position++];
  weights = runtime.run("physical" + suffix, physical);
  accumulate(weights, correction);
  for (const auto* name : {"stationarity_a", "stationarity_b"})
    for (double value : weights.at(name))
      result.stationarity_residual = std::max(result.stationarity_residual, std::abs(value));
  if (!std::isfinite(result.stationarity_residual) || result.stationarity_residual > 1e-7)
    throw std::runtime_error("fresh unrestricted full orbital stationarity failed");
  result.forces =
      derivative(reference, source.orbital(), weights, cuda, device, derivative_capacity, fitted,
                 independent_reference != nullptr, backend);
  return result;
}

}  // namespace generativeqc::cc
