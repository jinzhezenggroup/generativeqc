#include "dft/ao_grid.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <stdexcept>

#include "dft/generated_ao_axis.hpp"
#include "molecule/basis.hpp"

namespace generativeqc::dft {
namespace {
std::size_t multiply(std::size_t a, std::size_t b) {
  if (b && a > SIZE_MAX / b) throw std::invalid_argument("AO tile size overflow");
  return a * b;
}

// Derivative zero keeps its direct Horner path. The bounded dispatch has no
// method, molecule, basis-name, grid-size, cache, or backend-selection policy.
double differentiated_power(unsigned l, unsigned derivative, double alpha, double x) {
  if (derivative == 0) {
    // The initial polynomial is x^l. Keep the original Horner operations,
    // including addition of zero, without constructing a coefficient array.
    double result = 1;
    while (l) {
      result = result * x + 0.0;
      --l;
    }
    return result;
  }
  return generated_ao::axis_nonzero(l, derivative, alpha, x);
}
}  // namespace

AoBasis::AoBasis(const core::System& system) {
  natom = system.atoms.size();
  nao = molecule::ao_count(system);
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > 3) throw std::invalid_argument("AO grids support through f");
    nprimitive += shell.primitives.size();
  }
  if (!natom || !nao || natom > INT_MAX || nao > INT_MAX || nprimitive > INT_MAX)
    throw std::invalid_argument("invalid AO grid basis dimensions");
  packed.resize(3 * natom + 2 * nprimitive + 16 * nao, 0);
  for (std::size_t a = 0; a < natom; ++a)
    std::copy(system.atoms[a].position.begin(), system.atoms[a].position.end(),
              packed.begin() + 3 * a);
  auto* primitives = packed.data() + 3 * natom;
  auto* aos = primitives + 2 * nprimitive;
  std::size_t primitive_begin = 0, ao = 0;
  for (const auto& shell : system.shells) {
    for (std::size_t p = 0; p < shell.primitives.size(); ++p) {
      const auto& primitive = shell.primitives[p];
      if (!std::isfinite(primitive.exponent) || !std::isfinite(primitive.coefficient))
        throw std::invalid_argument("nonfinite normalized AO primitive");
      primitives[2 * (primitive_begin + p)] = primitive.exponent;
      primitives[2 * (primitive_begin + p) + 1] = primitive.coefficient;
    }
    for (const auto& expansion :
         molecule::ao_expansions(shell.angular_momentum, system.basis_representation)) {
      auto* record = aos + 16 * ao++;
      record[0] = shell.atom_index;
      record[1] = primitive_begin;
      record[2] = shell.primitives.size();
      record[3] = expansion.size();
      for (std::size_t t = 0; t < expansion.size(); ++t) {
        for (unsigned k = 0; k < 3; ++k) record[4 + 4 * t + k] = expansion[t].component[k];
        record[7 + 4 * t] = expansion[t].coefficient *
                            molecule::cartesian_component_normalization(expansion[t].component);
      }
    }
    primitive_begin += shell.primitives.size();
  }
}

void AoBasis::evaluate(const double* points, std::size_t npoint, unsigned order,
                       std::size_t ao_begin, std::size_t count, double* output,
                       std::size_t elements, const std::size_t* ao_ids) const {
  if (order > 3 || ao_begin > nao || count > nao - ao_begin)
    throw std::invalid_argument("invalid AO jet order or AO slice");
  const std::size_t jets = (order + 1) * (order + 2) * (order + 3) / 6;
  if (elements != multiply(multiply(jets, npoint), count) || (npoint && !points) ||
      (elements && !output))
    throw std::invalid_argument("invalid AO jet buffer");
  // A spatial mask selects records before scientific evaluation. Requiring
  // sorted unique IDs makes local D[I,I] and matrix scatter unambiguous.
  if (ao_ids)
    for (std::size_t i = 0; i < count; ++i)
      if (ao_ids[i] >= nao || (i && ao_ids[i] <= ao_ids[i - 1]))
        throw std::invalid_argument("invalid selected AO map");
  for (std::size_t i = 0; i < multiply(3, npoint); ++i)
    if (!std::isfinite(points[i])) throw std::invalid_argument("nonfinite grid point");
  const auto* primitives = packed.data() + 3 * natom;
  const auto* aos = primitives + 2 * nprimitive;

  // Derivative identities are tile invariants. Shell components also share
  // their displacement, radial contraction and one-axis polynomials. Reuse
  // those factors without changing any AO jet's primitive/term summation order.
  std::array<std::array<unsigned, 3>, 20> derivatives{};
  std::size_t derivative_count = 0;
  for (unsigned degree = 0; degree <= order; ++degree)
    for (const auto& derivative : molecule::cartesian_components(degree))
      derivatives[derivative_count++] = {derivative[0], derivative[1], derivative[2]};
  if (derivative_count != jets) throw std::logic_error("AO jet enumeration mismatch");

  // Ten components cover a Cartesian f shell. Chunking also bounds scratch
  // for packed records that share a contraction across more than one shell.
  // Form groups from selected records only: a sparse map never evaluates an
  // omitted component just to complete its shell.
  constexpr std::size_t maximum_components = 10;
  const auto evaluate_jets = [&]<std::size_t JetCount>() {
    constexpr unsigned Order = JetCount == 1 ? 0 : JetCount == 4 ? 1 : JetCount == 10 ? 2 : 3;
    for (std::size_t group_begin = 0; group_begin < count;) {
      std::array<const double*, maximum_components> records{};
      const auto* first_record = aos + 16 * (ao_ids ? ao_ids[group_begin] : ao_begin + group_begin);
      std::size_t components = 0;
      while (components < maximum_components && group_begin + components < count) {
        const auto selected = group_begin + components;
        const auto* record = aos + 16 * (ao_ids ? ao_ids[selected] : ao_begin + selected);
        if (record[0] != first_record[0] || record[1] != first_record[1] ||
            record[2] != first_record[2])
          break;
        records[components++] = record;
      }
      std::array<std::array<bool, 4>, 3> needed{};
      for (std::size_t component = 0; component < components; ++component) {
        const auto* record = records[component];
        for (unsigned term = 0; term < static_cast<unsigned>(record[3]); ++term)
          for (unsigned axis = 0; axis < 3; ++axis)
            needed[axis][static_cast<unsigned>(record[4 + 4 * term + axis])] = true;
      }
      const auto atom = static_cast<std::size_t>(first_record[0]);
      const auto first = static_cast<std::size_t>(first_record[1]);
      const auto end = first + static_cast<std::size_t>(first_record[2]);
      for (std::size_t point = 0; point < npoint; ++point) {
        std::array<double, 3> r{};
        double r2 = 0;
        for (unsigned k = 0; k < 3; ++k) {
          r[k] = points[3 * point + k] - packed[3 * atom + k];
          r2 += r[k] * r[k];
        }

        std::array<std::array<double, JetCount>, maximum_components> values{};
        for (auto p = first; p < end; ++p) {
          const double alpha = primitives[2 * p];
          const double radial = primitives[2 * p + 1] * std::exp(-alpha * r2);
          // Exact exponential underflow contributes zero, without evaluating
          // potentially overflowing far-field polynomial factors.
          if (radial == 0) continue;
          std::array<std::array<std::array<double, Order + 1>, 4>, 3> powers;
          for (unsigned axis = 0; axis < 3; ++axis)
            for (unsigned angular = 0; angular <= 3; ++angular)
              if (needed[axis][angular])
                for (unsigned derivative = 0; derivative <= Order; ++derivative)
                  powers[axis][angular][derivative] =
                      differentiated_power(angular, derivative, alpha, r[axis]);
          for (std::size_t component = 0; component < components; ++component) {
            const auto* record = records[component];
            for (unsigned t = 0; t < static_cast<unsigned>(record[3]); ++t) {
              const double weighted_radial = radial * record[7 + 4 * t];
              for (std::size_t jet = 0; jet < JetCount; ++jet) {
                double term = weighted_radial;
                for (unsigned axis = 0; axis < 3; ++axis)
                  term *= powers[axis][static_cast<unsigned>(record[4 + 4 * t + axis])]
                                [derivatives[jet][axis]];
                values[component][jet] += term;
              }
            }
          }
        }

        for (std::size_t component = 0; component < components; ++component)
          for (std::size_t jet = 0; jet < JetCount; ++jet) {
            const double value = values[component][jet];
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite AO jet result");
            output[(jet * npoint + point) * count + group_begin + component] = value;
          }
      }
      group_begin += components;
    }
  };
  switch (order) {
    case 0:
      evaluate_jets.template operator()<1>();
      break;
    case 1:
      evaluate_jets.template operator()<4>();
      break;
    case 2:
      evaluate_jets.template operator()<10>();
      break;
    case 3:
      evaluate_jets.template operator()<20>();
      break;
  }
}
}  // namespace generativeqc::dft
