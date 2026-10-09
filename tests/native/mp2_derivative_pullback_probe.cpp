// Bounded probe for the actual common derivative translation unit. The oracle
// deliberately uses a direct rank-eight loop, not its staged pullback algorithm.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "hf/reference.hpp"
#include "molecule/basis.hpp"
#include "posthf/mp2_derivative_common.hpp"
#include "posthf/mp2_gradient.hpp"
#include "probe_counts.hpp"

namespace allocation {
struct alignas(std::max_align_t) Header {
  std::size_t bytes;
  bool tracked;
};
bool enabled = false;
std::size_t live = 0, peak = 0, calls = 0, total = 0;
// Allocation ownership follows its creation epoch even while callbacks pause
// observation. Callback/oracle scratch never counts toward transform receipts.
struct Pause {
  bool previous = enabled;
  Pause() { enabled = false; }
  ~Pause() { enabled = previous; }
};
}  // namespace allocation
void* operator new(std::size_t bytes) {
  auto* header = static_cast<allocation::Header*>(std::malloc(sizeof(allocation::Header) + bytes));
  if (!header) throw std::bad_alloc();
  header->bytes = bytes;
  header->tracked = allocation::enabled;
  if (header->tracked) {
    allocation::live += bytes;
    allocation::peak = std::max(allocation::peak, allocation::live);
    allocation::total += bytes;
    ++allocation::calls;
  }
  return header + 1;
}
void operator delete(void* pointer) noexcept {
  if (!pointer) return;
  auto* header = static_cast<allocation::Header*>(pointer) - 1;
  if (header->tracked) allocation::live -= header->bytes;
  std::free(header);
}
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete(pointer); }

using namespace generativeqc;
using Quartet = std::array<std::size_t, 4>;
using Counts = std::array<std::size_t, 9>;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void close(double actual, double expected, const char* label) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > 3e-11 * (1.0 + std::abs(expected))) {
    std::cerr << label << ": actual=" << std::setprecision(17) << actual << " expected=" << expected
              << '\n';
    throw std::runtime_error(label);
  }
}
std::size_t index(std::size_t n, Quartet q) { return ((q[0] * n + q[1]) * n + q[2]) * n + q[3]; }
// Generate the group independently of the production projection expression.
std::array<Quartet, 8> permutations(Quartet q) {
  std::array<Quartet, 8> result{};
  for (unsigned bits = 0; bits < 8; ++bits) {
    auto item = q;
    if (bits & 1) std::swap(item[0], item[1]);
    if (bits & 2) std::swap(item[2], item[3]);
    if (bits & 4) {
      std::swap(item[0], item[2]);
      std::swap(item[1], item[3]);
    }
    result[bits] = item;
  }
  return result;
}
bool canonical(Quartet q) {
  const auto orbit = permutations(q);
  return q == *std::max_element(orbit.begin(), orbit.end());
}

struct Fixture {
  core::System system;
  hf::PhysicalReference reference;
  mp2::LagrangianWeights weights;
  std::vector<std::size_t> offsets;
  std::vector<double> dense;
};
Fixture fixture(unsigned id, unsigned representation, bool sparse) {
  const std::array<std::vector<unsigned>, 8> angular = {
      {{0, 0}, {1}, {0, 1}, {1, 0}, {0, 1, 0}, {2, 0}, {2, 0}, {3}}};
  const std::array<std::vector<unsigned>, 8> atoms = {
      {{0, 1}, {0}, {0, 1}, {1, 0}, {1, 0, 1}, {0, 1}, {0, 1}, {0}}};
  const std::array<std::size_t, 8> occupied = {1, 1, 1, 3, 2, 2, 2, 3};
  Fixture f;
  f.system.basis_representation =
      id >= 6 ? GENERATIVEQC_BASIS_SPHERICAL : GENERATIVEQC_BASIS_CARTESIAN;
  // Leave a spectator atom to verify scatter doesn't use shell index as atom index.
  f.system.atoms.resize(3);
  f.offsets.push_back(0);
  for (std::size_t shell = 0; shell < angular.at(id).size(); ++shell) {
    f.system.shells.push_back({atoms[id][shell], angular[id][shell], {}});
    const auto l = angular[id][shell];
    const auto width = id >= 6 ? 2 * l + 1 : (l + 1) * (l + 2) / 2;
    f.offsets.push_back(f.offsets.back() + width);
  }
  const auto n = f.offsets.back(), o = occupied[id], v = n - o;
  f.reference.nbf = n;
  f.reference.nocc = o;
  auto& c = f.reference.coefficients;
  c.resize(n * n);
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t p = 0; p < n; ++p)
      c[a * n + p] = sparse && (3 * a + p) % 4 == 0
                         ? 0.0
                         : 0.2 * std::sin(1.3 + 5 * a + 7 * p) + (a == p ? 0.7 : 0.0);
  auto& w = f.weights;
  w.orbitals = n;
  w.occupied = o;
  w.one_electron.resize(n * n);
  w.overlap.resize(n * n);
  for (std::size_t t = 0; t < n * n; ++t) {
    w.one_electron[t] = std::sin(0.3 + t);
    w.overlap[t] = std::cos(0.7 + 0.4 * t);
  }
  f.dense.resize(n * n * n * n);
  if (representation) {
    auto& factors = w.two_electron_factors;
    factors.orbitals = n;
    factors.occupied = o;
    factors.fock.resize(n * n);
    factors.correlation_iajb.resize(o * o * v * v);
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        const double value = sparse && (p + 2 * q) % 3 == 0 ? 0.0 : std::cos(0.8 + 3 * p + 2 * q);
        factors.fock[p * n + q] = value;
        // Independently expand the contract as tensor contributions, without
        // calling factorized_two_electron_weight or copying its lookup rule.
        for (std::size_t i = 0; i < o; ++i) {
          f.dense[index(n, {p, q, i, i})] += 2 * value;
          f.dense[index(n, {p, i, i, q})] -= value;
        }
      }
    std::size_t t = 0;
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b) {
            const double value = std::sin(0.2 + 0.6 * t);
            factors.correlation_iajb[t++] = value;
            f.dense[index(n, {i, o + a, j, o + b})] += value;
          }
  } else {
    for (std::size_t t = 0; t < f.dense.size(); ++t)
      f.dense[t] = 0.3 * std::sin(0.1 + 0.7 * t) + 0.2 * std::cos(0.6 + 0.4 * t);
    w.two_electron = f.dense;
  }
  if (representation == 2) w.two_electron = f.dense;
  return f;
}

std::vector<double> direct_pullback(const Fixture& f) {
  const auto n = f.reference.nbf;
  const auto& c = f.reference.coefficients;
  std::vector<double> result(n * n * n * n);
  for (std::size_t mu = 0; mu < n; ++mu)
    for (std::size_t nu = 0; nu < n; ++nu)
      for (std::size_t ka = 0; ka < n; ++ka)
        for (std::size_t la = 0; la < n; ++la) {
          long double sum = 0;
          for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = 0; q < n; ++q)
              for (std::size_t r = 0; r < n; ++r)
                for (std::size_t s = 0; s < n; ++s)
                  sum += static_cast<long double>(f.dense[index(n, {p, q, r, s})]) * c[mu * n + p] *
                         c[nu * n + q] * c[ka * n + r] * c[la * n + s];
          result[index(n, {mu, nu, ka, la})] = static_cast<double>(sum);
        }
  return result;
}
std::vector<double> direct_rank_two(const Fixture& f, const std::vector<double>& w) {
  const auto n = f.reference.nbf;
  std::vector<double> result(n * n);
  for (std::size_t mu = 0; mu < n; ++mu)
    for (std::size_t nu = 0; nu < n; ++nu) {
      long double sum = 0;
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q)
          sum += static_cast<long double>(f.reference.coefficients[mu * n + p]) * w[p * n + q] *
                 f.reference.coefficients[nu * n + q];
      result[mu * n + nu] = static_cast<double>(sum);
    }
  return result;
}
// Synthetic ERI = u*v + 0.17*u*u*v*v, u=a_mu+a_nu, v=a_ka+a_la.
// Each AO has distinct a and da/dR. Derivatives obey all eight ERI symmetries,
// but retain center-specific values before repeated-atom scattering.
std::array<double, 12> center_derivatives(Quartet ao) {
  std::array<double, 4> a{};
  for (unsigned slot = 0; slot < 4; ++slot) a[slot] = 0.15 + 0.07 * ao[slot];
  const double u = a[0] + a[1], v = a[2] + a[3];
  std::array<double, 12> result{};
  for (unsigned slot = 0; slot < 4; ++slot)
    for (unsigned axis = 0; axis < 3; ++axis) {
      const double da = std::sin(0.4 + 2 * ao[slot] + 3 * axis);
      result[3 * slot + axis] = da * (slot < 2 ? v + 0.34 * u * v * v : u + 0.34 * v * u * u);
    }
  return result;
}

struct Expected {
  Counts iterations{};
  std::size_t callbacks = 0, elements = 0, peak = 0, allocation_calls = 0, total_bytes = 0;
  std::size_t shell_peak = 0;
  std::vector<unsigned char> seen;
  std::vector<double> local, gradient, one, overlap;
};
Expected expected(const Fixture& f, const std::vector<double>& ao, bool factorized) {
  Expected e;
  const auto n = f.reference.nbf, h = f.system.shells.size(), coords = 3 * f.system.atoms.size();
  const auto& c = f.reference.coefficients;
  const auto& factors = f.weights.two_electron_factors;
  e.local.resize(ao.size());
  e.seen.resize(h * h * h * h);
  e.gradient.resize(coords);
  for (std::size_t x = 0; x < coords; ++x) e.gradient[x] = 0.01 * (x + 1);
  e.one = direct_rank_two(f, f.weights.one_electron);
  e.overlap = direct_rank_two(f, f.weights.overlap);
  std::vector<std::size_t> atom(n);
  for (std::size_t shell = 0; shell < h; ++shell)
    for (auto a = f.offsets[shell]; a < f.offsets[shell + 1]; ++a)
      atom[a] = f.system.shells[shell].atom_index;
  // Oracle gradient contracts every ordered AO quartet, even for dense mode.
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t k = 0; k < n; ++k)
        for (std::size_t l = 0; l < n; ++l) {
          const Quartet q{i, j, k, l};
          const auto center = center_derivatives(q);
          for (unsigned slot = 0; slot < 4; ++slot)
            for (unsigned axis = 0; axis < 3; ++axis)
              e.gradient[3 * atom[q[slot]] + axis] += ao[index(n, q)] * center[3 * slot + axis];
          if (factorized)
            e.local[index(n, q)] = ao[index(n, q)];
          else {
            long double sum = 0;
            for (auto permutation : permutations(q)) sum += ao[index(n, permutation)];
            e.local[index(n, q)] = static_cast<double>(sum / 8);
          }
        }
  std::set<std::array<std::size_t, 2>> pairs;
  std::set<std::array<std::size_t, 3>> triples;
  const auto persistent = sizeof(double) * 2 * n * n + sizeof(std::size_t) * (h + 1);
  e.allocation_calls = 5 + h;  // rank-two outputs/workspaces, offsets, first stages
  e.total_bytes = sizeof(double) * (4 * n * n + n * n * n * n) + sizeof(std::size_t) * (h + 1);
  for (std::size_t i = 0; i < h; ++i)
    for (std::size_t j = 0; j < h; ++j)
      for (std::size_t k = 0; k < h; ++k)
        for (std::size_t l = 0; l < h; ++l) {
          const Quartet q{i, j, k, l};
          if (!factorized && !canonical(q)) continue;
          pairs.insert({i, j});
          triples.insert({i, j, k});
          const auto di = f.offsets[i + 1] - f.offsets[i], dj = f.offsets[j + 1] - f.offsets[j];
          const auto dk = f.offsets[k + 1] - f.offsets[k], dl = f.offsets[l + 1] - f.offsets[l];
          const auto local = di * dj * dk * dl;
          ++e.callbacks;
          e.elements += local;
          e.iterations[6] += local * n;
          if (!factorized) e.iterations[8] += local;
          const auto shell =
              sizeof(double) * (di * n * n * n + di * dj * n * n + di * dj * dk * n + local);
          e.shell_peak = std::max(e.shell_peak, shell);
          e.peak = std::max(e.peak, persistent + shell);
          ++e.allocation_calls;
          e.total_bytes += sizeof(double) * local;
        }
  for (auto p : pairs) {
    const auto size =
        (f.offsets[p[0] + 1] - f.offsets[p[0]]) * (f.offsets[p[1] + 1] - f.offsets[p[1]]) * n * n;
    e.iterations[4] += size * n;
    ++e.allocation_calls;
    e.total_bytes += sizeof(double) * size;
  }
  for (auto t : triples) {
    const auto size = (f.offsets[t[0] + 1] - f.offsets[t[0]]) *
                      (f.offsets[t[1] + 1] - f.offsets[t[1]]) *
                      (f.offsets[t[2] + 1] - f.offsets[t[2]]) * n;
    e.iterations[5] += size * n;
    ++e.allocation_calls;
    e.total_bytes += sizeof(double) * size;
  }
  if (factorized) {
    const auto o = f.reference.nocc, v = n - o;
    for (std::size_t mu = 0; mu < n; ++mu)
      for (std::size_t p = 0; p < n; ++p) {
        if (c[mu * n + p] == 0) continue;
        for (std::size_t q = 0; q < n; ++q)
          if (factors.fock[p * n + q] != 0) {
            e.iterations[1] += o;
            e.iterations[2] += o;
          }
        if (p < o) e.iterations[3] += o * v * v;
      }
  } else {
    e.iterations[0] = n * n * n * n * n;
    e.iterations[7] = 8 * e.iterations[0];
  }
  return e;
}

int main(int argc, char** argv) {
  try {
    require(argc == 4, "expected case, factorized, sparse arguments");
    const unsigned id = std::stoul(argv[1]);
    const unsigned representation = std::stoul(argv[2]);
    const bool factorized = representation == 1, sparse = std::stoi(argv[3]);
    auto f = fixture(id, representation, sparse);
    const auto n = f.reference.nbf, h = f.system.shells.size();
    const auto ao = direct_pullback(f);
    auto e = expected(f, ao, factorized);
    std::size_t callbacks = 0, elements = 0, measured_shell_peak = 0;
    std::vector<double> initial_gradient(3 * f.system.atoms.size());
    for (std::size_t t = 0; t < initial_gradient.size(); ++t) initial_gradient[t] = 0.01 * (t + 1);
    const auto retained_output_bytes = sizeof(double) * initial_gradient.size();
    mp2::detail::OneElectronDerivativeContract one = [&](auto overlap, auto hcore) {
      allocation::Pause pause;
      for (std::size_t t = 0; t < n * n; ++t) {
        close(overlap[t], e.overlap[t], "overlap pullback");
        close(hcore[t], e.one[t], "one-electron pullback");
      }
      return std::move(initial_gradient);
    };
    mp2::detail::EriShellDerivativeContract shell = [&](const Quartet& q, auto local) {
      allocation::Pause pause;
      require(factorized || canonical(q), "noncanonical dense callback");
      require(!e.seen[index(h, q)]++, "duplicate callback");
      ++callbacks;
      elements += local.size();
      const auto di = f.offsets[q[0] + 1] - f.offsets[q[0]];
      const auto dj = f.offsets[q[1] + 1] - f.offsets[q[1]];
      const auto dk = f.offsets[q[2] + 1] - f.offsets[q[2]];
      const auto dl = f.offsets[q[3] + 1] - f.offsets[q[3]];
      require(local.size() == di * dj * dk * dl, "local shape");
      const auto shell_bytes =
          sizeof(double) * (di * n * n * n + di * dj * n * n + di * dj * dk * n + local.size());
      const auto persistent = sizeof(double) * 2 * n * n + sizeof(std::size_t) * (h + 1);
      require(allocation::live == persistent + shell_bytes, "active quartet allocation ownership");
      measured_shell_peak = std::max(measured_shell_peak, allocation::live - persistent);
      // Compute shell orbit before tracing, or without allocation: count unique
      // members in the fixed eight-element permutation array.
      const auto orbit = permutations(q);
      std::size_t multiplicity = 0;
      for (std::size_t i = 0; i < orbit.size(); ++i)
        if (std::find(orbit.begin(), orbit.begin() + i, orbit[i]) == orbit.begin() + i)
          ++multiplicity;
      std::array<double, 12> center{};
      std::size_t t = 0;
      for (auto mu = f.offsets[q[0]]; mu < f.offsets[q[0] + 1]; ++mu)
        for (auto nu = f.offsets[q[1]]; nu < f.offsets[q[1] + 1]; ++nu)
          for (auto ka = f.offsets[q[2]]; ka < f.offsets[q[2] + 1]; ++ka)
            for (auto la = f.offsets[q[3]]; la < f.offsets[q[3] + 1]; ++la) {
              const Quartet a{mu, nu, ka, la};
              const double expected_weight = e.local[index(n, a)] * (factorized ? 1 : multiplicity);
              close(local[t], expected_weight, "shell pullback weight");
              // Preallocated receipt is overwritten with actual production data.
              e.local[index(n, a)] = local[t];
              const auto derivative = center_derivatives(a);
              for (unsigned coordinate = 0; coordinate < 12; ++coordinate)
                center[coordinate] += local[t] * derivative[coordinate];
              ++t;
            }
      return center;
    };
    allocation::enabled = true;
    {
      auto result =
          mp2::detail::conventional_derivative(f.system, f.reference, f.weights, one, shell);
      allocation::enabled = false;
      require(allocation::live == 0, "common allocations escaped invocation");
      require(initial_gradient.empty(), "callback output was not ownership-transferred");
      require(allocation::peak == e.peak, "allocation high-water mismatch");
      require(allocation::calls == e.allocation_calls, "allocation call mismatch");
      require(allocation::total == e.total_bytes, "allocation total mismatch");
      require(measured_shell_peak == e.shell_peak, "shell allocation peak mismatch");
      require(callbacks == e.callbacks && elements == e.elements, "callback work mismatch");
      for (std::size_t t = 0; t < result.size(); ++t)
        close(result[t], e.gradient[t], "atom scatter");
      if (probe::instrumented)
        require(probe::counts == e.iterations, "executed accumulation count mismatch");
      // The production plan's numeric-buffer bound includes rank-two/output
      // owners and all four stages. Shell offsets are separate metadata bytes.
      response::GmresPlan response;
      response.dimension = f.reference.nocc * (n - f.reference.nocc);
      std::size_t maximum_shell = 0;
      for (std::size_t i = 0; i < h; ++i)
        maximum_shell = std::max(maximum_shell, f.offsets[i + 1] - f.offsets[i]);
      const auto plan =
          mp2::conventional_gradient_plan(n, f.reference.nocc, 0, response, maximum_shell,
                                          result.size(), 0, static_cast<std::size_t>(INT64_MAX));
      const auto numeric_peak =
          allocation::peak - sizeof(std::size_t) * (h + 1) + retained_output_bytes;
      require(numeric_peak <= plan.derivative_staging_bytes + plan.shell_cotangent_bytes,
              "common pullback exceeds production numeric staging plan");
      std::cout << std::setprecision(17) << "{\"n\":" << n << ",\"callbacks\":" << callbacks
                << ",\"local_elements\":" << elements
                << ",\"common_peak_bytes\":" << allocation::peak
                << ",\"retained_output_bytes\":" << retained_output_bytes
                << ",\"shell_peak_bytes\":" << measured_shell_peak
                << ",\"allocation_calls\":" << allocation::calls
                << ",\"allocated_bytes\":" << allocation::total
                << ",\"planned_numeric_staging_bytes\":"
                << plan.derivative_staging_bytes + plan.shell_cotangent_bytes
                << ",\"iterations\":[";
      for (unsigned i = 0; i < probe::counts.size(); ++i)
        std::cout << (i ? "," : "") << probe::counts[i];
      std::cout << "],\"gradient\":[";
      for (std::size_t t = 0; t < result.size(); ++t) std::cout << (t ? "," : "") << result[t];
      std::cout << "],\"weights\":[";
      for (std::size_t t = 0; t < e.local.size(); ++t) std::cout << (t ? "," : "") << e.local[t];
      std::cout << "]}\n";
    }
    require(allocation::live == 0, "leaked common allocations");
  } catch (const std::exception& error) {
    allocation::enabled = false;
    std::cerr << error.what() << '\n';
    return 1;
  }
}
