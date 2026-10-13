// Execute production point launch ABIs as host C++ with scheduling stubs.
// PBE/LDA algebra is real; unrelated generated providers reject if reached.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "dft/cuda_xc_point_program.hpp"
#include "dft/semilocal_family.hpp"
#include "dft/xc_point_response.hpp"

#define __device__
#define __global__
#define __forceinline__ inline
struct {
  long long x;
} blockIdx{0}, blockDim{1}, threadIdx{0}, gridDim{1};
using Kernel = void (*)(const double*, const double*, long long, long long, double*, double*, int*,
                        long long, double, double, const double*, long long);
Kernel observed_kernel{};
std::size_t observed_threads{};
bool execute_kernels = true;
std::vector<std::uint32_t> probe_split_codes, probe_split_mgga;
template <class Fn>
auto host_launch(Fn kernel, std::size_t grid, std::size_t threads, std::size_t, cudaStream_t) {
  observed_kernel = kernel;
  observed_threads = threads;
  return [=](auto... args) {
    if (!execute_kernels) return;  // Geometry-only probes do not claim point arithmetic.
    blockDim.x = threads;
    gridDim.x = grid;
    for (std::size_t block = 0; block < grid; ++block) {
      blockIdx.x = block;
      for (std::size_t thread = 0; thread < threads; ++thread) {
        threadIdx.x = thread;
        if constexpr (std::is_invocable_v<Fn, decltype(args)...>)
          kernel(args...);
        else
          kernel(args..., 0);  // Preserve the optional physical tile argument.
      }
    }
  };
}
int atomicCAS(int* error, int expected, int value) {
  const int old = *error;
  if (old == expected) *error = value;
  return old;
}
namespace generativeqc_tensor {
std::size_t blocks(std::size_t count, std::size_t threads) {
  return (count + threads - 1) / threads;
}
double finite(double value, int* error, int code) {
  if (!std::isfinite(value)) *error = code;
  return value;
}
}  // namespace generativeqc_tensor
namespace generativeqc::dft::generated {
struct UnusedValue {
  double energy_density{}, feature_derivative[7]{};
  bool matched{};
};
template <class... Args>
UnusedValue unused_point(Args...) {
  throw std::logic_error("PBE policy entered an unrelated mathematical provider");
}
template <class... Args>
UnusedValue b3lyp_device(Args... args) {
  return unused_point(args...);
}
template <class... Args>
UnusedValue b3lyp_device_equal_spin(Args... args) {
  return unused_point(args...);
}
template <class... Args>
UnusedValue wb97mv_device(Args... args) {
  return unused_point(args...);
}
template <class... Args>
UnusedValue wb97mv_device_equal_spin(Args... args) {
  return unused_point(args...);
}
template <class... Args>
UnusedValue r2scan_device(Args... args) {
  return unused_point(args...);
}
template <class... Args>
UnusedValue r2scan_device_equal_spin(Args... args) {
  return unused_point(args...);
}
template <class... Args>
UnusedValue evaluate_split_hybrid(Args... args) {
  return unused_point(args...);
}
constexpr double kWb97mvDeviceDensityThreshold = 1e-14;
constexpr double kWb97mvDeviceSigmaThreshold = 1e-14;
constexpr double kWb97mvDeviceTauThreshold = 1e-14;
bool split_hybrid_registered(std::uint32_t code) {
  return std::find(probe_split_codes.begin(), probe_split_codes.end(), code) !=
         probe_split_codes.end();
}
bool split_hybrid_is_mgga(std::uint32_t code) {
  return std::find(probe_split_mgga.begin(), probe_split_mgga.end(), code) !=
         probe_split_mgga.end();
}
const CudaXcPointProgram* automatic_cuda_point_entry(std::uint32_t) { return nullptr; }
}  // namespace generativeqc::dft::generated
namespace generativeqc::dft::cuda_xc_detail {
using I = long long;
using generativeqc_tensor::finite;
using std::isfinite;
using std::min;
struct CudaXcPointCapabilities {
  bool local_ao_selection{}, mixed_density_contraction{};
};
#include "point_dispatch.hpp"
}  // namespace generativeqc::dft::cuda_xc_detail
using namespace generativeqc::dft;
using namespace generativeqc::dft::cuda_xc_detail;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void policy(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value ? value : "");
#else
  if (value)
    setenv(name, value, 1);
  else
    unsetenv(name);
#endif
}
void near(double actual, double expected, double relative = 5e-10, double absolute = 1e-322) {
  if (!std::isfinite(actual) ||
      std::abs(actual - expected) > relative * std::abs(expected) + absolute) {
    std::cerr << "actual " << actual << " expected " << expected << '\n';
    throw std::runtime_error("point result differs from its numerical contract");
  }
}
struct Output {
  std::vector<double> coefficients, totals;
  int error{};
  Kernel kernel{};
  std::size_t threads{};
};
Output execute(unsigned functional, unsigned spins, const std::vector<double>& input,
               bool response = false, bool batched = false, double exchange = 1,
               const std::vector<double>& delta = {}, double correlation = 1) {
  const std::size_t terms = functional == 0 ? 1 : 4;
  const std::size_t count = input.size() / (terms * spins);
  std::vector<double> weights(count, 1), coefficients(input.size() + 2, 12345),
      totals(3 * count + 2, 54321);
  const auto original = input;
  auto launch = resolve_point_launcher(functional, response);
  int error = 0;
  if (batched) {
    auto batch = resolve_point_batch_launcher(functional, launch);
    batch(nullptr, input.data(), weights.data(), count, spins, coefficients.data() + 1,
          totals.data() + 1, &error, functional, exchange, correlation, 3);
  } else {
    launch(nullptr, input.data(), weights.data(), count, spins, coefficients.data() + 1,
           totals.data() + 1, &error, functional, exchange, correlation,
           response ? delta.data() : nullptr);
  }
  require(input == original, "point launch modified its immutable features");
  require(coefficients.front() == 12345 && coefficients.back() == 12345 &&
              totals.front() == 54321 && totals.back() == 54321,
          "point launch crossed buffer capacity");
  return {{coefficients.begin() + 1, coefficients.end() - 1},
          {totals.begin() + 1, totals.end() - 1},
          error,
          observed_kernel,
          observed_threads};
}
void fixtures(const char* filename, bool restricted) {
  std::ifstream input(filename);
  require(bool(input), "independent point fixture is missing");
  std::string line;
  std::size_t rows = 0;
  std::size_t ordinary_original_formula_failures = 0;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    int family = 1, oracle = 0;
    if (!restricted) row >> family >> oracle;
    const auto number = [&row]() {
      std::string token;
      require(bool(row >> token), "missing independent FP64 fixture value");
      char* end = nullptr;
      // The pinned corpus includes representable subnormals. strtod preserves
      // their FP64 rounding without an iostream range-error rejection.
      const double value = std::strtod(token.c_str(), &end);
      require(end == token.c_str() + token.size(), "malformed FP64 fixture value");
      return value;
    };
    double rho[2], gradient[2][3], expected[9];
    rho[0] = number();
    rho[1] = number();
    for (auto& spin : gradient)
      for (double& value : spin) value = number();
    for (double& value : expected) value = number();
    require(bool(row), "malformed independent point fixture");
    double comparison[9];
    std::copy_n(expected, 9, comparison);
    if (restricted) {
      require(rho[0] == rho[1], "restricted fixture lacks paired density");
      for (unsigned axis = 0; axis < 3; ++axis)
        require(gradient[0][axis] == gradient[1][axis], "restricted fixture lacks paired gradient");
      // This wider corpus qualifies the separately bound stable-subnormal
      // lowering. Ordinary/family/equal-spin entries keep their original
      // arithmetic, including retained independent tail failures.
      const auto bound = point::evaluate_pbe0_restricted_bound(rho[0], gradient[0]);
      require(bound.valid, "qualified bound fixture domain was rejected");
      const double bound_values[]{bound.energy,         bound.rho[0],         bound.rho[1],
                                  bound.gradient[0][0], bound.gradient[0][1], bound.gradient[0][2],
                                  bound.gradient[1][0], bound.gradient[1][1], bound.gradient[1][2]};
      for (unsigned component = 0; component < 9; ++component)
        near(bound_values[component], expected[component]);
      const auto original = point::evaluate(true, rho, gradient, .75, 1);
      require(original.valid, "ordinary fixture domain was rejected");
      const double original_values[]{
          original.energy,         original.rho[0],         original.rho[1],
          original.gradient[0][0], original.gradient[0][1], original.gradient[0][2],
          original.gradient[1][0], original.gradient[1][1], original.gradient[1][2]};
      bool independent_failure = false;
      for (unsigned component = 0; component < 9; ++component) {
        comparison[component] = original_values[component];
        const double tolerance = 5e-10 * std::abs(expected[component]) + 1e-322;
        if (!std::isfinite(original_values[component]) ||
            std::abs(original_values[component] - expected[component]) > tolerance)
          independent_failure = true;
        std::cout << std::setprecision(17) << "supplemental ordinary row=" << rows
                  << " component=" << component << " actual=" << original_values[component]
                  << " reference=" << expected[component] << " absolute_error="
                  << std::abs(original_values[component] - expected[component])
                  << " tolerance=" << tolerance << " bound=" << bound_values[component] << "\n";
      }
      if (independent_failure) {
        ++ordinary_original_formula_failures;
        std::cout << "ordinary supplemental original-formula qualification failed at row " << rows
                  << " (retained negative result; not ordinary numerical PASS)\n";
      }
    }
    const unsigned terms = family == 0 ? 1 : 4;
    // The shared ratio repair now qualifies ordinary tails independently.
    // Retain the original oracle and error rows, and require every emitted
    // physical route to satisfy it instead of substituting baseline parity.
    if (restricted) std::copy_n(expected, 9, comparison);
    for (unsigned spins : {1U, 2U}) {
      if (!restricted && spins == 1) continue;
      std::vector<double> features(terms * spins);
      for (unsigned spin = 0; spin < spins; ++spin) {
        features[spin * terms] = spins == 1 ? rho[0] + rho[1] : rho[spin];
        for (unsigned axis = 0; axis + 1 < terms; ++axis)
          features[spin * terms + 1 + axis] =
              spins == 1 ? gradient[0][axis] + gradient[1][axis] : gradient[spin][axis];
      }
      for (const char* equal : {"0", "1"})
        for (const char* specialized : {"0", "1"}) {
          policy("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", equal);
          policy("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", specialized);
          for (bool batched : {false, true}) {
            const auto result =
                execute(family, spins, features, false, batched, restricted ? .75 : 1);
            require(!result.error, "independent physical domain was rejected");
            require(result.threads == (family == 0 ? 128U : 32U), "physical launch width changed");
            near(result.totals[0], comparison[0]);
            for (unsigned spin = 0; spin < spins; ++spin) {
              near(result.coefficients[spin * terms],
                   spins == 1 ? .5 * (comparison[1] + comparison[2]) : comparison[1 + spin]);
              for (unsigned axis = 0; axis + 1 < terms; ++axis) {
                const double reference = spins == 1
                                             ? .5 * (comparison[3 + axis] + comparison[6 + axis])
                                             : comparison[3 + 3 * spin + axis];
                near(result.coefficients[spin * terms + 1 + axis], reference);
              }
            }
          }
        }
    }
    ++rows;
  }
  require(rows == (restricted ? 87U : 97U), "independent fixture corpus was shortened");
  if (restricted)
    std::cout << "bound PBE0 independent fixture PASS: " << rows
              << "; ordinary supplemental original-formula failed rows: "
              << ordinary_original_formula_failures << "\n";
  require(ordinary_original_formula_failures == 0,
          "shared repaired ordinary consumer failed the independent tail oracle");
}
std::vector<double> pack(const std::vector<double>& serial, unsigned spins, bool batched) {
  if (!batched) return serial;
  constexpr std::size_t points = 5, terms = 4;
  std::vector<double> output(serial.size());
  for (std::size_t begin = 0; begin < points; begin += 3) {
    const std::size_t count = std::min(std::size_t{3}, points - begin);
    for (std::size_t channel = 0; channel < spins * terms; ++channel)
      for (std::size_t p = 0; p < count; ++p)
        output[begin * spins * terms + channel * count + p] = serial[channel * points + begin + p];
  }
  return output;
}
void replay_and_response() {
  constexpr std::size_t points = 5;
  for (unsigned spins : {1U, 2U}) {
    std::vector<double> features(4 * spins * points), delta(features.size());
    for (std::size_t channel = 0; channel < 4 * spins; ++channel)
      for (std::size_t p = 0; p < points; ++p) {
        features[channel * points + p] =
            channel % 4 == 0 ? .2 + .1 * p + .05 * channel : .02 * (channel + 1);
        delta[channel * points + p] = .003 * (channel + 1);
      }
    for (const char* equal : {"0", "1"})
      for (const char* specialized : {"0", "1"}) {
        policy("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", equal);
        policy("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", specialized);
        const auto serial = execute(1, spins, features);
        const auto batched = execute(1, spins, pack(features, spins, true), false, true);
        require(!serial.error && !batched.error, "valid replay failed");
        require(serial.threads == 32 && batched.threads == 32,
                "PBE physical/batch geometry changed");
        const auto packed_coefficients = pack(serial.coefficients, spins, true);
        for (std::size_t i = 0; i < packed_coefficients.size(); ++i)
          near(batched.coefficients[i], packed_coefficients[i]);
        for (std::size_t begin = 0; begin < points; begin += 3) {
          const auto count = std::min(std::size_t{3}, points - begin);
          for (std::size_t channel = 0; channel < 3; ++channel)
            for (std::size_t p = 0; p < count; ++p)
              near(batched.totals[3 * begin + channel * count + p],
                   serial.totals[channel * points + begin + p]);
        }
        for (int direction : {-1, 0, 1}) {
          auto signed_delta = delta;
          for (std::size_t i = 0; i < signed_delta.size(); ++i)
            signed_delta[i] *= direction == 0 ? (i % 2 ? -1 : 1) : direction;
          const auto response = execute(1, spins, features, true, false, 1, signed_delta);
          require(!response.error, "valid signed response was rejected");
          require(response.threads == 128, "response geometry changed");
          auto plus = features, minus = features;
          constexpr double step = 1e-4;
          for (std::size_t i = 0; i < features.size(); ++i) {
            plus[i] += step * signed_delta[i];
            minus[i] -= step * signed_delta[i];
          }
          const auto high = execute(1, spins, plus), low = execute(1, spins, minus);
          for (std::size_t i = 0; i < features.size(); ++i)
            near(response.coefficients[i],
                 (high.coefficients[i] - low.coefficients[i]) / (2 * step), 1e-6, 2e-9);
        }
        try {
          resolve_point_batch_launcher(1, resolve_point_launcher(1, true));
          throw std::logic_error("response batch admitted");
        } catch (const std::invalid_argument&) {
        }
      }
  }
}
void weighted_response() {
  for (unsigned spins : {1U, 2U})
    for (const char* equal : {"0", "1"})
      for (const char* family : {"0", "1"}) {
        policy("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", equal);
        policy("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", family);
        const std::vector<double> features =
            spins == 1 ? std::vector<double>{1, .15, -.07, .12}
                       : std::vector<double>{.4, .15, -.07, .12, .6, -.04, .05, .09};
        const std::vector<double> delta =
            spins == 1 ? std::vector<double>{.05, .03, .02, -.01}
                       : std::vector<double>{.08, .03, .02, -.01, -.03, -.02, .04, .01};
        for (const auto& scales :
             {std::array<double, 2>{.75, 1}, {.37, .91}, {0, 1}, {1, 0}, {0, 0}}) {
          const auto response =
              execute(1, spins, features, true, false, scales[0], delta, scales[1]);
          require(!response.error, "valid MethodIR response weights were rejected");
          for (double step : {1e-4, 3e-5}) {
            auto plus = features, minus = features;
            for (std::size_t i = 0; i < delta.size(); ++i) {
              plus[i] += step * delta[i];
              minus[i] -= step * delta[i];
            }
            const auto high = execute(1, spins, plus, false, false, scales[0], {}, scales[1]);
            const auto low = execute(1, spins, minus, false, false, scales[0], {}, scales[1]);
            require(!high.error && !low.error, "weighted physical control was rejected");
            for (std::size_t i = 0; i < delta.size(); ++i)
              near(response.coefficients[i],
                   (high.coefficients[i] - low.coefficients[i]) / (2 * step), 1e-6, 2e-9);
          }
        }
        // Independent Dirac exchange derivative at zero gradient, with an
        // empty UKS alpha channel. The unused correlation jet must not poison X.
        const std::vector<double> density = spins == 1
                                                ? std::vector<double>{1, 0, 0, 0}
                                                : std::vector<double>{0, 0, 0, 0, 1, 0, 0, 0};
        const std::vector<double> direction = spins == 1
                                                  ? std::vector<double>{.1, 0, 0, 0}
                                                  : std::vector<double>{0, 0, 0, 0, .1, 0, 0, 0};
        const auto exchange = execute(1, spins, density, true, false, 1, direction, 0);
        require(!exchange.error, "unused singular correlation jet rejected exchange response");
        const double spin_rho = spins == 1 ? .5 : 1;
        const double spin_delta = spins == 1 ? .05 : .1;
        const double cx = .75 * std::cbrt(6 / std::acos(-1.0));
        const double expected = -cx * (4.0 / 9.0) * std::pow(spin_rho, -2.0 / 3.0) * spin_delta;
        near(exchange.coefficients[spins == 1 ? 0 : 4], expected, 5e-10, 1e-14);
        if (spins == 2) near(exchange.coefficients[0], 0, 0, 1e-322);
        for (const auto& scales : {std::array<double, 2>{-1, 1},
                                   {1, std::numeric_limits<double>::infinity()},
                                   {std::numeric_limits<double>::quiet_NaN(), 1}})
          require(execute(1, spins, features, true, false, scales[0], delta, scales[1]).error != 0,
                  "invalid MethodIR response weights were admitted");
      }
}
void consumer_identity() {
  for (unsigned spins : {1U, 2U})
    for (bool response : {false, true}) {
      for (bool batched : {false, true}) {
        if (batched && response) continue;
        const std::vector<double> features = spins == 1
                                                 ? std::vector<double>{.4, 0, 0, 0}
                                                 : std::vector<double>{.2, 0, 0, 0, .3, 0, 0, 0};
        const std::vector<double> delta(features.size(), 0);
        Kernel by_policy[2][2]{};
        for (unsigned equal = 0; equal < 2; ++equal)
          for (unsigned family = 0; family < 2; ++family) {
            policy("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", equal ? "1" : "0");
            policy("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", family ? "1" : "0");
            by_policy[equal][family] =
                execute(1, spins, features, response, batched, 1, delta).kernel;
          }
        require(by_policy[0][0] != by_policy[0][1],
                "ordinary family policies selected one consumer");
        if (spins == 1 && !response) {
          require(by_policy[1][0] == by_policy[1][1],
                  "equal-spin family policies duplicated one consumer");
          require(by_policy[1][0] != by_policy[0][0],
                  "equal-spin policy failed to select its physical consumer");
        } else {
          require(by_policy[1][0] == by_policy[0][0] && by_policy[1][1] == by_policy[0][1],
                  "equal-spin policy changed UKS or response routing");
        }
      }
    }
}
void invalid_domains_and_policy() {
  policy("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", "0");
  const auto generic = resolve_point_launcher(1, false);
  policy("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", "1");
  const auto specialized = resolve_point_launcher(1, false);
  require(generic != specialized, "prepared family identities collapsed");
  for (const char* equal : {"0", "1"})
    for (const char* selected : {"0", "1"}) {
      policy("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", equal);
      policy("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", selected);
      for (bool batched : {false, true}) {
        require(execute(1, 1, {0, 0, 0, 0}, false, batched).error == 0, "analytic vacuum rejected");
        for (double density : {-1e-30, std::numeric_limits<double>::infinity()})
          require(execute(1, 1, {density, 0, 0, 0}, false, batched).error != 0,
                  "invalid density accepted");
        require(execute(1, 1, {0, 1e-30, 0, 0}, false, batched).error != 0,
                "normal vacuum gradient accepted");
      }
      require(execute(1, 1, {0, 0, 0, 0}, true, false, 1, {1, 0, 0, 0}).error != 0,
              "undefined vacuum response accepted");
    }
  policy("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", "yes");
  try {
    resolve_point_launcher(1, false);
    throw std::logic_error("invalid family policy accepted");
  } catch (const std::invalid_argument&) {
  }
  policy("GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION", "0");
  policy("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", "yes");
  try {
    execute(1, 1, {.4, 0, 0, 0});
    throw std::logic_error("invalid equal-spin policy accepted");
  } catch (const std::invalid_argument&) {
  }
  require(execute(1, 2, {.2, 0, 0, 0, .3, 0, 0, 0}).error == 0, "RKS policy affected UKS");
  require(execute(1, 1, {.4, 0, 0, 0}, true, false, 1, {0, 0, 0, 0}).error == 0,
          "RKS policy affected response");
  try {
    resolve_point_batch_launcher(1, resolve_point_launcher(0, false));
    throw std::logic_error("mismatched prepared batch accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    resolve_point_launcher(99999, false);
    throw std::logic_error("unsupported functional admitted");
  } catch (const std::invalid_argument&) {
  }
}
int main(int argc, char** argv) {
  try {
    require(argc == 3, "two independent fixture paths required");
    fixtures(argv[1], false);
    fixtures(argv[2], true);
    replay_and_response();
    weighted_response();
    consumer_identity();
    invalid_domains_and_policy();
    std::cout
        << "scoped independent E/V, policy equivalence, response, ragged capacity and rejection "
           "contracts passed\n";
    std::cout
        << "weighted MethodIR response consistency and independent Dirac exchange checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
