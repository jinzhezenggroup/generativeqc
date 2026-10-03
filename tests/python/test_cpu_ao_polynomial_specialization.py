"""New AO reconstruction gates, independently identified from the lost candidate.

Compile the actual production constructor/evaluator and a frozen baseline from
9a5871dca1f371e5f6e9fcf9d6b076c83b894c14 into one small probe. Renaming baseline
symbols is the only transformation. Bitwise equality (NaN classification only)
protects operation order; a separate long-double Leibniz gate protects the math.
These standalone checks do not qualify complete DFT endpoints or performance.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/reference_data/cpu_ao_reconstruction"
FROZEN_HASHES = {
    "ao_grid_9a5871dc.cpp.txt": "fd38ffd99024f231f141ed8c6b27e5f1922551e81cee7ddb2845b8024d3a0eda",
    "ao_grid_9a5871dc.hpp.txt": "32f7216ba35c77468418cf61d23d6014c4d20f843b5a59bdf1549126d2b87e7f",
}

PROBE = r"""
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "dft/ao_grid.cpp"
#include "frozen.cpp"

using namespace generativeqc;
[[noreturn]] void fail(const std::string& message) {
  std::cerr << message << '\n';
  std::exit(1);
}
void require(bool condition, const std::string& message) {
  if (!condition) fail(message);  // Assertions must survive -DNDEBUG.
}
bool equivalent(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}
void compare(double actual, double expected, const std::string& message) {
  if (!equivalent(actual, expected)) {
    std::cerr << std::hexfloat << actual << " != " << expected << '\n';
    fail(message);
  }
}
void compare_vectors(const std::vector<double>& a, const std::vector<double>& b,
                     const std::string& message) {
  require(a.size() == b.size(), message + " size");
  for (std::size_t i = 0; i < a.size(); ++i) compare(a[i], b[i], message + "[" + std::to_string(i) + "]");
}

std::uint64_t random_bits(std::uint64_t& state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}
void axes() {
  const double denorm = std::numeric_limits<double>::denorm_min();
  const double minimum = std::numeric_limits<double>::min();
  const double maximum = std::numeric_limits<double>::max();
  const std::vector<double> alphas = {0.0, -0.0, denorm, minimum, 1e-100, .13, .5, 1.0,
                                    1.7, 2.0, 1e100, maximum / 4, maximum / 2, maximum};
  const std::vector<double> coordinates = {-maximum, -1e150, -1e50, -9.0, -1.0, -.5,
                                         -minimum, -denorm, -0.0, 0.0, denorm, minimum,
                                         .5, 1.0, 9.0, 1e50, 1e150, maximum};
  std::size_t comparisons = 0;
  auto check = [&](unsigned l, unsigned derivative, double alpha, double x) {
    const double actual = dft::differentiated_power(l, derivative, alpha, x);
    const double expected = dft::frozen_differentiated_power(l, derivative, alpha, x);
    if (!equivalent(actual, expected)) {
      std::cerr << "l=" << l << " derivative=" << derivative << std::hexfloat
                << " alpha=" << alpha << " x=" << x << '\n';
      compare(actual, expected, "axis operation-order mismatch");
    }
    ++comparisons;
  };
  // All legal public shapes, plus every bounded internal fallback shape. No
  // out-of-domain l + derivative > 6 call is made into the original buffer.
  for (unsigned l = 0; l <= 6; ++l)
    for (unsigned derivative = 0; derivative + l <= 6; ++derivative) {
      for (double alpha : alphas)
        for (double x : coordinates) check(l, derivative, alpha, x);
      std::uint64_t state = 0x8197752bc1bd6307ULL + 11 * l + derivative;
      for (unsigned repeat = 0; repeat < 1024; ++repeat) {
        // Random finite values spanning subnormal through near-overflow. The
        // exponent mask excludes NaN/Inf inputs, but not any finite exponent.
        const auto abits = random_bits(state) & 0x7fffffffffffffffULL;
        const auto xbits = random_bits(state);
        const double alpha = std::bit_cast<double>(abits);
        const double x = std::bit_cast<double>(xbits);
        if (std::isfinite(alpha) && std::isfinite(x)) check(l, derivative, alpha, x);
      }
    }
  // This case catches deleting a nominally zero coefficient operation.
  require(std::isnan(dft::differentiated_power(1, 1, maximum, 0.0)),
          "Inf * zero behavior was removed");
  std::cout << "axis_comparisons=" << comparisons << '\n';
}

long double leibniz(unsigned l, unsigned d, long double alpha, long double x) {
  const long double h[] = {1, -2 * alpha * x, 4 * alpha * alpha * x * x - 2 * alpha,
                          -8 * alpha * alpha * alpha * x * x * x + 12 * alpha * alpha * x};
  const unsigned choose[4][4] = {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 2, 1, 0}, {1, 3, 3, 1}};
  long double value = 0;
  for (unsigned k = 0; k <= std::min(l, d); ++k) {
    long double falling = 1;
    for (unsigned i = 0; i < k; ++i) falling *= l - i;
    value += choose[d][k] * falling * std::pow(x, l - k) * h[d - k];
  }
  return value;
}
void oracle() {
  for (unsigned l = 0; l <= 3; ++l)
    for (unsigned d = 0; d <= 3; ++d)
      for (double alpha : {0.0001, 0.13, 0.5, 1.0, 1.7, 8.0, 100.0})
        for (double x : {-4.0, -1.0, -0.5, -0.0001, -0.0, 0.0, 0.0001, 0.5, 1.0, 4.0}) {
          const double actual = dft::differentiated_power(l, d, alpha, x);
          const long double expected = leibniz(l, d, alpha, x);
          require(std::isfinite(actual), "nonfinite moderate axis");
          require(std::abs(actual - expected) <= 1e-12L + 5e-13L * std::abs(expected),
                  "independent long-double Leibniz mismatch");
        }
}
core::System system_fixture(bool spherical) {
  core::System system;
  system.atoms = {{8, {0.0, 0.0, 0.0}}, {8, {0.3, -0.2, 0.1}}};
  system.basis_representation = spherical ? GENERATIVEQC_BASIS_SPHERICAL : GENERATIVEQC_BASIS_CARTESIAN;
  for (unsigned atom = 0; atom < 2; ++atom)
    for (unsigned l = 0; l <= 3; ++l)
      system.shells.push_back({atom, l, {{1.7, .4}, {.51, -.17}, {.13, .8}, {1.0, -0.0}}});
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "fixture normalization: " + detail);
  return system;
}
struct Outcome {
  std::vector<double> output;
  std::string exception;
};
template <class Basis>
Outcome evaluate(const Basis& basis, const double* points, std::size_t npoint,
                 unsigned order, std::size_t begin, std::size_t count,
                 std::size_t elements, const std::size_t* ids, bool null_output = false) {
  Outcome result{std::vector<double>(elements + 2, 987654.125), {}};
  try {
    basis.evaluate(points, npoint, order, begin, count,
                   null_output ? nullptr : result.output.data() + 1, elements, ids);
  } catch (const std::invalid_argument& error) {
    result.exception = "invalid_argument: " + std::string(error.what());
  } catch (const std::runtime_error& error) {
    result.exception = "runtime_error: " + std::string(error.what());
  } catch (const std::logic_error& error) {
    result.exception = "logic_error: " + std::string(error.what());
  }
  compare(result.output.front(), 987654.125, "left output canary");
  compare(result.output.back(), 987654.125, "right output canary");
  return result;
}
void compare_outcomes(const Outcome& actual, const Outcome& expected) {
  require(actual.exception == expected.exception, "exception mismatch: " + actual.exception + " / " + expected.exception);
  compare_vectors(actual.output, expected.output, "AO output");
}
void whole(bool spherical, unsigned order, unsigned mode) {
  const auto system = system_fixture(spherical);
  const dft::AoBasis actual(system);
  const dft::FrozenAoBasis expected(system);
  compare_vectors(actual.packed, expected.packed, "constructed packed basis");
  const auto saved = actual.packed;
  require(actual.nao == expected.nao && actual.natom == expected.natom &&
              actual.nprimitive == expected.nprimitive, "basis dimensions");
  std::vector<double> points = {0.0, -0.0, 0.0, .1, -.2, .3, 1.1, .7, -.9,
                                -2.1, .5, 1.7, .3, -.2, .1};
  const auto saved_points = points;
  std::vector<std::size_t> ids = {0, 3, 8, actual.nao - 1};
  std::size_t begin = 0, count = actual.nao;
  const std::size_t* selected = nullptr;
  if (mode == 1) { begin = 1; count = actual.nao - 2; }
  if (mode == 2) { selected = ids.data(); count = ids.size(); }
  if (mode == 3) points.clear();
  if (mode == 4) { count = 0; selected = ids.data(); }
  const std::size_t npoint = points.size() / 3;
  const std::size_t elements = (order + 1) * (order + 2) * (order + 3) / 6 * npoint * count;
  const auto a = evaluate(actual, points.data(), npoint, order, begin, count, elements, selected);
  const auto b = evaluate(expected, points.data(), npoint, order, begin, count, elements, selected);
  require(a.exception.empty(), "ordinary evaluation rejected: " + a.exception);
  compare_outcomes(a, b);
  compare_vectors(actual.packed, saved, "input basis changed");
  if (mode != 3) compare_vectors(points, saved_points, "points changed");
}
void invalid() {
  const auto system = system_fixture(true);
  const dft::AoBasis actual(system);
  const dft::FrozenAoBasis expected(system);
  for (unsigned mode = 0; mode < 10; ++mode) {
    std::vector<double> points = {.1, -.2, .3};
    std::vector<std::size_t> ids = {0, 3};
    const std::size_t* selected = nullptr;
    unsigned order = 1;
    std::size_t begin = 0, count = 2, elements = 8;
    if (mode == 0) order = 4;
    if (mode == 1) begin = actual.nao + 1;
    if (mode == 2) count = actual.nao + 1;
    if (mode == 3) ++elements;
    if (mode == 6) points[1] = std::numeric_limits<double>::quiet_NaN();
    if (mode >= 7) {
      selected = ids.data();
      if (mode == 7) ids = {1, 1};
      if (mode == 8) ids = {2, 1};
      if (mode == 9) ids = {0, actual.nao};
      selected = ids.data();
    }
    const auto a = evaluate(actual, mode == 4 ? nullptr : points.data(), 1, order,
                            begin, count, elements, selected, mode == 5);
    const auto b = evaluate(expected, mode == 4 ? nullptr : points.data(), 1, order,
                            begin, count, elements, selected, mode == 5);
    require(!a.exception.empty(), "invalid input unexpectedly accepted");
    compare_outcomes(a, b);
  }
  for (unsigned mode = 0; mode < 2; ++mode) {
    auto bad = system;
    if (mode == 0) bad.shells[0].angular_momentum = 4;
    else bad.shells[0].primitives[0].exponent = std::numeric_limits<double>::infinity();
    auto check = [&]<class Basis>() {
      try { const Basis basis(bad); }
      catch (const std::invalid_argument& e) { return std::string(e.what()); }
      return std::string();
    };
    const auto a = check.template operator()<dft::AoBasis>();
    const auto b = check.template operator()<dft::FrozenAoBasis>();
    require(!a.empty() && a == b, "constructor rejection changed");
  }
}
void extremes() {
  const double maximum = std::numeric_limits<double>::max();
  for (unsigned mode = 0; mode < 6; ++mode) {
    core::System system;
    system.atoms = {{1, {0, 0, 0}}};
    system.shells = {{0, 0, {{1.0, 1.0}}}, {0, 3, {{maximum, mode == 2 ? -0.0 : 1.0}}}};
    // AoBasis expects normalized finite coefficients; these deliberate edge
    // fixtures bypass normalization to isolate its established radial guard.
    const dft::AoBasis actual(system);
    const dft::FrozenAoBasis expected(system);
    std::vector<double> points = {0.0, 0.0, 0.0};
    if (mode == 0) points = {1e150, 1e150, 1e150};
    if (mode == 4) points = {1.0, 1.0, 1.0};
    if (mode == 5) points = {std::numeric_limits<double>::denorm_min(), 0.0, 0.0};
    const std::size_t selected_id = 0;
    const std::size_t* selected = mode == 3 ? &selected_id : nullptr;
    const std::size_t count = selected ? 1 : actual.nao;
    const auto a = evaluate(actual, points.data(), 1, 3, 0, count, 20 * count, selected);
    const auto b = evaluate(expected, points.data(), 1, 3, 0, count, 20 * count, selected);
    compare_outcomes(a, b);
    if (mode == 1 || mode == 5)
      require(a.exception == "runtime_error: nonfinite AO jet result", "nonfinite edge was not rejected");
    else
      require(a.exception.empty(), "radial-zero/omitted-AO guard moved: " + a.exception);
    if (mode == 0)
      for (std::size_t i = 1; i + 1 < a.output.size(); ++i) compare(a.output[i], 0.0, "underflow output");
  }
}
int main(int argc, char** argv) {
  require(argc >= 2, "missing mode");
  const std::string mode = argv[1];
  if (mode == "axes") axes();
  else if (mode == "oracle") oracle();
  else if (mode == "invalid") invalid();
  else if (mode == "extremes") extremes();
  else if (mode == "whole") {
    require(argc == 5, "whole requires representation/order/selection");
    whole(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]));
  } else fail("unknown mode");
}
"""


def test_frozen_baseline_bytes() -> None:
    for name, expected in FROZEN_HASHES.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == expected


@pytest.fixture(scope="module", params=["production-o2", "production-o3", "strict-o3"])
def specialization_probe(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    ccache = shutil.which("ccache")
    if not ccache:
        pytest.fail(
            "AO qualification requires ccache; install/activate it before compiling"
        )
    subprocess.run([ccache, "--version"], check=True, capture_output=True, text=True)
    compiler = os.environ.get("AO_PROBE_CXX", "c++")
    if not shutil.which(compiler):
        pytest.fail(f"AO qualification requires a C++20 compiler: {compiler}")
    directory = tmp_path_factory.mktemp(f"ao-specialization-{request.param}")
    test_frozen_baseline_bytes()
    header = (
        (FIXTURE / "ao_grid_9a5871dc.hpp.txt")
        .read_text()
        .replace("AoBasis", "FrozenAoBasis")
    )
    frozen = (FIXTURE / "ao_grid_9a5871dc.cpp.txt").read_text()
    frozen = frozen.replace('"dft/ao_grid.hpp"', '"frozen.hpp"')
    for original, renamed in [
        ("AoBasis", "FrozenAoBasis"),
        ("differentiated_power", "frozen_differentiated_power"),
        ("multiply", "frozen_multiply"),
    ]:
        frozen = frozen.replace(original, renamed)
    (directory / "frozen.hpp").write_text(header)
    (directory / "frozen.cpp").write_text(frozen)
    cpp, executable = directory / "probe.cpp", directory / "probe"
    cpp.write_text(PROBE)
    flags = {
        "production-o2": ["-O2", "-DNDEBUG"],
        "production-o3": ["-O3", "-DNDEBUG", "-fPIC"],
        "strict-o3": ["-O3", "-ffp-contract=off"],
    }[request.param]
    # Compile one translation unit per ccache call; invoking ccache with a
    # multi-source compile/link command would silently make every run uncacheable.
    objects = []
    for source in (cpp, ROOT / "src/molecule/basis.cpp"):
        obj = directory / f"{source.stem}.o"
        command = [
            ccache,
            compiler,
            "-std=c++20",
            *flags,
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
            "-c",
            str(source),
            "-o",
            str(obj),
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=90, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(obj)
    result = subprocess.run(
        [compiler, *(str(obj) for obj in objects), "-o", str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("mode", ["axes", "oracle", "invalid", "extremes"])
def test_polynomial_and_failure_gates(specialization_probe: Path, mode: str) -> None:
    subprocess.run([str(specialization_probe), mode], check=True, timeout=20)


@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("order", range(4))
@pytest.mark.parametrize("selection", range(5))
def test_whole_ao_bitwise(
    specialization_probe: Path, spherical: bool, order: int, selection: int
) -> None:
    subprocess.run(
        [
            str(specialization_probe),
            "whole",
            str(int(spherical)),
            str(order),
            str(selection),
        ],
        check=True,
        timeout=20,
    )
