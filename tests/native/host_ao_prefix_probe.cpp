#include <cfenv>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <string>

#include "dft/host_ao_prefix.hpp"
#include "molecule/basis.hpp"

namespace {
bool fail_numeric_allocation{};
}

// Inject a real failure of the optional numeric allocation, not its planner.
void* operator new[](std::size_t bytes) {
  if (fail_numeric_allocation) {
    fail_numeric_allocation = false;
    throw std::bad_alloc();
  }
  if (auto* allocation = std::malloc(bytes ? bytes : 1)) return allocation;
  throw std::bad_alloc();
}
void operator delete[](void* allocation) noexcept { std::free(allocation); }
void operator delete[](void* allocation, std::size_t) noexcept { std::free(allocation); }

namespace {
using namespace generativeqc;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Error = std::invalid_argument, class Operation>
void rejects(Operation operation) {
  bool rejected{};
  try {
    operation();
  } catch (const Error&) {
    rejected = true;
  }
  require(rejected, "invalid operation was accepted");
}

core::System system_for() {
  core::System system;
  system.atoms = {{1, {0.1, -0.2, 0.3}, 0}, {1, {1.7, 0.2, -0.4}, 0}};
  system.shells = {{0, 0, {{1.2, 1}, {.35, .4}}}, {1, 1, {{.8, 1}}}};
  std::string detail;
  if (molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::invalid_argument(detail);
  return system;
}

dft::GridSpec small_grid() {
  dft::GridSpec spec;
  spec.radial_points = 3;
  spec.angular_polar = 3;
  spec.angular_azimuth = 4;
  return spec;
}

struct Fixture {
  core::System system{system_for()};
  dft::AoBasis basis{system};
  dft::MolecularGrid grid{system, small_grid()};
  static constexpr std::size_t tile = 7;

  std::size_t bytes(std::size_t points) const {
    return (basis.packed.size() + points * (4 * basis.nao + 3)) * sizeof(double);
  }

  /** Independent call to the unchanged native AO evaluator, byte comparison. */
  void parity(dft::HostAoPrefix& cache, std::size_t begin, std::size_t count) const {
    std::vector<double> expected(4 * count * basis.nao), actual(expected.size()), weights(count);
    std::fenv_t environment{};
    require(std::fegetenv(&environment) == 0, "cannot read floating environment");
    if (count)
      basis.evaluate(grid.points().data() + 3 * begin, count, 1, 0, basis.nao, expected.data(),
                     expected.size());
    const auto expected_flags = std::fetestexcept(FE_ALL_EXCEPT);
    require(std::fesetenv(&environment) == 0, "cannot restore floating environment");
    cache.read(begin, count, actual, weights);
    require(std::fetestexcept(FE_ALL_EXCEPT) == expected_flags, "floating flags changed");
    require(actual.empty() || std::memcmp(actual.data(), expected.data(), actual.size() * 8) == 0,
            "AO bytes changed");
    require(std::equal(weights.begin(), weights.end(), grid.weights().begin() + begin),
            "weights are stale");
  }
};

void check(const std::string& scenario) {
  Fixture frame;
  dft::HostAoPrefix cache(frame.basis, frame.grid, Fixture::tile);
  const auto total = frame.grid.point_count();
  if (scenario == "capacity") {
    require(cache.admit(0) == 0 && cache.admit(frame.bytes(6)) == 0, "short cap admitted");
    require(cache.admit(frame.bytes(10)) == frame.bytes(7), "partial tile admitted");
    require(cache.prefix_points() == 7 && cache.admit(frame.bytes(total)) == 0,
            "admitted prefix grew");
    dft::HostAoPrefix full(frame.basis, frame.grid, Fixture::tile);
    require(full.admit(frame.bytes(total)) == frame.bytes(total), "full cap not admitted");
    for (std::size_t begin = 0; begin < total; begin += Fixture::tile) {
      const auto count = std::min(Fixture::tile, total - begin);
      frame.parity(full, begin, count);
      frame.parity(full, begin, count);
    }
    require(full.cached_points() > 0, "full prefix never replayed");
  } else if (scenario == "tails") {
    cache.admit(frame.bytes(14));
    for (int trial = 0; trial < 3; ++trial)
      for (std::size_t begin = 0; begin < total; begin += Fixture::tile)
        frame.parity(cache, begin, std::min(Fixture::tile, total - begin));
    require(cache.cached_points() > 0 && cache.evaluated_points() > total,
            "prefix/tail work census is wrong");
    require(cache.cached_points() + cache.evaluated_points() == 3 * total,
            "point work disappeared");
  } else if (scenario == "streaming") {
    frame.parity(cache, 0, 7);
    frame.parity(cache, 0, 7);
    require(
        cache.numeric_bytes() == 0 && cache.evaluated_points() == 14 && cache.cached_points() == 0,
        "streaming unexpectedly cached");
  } else if (scenario == "allocation") {
    fail_numeric_allocation = true;
    require(cache.admit(frame.bytes(total)) == 0 && !fail_numeric_allocation,
            "numeric allocation failure not exercised");
    require(cache.numeric_bytes() == 0 && cache.prefix_points() == 0, "failed cache published");
    frame.parity(cache, 0, 7);
  } else if (scenario == "basis" || scenario == "points" || scenario == "weights") {
    cache.admit(frame.bytes(14));
    frame.parity(cache, 0, 7);
    auto& value = scenario == "basis" ? frame.basis.packed[0]
                  : scenario == "points"
                      ? const_cast<std::vector<double>&>(frame.grid.points())[0]
                      : const_cast<std::vector<double>&>(frame.grid.weights())[0];
    value += .125;
    const auto evaluated = cache.evaluated_points();
    frame.parity(cache, 0, 7);
    if (scenario != "weights")
      require(cache.evaluated_points() == evaluated + 7, "mutated input replayed");
  } else if (scenario == "environment") {
    cache.admit(frame.bytes(14));
    std::fenv_t original{};
    std::fegetenv(&original);
    std::feclearexcept(FE_ALL_EXCEPT);
    frame.parity(cache, 0, 7);
    std::feclearexcept(FE_ALL_EXCEPT);
    frame.parity(cache, 0, 7);
    require(cache.cached_points() == 7, "same environment did not replay");
    require(std::feraiseexcept(FE_DIVBYZERO) == 0, "exception control unavailable");
    const auto before_flags = cache.evaluated_points();
    frame.parity(cache, 0, 7);
    require(cache.evaluated_points() == before_flags + 7, "changed flags replayed");
    require(std::fesetround(FE_DOWNWARD) == 0, "rounding control unavailable");
    const auto evaluated = cache.evaluated_points();
    frame.parity(cache, 0, 7);
    require(cache.evaluated_points() == evaluated + 7, "changed rounding replayed");
    std::fesetenv(&original);
  } else if (scenario == "failure") {
    cache.admit(frame.bytes(14));
    auto& point = const_cast<std::vector<double>&>(frame.grid.points())[0];
    const auto saved = point;
    point = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> ao(4 * 7 * frame.basis.nao), weights(7);
    rejects([&] { cache.read(0, 7, ao, weights); });
    require(cache.evaluated_points() == 0 && cache.cached_points() == 0,
            "failed evaluation committed");
    point = saved;
    frame.parity(cache, 0, 7);
    frame.parity(cache, 0, 7);
    require(cache.evaluated_points() >= 7, "failed entry incorrectly replayed");
  } else if (scenario == "nonfinite") {
    frame.basis.packed[3 * frame.basis.natom + 1] = std::numeric_limits<double>::quiet_NaN();
    dft::HostAoPrefix invalid(frame.basis, frame.grid, Fixture::tile);
    invalid.admit(frame.bytes(14));
    std::vector<double> ao(4 * 7 * frame.basis.nao), weights(7);
    rejects<std::runtime_error>([&] { invalid.read(0, 7, ao, weights); });
    rejects<std::runtime_error>([&] { invalid.read(0, 7, ao, weights); });
    require(invalid.cached_points() == 0 && invalid.evaluated_points() == 0,
            "nonfinite evaluation committed or replayed");
  } else if (scenario == "shape") {
    cache.admit(frame.bytes(14));
    for (auto* dimension : {&frame.basis.nao, &frame.basis.natom, &frame.basis.nprimitive}) {
      const auto saved = *dimension;
      *dimension = 0;
      rejects([&] { cache.read(total, 0, {}, {}); });
      *dimension = saved;
    }
    frame.basis.packed.push_back(0);
    rejects([&] { cache.read(0, 0, {}, {}); });
    frame.basis.packed.pop_back();
    auto& points = const_cast<std::vector<double>&>(frame.grid.points());
    points.pop_back();
    rejects([&] { cache.read(0, 0, {}, {}); });
    points.push_back(0);
    auto& weights = const_cast<std::vector<double>&>(frame.grid.weights());
    weights.pop_back();
    rejects([&] { cache.read(0, 0, {}, {}); });
    weights.push_back(0);
    const auto saved = frame.basis.nao;
    frame.basis.nao = std::numeric_limits<std::size_t>::max();
    rejects([&] { dft::HostAoPrefix invalid(frame.basis, frame.grid, 7); });
    frame.basis.nao = saved;
    rejects([&] { dft::HostAoPrefix invalid(frame.basis, frame.grid, 0); });
  } else if (scenario == "empty") {
    cache.read(total, 0, {}, {});
    rejects<std::logic_error>([&] { cache.admit(frame.bytes(total)); });
    rejects([&] { cache.read(total + 1, 0, {}, {}); });
    rejects([&] { cache.read(0, std::numeric_limits<std::size_t>::max(), {}, {}); });
    rejects([&] { cache.read(0, 1, {}, {}); });
  } else if (scenario == "unaligned") {
    cache.admit(frame.bytes(total));
    frame.parity(cache, 1, 3);
    frame.parity(cache, 1, 3);
    frame.parity(cache, 0, 3);
    require(cache.cached_points() == 0 && cache.evaluated_points() == 9,
            "noncanonical tile replayed");
  } else {
    throw std::invalid_argument("unknown AO-prefix scenario");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("expected scenario");
    check(argv[1]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
