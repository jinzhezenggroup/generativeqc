#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "runtime/resource_planning.hpp"

extern "C" int generativeqc_resource_compose_v1(const std::uint64_t*, std::size_t,
                                                const std::uint64_t*, std::size_t,
                                                const std::uint64_t*, std::size_t,
                                                const std::uint64_t*, std::size_t, std::size_t,
                                                double, std::uint64_t, std::uint64_t*, std::size_t);

namespace {
using namespace generativeqc::runtime::resources;
constexpr auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Callback>
void rejects(Callback callback) {
  bool rejected = false;
  try {
    callback();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "invalid resource metadata was accepted");
}

void test_headroom_and_lifetimes() {
  require(headroom_bytes(100, 0.125) == 13, "fractional headroom rounding changed");
  require(headroom_bytes(maximum, 0.5) == (std::uint64_t{1} << 62),
          "int64 headroom rounded through double");
  require(headroom_bytes(maximum, std::numeric_limits<double>::denorm_min()) == 1,
          "subnormal headroom was lost");
  for (double fraction : {-1.0, 1.0, std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()})
    rejects([&] { (void)headroom_bytes(1, fraction); });
  rejects([] { (void)headroom_bytes(maximum + 1, 0); });

  const std::vector<Buffer> buffers{{maximum, kPageableSpace, 0, 0, true},
                                    {maximum, kPageableSpace, 1, 1, false},
                                    {10, kPinnedSpace, maximum, maximum, true},
                                    {4, kFirstDeviceSpace, maximum, maximum, true},
                                    {6, kFirstDeviceSpace + 1, maximum, maximum, false}};
  const auto result = account(buffers, 6);
  require(result.peaks[kHostSpace].bytes == maximum && result.peaks[kHostSpace].first_phase == 0,
          "adjacent maximum buffers should not overlap or change earliest peak");
  require(
      result.peaks[kDeviceSpace].bytes == 10 && result.peaks[kDeviceSpace].first_phase == maximum,
      "sparse final phase lost device overlap");
  require(result.resident[kDeviceSpace].bytes == 4 && !result.resident[5].present,
          "workspace was incorrectly retained");
  rejects([] {
    (void)account(std::array<Buffer, 2>{{{maximum, kPageableSpace, 0, 0, true},
                                         {1, kPinnedSpace, 0, 0, true}}},
                  4);
  });
  rejects([] { (void)account(std::array<Buffer, 1>{{{1, kHostSpace, 0, 0, true}}}, 4); });
}

void test_coupled_choice_and_wide_objectives() {
  const std::vector<Request> requests{{{{0, 1, {{60, 3, 0, 0, true}, {80, 4, 0, 0, true}}},
                                        {1, 0, {{90, 3, 0, 0, true}, {20, 4, 0, 0, true}}}}},
                                      {{{0, 0, {{10, 3, 0, 0, true}, {40, 4, 0, 0, true}}}}}};
  const auto selected = compose(requests, std::array<Limit, 2>{{{0, 100, 0}, {1, 60, 0}}}, 5, 0, 2);
  require(
      selected.status == Status::feasible && selected.selections == std::vector<std::size_t>{1, 0},
      "coupled host/device caps selected a greedy unsupported combination");
  require(selected.accounting.peaks[0].bytes == 100 && selected.accounting.peaks[1].bytes == 60,
          "exact coupled limits changed");
  const auto shortage = compose(requests, std::array<Limit, 2>{{{0, 99, 0}, {1, 59, 0}}}, 5, 0, 2);
  require(shortage.status == Status::infeasible && shortage.selections == selected.selections,
          "below-limit closest combination changed");
  require(compose(requests, {}, 5, 0, 1).status == Status::search_limit,
          "finite product guard changed");

  const std::vector<Request> costly{{{{1, 1, {{1, 3, 0, 0, true}}}, {2, 0, {{1, 3, 0, 0, true}}}}},
                                    {{{maximum, 0, {{1, 3, 0, 0, true}}}}},
                                    {{{maximum, 0, {{1, 3, 0, 0, true}}}}}};
  const auto wide = compose(costly, {}, 4, 0, 2);
  require(wide.selections.front() == 0, "cost sum wrapped uint64 or lexical rank overrode cost");
  auto tied = costly;
  tied.front().candidates[1].relative_cost = 1;
  require(compose(tied, {}, 4, 0, 2).selections.front() == 1,
          "equal-cost names were not ordered lexically");
}

void test_wire_transaction() {
  std::vector<std::uint64_t> requests{1}, candidates{0, 0, 1}, buffers{64, 3, 0, 0, 1},
      limits{0, 64, 0};
  std::vector<std::uint64_t> output(35, 99);
  const auto query = [&](std::uint64_t* target, std::size_t count) {
    return generativeqc_resource_compose_v1(
        requests.data(), requests.size(), candidates.data(), candidates.size(), buffers.data(),
        buffers.size(), limits.data(), limits.size(), 4, 0, 65536, target, count);
  };
  require(query(output.data(), output.size()) == 0 && output[0] == 0 && output[1] == 1 &&
              output[3] == 64 && output[4] == 1,
          "valid ABI composition changed");
  const auto unchanged = [&](auto callback) {
    std::fill(output.begin(), output.end(), 99);
    const auto previous = output;
    require(callback() != 0 && output == previous, "failed ABI query partially published output");
  };
  buffers.back() = 2;
  unchanged([&] { return query(output.data(), output.size()); });
  buffers.back() = 1;
  unchanged([&] { return query(output.data(), output.size() - 1); });
  unchanged([&] { return query(nullptr, output.size()); });
  limits.insert(limits.end(), {0, 65, 0});
  unchanged([&] { return query(output.data(), output.size()); });
  limits.resize(3);
  unchanged([&] {
    return generativeqc_resource_compose_v1(
        requests.data(), 1, candidates.data(), std::numeric_limits<std::size_t>::max() / 3 * 3,
        buffers.data(), buffers.size(), limits.data(), limits.size(), 4, 0, 65536, output.data(),
        output.size());
  });
  output[0] = 1;
  const auto alias_original = output;
  require(generativeqc_resource_compose_v1(output.data(), 1, candidates.data(), 3, buffers.data(),
                                           5, limits.data(), 3, 4, 0, 65536, output.data(),
                                           output.size()) != 0 &&
              output == alias_original,
          "aliased buffers were modified");
  std::vector<std::uint64_t> adjacent(3 + output.size(), 99);
  std::copy(candidates.begin(), candidates.end(), adjacent.begin());
  require(generativeqc_resource_compose_v1(requests.data(), 1, adjacent.data(), 3, buffers.data(),
                                           5, limits.data(), 3, 4, 0, 65536, adjacent.data() + 3,
                                           output.size()) == 0 &&
              adjacent[0] == 0 && adjacent[1] == 0 && adjacent[2] == 1,
          "adjacent disjoint buffers rejected");
  requests = {2};
  candidates = {0, 0, 1, 1, 1, 2};
  buffers = {64, 3, 0, 0, 1, maximum, 3, 0, 0, 1, 1, 3, 0, 0, 1};
  unchanged([&] { return query(output.data(), output.size()); });
}
}  // namespace

int main() {
  try {
    test_headroom_and_lifetimes();
    test_coupled_choice_and_wide_objectives();
    test_wire_transaction();
    std::cout
        << "Native resource composition, lifetimes, exact budgets and transactional ABI passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
