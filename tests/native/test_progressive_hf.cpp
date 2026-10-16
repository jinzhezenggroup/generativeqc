#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "scf/progressive_hf.hpp"

extern "C" generativeqc_status generativeqc_progressive_hf_run_v1(
    generativeqc::scf::ProgressiveHFCallback, void*, std::uint64_t*, std::size_t);

namespace {
using Event = generativeqc::scf::ProgressiveHFEvent;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Run {
  std::vector<Event> events;
  std::uint32_t source{1}, target{3}, retained{1};
  int failure{-1}, malformed{-1};
};

int callback(void* opaque, std::uint32_t operation, std::uint32_t* output) {
  auto& state = *static_cast<Run*>(opaque);
  const auto event = static_cast<Event>(operation);
  state.events.push_back(event);
  if (static_cast<int>(operation) == state.failure) return 1;
  *output = event == Event::execute_source   ? state.source
            : event == Event::execute_target ? state.target
            : event == Event::retain_target  ? state.retained
                                             : 0;
  if (static_cast<int>(operation) == state.malformed) *output = 32;
  return 0;
}

bool contains(const Run& run, Event event) {
  for (const auto seen : run.events)
    if (seen == event) return true;
  return false;
}

void test_schedules() {
  for (const auto source : {0u, 1u}) {
    for (const auto target : {0u, 1u, 3u}) {
      for (const auto retained : {0u, 1u}) {
        Run run{{}, source, target, retained};
        std::array<std::uint64_t, 2> receipt{99, 98};
        require(generativeqc_progressive_hf_run_v1(callback, &run, receipt.data(), 2) == 0,
                "valid schedule failed");
        std::vector<Event> expected{Event::prepare_source, Event::execute_source,
                                    Event::prepare_target};
        if (source) expected.push_back(Event::project);
        expected.push_back(Event::execute_target);
        if (target == 3) expected.push_back(Event::retain_target);
        expected.push_back(Event::close_target);
        expected.push_back(Event::close_source);
        if (target == 3 && retained) expected.push_back(Event::audit_target);
        require(run.events == expected, "source replaced target or audit preceded cleanup");
        std::uint64_t mask{};
        for (const auto event : expected) mask |= std::uint64_t{1} << static_cast<unsigned>(event);
        require(receipt[0] == mask, "completed-operation receipt is inconsistent");
      }
    }
  }
}

void test_failure_cleanup() {
  for (int failure = 0; failure != 9; ++failure) {
    for (const auto malformed : {false, true}) {
      Run run;
      if (malformed)
        run.malformed = failure;
      else
        run.failure = failure;
      std::array<std::uint64_t, 2> receipt{};
      const auto status = generativeqc_progressive_hf_run_v1(callback, &run, receipt.data(), 2);
      require(status != 0, "failed callback was silently accepted");
      require(contains(run, Event::close_source) == (failure != 0 || malformed),
              "source lifetime was lost");
      require(contains(run, Event::close_target) == (failure > 2 || (failure == 2 && malformed)),
              "target lifetime was lost");
      require(!contains(run, Event::audit_target) || failure == 8,
              "failure admitted a physical audit");
      if (contains(run, Event::close_target))
        require(run.events.back() == Event::close_source || failure == 8,
                "cleanup did not close source last");
    }
  }
  Run run;
  run.target = 2;
  std::array<std::uint64_t, 2> receipt{};
  require(generativeqc_progressive_hf_run_v1(callback, &run, receipt.data(), 2) != 0 &&
              !contains(run, Event::retain_target) && contains(run, Event::close_source),
          "convergence without target success was admitted");
}

void test_transactional_arguments_and_recovery() {
  Run run;
  std::array<std::uint64_t, 2> receipt{91, 92};
  require(generativeqc_progressive_hf_run_v1(nullptr, &run, receipt.data(), 2) != 0 &&
              receipt == std::array<std::uint64_t, 2>{91, 92},
          "null callback changed publication");
  require(generativeqc_progressive_hf_run_v1(callback, &run, receipt.data(), 1) != 0 &&
              receipt == std::array<std::uint64_t, 2>{91, 92} && run.events.empty(),
          "invalid count executed work or changed publication");
  require(generativeqc_progressive_hf_run_v1(callback, &run, nullptr, 2) != 0 && run.events.empty(),
          "null receipt executed work");
  for (int round = 0; round != 1000; ++round) {
    run = Run{};
    run.failure = 4;
    require(generativeqc_progressive_hf_run_v1(callback, &run, receipt.data(), 2) != 0,
            "injected failure succeeded");
    run = Run{};
    require(generativeqc_progressive_hf_run_v1(callback, &run, receipt.data(), 2) == 0 &&
                run.events.back() == Event::audit_target,
            "prior failure contaminated a fresh controller");
  }
}
}  // namespace

int main() {
  try {
    test_schedules();
    test_failure_cleanup();
    test_transactional_arguments_and_recovery();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
