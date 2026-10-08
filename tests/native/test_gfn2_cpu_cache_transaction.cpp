// Exercise the unmodified production cache transaction with a plan-builder stub.
// The injected failure is a real std::vector allocation, not a mocked key copy.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "runtime/types.hpp"

namespace injection {
bool armed = false;
bool copying_plan_key = false;
std::size_t target_bytes = 0;
unsigned failures = 0;

struct PlanCopyScope {
  PlanCopyScope() { copying_plan_key = true; }
  ~PlanCopyScope() { copying_plan_key = false; }
};
}  // namespace injection

void* operator new(std::size_t bytes) {
  if (injection::armed && !injection::copying_plan_key &&
      (injection::target_bytes == 0 || bytes == injection::target_bytes)) {
    injection::armed = false;
    ++injection::failures;
    throw std::bad_alloc();
  }
  if (void* pointer = std::malloc(bytes == 0 ? 1 : bytes)) return pointer;
  throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

constexpr std::int32_t kDefaultMixerHistory = 8;
constexpr double kDefaultMixerDamping = 0.4;
#include "gfn2_cpu_cache_key.inc"

SystemKey copy_plan_key(const SystemKey& key) {
  injection::PlanCopyScope scope;
  return key;
}

struct SystemExecution {
  explicit SystemExecution(const SystemKey& value, int) : key(copy_plan_key(value)) { ++live; }
  ~SystemExecution() { --live; }
  generativeqc_xtb_status_t build(std::string&) {
    ++builds;
    if (throw_build) throw std::bad_alloc();
    return fail_build ? GENERATIVEQC_XTB_STATUS_ALLOCATION_FAILED : GENERATIVEQC_XTB_STATUS_SUCCESS;
  }

  SystemKey key;
  static inline unsigned live = 0;
  static inline unsigned builds = 0;
  static inline bool fail_build = false;
  static inline bool throw_build = false;
};

struct Cache {
  int mulliken_kernels = 0;
  std::vector<SystemKey> keys;
  std::vector<std::unique_ptr<SystemExecution>> systems;
#include "gfn2_cpu_cache_ensure.inc"
};

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

SystemKey make_key(std::initializer_list<std::int32_t> elements) {
  SystemKey key;
  key.atomic_numbers = elements;
  return key;
}

void require_identity(const Cache& cache, const std::vector<SystemKey>& requested) {
  require(cache.keys == requested, "committed cache key differs from request");
  require(cache.systems.size() == requested.size(), "committed system count differs");
  for (std::size_t i = 0; i < requested.size(); ++i) {
    require(cache.systems[i]->key == requested[i],
            "cached plan topology differs from its committed key after recovery");
  }
}

int main(int argc, char** argv) {
  require(argc == 2, "one scenario is required");
  const std::string scenario = argv[1];
  const std::vector<SystemKey> a{make_key({8, 1, 1})};
  const std::vector<SystemKey> b{make_key({6, 1, 1, 1, 1})};
  std::string error;
  {
    Cache cache;
    if (scenario == "first-key-allocation") {
      injection::target_bytes = a[0].atomic_numbers.size() * sizeof(std::int32_t);
      injection::armed = true;
      bool failed = false;
      try {
        (void)cache.ensure_systems(a, error);
      } catch (const std::bad_alloc&) {
        failed = true;
      }
      injection::armed = false;
      require(failed && injection::failures == 1, "first key allocation was not injected");
      require(cache.keys.empty() && cache.systems.empty(),
              "failed first preparation published a partial cache");
    }

    require(cache.ensure_systems(a, error) == GENERATIVEQC_XTB_STATUS_SUCCESS,
            "initial A preparation failed");
    require_identity(cache, a);
    const auto* original = cache.systems[0].get();
    const auto builds = SystemExecution::builds;

    if (scenario == "key-allocation") {
      // Plan construction copies the same key under PlanCopyScope. Only the
      // independent committed-key copy can consume this exact allocation.
      injection::target_bytes = b[0].atomic_numbers.size() * sizeof(std::int32_t);
      injection::armed = true;
      bool failed = false;
      try {
        (void)cache.ensure_systems(b, error);
      } catch (const std::bad_alloc&) {
        failed = true;
      }
      injection::armed = false;
      require(failed && injection::failures == 1, "replacement key allocation was not injected");
      const bool retained = cache.keys == a && cache.systems.size() == 1 &&
                            cache.systems[0].get() == original && cache.systems[0]->key == a[0];
      const auto builds_after_failure = SystemExecution::builds;
      require(cache.ensure_systems(a, error) == GENERATIVEQC_XTB_STATUS_SUCCESS,
              "A recovery after B allocation failure failed");
      require_identity(cache, a);
      require(retained && cache.systems[0].get() == original && SystemExecution::live == 1 &&
                  SystemExecution::builds == builds_after_failure,
              "key allocation failure replaced or leaked the retained A plan");
    } else if (scenario == "plan-build" || scenario == "plan-throw") {
      SystemExecution::fail_build = true;
      SystemExecution::throw_build = scenario == "plan-throw";
      generativeqc_xtb_status_t status = GENERATIVEQC_XTB_STATUS_SUCCESS;
      bool threw = false;
      try {
        status = cache.ensure_systems(b, error);
      } catch (const std::bad_alloc&) {
        threw = true;
      }
      SystemExecution::fail_build = false;
      SystemExecution::throw_build = false;
      require(scenario == "plan-throw"
                  ? threw
                  : !threw && status == GENERATIVEQC_XTB_STATUS_ALLOCATION_FAILED,
              "plan build failure was not propagated");
      require_identity(cache, a);
      require(cache.systems[0].get() == original && SystemExecution::live == 1,
              "plan build failure replaced or leaked the retained A plan");
    } else if (scenario == "reuse") {
      injection::target_bytes = 0;
      injection::armed = true;
      require(cache.ensure_systems(a, error) == GENERATIVEQC_XTB_STATUS_SUCCESS,
              "unchanged A preparation failed");
      injection::armed = false;
      require(injection::failures == 0 && SystemExecution::builds == builds &&
                  cache.systems[0].get() == original,
              "unchanged topology allocated, rebuilt or replaced its plan");
    } else {
      require(scenario == "first-key-allocation", "unknown scenario");
    }

    require(cache.ensure_systems(b, error) == GENERATIVEQC_XTB_STATUS_SUCCESS, "B retry failed");
    require_identity(cache, b);
    require(cache.ensure_systems(a, error) == GENERATIVEQC_XTB_STATUS_SUCCESS,
            "normal B to A transition failed");
    require_identity(cache, a);
    require(SystemExecution::live == 1, "replaced plans were not released");
  }
  require(SystemExecution::live == 0, "cache destruction leaked a plan");
  std::cout << "PASS " << scenario << " (production cache transaction; plan builder is a stub)\n";
}
