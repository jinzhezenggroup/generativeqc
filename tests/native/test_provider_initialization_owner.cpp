#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "scf/provider_initialization_generation.hpp"

#if __has_include("scf/provider_initialization_owner.hpp")
#include "scf/provider_initialization_owner.hpp"
#endif

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
}  // namespace

#if __has_include("scf/provider_initialization_owner.hpp")
namespace {
using namespace generativeqc;
using namespace generativeqc::scf;
int destroyed{}, destroyed_live{}, calls{};
runtime::CudaProviderIdentityObservation observation() {
  runtime::CudaProviderIdentity identity;
  identity.provider = runtime::CudaProviderKind::Nvidia;
  identity.providers = 3;
  identity.context = {3, 41, {}, 13020, 12090};
  identity.build_ids = {"aa", "bb", "cc", "dd"};
  runtime::CudaProviderIdentityObservation result;
  result.status = runtime::ContextIdentityStatus::available;
  result.loaded_status = runtime::LoadedIdentityStatus::available;
  result.identity = identity;
  for (auto& pin : result.library_pins) pin = std::make_shared<int>(1);
  return result;
}
struct WholeOwner {
  PendingProviderInitializationGeneration binding{ProviderInitializationIntent(512ULL << 20),
                                                  {true, true},
                                                  ProviderInitializationEigenFamily::xsyevd,
                                                  3};
  ProviderOwnerInventory state{{{"stream", ProviderOwnedResourceKind::stream, 1},
                                {"vendor", ProviderOwnedResourceKind::status_resource, 2},
                                {"arena", ProviderOwnedResourceKind::device_buffer, 3},
                                {"workspace", ProviderOwnedResourceKind::device_buffer, 4},
                                {"host", ProviderOwnedResourceKind::host_buffer, 5},
                                {"unused", ProviderOwnedResourceKind::device_buffer, 0}}};
  bool fail_drain{}, omit_operation{}, managed{}, closed{};
  ~WholeOwner() {
    ++destroyed;
    for (const auto& resource : state.resources)
      if (resource.token) ++destroyed_live;
  }
  PendingProviderInitializationGeneration& generation() { return binding; }
  ProviderOwnerSchema inventory_schema() const {
    return {{{"stream", ProviderOwnedResourceKind::stream},
             {"vendor", ProviderOwnedResourceKind::status_resource},
             {"arena", ProviderOwnedResourceKind::device_buffer},
             {"workspace", ProviderOwnedResourceKind::device_buffer},
             {"host", ProviderOwnedResourceKind::host_buffer},
             {"unused", ProviderOwnedResourceKind::device_buffer}}};
  }
  ProviderOwnerInventory inventory() const { return state; }
  void enter_explicit_retirement_protocol() { managed = true; }
  void mark_teardown_complete() { closed = true; }
  ProviderInitializationTeardownReport close_current_context() {
    ProviderInitializationTeardownReport report;
    const std::vector<std::string> actions{
        "stream/drain", "vendor/release", "arena/release",        "workspace/release",
        "host/release", "unused/release", "stream/release_fence", "stream/release"};
    bool stopped = false;
    for (const auto& action : actions) {
      const auto slot = action.substr(0, action.find('/'));
      auto& resource = *std::find_if(state.resources.begin(), state.resources.end(),
                                     [&](const auto& item) { return item.slot == slot; });
      if (!resource.token) {
        report.steps.push_back({action, ProviderTeardownStepState::not_created, {}});
      } else if (stopped) {
        report.steps.push_back({action, ProviderTeardownStepState::unavailable, {}});
      } else {
        ++calls;
        if (action == "stream/drain" && fail_drain) {
          report.steps.push_back({action, ProviderTeardownStepState::returned, 700});
          stopped = true;
        } else if (slot == "host") {
          report.steps.push_back({action, ProviderTeardownStepState::completed_without_status, {}});
          resource.token = 0;
        } else {
          report.steps.push_back({action, ProviderTeardownStepState::returned, 0});
          if (action.ends_with("/release")) resource.token = 0;
        }
      }
    }
    if (omit_operation) report.steps.erase(report.steps.begin() + 1);
    return report;
  }
};

int run_owner(const std::string& mode) {
  if (mode == "duplicate-schema") {
    WholeOwner owner;
    const auto before = owner.inventory();
    auto raw = owner.close_current_context();
    auto schema = owner.inventory_schema();
    schema.slots.back() = schema.slots.front();
    require(!verify_provider_owner_teardown(schema, before, owner.inventory(), raw).confirmed(),
            "duplicate schema hid an omitted resource slot");
    return 0;
  }
  if (mode == "inventory") {
    WholeOwner owner;
    const auto before = owner.inventory();
    auto raw = owner.close_current_context();
    const auto after = owner.inventory();
    auto valid = verify_provider_owner_teardown(owner.inventory_schema(), before, after, raw);
    require(valid.confirmed(), "complete inventory close refused");
    auto missing = raw;
    missing.steps.erase(missing.steps.begin() + 1);
    require(!verify_provider_owner_teardown(owner.inventory_schema(), before, after, missing)
                 .confirmed(),
            "missing operation confirmed");
    auto duplicate = raw;
    duplicate.steps.push_back(duplicate.steps.front());
    require(!verify_provider_owner_teardown(owner.inventory_schema(), before, after, duplicate)
                 .confirmed(),
            "duplicate operation confirmed");
    auto wrong_absence = raw;
    wrong_absence.steps[2] = {"arena/release", ProviderTeardownStepState::not_created, {}};
    require(!verify_provider_owner_teardown(owner.inventory_schema(), before, after, wrong_absence)
                 .confirmed(),
            "created arena was declared not-created");
    auto wrong_statusless = raw;
    wrong_statusless.steps[4] = {"host/release", ProviderTeardownStepState::returned, 0};
    require(
        !verify_provider_owner_teardown(owner.inventory_schema(), before, after, wrong_statusless)
             .confirmed(),
        "status-less free was fabricated as API status zero");
    auto missing_inventory = before;
    missing_inventory.resources.pop_back();
    require(!verify_provider_owner_teardown(owner.inventory_schema(), missing_inventory, after, raw)
                 .confirmed(),
            "incomplete owner inventory confirmed");
    return 0;
  }
  auto current = observation();
  auto owner = std::make_shared<WholeOwner>();
  owner->binding.bind_current([&] { return current; });
  std::array<std::weak_ptr<void>, 4> pins;
  for (unsigned i = 0; i < 4; ++i) pins[i] = current.library_pins[i];
  current.library_pins = {};
  const auto observe = [&] {
    auto result = current;
    for (unsigned i = 0; i < 4; ++i) result.library_pins[i] = pins[i].lock();
    return result;
  };
  std::weak_ptr<WholeOwner> weak = owner;
  ProviderOwnerRetentionCell<WholeOwner> cell;
  const ProviderOwnerOriginalBudget budget{512ULL << 20, 1ULL << 30, 2ULL << 30};
  require(cell.adopt(owner, budget), "owner was not retained");
  owner.reset();
  require(!weak.expired() && !cell.can_destroy() &&
              cell.retained_budget()->allowance_bytes == budget.allowance_bytes,
          "retention lost whole owner or original budget before close");
  if (mode == "uncertain-close-retry") {
    weak.lock()->fail_drain = true;
    auto failed = cell.try_close_current(observe);
    require(!failed.confirmed() && failed.steps.front().status == 700,
            "injected cleanup failure was not preserved");
    const auto previous_calls = calls;
    weak.lock()->fail_drain = false;
    auto retried = cell.try_close_current(observe);
    require(!retried.confirmed() && calls == previous_calls && !weak.expired() &&
                !pins[1].expired() && !cell.can_destroy() && cell.retained_budget() == budget,
            "unconfirmed physical close was retried or original custody was released");
    return 0;
  }
  if (mode == "foreign-owner") {
    current.identity->context.context_id = 42;
    const auto result = cell.try_close_current(observe);
    require(!result.confirmed() && result.steps.empty() && calls == 0 && destroyed == 0 &&
                !pins[1].expired() && !cell.can_destroy(),
            "foreign-context deletion/cleanup occurred");
    int new_generations = 0;
    require(!cell.create(
                [&] {
                  ++new_generations;
                  return std::make_shared<WholeOwner>();
                },
                budget) &&
                new_generations == 0,
            "new generation replaced retained owner");
    require(cell.retained_budget()->allowance_bytes == budget.allowance_bytes,
            "refusal released selected A");
    current.identity->context.context_id = 41;
  } else if (mode == "failed-close") {
    weak.lock()->fail_drain = true;
    auto result = cell.try_close_current(observe);
    require(!result.confirmed() && result.steps.front().status == 700 && !cell.can_destroy() &&
                !weak.expired() && !pins[1].expired(),
            "failed close destroyed owner or repaired status");
    const auto calls_before = calls;
    weak.lock()->fail_drain = false;
    auto refused = cell.try_close_current(observe);
    require(!refused.confirmed() && refused.steps.front().status == 700 && calls == calls_before &&
                !weak.expired() && cell.retained_budget() == budget,
            "failed close was retried or its original budget was released");
    return 0;
  } else if (mode == "missing-operation") {
    weak.lock()->omit_operation = true;
    auto result = cell.try_close_current(observe);
    require(!result.confirmed() && !weak.expired() && !cell.can_destroy(),
            "partial report deleted owner");
    const auto calls_before = calls;
    weak.lock()->omit_operation = false;
    auto refused = cell.try_close_current(observe);
    require(!refused.confirmed() && calls == calls_before && !weak.expired() &&
                !pins[1].expired() && cell.retained_budget() == budget,
            "incomplete receipt was silently retried or released");
    return 0;
  } else if (mode != "owner")
    throw std::runtime_error("unknown owner mode");
  auto result = cell.try_close_current(observe);
  require(result.confirmed() && weak.expired() && cell.can_destroy() && destroyed_live == 0,
          "complete close failed or automatic members were destroyed live");
  require(!cell.retained_budget() && cell.destroy_closed(), "closed cell did not release custody");
  return 0;
}
}  // namespace
#endif

int main(int argc, char** argv) {
  using namespace generativeqc::scf;
  try {
    require(argc == 2, "one owner probe mode required");
    const std::string mode = argv[1];
    if (mode == "partial-report") {
      ProviderInitializationTeardownReport report{
          {{"vendor", ProviderTeardownStepState::returned, 0}}, {}};
      require(!report.confirmed(),
              "partial reported success falsely confirmed whole-owner teardown");
      return 0;
    }
#if __has_include("scf/provider_initialization_owner.hpp")
    return run_owner(mode);
#else
    std::cerr << "concrete provider owner retention port is unavailable\n";
    return 7;
#endif
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
