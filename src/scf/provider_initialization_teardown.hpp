#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace generativeqc::scf {

inline constexpr std::size_t kProviderOwnerInventoryLimit = 64;

enum class ProviderTeardownStepState {
  unavailable,
  not_created,
  returned,
  completed_without_status
};

struct ProviderInitializationTeardownStep {
  std::string operation;
  ProviderTeardownStepState state{ProviderTeardownStepState::unavailable};
  std::optional<std::uint32_t> status;
};

enum class ProviderOwnedResourceKind {
  stream,
  status_resource,
  device_buffer,
  host_buffer,
  host_object,
  event
};

/** Populated by the concrete owner from its real member state, including every
 * known slot that was never created. A nonzero token marks a created slot whose
 * retirement is not proven; an error may leave its physical state unknown. */
struct ProviderOwnedResource {
  std::string slot;
  ProviderOwnedResourceKind kind{};
  std::uintptr_t token{};
  bool operator==(const ProviderOwnedResource&) const = default;
};

struct ProviderOwnerInventory {
  std::vector<ProviderOwnedResource> resources;
};

struct ProviderResourceSlotSchema {
  std::string slot;
  ProviderOwnedResourceKind kind{};
};

struct ProviderOwnerSchema {
  std::vector<ProviderResourceSlotSchema> slots;
};

/** Additional real resources closed under the Resources measurement lock.
 * Callbacks must not acquire that same nonrecursive mutex again. */
class ProviderOwnerCloseExtension {
 public:
  virtual ~ProviderOwnerCloseExtension() = default;
  virtual ProviderOwnerSchema extension_schema() const = 0;
  virtual ProviderOwnerInventory extension_inventory() const = 0;
  virtual ProviderInitializationTeardownStep close_extension_operation(const std::string&) = 0;
};

inline bool provider_inventory_matches_schema(const ProviderOwnerSchema& schema,
                                              const ProviderOwnerInventory& inventory) {
  if (schema.slots.empty() || schema.slots.size() != inventory.resources.size()) return false;
  for (const auto& expected : schema.slots) {
    if (expected.slot.empty() || expected.slot.find('/') != std::string::npos ||
        std::count_if(schema.slots.begin(), schema.slots.end(),
                      [&](const auto& slot) { return slot.slot == expected.slot; }) != 1)
      return false;
    const auto count = std::count_if(
        inventory.resources.begin(), inventory.resources.end(),
        [&](const auto& item) { return item.slot == expected.slot && item.kind == expected.kind; });
    if (count != 1) return false;
  }
  return true;
}

struct ProviderTeardownRequirement {
  std::string operation;
  bool created{};
  bool returns_status{};
};

inline std::vector<ProviderTeardownRequirement> provider_teardown_plan(
    const ProviderOwnerInventory& inventory) {
  if (inventory.resources.empty() || inventory.resources.size() > kProviderOwnerInventoryLimit)
    throw std::invalid_argument("invalid provider owner inventory size");
  std::vector<std::string> slots;
  const ProviderOwnedResource* stream = nullptr;
  for (const auto& resource : inventory.resources) {
    if (resource.slot.empty() || resource.slot.find('/') != std::string::npos ||
        std::find(slots.begin(), slots.end(), resource.slot) != slots.end())
      throw std::invalid_argument("duplicate or invalid provider owner inventory slot");
    slots.push_back(resource.slot);
    if (resource.kind == ProviderOwnedResourceKind::stream) {
      if (stream) throw std::invalid_argument("provider owner has multiple owning streams");
      stream = &resource;
    }
  }
  if (!stream) throw std::invalid_argument("provider owner inventory omitted its stream slot");
  std::vector<ProviderTeardownRequirement> result;
  result.push_back({stream->slot + "/drain", stream->token != 0, true});
  for (const auto& resource : inventory.resources) {
    if (resource.kind == ProviderOwnedResourceKind::stream) continue;
    if (resource.kind == ProviderOwnedResourceKind::device_buffer && resource.token &&
        !stream->token)
      throw std::invalid_argument("owned async buffer has no owning stream");
    if (resource.kind == ProviderOwnedResourceKind::event)
      result.push_back({resource.slot + "/drain", resource.token != 0, true});
    const bool status = resource.kind == ProviderOwnedResourceKind::status_resource ||
                        resource.kind == ProviderOwnedResourceKind::device_buffer ||
                        resource.kind == ProviderOwnedResourceKind::event;
    result.push_back({resource.slot + "/release", resource.token != 0, status});
  }
  result.push_back({stream->slot + "/release_fence", stream->token != 0, true});
  result.push_back({stream->slot + "/release", stream->token != 0, true});
  return result;
}

class ProviderInitializationTeardownReport;
ProviderInitializationTeardownReport verify_provider_owner_teardown(
    const ProviderOwnerSchema&, const ProviderOwnerInventory&, const ProviderOwnerInventory&,
    ProviderInitializationTeardownReport);

/** Raw operation success and whole-owner closure are separate claims. A raw
 * nonempty list of zero statuses is never a complete-owner receipt. */
class ProviderInitializationTeardownReport {
 public:
  std::vector<ProviderInitializationTeardownStep> steps;
  std::string detail;
  ProviderInitializationTeardownReport() = default;
  ProviderInitializationTeardownReport(std::vector<ProviderInitializationTeardownStep> values,
                                       std::string description)
      : steps(std::move(values)), detail(std::move(description)) {}
  bool reported_steps_succeeded() const noexcept {
    if (steps.empty()) return false;
    for (const auto& step : steps) {
      if (step.state == ProviderTeardownStepState::returned) {
        if (!step.status || *step.status != 0) return false;
      } else if (step.state == ProviderTeardownStepState::not_created ||
                 step.state == ProviderTeardownStepState::completed_without_status) {
        if (step.status) return false;
      } else
        return false;
    }
    return true;
  }
  bool confirmed() const noexcept {
    if (!schema_ || !before_ || !after_) return false;
    try {
      if (!provider_inventory_matches_schema(*schema_, *before_) ||
          !provider_inventory_matches_schema(*schema_, *after_))
        return false;
      const auto expected = provider_teardown_plan(*before_);
      (void)provider_teardown_plan(*after_);
      if (before_->resources.size() != after_->resources.size() || steps.size() != expected.size())
        return false;
      for (const auto& original : before_->resources) {
        const auto found =
            std::find_if(after_->resources.begin(), after_->resources.end(),
                         [&](const auto& item) { return item.slot == original.slot; });
        if (found == after_->resources.end() || found->kind != original.kind || found->token)
          return false;
      }
      for (std::size_t index = 0; index < expected.size(); ++index) {
        const auto& requirement = expected[index];
        const auto& step = steps[index];
        if (step.operation != requirement.operation) return false;
        if (!requirement.created) {
          if (step.state != ProviderTeardownStepState::not_created || step.status) return false;
        } else if (requirement.returns_status) {
          if (step.state != ProviderTeardownStepState::returned || !step.status || *step.status)
            return false;
        } else if (step.state != ProviderTeardownStepState::completed_without_status || step.status)
          return false;
      }
      return true;
    } catch (...) {
      return false;
    }
  }

 private:
  friend ProviderInitializationTeardownReport verify_provider_owner_teardown(
      const ProviderOwnerSchema&, const ProviderOwnerInventory&, const ProviderOwnerInventory&,
      ProviderInitializationTeardownReport);
  std::optional<ProviderOwnerSchema> schema_;
  std::optional<ProviderOwnerInventory> before_, after_;
};

/** Describe the first observed raw API failure without assigning a vendor/
 * runtime domain the generic report does not know. Missing status stays unknown. */
inline std::string provider_teardown_failure_detail(
    const ProviderInitializationTeardownReport& report) {
  for (const auto& step : report.steps) {
    if (step.state == ProviderTeardownStepState::returned && step.status && *step.status != 0)
      return "operation=" + step.operation +
             "; state=returned; raw_status=" + std::to_string(*step.status);
  }
  for (const auto& step : report.steps) {
    if (step.state == ProviderTeardownStepState::unavailable)
      return "operation=" + step.operation + "; state=unavailable; raw_status=UNAVAILABLE";
    if (step.state == ProviderTeardownStepState::returned && !step.status)
      return "operation=" + step.operation + "; state=returned; raw_status=UNAVAILABLE";
  }
  return {};
}

inline ProviderInitializationTeardownReport verify_provider_owner_teardown(
    const ProviderOwnerSchema& schema, const ProviderOwnerInventory& before,
    const ProviderOwnerInventory& after, ProviderInitializationTeardownReport report) {
  report.schema_ = schema;
  report.before_ = before;
  report.after_ = after;
  if (!report.confirmed()) {
    const auto failure = provider_teardown_failure_detail(report);
    if (!failure.empty() && report.detail.find(failure) == std::string::npos) {
      if (!report.detail.empty()) report.detail += "; ";
      report.detail += failure;
    } else if (report.detail.empty()) {
      report.detail = "whole-owner teardown unavailable: inventory/operation coverage incomplete";
    }
  }
  return report;
}

}  // namespace generativeqc::scf
