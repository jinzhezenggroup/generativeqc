#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <utility>

namespace generativeqc::runtime {

struct HostResourceSnapshot {
  std::uint64_t maximum_bytes;
  std::uint64_t retained_bytes;
  std::uint64_t workspace_bytes;
  std::uint64_t owner_count;
};

/** Modeled host capacity, independent of an observer's lifetime. Owners are
 * admitted against the sum of their retained data and largest scratch need.
 * The workspace lock makes the maximum valid even across separate contexts.
 */
class HostResourceLedger {
 public:
  explicit HostResourceLedger(std::uint64_t maximum_bytes) : maximum_bytes_(maximum_bytes) {}

  std::unique_lock<std::mutex> use_workspace() { return std::unique_lock(workspace_mutex_); }

  HostResourceSnapshot snapshot() const {
    std::lock_guard lock(state_mutex_);
    return {maximum_bytes_, retained_bytes_, largest_workspace(), workspaces_.size()};
  }

 private:
  friend class HostResourceLease;
  using Workspaces = std::multiset<std::uint64_t>;

  std::uint64_t largest_workspace() const noexcept {
    return workspaces_.empty() ? 0 : *workspaces_.rbegin();
  }

  const std::uint64_t maximum_bytes_;
  std::mutex workspace_mutex_;
  mutable std::mutex state_mutex_;
  std::uint64_t retained_bytes_{};
  Workspaces workspaces_;
};

/** A scientific owner's reservation. Destroy it only after destroying the
 * retained providers. An unsuccessful reservation changes no ledger state.
 */
class HostResourceLease {
 public:
  HostResourceLease(std::shared_ptr<HostResourceLedger> ledger, std::uint64_t retained_bytes,
                    std::uint64_t workspace_bytes)
      : ledger_(std::move(ledger)), retained_bytes_(retained_bytes) {
    std::lock_guard lock(ledger_->state_mutex_);
    const auto available = ledger_->maximum_bytes_ - ledger_->retained_bytes_;
    const auto current_workspace = ledger_->largest_workspace();
    if (retained_bytes > available || current_workspace > available - retained_bytes ||
        workspace_bytes > available - retained_bytes)
      throw std::bad_alloc();
    workspace_ = ledger_->workspaces_.insert(workspace_bytes);
    ledger_->retained_bytes_ += retained_bytes;
  }

  HostResourceLease(const HostResourceLease&) = delete;
  HostResourceLease& operator=(const HostResourceLease&) = delete;

  ~HostResourceLease() {
    std::lock_guard lock(ledger_->state_mutex_);
    ledger_->workspaces_.erase(workspace_);
    ledger_->retained_bytes_ -= retained_bytes_;
  }

  std::unique_lock<std::mutex> use_workspace() const { return ledger_->use_workspace(); }

 private:
  std::shared_ptr<HostResourceLedger> ledger_;
  std::uint64_t retained_bytes_;
  HostResourceLedger::Workspaces::iterator workspace_;
};

}  // namespace generativeqc::runtime
