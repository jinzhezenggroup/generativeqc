#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "scf/provider_initialization_generation.hpp"

namespace generativeqc::scf {

/** Original reservations are retained labels, not a second charge or voucher. */
struct ProviderOwnerOriginalBudget {
  std::size_t allowance_bytes{};
  std::size_t reference_envelope_bytes{};
  std::size_t endpoint_envelope_bytes{};
  bool operator==(const ProviderOwnerOriginalBudget&) const = default;
};

/** One explicit-close holder. Its destructor is private, so ordinary C++ member
 * destruction cannot start deletion of an unconfirmed physical owner. The C ABI
 * context/calculation destroy adapters must consult the cell BEFORE delete.
 * A failed close retains the entire owner, generation leases and original budget.
 * No global quarantine or context create/select/retain/reset exists here. */
template <class Owner>
class ProviderOwnerRetentionCell {
 private:
  class Holder {
   public:
    Holder(std::shared_ptr<Owner> owner, ProviderOwnerOriginalBudget budget)
        : owner_(std::move(owner)), budget_(budget) {
      if (!owner_ || budget_.allowance_bytes != owner_->generation().declared_allowance())
        throw std::invalid_argument("provider retention lost original allowance");
      owner_->enter_explicit_retirement_protocol();
    }
    Holder(const Holder&) = delete;
    Holder& operator=(const Holder&) = delete;
    bool closed() const noexcept { return !owner_; }
    const ProviderOwnerOriginalBudget& budget() const noexcept { return budget_; }

    template <class Observe>
    ProviderInitializationTeardownReport close_current(Observe&& observe) {
      if (!owner_) return last_;
      // A failed CUDA/vendor call may already have consumed a handle or queued
      // a free. Its unknown side effects cannot be repaired by repeating close.
      // Observation-only refusals occur before this latch and remain retryable
      // when the original context is normally current again.
      if (cleanup_attempted_) return last_;
      const auto schema = owner_->inventory_schema();
      const auto before = owner_->inventory();
      if (!provider_inventory_matches_schema(schema, before))
        return last_ = {{}, "teardown unavailable: concrete owner inventory is incomplete"};
      const auto close = [&] {
        cleanup_attempted_ = true;
        auto raw = owner_->close_current_context();
        return verify_provider_owner_teardown(schema, before, owner_->inventory(), std::move(raw));
      };
      const bool empty = std::none_of(before.resources.begin(), before.resources.end(),
                                      [](const auto& resource) { return resource.token != 0; });
      // An unbound first candidate with a verifiably empty real inventory owns
      // no hardware object. Its close must report not_created, never API zero.
      bool host_only_unbound = empty;
      if constexpr (requires { owner_->unbound_host_only_inventory(); })
        host_only_unbound = owner_->unbound_host_only_inventory();
      if (host_only_unbound && !owner_->generation().has_bound_identity())
        last_ = close();
      else
        last_ = owner_->generation().retire_current(std::forward<Observe>(observe), close);
      if (last_.confirmed()) {
        owner_->mark_teardown_complete();
        owner_.reset();  // Deletion begins only after real full inventory closure.
      }
      return last_;
    }
    static bool delete_if_closed(Holder*& holder) noexcept {
      if (holder && !holder->closed()) return false;
      delete holder;
      holder = nullptr;
      return true;
    }

   private:
    ~Holder() = default;
    std::shared_ptr<Owner> owner_;
    const ProviderOwnerOriginalBudget budget_;
    ProviderInitializationTeardownReport last_;
    bool cleanup_attempted_{};
  };

 public:
  ProviderOwnerRetentionCell() = default;
  ProviderOwnerRetentionCell(const ProviderOwnerRetentionCell&) = delete;
  ProviderOwnerRetentionCell& operator=(const ProviderOwnerRetentionCell&) = delete;
  /** This cell is a manual C-handle lifetime boundary. Destroying the surrounding
   * SDK context is forbidden until destroy_closed() succeeds. It intentionally
   * has no implicit deleter for an open Holder: skipping a dtor body would not
   * retain its automatic members. The live Holder remains an explicit retained
   * allocation, not a fabricated cleanup. Normal ABI adapters must keep this
   * bounded cell reachable on refused void destroy/failed candidate rollback. */
  ~ProviderOwnerRetentionCell() = default;

  bool adopt(std::shared_ptr<Owner> owner, ProviderOwnerOriginalBudget budget) {
    if (holder_ && !holder_->closed()) return false;
    (void)Holder::delete_if_closed(holder_);
    holder_ = new Holder(std::move(owner), budget);
    return true;
  }
  template <class Create>
  bool create(Create&& factory, ProviderOwnerOriginalBudget budget) {
    if (holder_ && !holder_->closed()) return false;
    return adopt(std::forward<Create>(factory)(), budget);
  }
  template <class Observe>
  ProviderInitializationTeardownReport try_close_current(Observe&& observe) {
    if (!holder_) return {{}, "no explicit physical owner in this cell"};
    return holder_->close_current(std::forward<Observe>(observe));
  }
  bool can_destroy() const noexcept { return !holder_ || holder_->closed(); }
  bool destroy_closed() noexcept { return Holder::delete_if_closed(holder_); }
  std::optional<ProviderOwnerOriginalBudget> retained_budget() const noexcept {
    return holder_ && !holder_->closed() ? std::optional{holder_->budget()} : std::nullopt;
  }

 private:
  Holder* holder_{};  // bounded one-slot manual holder; never auto-delete open state
};

}  // namespace generativeqc::scf
