#include <cstring>
#include <memory>
#include <thread>

#include "api/handles.hpp"
#include "generativeqc/warm_seed.h"
#include "methods/batch_capabilities.hpp"
#include "native_test_require.hpp"
#include "runtime/resource_ledger.hpp"

namespace {
using namespace generativeqc;

struct WithoutSeeds : methods::PreparedBatch {
  std::size_t size() const noexcept override { return 2; }
  std::vector<methods::BatchItemResult> execute_impl(const methods::Coordinates&, bool) override {
    throw std::runtime_error("snapshot transport executed science");
  }
};

/** Failures exercise publication/ownership boundaries, not implementation spelling. */
struct WithSeeds : WithoutSeeds, methods::BatchWarmStarts {
  std::size_t dimensions[2]{4, 18};
  std::vector<std::optional<scf::HfWarmState>> states{
      scf::HfWarmState{{1, 0, 0, 1}, {0, 0, -0.7, 0, 0, 0.7}, -1, 1e-12, 1e-9, 8}, std::nullopt};
  bool forbid_export{}, fail_second_export{}, reject_restore{};
  std::size_t validation_allowance{};
  std::size_t warm_density_size(std::size_t i) const override { return dimensions[i]; }
  const std::optional<scf::HfWarmState>& warm_state(std::size_t i) const override {
    if (forbid_export) throw std::runtime_error("unadmitted seed materialization");
    if (fail_second_export && i == 1) throw std::bad_alloc();
    return states.at(i);
  }
  void clear_warm_starts() override { states.assign(2, std::nullopt); }
  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>> value) override {
    // A shape-only provider lease exercises real admission without a GPU.
    std::optional<runtime::DeviceProviderAllowance> allowance;
    if (validation_allowance) allowance.emplace(0, validation_allowance);
    if (reject_restore) throw std::invalid_argument("physical seed validation failed");
    for (std::size_t i = 0; i < value.size(); ++i)
      if (value[i]) states[i].swap(value[i]);
  }
  void set_warm_start_updates(bool) override {
    throw std::runtime_error("unexpected policy change");
  }
};

using Snapshot = std::unique_ptr<generativeqc_warm_seed_snapshot,
                                 decltype(&generativeqc_warm_seed_snapshot_destroy)>;
}  // namespace

int main() {
  Snapshot snapshot(nullptr, generativeqc_warm_seed_snapshot_destroy);
  {
    generativeqc_context context;
    generativeqc_batch source;
    source.context = &context;
    source.atom_counts = {2, 3};
    source.flags = GENERATIVEQC_BATCH_ENABLE_WARM_STARTS;
    auto plan = std::make_unique<WithSeeds>();
    auto* owner = plan.get();
    source.plan = std::move(plan);
    generativeqc_warm_seed_snapshot_info_v1 info{sizeof(info), GENERATIVEQC_ABI_VERSION};
    info.numeric_bytes = 991;
    const auto before = info;
    auto* sentinel = reinterpret_cast<generativeqc_warm_seed_snapshot*>(std::uintptr_t{1});
    auto* output = sentinel;
    owner->forbid_export = true;
    require_test(generativeqc_batch_capture_warm_seeds_v1(&source, 295, &info, &output) ==
                 GENERATIVEQC_STATUS_OUT_OF_MEMORY);
    require_test(output == sentinel && std::memcmp(&info, &before, sizeof(info)) == 0);
    owner->forbid_export = false;
    owner->fail_second_export = true;
    require_test(generativeqc_batch_capture_warm_seeds_v1(&source, 296, &info, &output) ==
                 GENERATIVEQC_STATUS_OUT_OF_MEMORY);
    require_test(output == sentinel && std::memcmp(&info, &before, sizeof(info)) == 0);
    owner->fail_second_export = false;
    require_test(generativeqc_batch_capture_warm_seeds_v1(&source, 296, &info, &output) == 0);
    snapshot.reset(output);
    require_test(info.system_count == 2 && info.numeric_bytes == 80 &&
                 info.largest_density_bytes == 32);
    owner->states[0]->density[0] = 9;
    owner->clear_warm_starts();
    // A cold capture still requires the conservative all-slot envelope, while
    // its actual retained numeric receipt correctly reports zero.
    generativeqc_warm_seed_snapshot* cold{};
    require_test(generativeqc_batch_capture_warm_seeds_v1(&source, 296, &info, &cold) == 0);
    require_test(info.numeric_bytes == 0 && info.largest_density_bytes == 0);
    generativeqc_warm_seed_snapshot_destroy(cold);
    info.abi_version += 1;
    require_test(generativeqc_batch_capture_warm_seeds_v1(&source, 296, &info, &output) ==
                 GENERATIVEQC_STATUS_ABI_MISMATCH);
    info.abi_version -= 1;
    source.plan = std::make_unique<WithoutSeeds>();
    require_test(generativeqc_batch_capture_warm_seeds_v1(&source, 296, &info, &output) ==
                 GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
  }
  // Source plan and context are gone. Read-only views are independently owned.
  generativeqc_warm_seed_view_v1 view{sizeof(view), GENERATIVEQC_ABI_VERSION};
  require_test(generativeqc_warm_seed_snapshot_read_v1(snapshot.get(), 0, &view) == 0);
  require_test(view.present && view.density_count == 4 && view.coordinate_count == 6);
  require_test(view.density[0] == 1 && view.coordinates[2] == -0.7 && view.energy == -1);
  const auto before = view;
  require_test(generativeqc_warm_seed_snapshot_read_v1(snapshot.get(), 2, &view) ==
               GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  require_test(std::memcmp(&view, &before, sizeof(view)) == 0);
  require_test(generativeqc_warm_seed_snapshot_read_v1(snapshot.get(), 1, &view) == 0);
  require_test(!view.present && !view.density && !view.coordinates && !view.density_count);
  auto reader = [&] {
    for (int i = 0; i != 64; ++i) {
      generativeqc_warm_seed_view_v1 local{sizeof(local), GENERATIVEQC_ABI_VERSION};
      require_test(generativeqc_warm_seed_snapshot_read_v1(snapshot.get(), 0, &local) == 0);
      require_test(local.density[0] == 1);
    }
  };
  std::thread first(reader), second(reader);
  first.join();
  second.join();

  generativeqc_context context;
  generativeqc_batch target;
  target.context = &context;
  target.atom_counts = {2, 3};
  target.flags = GENERATIVEQC_BATCH_ENABLE_WARM_STARTS;
  auto plan = std::make_unique<WithSeeds>();
  auto* owner = plan.get();
  owner->states[0]->density[0] = 7;
  target.plan = std::move(plan);
  owner->dimensions[0] = 9;
  require_test(generativeqc_batch_restore_warm_seeds_v1(&target, snapshot.get()) ==
               GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  require_test(owner->states[0]->density[0] == 7);
  owner->dimensions[0] = 4;
  owner->reject_restore = true;
  require_test(generativeqc_batch_restore_warm_seeds_v1(&target, snapshot.get()) ==
               GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  require_test(owner->states[0]->density[0] == 7);
  owner->reject_restore = false;
  target.resource_ledger = std::make_shared<runtime::DeviceResourceLedger>();
  target.resource_ledger->limit = 15;
  owner->validation_allowance = 16;
  require_test(generativeqc_batch_restore_warm_seeds_v1(&target, snapshot.get()) ==
               GENERATIVEQC_STATUS_OUT_OF_MEMORY);
  require_test(owner->states[0]->density[0] == 7 && target.resource_ledger->rejected > 0);
  require_test(target.resource_ledger->provider_reserved == 0 && !target.resource_ledger->active);
  target.resource_ledger->limit = 16;
  for (int replay = 0; replay != 2; ++replay) {
    require_test(generativeqc_batch_restore_warm_seeds_v1(&target, snapshot.get()) == 0);
    require_test(owner->states[0]->density[0] == 1);
    require_test(target.resource_ledger->provider_reservation_peak == 16);
    require_test(target.resource_ledger->provider_reserved == 0 && !target.resource_ledger->active);
    owner->clear_warm_starts();
  }
  require_test(generativeqc_batch_restore_warm_seeds_v1(nullptr, snapshot.get()) ==
               GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  generativeqc_warm_seed_snapshot_destroy(nullptr);
}
