#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

#include "api/handles.hpp"
#include "generativeqc/warm_seed.h"
#include "methods/batch_capabilities.hpp"
#include "native_test_require.hpp"

namespace {
using namespace generativeqc;

/** Fault-injected prepared owners prove metadata queries cannot export seeds. */
struct WithoutSeeds : methods::PreparedBatch {
  std::size_t size() const noexcept override { return 2; }
  std::vector<methods::BatchItemResult> execute_impl(const methods::Coordinates&, bool) override {
    throw std::runtime_error("capacity query executed science");
  }
};

struct WithSeeds : WithoutSeeds, methods::BatchWarmStarts {
  std::size_t dimensions[2]{4, 18};
  std::size_t warm_density_size(std::size_t index) const override { return dimensions[index]; }
  const std::optional<scf::HfWarmState>& warm_state(std::size_t) const override {
    throw std::runtime_error("capacity query materialized a seed");
  }
  void clear_warm_starts() override { throw std::runtime_error("seed mutation"); }
  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>>) override {
    throw std::runtime_error("seed mutation");
  }
  void set_warm_start_updates(bool) override { throw std::runtime_error("seed mutation"); }
};
}  // namespace

int main() {
  generativeqc_context context;
  generativeqc_batch batch;
  batch.context = &context;
  batch.atom_counts = {2, 3};
  auto plan = std::make_unique<WithSeeds>();
  auto* owner = plan.get();
  batch.plan = std::move(plan);
  generativeqc_warm_seed_capacity_v1 output{sizeof(output), GENERATIVEQC_ABI_VERSION};
  const auto query = [&] {
    return generativeqc_batch_query_warm_seed_capacity_v1(&batch, &output);
  };
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS);
  require_test(output.density_bytes == 176 && output.coordinate_bytes == 120 &&
               output.total_bytes == 296 && output.largest_density_bytes == 144);
  const auto before = output;
  owner->dimensions[1] = std::numeric_limits<std::size_t>::max();
  require_test(query() == GENERATIVEQC_STATUS_OUT_OF_MEMORY);
  require_test(std::memcmp(&before, &output, sizeof(output)) == 0);
  // Each item fits, but the aggregate byte count must not wrap.
  owner->dimensions[0] = owner->dimensions[1] = std::numeric_limits<std::size_t>::max() / 8;
  require_test(query() == GENERATIVEQC_STATUS_OUT_OF_MEMORY);
  require_test(std::memcmp(&before, &output, sizeof(output)) == 0);
  batch.plan = std::make_unique<WithoutSeeds>();
  require_test(query() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
  require_test(std::memcmp(&before, &output, sizeof(output)) == 0);
  output.abi_version += 1;
  require_test(query() == GENERATIVEQC_STATUS_ABI_MISMATCH);
  require_test(generativeqc_batch_query_warm_seed_capacity_v1(nullptr, &output) ==
               GENERATIVEQC_STATUS_INVALID_ARGUMENT);
}
