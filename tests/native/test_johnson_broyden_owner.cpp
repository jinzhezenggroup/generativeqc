// Standalone contract test: includes only the method-independent CPU owner.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "solver/cpu/johnson_broyden.hpp"

namespace {
std::size_t allocations = 0;
bool count_allocations = false;
void* allocate(std::size_t size) {
  if (count_allocations) ++allocations;
  if (void* pointer = std::malloc(size ? size : 1)) return pointer;
  throw std::bad_alloc{};
}
void* allocate_aligned(std::size_t size, std::size_t alignment) {
  if (count_allocations) ++allocations;
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size ? size : alignment) == 0) return pointer;
  throw std::bad_alloc{};
}
}  // namespace
void* operator new(std::size_t size) { return allocate(size); }
void* operator new[](std::size_t size) { return allocate(size); }
void* operator new(std::size_t size, std::align_val_t alignment) {
  return allocate_aligned(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return allocate_aligned(size, static_cast<std::size_t>(alignment));
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}

namespace {
using namespace generativeqc::solver::cpu;
using generativeqc::solver::BroydenPolicy;
using generativeqc::solver::BroydenResult;
using generativeqc::solver::BroydenStatusEncoding;
constexpr auto success = BroydenResult::success;
constexpr auto invalid = BroydenResult::invalid_argument;
constexpr auto numerical = BroydenResult::numerical_failure;
constexpr std::size_t capacity = 4;
constexpr std::array<std::int64_t, 4> packed_offsets{0, 3, 7, 12};
constexpr std::array<std::array<std::int64_t, 4>, 2> field_offsets{{{0, 2, 3, 6}, {0, 1, 4, 6}}};
constexpr BroydenStatusEncoding encoding{7, -3, 41};

struct Aligned {
  const std::size_t size;
  void* data;
  explicit Aligned(std::size_t bytes) : size(bytes), data(allocate_aligned(bytes, 64)) {
    std::memset(data, 0, size);
  }
  ~Aligned() { std::free(data); }
  Aligned(const Aligned&) = delete;
};
using Bytes = std::vector<unsigned char>;
Bytes snapshot(const Aligned& buffer) {
  const auto* first = static_cast<const unsigned char*>(buffer.data);
  return {first, first + buffer.size};
}
void unchanged(const Bytes& before, const Aligned& buffer) {
  assert(before.size() == buffer.size);
  assert(std::memcmp(before.data(), buffer.data, buffer.size) == 0);
}

struct Fixture {
  BroydenVectorLayoutView layout;
  BroydenPlan plan;
  Aligned raw{128};
  std::unique_ptr<Aligned> persistent, temporary, staging;
  BroydenVectorView vector;
  BroydenState state, staged;
  BroydenWorkspace workspace;
  std::string error;

  Fixture() {
    error.reserve(256);
    layout.batch_size = 3;
    layout.workspace_size_bytes = raw.size;
    layout.workspace_alignment = 64;
    layout.field_count = 2;
    vector.workspace_base = raw.data;
    vector.workspace_size_bytes = raw.size;
    vector.field_count = 2;
    for (std::size_t field = 0; field < 2; ++field) {
      layout.fields[field] = {field * 64, 48, 6, field_offsets[field].data(), 4};
      vector.fields[field] = reinterpret_cast<double*>(static_cast<char*>(raw.data) + field * 64);
      for (std::size_t element = 0; element < 6; ++element)
        vector.fields[field][element] = 0.1 * (1 + field * 6 + element);
    }
    assert(make_broyden_plan(layout, BroydenPolicy{capacity, .3, .04, .08}, encoding, plan,
                             error) == success);
    persistent = std::make_unique<Aligned>(plan.state_size_bytes());
    staging = std::make_unique<Aligned>(plan.state_size_bytes());
    temporary = std::make_unique<Aligned>(plan.workspace_size_bytes());
    assert(bind_broyden_state(plan, persistent->data, persistent->size, state, error) == success);
    assert(bind_broyden_state(plan, staging->data, staging->size, staged, error) == success);
    assert(bind_broyden_workspace(plan, temporary->data, temporary->size, workspace, error) ==
           success);
    for (std::size_t system = 0; system < 3; ++system) {
      assert(state.initialized[system] == 0);
      assert(state.system_statuses[system] == encoding.uninitialized);
    }
  }
  void initialize() { assert(initialize_broyden_state(plan, vector, state, error) == success); }
  void next_raw(unsigned step) {
    for (std::size_t system = 0; system < 3; ++system) {
      std::size_t packed = packed_offsets[system];
      for (std::size_t field = 0; field < 2; ++field) {
        for (auto element = field_offsets[field][system];
             element < field_offsets[field][system + 1]; ++element, ++packed) {
          vector.fields[field][element] =
              state.current_inputs[packed] + .01 * std::sin(.37 * (packed + 1) + .63 * step);
        }
      }
    }
  }
};

// Account for all state fields, including system-local flags. Compare every
// byte outside the selected system, including padding, against its preimage.
void peers_unchanged(const Bytes& before, const Aligned& buffer, const BroydenState& state,
                     std::size_t system) {
  std::vector<bool> selected(buffer.size, false);
  auto mark = [&](const auto* pointer, std::size_t count) {
    const auto start = reinterpret_cast<const unsigned char*>(pointer) -
                       static_cast<const unsigned char*>(buffer.data);
    const auto bytes = sizeof(*pointer) * count;
    assert(start >= 0 && static_cast<std::size_t>(start) + bytes <= buffer.size);
    std::fill_n(selected.begin() + start, bytes, true);
  };
  const auto offset = packed_offsets[system];
  const auto dimension = packed_offsets[system + 1] - offset;
  mark(state.current_inputs + offset, dimension);
  mark(state.previous_inputs + offset, dimension);
  mark(state.previous_residuals + offset, dimension);
  mark(state.df_history + capacity * offset, capacity * dimension);
  mark(state.u_history + capacity * offset, capacity * dimension);
  mark(state.omega + capacity * system, capacity);
  mark(state.residual_rms + system, 1);
  mark(state.residual_maximum + system, 1);
  mark(state.iterations + system, 1);
  mark(state.restart_counts + system, 1);
  mark(state.system_statuses + system, 1);
  mark(state.initialized + system, 1);
  mark(state.converged + system, 1);
  const auto* actual = static_cast<const unsigned char*>(buffer.data);
  for (std::size_t byte = 0; byte < before.size(); ++byte)
    if (!selected[byte]) assert(actual[byte] == before[byte]);
}

void test_plan_identity_and_validation() {
  Fixture f;
  f.initialize();
  assert(f.plan.vector_offsets() ==
         std::vector<std::int64_t>(packed_offsets.begin(), packed_offsets.end()));
  assert(f.plan.total_vector_elements() == 12 && f.plan.maximum_vector_elements() == 5);
  const auto* identity = f.plan.identity();
  BroydenPlan copied = f.plan;
  BroydenPlan moved = std::move(copied);
  assert(moved.identity() == identity && !copied.sealed());
  BroydenPlan assigned;
  assigned = moved;
  BroydenPlan move_assigned;
  move_assigned = std::move(assigned);
  assert(move_assigned.identity() == identity && !assigned.sealed());
  assert(validate_broyden_state_binding(move_assigned, f.state, f.error) == success);
  assert(validate_broyden_workspace_binding(moved, f.workspace, f.error) == success);
  f.next_raw(1);
  assert(mix_broyden_system(moved, 0, f.vector, f.state, f.workspace, f.error) == success);

  BroydenPlan other;
  assert(make_broyden_plan(f.layout, BroydenPolicy{capacity, .3, .04, .08}, encoding, other,
                           f.error) == success);
  assert(other.identity() != identity);
  const auto admission_before = snapshot(*f.persistent);
  assert(validate_broyden_state_binding(other, f.state, f.error) == invalid);
  assert(validate_broyden_workspace_binding(other, f.workspace, f.error) == invalid);
  assert(mix_broyden_system(other, 0, f.vector, f.state, f.workspace, f.error) == invalid);
  assert(make_broyden_plan(f.layout, BroydenPolicy{0, .3, .04, .08}, encoding, f.plan, f.error) ==
         invalid);
  assert(f.plan.identity() == identity);
  assert(make_broyden_plan(f.layout, BroydenPolicy{capacity, .3, .04, .08}, {7, 7, 41}, f.plan,
                           f.error) == invalid);
  assert(f.plan.identity() == identity);
  auto overlapping = f.layout;
  overlapping.fields[1].offset_bytes = 0;
  assert(make_broyden_plan(overlapping, BroydenPolicy{capacity, .3, .04, .08}, encoding, f.plan,
                           f.error) == invalid);
  assert(f.plan.identity() == identity);
  // The generic API rejects misaligned FP64 fields before any dereference.
  // These are deliberately stricter invalid-admission cases than the legacy
  // method owner, whose normal field layouts were always naturally aligned.
  for (std::size_t alignment : {std::size_t{1}, std::size_t{4}}) {
    auto misaligned = f.layout;
    misaligned.workspace_alignment = alignment;
    assert(make_broyden_plan(misaligned, BroydenPolicy{capacity, .3, .04, .08}, encoding, f.plan,
                             f.error) == invalid);
    assert(f.plan.identity() == identity);
  }
  for (std::size_t offset : {std::size_t{1}, std::size_t{4}}) {
    auto misaligned = f.layout;
    misaligned.fields[0].offset_bytes = offset;
    assert(make_broyden_plan(misaligned, BroydenPolicy{capacity, .3, .04, .08}, encoding, f.plan,
                             f.error) == invalid);
    assert(f.plan.identity() == identity);
  }
  assert(f.plan.matches_vector_layout(f.layout));
  auto incorrect_count = f.layout;
  --incorrect_count.fields[0].element_count;
  assert(!f.plan.matches_vector_layout(incorrect_count));
  unchanged(admission_before, *f.persistent);

  const auto before = snapshot(*f.persistent);
  for (double* BroydenState::* member :
       {&BroydenState::current_inputs, &BroydenState::previous_inputs,
        &BroydenState::previous_residuals, &BroydenState::df_history, &BroydenState::u_history,
        &BroydenState::omega, &BroydenState::residual_rms, &BroydenState::residual_maximum}) {
    auto forged = f.state;
    ++(forged.*member);
    assert(validate_broyden_state_binding(f.plan, forged, f.error) == invalid);
    assert(initialize_broyden_state(f.plan, f.vector, forged, f.error) == invalid);
  }
  for (std::uint64_t* BroydenState::* member :
       {&BroydenState::iterations, &BroydenState::restart_counts}) {
    auto forged = f.state;
    ++(forged.*member);
    assert(validate_broyden_state_binding(f.plan, forged, f.error) == invalid);
  }
  for (std::uint8_t* BroydenState::* member :
       {&BroydenState::initialized, &BroydenState::converged}) {
    auto forged = f.state;
    ++(forged.*member);
    assert(validate_broyden_state_binding(f.plan, forged, f.error) == invalid);
  }
  auto forged = f.state;
  ++forged.system_statuses;
  assert(validate_broyden_state_binding(f.plan, forged, f.error) == invalid);
  forged = f.state;
  forged.status_encoding.numerical_failure = 42;
  assert(validate_broyden_state_binding(f.plan, forged, f.error) == invalid);
  forged = f.state;
  --forged.workspace_size_bytes;
  assert(validate_broyden_state_binding(f.plan, forged, f.error) == invalid);
  for (double* BroydenWorkspace::* member :
       {&BroydenWorkspace::residual, &BroydenWorkspace::mixed, &BroydenWorkspace::delta_f,
        &BroydenWorkspace::new_u, &BroydenWorkspace::beta, &BroydenWorkspace::coefficients}) {
    auto bad = f.workspace;
    ++(bad.*member);
    assert(validate_broyden_workspace_binding(f.plan, bad, f.error) == invalid);
    assert(mix_broyden_system(f.plan, 0, f.vector, f.state, bad, f.error) == invalid);
  }
  auto bad_workspace = f.workspace;
  ++bad_workspace.history_slots;
  assert(validate_broyden_workspace_binding(f.plan, bad_workspace, f.error) == invalid);
  bad_workspace = f.workspace;
  --bad_workspace.workspace_size_bytes;
  assert(validate_broyden_workspace_binding(f.plan, bad_workspace, f.error) == invalid);
  auto bad_vector = f.vector;
  ++bad_vector.fields[0];
  assert(initialize_broyden_state(f.plan, bad_vector, f.state, f.error) == invalid);
  const auto raw_before = snapshot(f.raw);
  const auto scratch_before = snapshot(*f.temporary);
  // Keep the first field apparently canonical so validation must reject the
  // base/range before evaluating the second field's nonzero 64-byte offset.
  // Construct fake addresses with defined unsigned arithmetic, never pointer
  // arithmetic on the null or wrapping base itself.
  for (const auto address : {std::uintptr_t{0}, std::numeric_limits<std::uintptr_t>::max() - 63}) {
    bad_vector = f.vector;
    bad_vector.workspace_base = reinterpret_cast<void*>(address);
    bad_vector.fields[0] = reinterpret_cast<double*>(address);
    bad_vector.fields[1] = reinterpret_cast<double*>(address + std::uintptr_t{64});
    assert(initialize_broyden_state(f.plan, bad_vector, f.state, f.error) == invalid);
    assert(mix_broyden_system(f.plan, 0, bad_vector, f.state, f.workspace, f.error) == invalid);
    unchanged(before, *f.persistent);
    unchanged(raw_before, f.raw);
    unchanged(scratch_before, *f.temporary);
  }
  unchanged(before, *f.persistent);
}

void test_storage_and_initialization_atomicity() {
  Fixture f;
  const auto before = snapshot(*f.persistent);
  BroydenState rejected = f.state;
  assert(bind_broyden_state(f.plan, f.persistent->data, f.persistent->size - 1, rejected,
                            f.error) == invalid);
  assert(rejected.plan_identity == f.state.plan_identity &&
         rejected.current_inputs == f.state.current_inputs);
  assert(bind_broyden_state(f.plan, static_cast<char*>(f.persistent->data) + 1, f.persistent->size,
                            rejected, f.error) == invalid);
  unchanged(before, *f.persistent);
  BroydenWorkspace rejected_workspace = f.workspace;
  assert(bind_broyden_workspace(f.plan, f.temporary->data, f.temporary->size - 1,
                                rejected_workspace, f.error) == invalid);
  assert(rejected_workspace.beta == f.workspace.beta);
  assert(bind_broyden_workspace(f.plan, static_cast<char*>(f.temporary->data) + 1,
                                f.temporary->size, rejected_workspace, f.error) == invalid);

  // A later ragged system must be validated before any earlier one is reset.
  f.vector.fields[1][5] = std::numeric_limits<double>::quiet_NaN();
  assert(initialize_broyden_state(f.plan, f.vector, f.state, f.error) == invalid);
  unchanged(before, *f.persistent);
  f.vector.fields[1][5] = 1.2;
  f.initialize();
  f.next_raw(1);
  assert(mix_broyden_batch(f.plan, f.vector, f.state, f.workspace, f.error) == success);
  const auto live = snapshot(*f.persistent);
  f.vector.fields[1][5] = std::numeric_limits<double>::infinity();
  assert(initialize_broyden_state(f.plan, f.vector, f.state, f.error) == invalid);
  unchanged(live, *f.persistent);

  // Canonically bound individual views can still overlap one another.
  auto alias_vector = f.vector;
  alias_vector.workspace_base = f.persistent->data;
  for (std::size_t field = 0; field < 2; ++field)
    alias_vector.fields[field] =
        reinterpret_cast<double*>(static_cast<char*>(f.persistent->data) + 64 * field);
  assert(initialize_broyden_state(f.plan, alias_vector, f.state, f.error) == invalid);
  assert(f.persistent->size >= f.temporary->size);
  BroydenWorkspace alias_workspace;
  assert(bind_broyden_workspace(f.plan, f.persistent->data, f.persistent->size, alias_workspace,
                                f.error) == success);
  assert(mix_broyden_system(f.plan, 0, f.vector, f.state, alias_workspace, f.error) == invalid);
  assert(prepare_broyden_system_transaction(f.plan, 0, f.state, f.state, f.error) == invalid);
  assert(commit_broyden_system_transaction(f.plan, 0, f.state, f.state, f.error) == invalid);
  unchanged(live, *f.persistent);
}

void test_descriptor_storage_aliases() {
  Fixture f;
  f.initialize();
  f.next_raw(1);
  const auto raw_before = snapshot(f.raw);
  const auto persistent_before = snapshot(*f.persistent);

  // Copy a canonical descriptor into its own numerical storage. Its member
  // pointers still match the plan, but executing with it would destroy the
  // descriptor during the first numerical write.
  assert(f.temporary->size >= sizeof(BroydenWorkspace));
  auto* scratch_inside_itself = ::new (f.temporary->data) BroydenWorkspace(f.workspace);
  const auto scratch_with_descriptor = snapshot(*f.temporary);
  assert(validate_broyden_workspace_binding(f.plan, *scratch_inside_itself, f.error) == invalid);
  assert(mix_broyden_system(f.plan, 0, f.vector, f.state, *scratch_inside_itself, f.error) ==
         invalid);
  unchanged(scratch_with_descriptor, *f.temporary);
  unchanged(persistent_before, *f.persistent);
  unchanged(raw_before, f.raw);

  assert(f.persistent->size >= sizeof(BroydenState));
  auto* state_inside_itself = ::new (f.persistent->data) BroydenState(f.state);
  const auto state_with_descriptor = snapshot(*f.persistent);
  assert(validate_broyden_state_binding(f.plan, *state_inside_itself, f.error) == invalid);
  assert(mix_broyden_system(f.plan, 0, f.vector, *state_inside_itself, f.workspace, f.error) ==
         invalid);
  assert(initialize_broyden_state(f.plan, f.vector, *state_inside_itself, f.error) == invalid);
  unchanged(state_with_descriptor, *f.persistent);
  unchanged(scratch_with_descriptor, *f.temporary);
  unchanged(raw_before, f.raw);
}

void test_restart_and_peer_failures() {
  Fixture f;
  f.initialize();
  for (unsigned step = 1; step <= 7; ++step) {
    f.next_raw(step);
    assert(mix_broyden_batch(f.plan, f.vector, f.state, f.workspace, f.error) == success);
  }
  f.next_raw(8);
  const auto before = snapshot(*f.persistent);
  assert(restart_broyden_system(f.plan, 1, f.vector, f.state, f.error) == success);
  peers_unchanged(before, *f.persistent, f.state, 1);
  assert(f.state.iterations[1] == 0 && f.state.restart_counts[1] == 1);
  assert(f.state.system_statuses[1] == encoding.success && f.state.initialized[1] == 1);
  assert(f.state.converged[1] == 0 && f.state.residual_rms[1] == 0 &&
         f.state.residual_maximum[1] == 0);
  for (auto i = packed_offsets[1]; i < packed_offsets[2]; ++i) {
    assert(f.state.previous_inputs[i] == 0 && f.state.previous_residuals[i] == 0);
    for (std::size_t h = 0; h < capacity; ++h) {
      const auto index = capacity * packed_offsets[1] + h * 4 + i - packed_offsets[1];
      assert(f.state.df_history[index] == 0 && f.state.u_history[index] == 0);
    }
  }
  for (std::size_t h = 0; h < capacity; ++h) assert(f.state.omega[capacity + h] == 0);
  std::size_t packed = packed_offsets[1];
  for (std::size_t field = 0; field < 2; ++field)
    for (auto i = field_offsets[field][1]; i < field_offsets[field][2]; ++i)
      assert(f.state.current_inputs[packed++] == f.vector.fields[field][i]);
  f.state.restart_counts[1] = std::numeric_limits<std::uint64_t>::max();
  const auto overflow = snapshot(*f.persistent);
  assert(restart_broyden_system(f.plan, 1, f.vector, f.state, f.error) == invalid);
  unchanged(overflow, *f.persistent);
  f.state.restart_counts[1] = 1;
  f.vector.fields[0][2] = std::numeric_limits<double>::quiet_NaN();
  const auto nonfinite = snapshot(*f.persistent);
  const auto raw_before = snapshot(f.raw);
  assert(restart_broyden_system(f.plan, 1, f.vector, f.state, f.error) == invalid);
  unchanged(nonfinite, *f.persistent);
  unchanged(raw_before, f.raw);
  assert(mix_broyden_system(f.plan, 1, f.vector, f.state, f.workspace, f.error) == numerical);
  assert(f.state.system_statuses[1] == encoding.numerical_failure);
  f.state.system_statuses[1] = encoding.success;
  unchanged(nonfinite, *f.persistent);
  unchanged(raw_before, f.raw);

  // A failed middle peer must not prevent the serial wrapper advancing others.
  const auto iteration0 = f.state.iterations[0], iteration2 = f.state.iterations[2];
  assert(mix_broyden_batch(f.plan, f.vector, f.state, f.workspace, f.error) == numerical);
  assert(f.state.iterations[0] == iteration0 + 1 && f.state.iterations[2] == iteration2 + 1);
  assert(f.state.iterations[1] == 0 && f.state.system_statuses[1] == encoding.numerical_failure);
  assert(f.state.system_statuses[0] == encoding.success &&
         f.state.system_statuses[2] == encoding.success);
}

void test_transactions_and_no_allocations() {
  Fixture f;
  f.initialize();
  for (unsigned step = 1; step <= 7; ++step) {
    f.next_raw(step);
    assert(mix_broyden_batch(f.plan, f.vector, f.state, f.workspace, f.error) == success);
  }
  const auto persistent_before = snapshot(*f.persistent);
  const auto staged_before = snapshot(*f.staging);
  assert(prepare_broyden_system_transaction(f.plan, 1, f.state, f.staged, f.error) == success);
  unchanged(persistent_before, *f.persistent);
  peers_unchanged(staged_before, *f.staging, f.staged, 1);
  f.next_raw(8);
  f.vector.fields[0][2] = std::numeric_limits<double>::quiet_NaN();
  const auto failed_raw = snapshot(f.raw);
  assert(mix_broyden_system(f.plan, 1, f.vector, f.staged, f.workspace, f.error) == numerical);
  unchanged(persistent_before, *f.persistent);
  unchanged(failed_raw, f.raw);
  peers_unchanged(staged_before, *f.staging, f.staged, 1);
  // Rollback discards the failed staging record by preparing again.
  assert(prepare_broyden_system_transaction(f.plan, 1, f.state, f.staged, f.error) == success);
  f.next_raw(9);
  assert(mix_broyden_system(f.plan, 1, f.vector, f.staged, f.workspace, f.error) == success);
  unchanged(persistent_before, *f.persistent);
  const auto ready = snapshot(*f.staging);
  assert(commit_broyden_system_transaction(f.plan, 1, f.staged, f.state, f.error) == success);
  peers_unchanged(persistent_before, *f.persistent, f.state, 1);
  unchanged(ready, *f.staging);
  assert(f.state.iterations[1] == 8);
  assert(std::memcmp(f.state.current_inputs + 3, f.staged.current_inputs + 3, 4 * sizeof(double)) ==
         0);

  // Count every replaceable C++ allocation, including aligned/array forms.
  // Plan construction and test snapshots deliberately precede this interval.
  allocations = 0;
  count_allocations = true;
  assert(validate_broyden_state_binding(f.plan, f.state, f.error) == success);
  assert(validate_broyden_workspace_binding(f.plan, f.workspace, f.error) == success);
  assert(initialize_broyden_state(f.plan, f.vector, f.state, f.error) == success);
  for (unsigned step = 1; step <= 2 * capacity + 5; ++step) {
    f.next_raw(step);
    assert(mix_broyden_batch(f.plan, f.vector, f.state, f.workspace, f.error) == success);
    assert(prepare_broyden_system_transaction(f.plan, 1, f.state, f.staged, f.error) == success);
    assert(commit_broyden_system_transaction(f.plan, 1, f.staged, f.state, f.error) == success);
  }
  assert(restart_broyden_system(f.plan, 1, f.vector, f.state, f.error) == success);
  count_allocations = false;
  assert(allocations == 0);
}
}  // namespace

int main() {
  static_assert(kBroydenWorkspaceAlignment == 64 && kBroydenMaximumFields == 4);
  static_assert(sizeof(*BroydenState{}.system_statuses) == sizeof(std::int32_t));
  test_plan_identity_and_validation();
  test_storage_and_initialization_atomicity();
  test_descriptor_storage_aliases();
  test_restart_and_peer_failures();
  test_transactions_and_no_allocations();
}
