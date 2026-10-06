// Compile-time ownership proofs for the actual queue implementation. No CUDA
// execution or numerical/performance qualification is implied by these checks.
#include "scf/cuda/direct_warp_queue.cuh"

using namespace generativeqc::scf::cuda_execution;
using Queue = BoundedClassWarpQueue<256U, 55U>;

struct SerialClaim {
  constexpr std::uint32_t operator()(std::uint32_t* cursor) const { return (*cursor)++; }
};

constexpr bool check_packet(unsigned count, unsigned mode, unsigned workers) {
  Queue queue{};
  for (unsigned slot = 0; slot < count; ++slot) {
    queue.classes[slot] = mode == 0 ? 54U : (slot * 17U + mode * 11U) % 55U;
  }
  if (!queue.prepare(count)) return false;
  // Independent stable-bucket oracle; preparation must not omit or reorder ties.
  unsigned position = 0;
  for (unsigned c = 0; c < 55U; ++c) {
    if (queue.boundaries[c] != position) return false;
    for (unsigned slot = 0; slot < count; ++slot) {
      if (queue.classes[slot] == c && queue.slots[position++] != slot) return false;
    }
  }
  if (position != count || queue.boundaries[55] != count) return false;
  unsigned current_class[32]{};
  unsigned seen[256]{};
  // Deliberately uneven worker progress, including a delayed worker zero.
  for (unsigned step = 0; step < count + workers * 2U; ++step) {
    const unsigned worker = step < count / 2U ? workers - 1U : step % workers;
    const auto slot = queue.pop(current_class[worker], SerialClaim{});
    if (slot != Queue::exhausted && (slot >= count || ++seen[slot] != 1U)) return false;
  }
  for (unsigned worker = 0; worker < workers; ++worker) {
    while (true) {
      const auto slot = queue.pop(current_class[worker], SerialClaim{});
      if (slot == Queue::exhausted) break;
      if (slot >= count || ++seen[slot] != 1U) return false;
    }
    if (queue.pop(current_class[worker], SerialClaim{}) != Queue::exhausted) return false;
  }
  for (unsigned slot = 0; slot < count; ++slot) {
    if (seen[slot] != 1U) return false;
  }
  for (unsigned c = 0; c < 55U; ++c) {
    if (queue.cursors[c] > queue.boundaries[c + 1U] + workers) return false;
  }
  // Shared index reuse must reset all claims, including previously empty classes.
  return queue.prepare(count) && queue.cursors[54] == queue.boundaries[54];
}

constexpr bool check_recovery() {
  Queue queue{};
  if (queue.prepare(257U) || queue.ready) return false;
  if (!queue.prepare(0U)) return false;
  queue.classes[0] = 55U;
  if (queue.prepare(1U) || queue.ready) return false;
  unsigned c = 0;
  if (queue.pop(c, SerialClaim{}) != Queue::exhausted) return false;
  queue.classes[0] = 54U;
  return queue.prepare(1U) && queue.pop(c, SerialClaim{}) == 0U &&
         queue.pop(c, SerialClaim{}) == Queue::exhausted;
}

static_assert(check_recovery());
#define CHECK_PACKET(N)                      \
  static_assert(check_packet(N, 0U, 1U));     \
  static_assert(check_packet(N, 1U, 4U));     \
  static_assert(check_packet(N, 2U, 8U));     \
  static_assert(check_packet(N, 3U, 32U))
CHECK_PACKET(0U);
CHECK_PACKET(1U);
CHECK_PACKET(2U);
CHECK_PACKET(7U);
CHECK_PACKET(8U);
CHECK_PACKET(31U);
CHECK_PACKET(32U);
CHECK_PACKET(33U);
CHECK_PACKET(63U);
CHECK_PACKET(64U);
CHECK_PACKET(65U);
CHECK_PACKET(127U);
CHECK_PACKET(128U);
CHECK_PACKET(129U);
CHECK_PACKET(255U);
CHECK_PACKET(256U);
#undef CHECK_PACKET

// The diagnostic switch is exact; unsupported launch shapes retain the incumbent.
using generativeqc::scf::cuda_execution::bounded_force_warp_schedule;
static_assert(bounded_force_warp_schedule(nullptr, 256, 1, 1).threads == 256);
static_assert(!bounded_force_warp_schedule(nullptr, 256, 1, 1).dynamic);
static_assert(!bounded_force_warp_schedule("", 256, 1, 1).dynamic);
static_assert(!bounded_force_warp_schedule("0", 256, 1, 1).dynamic);
static_assert(!bounded_force_warp_schedule("invalid", 256, 1, 1).dynamic);
static_assert(!bounded_force_warp_schedule("warp128-extra", 256, 1, 1).dynamic);
static_assert(!bounded_force_warp_schedule("warp128", 128, 1, 1).dynamic);
static_assert(!bounded_force_warp_schedule("warp128", 256, 2, 1).dynamic);
static_assert(!bounded_force_warp_schedule("warp128", 256, 1, 2).dynamic);
static_assert(bounded_force_warp_schedule("static128", 256, 1, 1).threads == 128);
static_assert(!bounded_force_warp_schedule("static128", 256, 1, 1).dynamic);
static_assert(bounded_force_warp_schedule("warp128", 256, 1, 1).threads == 128);
static_assert(bounded_force_warp_schedule("warp128", 256, 1, 1).dynamic);
static_assert(bounded_force_warp_schedule("warp256", 256, 1, 1).threads == 256);
static_assert(bounded_force_warp_schedule("warp256", 256, 1, 1).dynamic);
