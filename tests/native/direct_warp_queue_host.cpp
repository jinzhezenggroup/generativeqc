// Host concurrency test of the same queue algorithm used by the CUDA wrapper.
// This is not scientific derivative, CUDA synchronization, or performance evidence.
#include "scf/direct_warp_queue.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using Queue = generativeqc::scf::detail::BoundedClassWarpQueue<55, 256>;

struct HostAtomics {
  std::mutex mutex;
  std::uint32_t load(std::uint32_t* address) {
    const std::lock_guard<std::mutex> lock(mutex);
    return *address;
  }
  std::uint32_t exchange(std::uint32_t* address, std::uint32_t value) {
    const std::lock_guard<std::mutex> lock(mutex);
    const auto previous = *address;
    *address = value;
    return previous;
  }
  std::uint32_t compare_exchange(std::uint32_t* address, std::uint32_t expected,
                                 std::uint32_t desired) {
    const std::lock_guard<std::mutex> lock(mutex);
    const auto previous = *address;
    if (previous == expected) *address = desired;
    return previous;
  }
};

int main() {
  Queue queue;
  HostAtomics atomics;
  std::size_t checked = 0;
  for (unsigned workers : {1U, 2U, 4U, 8U, 16U}) {
    for (unsigned count : {0U, 1U, 31U, 32U, 33U, 63U, 64U, 65U, 127U, 128U, 129U, 255U, 256U}) {
      // Reuse the same allocation, intentionally leaving stale next[] words.
      for (unsigned repeat = 0; repeat < 4; ++repeat) {
        for (auto& head : queue.heads) head = Queue::empty;
        std::array<std::atomic<unsigned>, 256> visits{};
        std::array<unsigned, 256> classes{};
        for (auto& visit : visits) visit.store(0);
        for (unsigned slot = 0; slot < count; ++slot) {
          classes[slot] = repeat == 0 ? 20U : (slot * 17U + repeat) % 61U;
        }
        std::vector<std::thread> producers;
        for (unsigned worker = 0; worker < workers; ++worker) {
          producers.emplace_back([&, worker] {
            for (unsigned slot = worker; slot < count; slot += workers) {
              queue.publish(classes[slot], slot, atomics);
            }
          });
        }
        for (auto& producer : producers) producer.join();
        // This join is the host equivalent of publication-phase synchronization.
        std::vector<std::thread> consumers;
        for (unsigned worker = 0; worker < workers; ++worker) {
          consumers.emplace_back([&, worker] {
            unsigned preferred = worker;
            for (;;) {
              const auto slot = queue.take_any(preferred, atomics);
              if (slot == Queue::empty) break;
              assert(slot < count);
              const auto expected_class = classes[slot] < 55U ? classes[slot] : 55U;
              assert(preferred == expected_class);
              assert(visits[slot].fetch_add(1) == 0U);
              if ((slot % 7U) == 0U) std::this_thread::yield();
            }
          });
        }
        for (auto& consumer : consumers) consumer.join();
        for (unsigned slot = 0; slot < 256U; ++slot) {
          assert(visits[slot].load() == (slot < count ? 1U : 0U));
        }
        for (auto head : queue.heads) assert(head == Queue::empty);
        ++checked;
      }
    }
  }
  std::cout << checked << " concurrent host batches passed\n";
}
