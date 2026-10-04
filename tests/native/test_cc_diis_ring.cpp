// Compare real physical-slot bookkeeping with an independent chronological deque.
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

#include "solver/diis_ring.hpp"

int main() {
  using generativeqc::solver::DiisRing;
  for (std::size_t capacity : {1, 2, 3, 8, 20}) {
    DiisRing ring(capacity);
    std::vector<unsigned> physical(capacity, 0);
    std::deque<unsigned> chronological;
    for (unsigned step = 1; step <= 20 * capacity; ++step) {
      const auto old = physical;
      const auto slot = ring.push();
      physical[slot] = step;
      chronological.push_back(step);
      if (chronological.size() > capacity) chronological.pop_front();
      for (std::size_t i = 0; i < capacity; ++i)
        if (i != slot && physical[i] != old[i]) throw std::runtime_error("history moved");
      if (step % 4 == 0) {
        ring.retire_oldest();
        chronological.pop_front();
      }
      if (ring.size() != chronological.size()) throw std::runtime_error("retirement count");
      for (std::size_t i = 0; i < ring.size(); ++i)
        if (physical[ring.slot(i)] != chronological[i]) throw std::runtime_error("chronology");
      if (step % 13 == 0) {
        ring.clear();
        chronological.clear();
      }
    }
    ring.clear();
    ring.retire_oldest();
    if (ring.size() || ring.first() || ring.push()) throw std::runtime_error("restart");
    bool rejected = false;
    try {
      (void)ring.slot(ring.size());
    } catch (const std::out_of_range&) {
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("dead row accepted");
  }
  DiisRing disabled(0);
  try {
    disabled.push();
  } catch (const std::invalid_argument&) {
    return 0;
  }
  throw std::runtime_error("disabled history accepted an insertion");
}
