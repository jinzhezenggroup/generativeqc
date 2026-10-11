#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace {
struct alignas(std::max_align_t) Header {
  std::size_t bytes;
};
std::atomic<std::size_t> live{}, peak{}, calls{}, total{};
}  // namespace
void* operator new(std::size_t bytes) {
  auto* header = static_cast<Header*>(std::malloc(sizeof(Header) + bytes));
  if (!header) throw std::bad_alloc();
  header->bytes = bytes;
  const auto current = live.fetch_add(bytes) + bytes;
  auto high = peak.load();
  while (high < current && !peak.compare_exchange_weak(high, current)) {
  }
  calls.fetch_add(1);
  total.fetch_add(bytes);
  return header + 1;
}
void operator delete(void* pointer) noexcept {
  if (!pointer) return;
  auto* header = static_cast<Header*>(pointer) - 1;
  live.fetch_sub(header->bytes);
  std::free(header);
}
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
extern "C" {
void qc_reset() {
  peak.store(live.load());
  calls.store(0);
  total.store(0);
}
std::size_t qc_live() { return live.load(); }
std::size_t qc_peak() { return peak.load(); }
std::size_t qc_calls() { return calls.load(); }
std::size_t qc_total() { return total.load(); }
}
