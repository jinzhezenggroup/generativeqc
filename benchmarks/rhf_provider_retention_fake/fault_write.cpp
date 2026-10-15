/** Fail exactly one observer JSONL write, then forward every later write. */

#include <dlfcn.h>
#include <errno.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace {

std::atomic<bool> failed{};
thread_local bool resolving{};

bool contains(const char* data, std::size_t size, const char* needle) noexcept {
  const std::size_t length = std::strlen(needle);
  if (length > size) return false;
  for (std::size_t offset = 0; offset + length <= size; ++offset) {
    if (std::memcmp(data + offset, needle, length) == 0) return true;
  }
  return false;
}

}  // namespace

extern "C" ssize_t write(int descriptor, const void* buffer, std::size_t size) {
  using Write = ssize_t (*)(int, const void*, std::size_t);
  static Write real{};
  if (!real && !resolving) {
    resolving = true;
    real = reinterpret_cast<Write>(::dlsym(RTLD_NEXT, "write"));
    resolving = false;
  }
  if (!real) {
    errno = EIO;
    return -1;
  }
  const char* enabled = std::getenv("FAKE_FAIL_ONE_TRACE_WRITE");
  bool expected = false;
  if (enabled && std::strcmp(enabled, "1") == 0 && buffer &&
      contains(static_cast<const char*>(buffer), size,
               "generativeqc.rhf-provider-probe.v1") &&
      failed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    errno = EIO;
    return -1;
  }
  return real(descriptor, buffer, size);
}
