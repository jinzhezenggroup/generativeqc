#include <cstddef>
#include <cstdlib>

/** Ordinary PLT calls are visible to the admitted native heap profiler.
 * Direct ctypes/dlsym calls to libc bypass its patched call sites. Markers are
 * bounded observer payloads, not replacements for any scientific allocator. */
extern "C" void* generativeqc_audit_marker_allocate(std::size_t bytes) {
  if (bytes == 0 || bytes > 4096) return nullptr;
  return std::malloc(bytes);
}

extern "C" void generativeqc_audit_marker_release(void* pointer) { std::free(pointer); }
