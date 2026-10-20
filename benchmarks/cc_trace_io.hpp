#pragma once
#include <unistd.h>

#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdint>

namespace cc_trace_io {
static_assert(std::endian::native == std::endian::little,
              "CC trace binary v1 is explicitly uint64 little-endian");
inline int write_all(int fd, const void* data, std::size_t bytes) noexcept {
  auto* cursor = static_cast<const unsigned char*>(data);
  while (bytes) {
    const auto count = ::write(fd, cursor, bytes);
    if (count < 0) {
      if (errno == EINTR) continue;
      return -errno;
    }
    if (!count) return -EIO;
    cursor += count;
    bytes -= static_cast<std::size_t>(count);
  }
  return 0;
}
}  // namespace cc_trace_io
