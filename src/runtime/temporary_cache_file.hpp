#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/statfs.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace generativeqc::runtime {

/** Unlinked, physically reserved local scratch storage for reproducible values.
 * This is not a persistent cache: the open descriptor is its only lifetime.
 * RAM filesystems and common network filesystems are refused because spilling
 * under RAM pressure must not consume tmpfs or a shared high-frequency store.
 * No sparse-file promise is treated as capacity. Unsupported platforms or
 * reservation failures return false, allowing the original exact provider.
 */
class TemporaryCacheFile {
 public:
  TemporaryCacheFile() = default;
  TemporaryCacheFile(const TemporaryCacheFile&) = delete;
  TemporaryCacheFile& operator=(const TemporaryCacheFile&) = delete;
  ~TemporaryCacheFile() {
#if defined(__linux__)
    if (descriptor_ >= 0) (void)::close(descriptor_);
#endif
  }

  /** Honor the explicit spill directory, then the platform temporary directory.
   * An unavailable directory disables optional spilling instead of falling back
   * to an unrelated filesystem. An explicit empty override also stays disabled.
   */
  static std::filesystem::path default_directory(const char* configured = nullptr) {
    if (configured) return configured;
    std::error_code error;
    auto directory = std::filesystem::temp_directory_path(error);
    return error ? std::filesystem::path{} : directory;
  }

  /** Leave three quarters of local free space available to other users. */
  static std::size_t allowance(const std::filesystem::path& directory) noexcept {
#if defined(__linux__)
    struct statfs type{};
    struct statvfs space{};
    if (::statfs(directory.c_str(), &type) || ::statvfs(directory.c_str(), &space)) return 0;
    // tmpfs, ramfs, NFS, CIFS/SMB2, Lustre and FUSE (unknown backing).
    for (const unsigned long rejected : {0x01021994UL, 0x858458f6UL, 0x6969UL, 0xff534d42UL,
                                         0xfe534d42UL, 0x0bd00bd0UL, 0x65735546UL})
      if (static_cast<unsigned long>(type.f_type) == rejected) return 0;
    if (!space.f_frsize ||
        space.f_bavail > std::numeric_limits<std::size_t>::max() / space.f_frsize)
      return 0;
    return std::min<std::size_t>(64ULL << 30, space.f_bavail * space.f_frsize / 4);
#else
    return 0;
#endif
  }

  bool reserve(const std::filesystem::path& directory, std::size_t bytes) noexcept {
#if defined(__linux__)
    if (descriptor_ >= 0 || !bytes || bytes > allowance(directory) ||
        bytes > static_cast<std::size_t>(std::numeric_limits<off_t>::max()))
      return false;
    try {
      const auto name = (directory / "generativeqc-eri-XXXXXX").string();
      std::vector<char> path(name.begin(), name.end());
      path.push_back('\0');
      const int file = ::mkstemp(path.data());
      if (file < 0) return false;
      // Unlink immediately: failures, normal close and process termination all
      // release the file without a pathname cleanup race or stale cache reuse.
      if (::unlink(path.data())) {
        (void)::close(file);
        return false;
      }
      if (::fcntl(file, F_SETFD, FD_CLOEXEC) == -1 ||
          ::posix_fallocate(file, 0, static_cast<off_t>(bytes))) {
        (void)::close(file);
        return false;
      }
      descriptor_ = file;
      bytes_ = bytes;
      (void)::posix_fadvise(file, 0, 0, POSIX_FADV_SEQUENTIAL);
      return true;
    } catch (...) {
      return false;
    }
#else
    return false;
#endif
  }

  bool write(std::size_t offset, const void* data, std::size_t bytes) noexcept {
    return transfer(offset, const_cast<void*>(data), bytes, true);
  }
  bool read(std::size_t offset, void* data, std::size_t bytes) noexcept {
    return transfer(offset, data, bytes, false);
  }

  /** Flush construction before dropping page-cache residency. This makes disk
   * backing reclaimable even when the file is smaller than available RAM.
   * Replay reads similarly advise eviction after consuming each bounded tile.
   */
  bool finish_writes() noexcept {
#if defined(__linux__)
    if (descriptor_ < 0) return false;
    int status;
    do {
      status = ::fdatasync(descriptor_);
    } while (status && errno == EINTR);
    return !status && !::posix_fadvise(descriptor_, 0, 0, POSIX_FADV_DONTNEED);
#else
    return false;
#endif
  }

 private:
  bool transfer(std::size_t offset, void* data, std::size_t bytes, bool writing) noexcept {
#if defined(__linux__)
    if (descriptor_ < 0 || offset > bytes_ || bytes > bytes_ - offset || (!data && bytes))
      return false;
    const auto begin = offset, length = bytes;
    auto* cursor = static_cast<char*>(data);
    while (bytes) {
      const auto tile = std::min<std::size_t>(bytes, std::numeric_limits<ssize_t>::max());
      const auto done = writing ? ::pwrite(descriptor_, cursor, tile, static_cast<off_t>(offset))
                                : ::pread(descriptor_, cursor, tile, static_cast<off_t>(offset));
      if (done < 0 && errno == EINTR) continue;
      if (done <= 0) return false;
      bytes -= static_cast<std::size_t>(done);
      offset += static_cast<std::size_t>(done);
      cursor += done;
    }
    if (!writing)
      (void)::posix_fadvise(descriptor_, static_cast<off_t>(begin), static_cast<off_t>(length),
                            POSIX_FADV_DONTNEED);
    return true;
#else
    return false;
#endif
  }
  int descriptor_{-1};
  std::size_t bytes_{};
};

}  // namespace generativeqc::runtime
