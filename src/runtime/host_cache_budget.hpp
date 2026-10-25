#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace generativeqc::runtime {

/** Conservative optional cache allowance, not an allocation guarantee.
 * Keep three quarters of currently available RAM for the mandatory owner,
 * other processes and allocator growth. Linux cgroup limits apply at every
 * ancestor, including a Slurm job parent; machine MemAvailable alone can be
 * much larger than the job's allowance. Unknown/malformed observations refuse
 * automatic caching. Explicit user ceilings remain a separate caller policy.
 * Path arguments allow executable capacity tests without inspecting the host.
 */
inline std::size_t automatic_host_cache_bytes(
    const std::filesystem::path& proc = "/proc",
    const std::filesystem::path& cgroups = "/sys/fs/cgroup") noexcept {
  try {
    const auto number = [](const std::string& text, std::size_t& value) {
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
      return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
    };
    std::ifstream memory(proc / "meminfo");
    std::string key, text, unit;
    std::size_t available = 0;
    while (memory >> key >> text >> unit)
      if (key == "MemAvailable:") {
        if (unit != "kB" || !number(text, available) ||
            available > std::numeric_limits<std::size_t>::max() / 1024)
          return 0;
        available *= 1024;
        break;
      }
    if (!available) return 0;
    std::ifstream membership(proc / "self/cgroup");
    if (!membership) return 0;
    bool found = false;
    std::string line;
    while (std::getline(membership, line)) {
      const auto first = line.find(':'), last = line.find(':', first + 1);
      if (first == std::string::npos || last == std::string::npos) return 0;
      const auto controllers = "," + line.substr(first + 1, last - first - 1) + ",";
      const bool unified = first == 1 && line.front() == '0' && controllers == ",,";
      if (!unified && controllers.find(",memory,") == std::string::npos) continue;
      found = true;
      const auto root = unified ? cgroups : cgroups / "memory";
      const std::filesystem::path relative = line.substr(last + 1);
      if (!relative.is_absolute()) return 0;
      for (const auto& part : relative)
        if (part == "..") return 0;
      auto directory = (root / relative.relative_path()).lexically_normal();
      if (!std::filesystem::is_directory(directory)) return 0;
      for (;;) {
        const auto limit_path = directory / (unified ? "memory.max" : "memory.limit_in_bytes");
        if (std::filesystem::exists(limit_path)) {
          std::ifstream limit_file(limit_path);
          if (!(limit_file >> text)) return 0;
          if (text != "max") {
            std::size_t limit = 0, used = 0;
            if (!number(text, limit)) return 0;
            std::ifstream usage(directory / (unified ? "memory.current" : "memory.usage_in_bytes"));
            if (!(usage >> text) || !number(text, used)) return 0;
            available = std::min(available, used < limit ? limit - used : 0);
          }
        }
        if (directory == root) break;
        directory = directory.parent_path();
      }
    }
    return found ? std::min<std::size_t>(32ULL << 30, available / 4) : 0;
  } catch (...) {
    // Filesystem/observation failure must not invalidate the exact fallback.
    return 0;
  }
}

}  // namespace generativeqc::runtime
