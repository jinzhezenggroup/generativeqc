// Native wheel-installed PATH entry: locate the packaged C++ CLI without Python.
// The linked scientific runtime remains next to generativeqc/bin/generativeqc,
// where the executable's $ORIGIN RPATH resolves the packaged native library.
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <vector>
#endif
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;

fs::path executable_path() {
#if defined(__linux__)
  std::string buffer(4096, '\0');
  const auto size = ::readlink("/proc/self/exe", buffer.data(), buffer.size());
  if (size <= 0 || static_cast<std::size_t>(size) == buffer.size())
    throw std::runtime_error("cannot locate the wheel launcher executable");
  buffer.resize(static_cast<std::size_t>(size));
  return fs::canonical(fs::path(buffer));
#elif defined(__APPLE__)
  std::uint32_t count = 0;
  (void)::_NSGetExecutablePath(nullptr, &count);
  std::vector<char> buffer(count + 1, '\0');
  if (::_NSGetExecutablePath(buffer.data(), &count) != 0)
    throw std::runtime_error("cannot locate the wheel launcher executable");
  return fs::canonical(fs::path(buffer.data()));
#else
#error "The installed wheel launcher requires a supported POSIX executable path."
#endif
}

void admit(fs::path candidate, const fs::path& launcher, std::set<fs::path>& found) {
  std::error_code error;
  if (!fs::is_regular_file(candidate, error) || error) return;
  if (::access(candidate.c_str(), X_OK) != 0) return;
  candidate = fs::canonical(candidate, error);
  if (!error && candidate != launcher) found.insert(std::move(candidate));
}

void search_site_packages(const fs::path& directory, const fs::path& launcher,
                          std::set<fs::path>& found) {
  admit(directory / "generativeqc/bin/generativeqc", launcher, found);
}

std::set<fs::path> find_candidates(const fs::path& launcher) {
  // pip/venv/user installations place scripts in PREFIX/bin and the native
  // package in PREFIX/lib/pythonX.Y/site-packages. Never consult PATH and
  // never select a binary from another environment.
  const fs::path prefix = launcher.parent_path().parent_path();
  std::set<fs::path> found;
  for (const char* root : {"lib", "lib64", "Lib"}) {
    const fs::path base = prefix / root;
    search_site_packages(base / "site-packages", launcher, found);
    search_site_packages(base / "dist-packages", launcher, found);
    std::error_code error;
    fs::directory_iterator entries(base, error);
    if (error) continue;
    for (const auto& entry : entries) {
      if (!entry.is_directory(error) || error) {
        error.clear();
        continue;
      }
      const auto name = entry.path().filename().string();
      if (!name.starts_with("python")) continue;
      search_site_packages(entry.path() / "site-packages", launcher, found);
      search_site_packages(entry.path() / "dist-packages", launcher, found);
    }
  }
  return found;
}

}  // namespace

int main(int /*argc*/, char* argv[]) {
  try {
    const fs::path launcher = executable_path();
    const auto candidates = find_candidates(launcher);
    if (candidates.size() != 1) {
      std::cerr << "generativeqc: expected exactly one packaged native CLI adjacent "
                   "to this wheel installation; found "
                << candidates.size()
                << ". Reinstall the matching GenerativeQC wheel/SDK.\n";
      return 127;
    }
    const std::string binary = candidates.begin()->string();
    argv[0] = const_cast<char*>(binary.c_str());
    ::execv(binary.c_str(), argv);
    std::cerr << "generativeqc: cannot execute native CLI: " << std::strerror(errno) << '\n';
    return 126;
  } catch (const std::exception& error) {
    std::cerr << "generativeqc: " << error.what() << '\n';
    return 127;
  }
}
