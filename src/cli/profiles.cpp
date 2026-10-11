#include "cli/profiles.hpp"

#include <archive.h>
#include <archive_entry.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "cli/options.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__GLIBC__)
#include <gnu/libc-version.h>
#endif
#endif

namespace generativeqc::cli {
namespace {
constexpr std::array<const char*, 4> bundle_files{"profile.json", "libgenerativeqc.so",
                                                  "manifest.json", "evidence.json"};
constexpr std::uint64_t artifact_limit = std::uint64_t{2} * 1024 * 1024 * 1024;
const Json policy = {
    {"precision", "fp64"}, {"spin", {"rhf", "uhf"}}, {"consumers", {"fock", "force"}}};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

bool has_evidence(const Json& value) {
  if (value.is_null()) return false;
  if (value.is_boolean()) return value.get<bool>();
  if (value.is_number()) return value != 0;
  if (value.is_string()) return !value.get<std::string>().empty();
  return !value.empty();
}

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(const std::filesystem::path& parent) {
#if defined(__unix__) || defined(__APPLE__)
    std::string pattern = (parent / ".gqc-profile-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    require(::mkdtemp(buffer.data()) != nullptr, "cannot create profile staging directory");
    path = buffer.data();
#else
    (void)parent;
    throw std::runtime_error("profile staging requires POSIX");
#endif
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
  std::filesystem::path path;
};

class CacheLock {
 public:
  explicit CacheLock(const std::filesystem::path& root) {
#if defined(__unix__) || defined(__APPLE__)
    descriptor_ = ::open((root / ".lock").c_str(), O_CREAT | O_RDWR | O_NOFOLLOW, 0600);
    require(descriptor_ >= 0, "cannot open profile cache lock");
    if (::flock(descriptor_, LOCK_EX)) {
      ::close(descriptor_);
      throw std::runtime_error("cannot acquire profile cache lock");
    }
#else
    (void)root;
    throw std::runtime_error("profile locking requires POSIX");
#endif
  }
  ~CacheLock() {
#if defined(__unix__) || defined(__APPLE__)
    ::close(descriptor_);
#endif
  }
  CacheLock(const CacheLock&) = delete;
  CacheLock& operator=(const CacheLock&) = delete;

 private:
  int descriptor_{-1};
};

Json index_at(const std::filesystem::path& root) {
  const auto path = root / "active.json";
  if (!std::filesystem::exists(path)) return Json::object();
  auto index = read_json(path);
  require(index.is_object(), "invalid profile index: expected an object");
  for (const auto& value : index)
    require(value.is_string(), "invalid profile index: expected string bundle IDs");
  return index;
}

Json gates(std::initializer_list<const char*> names) {
  Json result = Json::object();
  for (const auto name : names) result[name] = "pass";
  return result;
}

void validate_dft(const Json& winners, const Json& evidence, const Json& identity) {
  std::set<std::pair<std::string, std::string>> accepted, selected;
  for (const auto& row : evidence.value("dft_schedules", Json::array()))
    if (row.value("accepted", false))
      accepted.emplace(row.at("workload_hash"), row.at("schedule_hash"));
  std::set<std::string> workloads;
  for (const auto& winner : winners) {
    const auto& workload = winner.at("workload");
    const auto& schedule = winner.at("schedule");
    const std::string wh = winner.at("workload_hash"), sh = winner.at("schedule_hash");
    const std::string source = winner.at("source_hash");
    require(wh == canonical_hash(workload) && sh == canonical_hash(schedule),
            "DFT workload/schedule hash differs");
    require(workloads.insert(wh).second, "duplicate DFT workload winner");
    selected.emplace(wh, sh);
    require(workload.at("schema") == "generativeqc.grid-xc-scientific.v1" &&
                schedule.at("schema") == "generativeqc.grid-xc-schedule.v1",
            "unsupported DFT identity schema");
    const std::string name = schedule.at("name");
    require(
        schedule.size() == 6 && (name == "device_fused" || name == "host_unfused") &&
            schedule.at("fusion") == (name == "device_fused" ? "device_xc_vxc" : "host_xc_vxc") &&
            schedule.at("resident_jets").is_boolean() &&
            schedule.at("resident_jets") == (name == "device_fused") &&
            schedule.at("matrix_accumulation") == "symmetric",
        "invalid executable DFT schedule");
    const auto& tile = schedule.at("point_tile");
    require(tile.is_null() || (tile.is_number_integer() && tile.get<std::int64_t>() > 0),
            "invalid DFT point tile");
    const auto& device = identity.at("device");
    require(workload.at("architecture") == "sm_" + std::to_string(device.at("major").get<int>()) +
                                               std::to_string(device.at("minor").get<int>()) &&
                workload.at("precision") == "fp64" &&
                workload.at("source_identity") == identity.at("source_identity"),
            "incompatible DFT target/source");
    for (const auto field :
         {"functional", "functional_identity", "ingredients", "jet_outputs", "grid_identity",
          "grid_model", "screening_identity", "spin", "observable", "density_route"}) {
      require(workload.contains(field), std::string("missing DFT workload field: ") + field);
      if (std::string_view(field) != "screening_identity")
        require(!workload.at(field).is_null() && workload.at(field) != "" &&
                    workload.at(field) != Json::array(),
                "empty DFT workload identity");
    }
    require(is_digest(source) && winner.at("gates") == gates({"legality", "resources", "numerical",
                                                              "performance", "endpoint"}),
            "DFT candidate did not pass promotion gates");
    Json matched;
    for (const auto& row : evidence.value("dft_schedules", Json::array()))
      if (row.value("accepted", false) && row.value("workload_hash", "") == wh &&
          row.value("schedule_hash", "") == sh)
        matched = row;
    const auto& legality = matched.at("legality");
    const auto& resources = matched.at("resources");
    const auto& numerical = matched.at("numerical");
    const auto& endpoint = matched.at("endpoint");
    require(legality.value("legal", false) && legality.at("schedule_hash") == sh &&
                resources.value("passed", false) && resources.at("schedule_hash") == sh &&
                resources.at("source_hash") == source && numerical.value("passed", false) &&
                has_evidence(numerical.value("independent_reference", Json())) &&
                numerical.at("schedule_hash") == sh && numerical.at("source_hash") == source &&
                endpoint.value("passed", false) && endpoint.value("complete_energy_force", false) &&
                endpoint.at("scientific_identity") == wh &&
                endpoint.at("candidate_schedule_hash") == sh &&
                endpoint.at("candidate_source_hash") == source &&
                has_evidence(endpoint.at("baseline_schedule_hash")) &&
                endpoint.at("baseline_schedule_hash") != sh &&
                endpoint.value("interleaved", false) && endpoint.value("synchronized", false) &&
                endpoint.at("baseline").is_array() && endpoint.at("candidate").is_array() &&
                endpoint.at("baseline").size() >= 5 &&
                endpoint.at("baseline").size() == endpoint.at("candidate").size(),
            "DFT winner lacks independent numerical/resource/endpoint evidence");
  }
  require(accepted == selected, "DFT winners differ from accepted evidence");
}

using ProbeFunction = generativeqc_status (*)(std::int32_t,
                                              generativeqc_cuda_tuning_device_descriptor*);
using SourceFunction = const char* (*)();

Json probe_device(int ordinal, ProbeFunction probe = generativeqc_cuda_tuning_device,
                  SourceFunction source = generativeqc_get_source_identity) {
  generativeqc_cuda_tuning_device_descriptor device{};
  device.struct_size = sizeof(device);
  device.abi_version = GENERATIVEQC_ABI_VERSION;
  const auto status = probe(ordinal, &device);
  require(status == GENERATIVEQC_STATUS_SUCCESS,
          "CUDA tuning device probe failed (status " + std::to_string(status) + ")");
  Json hardware;
#define GQC_DEVICE_FIELD(field) hardware[#field] = device.field
  GQC_DEVICE_FIELD(name);
  GQC_DEVICE_FIELD(major);
  GQC_DEVICE_FIELD(minor);
  GQC_DEVICE_FIELD(warp_size);
  GQC_DEVICE_FIELD(maximum_threads_per_block);
  GQC_DEVICE_FIELD(maximum_threads_per_sm);
  GQC_DEVICE_FIELD(maximum_blocks_per_sm);
  GQC_DEVICE_FIELD(registers_per_sm);
  GQC_DEVICE_FIELD(maximum_registers_per_thread);
  GQC_DEVICE_FIELD(sm_count);
  GQC_DEVICE_FIELD(shared_memory_per_block);
  GQC_DEVICE_FIELD(shared_memory_per_block_optin);
  GQC_DEVICE_FIELD(shared_memory_per_sm);
  GQC_DEVICE_FIELD(runtime_version);
  GQC_DEVICE_FIELD(driver_version);
  GQC_DEVICE_FIELD(toolkit_version);
  GQC_DEVICE_FIELD(release_build);
  GQC_DEVICE_FIELD(fast_compile);
  GQC_DEVICE_FIELD(official_profile);
  GQC_DEVICE_FIELD(portable);
#undef GQC_DEVICE_FIELD
  Json host;
#if defined(__unix__) || defined(__APPLE__)
  struct utsname system{};
  require(::uname(&system) == 0, "cannot identify native host");
  host = {{"system", system.sysname}, {"machine", system.machine}, {"libc", {"", ""}}};
#if defined(__GLIBC__)
  host["libc"] = {"glibc", ::gnu_get_libc_version()};
#endif
#endif
  return {{"source_identity", source()},
          {"native_abi", GENERATIVEQC_ABI_VERSION},
          {"schema", 1},
          {"policy", policy},
          {"host", host},
          {"device", hardware}};
}

Json compatible_identity(Json probe) {
  probe["device"].erase("official_profile");
  probe["device"].erase("portable");
  return probe;
}

void verify_library(const std::filesystem::path& directory, const Json& probe, int ordinal,
                    bool tuned) {
#if defined(__unix__) || defined(__APPLE__)
  // Resolve entry points from the candidate handle, not the already-linked baseline.
  const auto path = std::filesystem::absolute(directory / "libgenerativeqc.so");
  void* raw = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  require(raw != nullptr, std::string("cannot load profile library: ") + (raw ? "" : ::dlerror()));
  const auto close_library = [](void* pointer) { ::dlclose(pointer); };
  std::unique_ptr<void, decltype(close_library)> handle(raw, close_library);
  const auto device =
      reinterpret_cast<ProbeFunction>(::dlsym(raw, "generativeqc_cuda_tuning_device"));
  const auto source =
      reinterpret_cast<SourceFunction>(::dlsym(raw, "generativeqc_get_source_identity"));
  require(device && source, "profile library is missing native identity entry points");
  const auto actual = probe_device(ordinal, device, source);
  require(!tuned || actual["device"]["portable"] == 0,
          "cached binary has no tuned profile for allocated GPU");
  require(compatible_identity(actual) == compatible_identity(probe),
          "profile binary has incompatible source/ABI/toolkit identity");
#else
  (void)directory;
  (void)probe;
  (void)ordinal;
  (void)tuned;
  throw std::runtime_error("native profile loading requires POSIX");
#endif
}

// Compiler identity is optional for replay, but a present toolkit must match.
std::filesystem::path executable(std::string_view name) {
  if (name.empty()) return {};
  if (name.find('/') != name.npos) {
    const std::filesystem::path path{name};
#if defined(__unix__) || defined(__APPLE__)
    if (::access(path.c_str(), X_OK) == 0) return std::filesystem::canonical(path);
#endif
    return {};
  }
  const char* search = std::getenv("PATH");
  if (!search) return {};
  std::string paths(search);
  std::size_t start = 0;
  do {
    const auto end = paths.find(':', start);
    const auto directory = paths.substr(start, end - start);
    auto found =
        executable((std::filesystem::path(directory.empty() ? "." : directory) / name).string());
    if (!found.empty()) return found;
    if (end == paths.npos) break;
    start = end + 1;
  } while (true);
  return {};
}

std::string version_output(const std::filesystem::path& tool) {
#if defined(__unix__) || defined(__APPLE__)
  int descriptors[2];
  require(::pipe(descriptors) == 0, "cannot create compiler identity pipe");
  const pid_t child = ::fork();
  if (child < 0) {
    ::close(descriptors[0]);
    ::close(descriptors[1]);
    throw std::runtime_error("cannot probe compiler identity");
  }
  if (child == 0) {
    ::close(descriptors[0]);
    ::dup2(descriptors[1], STDOUT_FILENO);
    ::dup2(descriptors[1], STDERR_FILENO);
    ::close(descriptors[1]);
    ::execl(tool.c_str(), tool.c_str(), "--version", static_cast<char*>(nullptr));
    ::_exit(127);
  }
  ::close(descriptors[1]);
  ::fcntl(descriptors[0], F_SETFL, O_NONBLOCK);
  std::string output;
  std::array<char, 4096> buffer{};
  int status = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  for (;;) {
    const auto count = ::read(descriptors[0], buffer.data(), buffer.size());
    if (count > 0) output.append(buffer.data(), static_cast<std::size_t>(count));
    const auto waited = ::waitpid(child, &status, WNOHANG);
    if (waited == child) {
      while (const auto remaining = ::read(descriptors[0], buffer.data(), buffer.size())) {
        if (remaining < 0) break;
        output.append(buffer.data(), static_cast<std::size_t>(remaining));
      }
      ::close(descriptors[0]);
      require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "compiler identity probe failed");
      break;
    }
    if (output.size() > 1024 * 1024 || std::chrono::steady_clock::now() >= deadline || waited < 0) {
      ::kill(child, SIGKILL);
      ::waitpid(child, &status, 0);
      ::close(descriptors[0]);
      throw std::runtime_error("compiler identity probe timed out or exceeded output limit");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  const auto first = output.find_first_not_of(" \t\r\n");
  require(first != output.npos, "empty compiler identity");
  return output.substr(first, output.find_last_not_of(" \t\r\n") - first + 1);
#else
  (void)tool;
  throw std::runtime_error("compiler identity probing requires POSIX");
#endif
}

void check_toolchain(const Json& profile) {
  std::filesystem::path nvcc;
  for (const auto variable : {"GENERATIVEQC_NVCC", "CUDACXX"})
    if (const char* value = std::getenv(variable); value && nvcc.empty()) nvcc = executable(value);
  if (nvcc.empty())
    if (const char* value = std::getenv("CUDA_PATH"))
      nvcc = executable((std::filesystem::path(value) / "bin/nvcc").string());
  if (nvcc.empty()) nvcc = executable("nvcc");
  if (!nvcc.empty()) {
    const Json identity = {{"nvcc", version_output(nvcc)},
                           {"ptxas", version_output(nvcc.parent_path() / "ptxas")}};
    require(identity == profile.at("toolchain"), "NVCC/PTXAS identity changed");
  }
}

void extract_archive(const std::filesystem::path& input, const std::filesystem::path& target) {
  std::unique_ptr<archive, decltype(&archive_read_free)> reader(archive_read_new(),
                                                                archive_read_free);
  require(reader != nullptr, "cannot allocate profile archive reader");
  archive_read_support_format_zip(reader.get());
  require(archive_read_open_filename(reader.get(), input.c_str(), 65536) == ARCHIVE_OK,
          "cannot open profile ZIP archive");
  std::set<std::string> seen;
  archive_entry* entry = nullptr;
  int status;
  while ((status = archive_read_next_header(reader.get(), &entry)) == ARCHIVE_OK) {
    const char* raw_name = archive_entry_pathname(entry);
    const std::string name = raw_name ? raw_name : "";
    require(std::find(bundle_files.begin(), bundle_files.end(), name) != bundle_files.end() &&
                seen.insert(name).second,
            "profile archive must contain exactly the four bundle files");
    require(archive_entry_filetype(entry) == AE_IFREG && !archive_entry_symlink(entry) &&
                !archive_entry_hardlink(entry) && archive_entry_size(entry) >= 0 &&
                static_cast<std::uint64_t>(archive_entry_size(entry)) <= artifact_limit,
            "unsafe profile archive entry or artifact size");
    std::ofstream output(target / name, std::ios::binary);
    require(static_cast<bool>(output), "cannot stage profile artifact");
    std::array<char, 65536> buffer{};
    std::uint64_t total = 0;
    la_ssize_t count;
    while ((count = archive_read_data(reader.get(), buffer.data(), buffer.size())) > 0) {
      total += static_cast<std::uint64_t>(count);
      require(total <= artifact_limit, "profile artifact exceeds import size limit");
      output.write(buffer.data(), count);
    }
    output.close();
    require(count == 0 && output.good() &&
                total == static_cast<std::uint64_t>(archive_entry_size(entry)),
            "corrupt profile archive artifact");
  }
  require(status == ARCHIVE_EOF && seen.size() == bundle_files.size(),
          "incomplete or corrupt profile archive");
}

void export_archive(const std::filesystem::path& input, const std::filesystem::path& destination) {
  validate_profile_bundle(input);
  const auto parent = std::filesystem::absolute(destination).parent_path();
  std::filesystem::create_directories(parent);
  TemporaryDirectory staging(parent);
  const auto pending = staging.path / "bundle.zip";
  std::unique_ptr<archive, decltype(&archive_write_free)> writer(archive_write_new(),
                                                                 archive_write_free);
  require(writer != nullptr, "cannot allocate profile archive writer");
  require(archive_write_set_format_zip(writer.get()) == ARCHIVE_OK &&
              archive_write_open_filename(writer.get(), pending.c_str()) == ARCHIVE_OK,
          "cannot open profile export");
  for (const auto name : bundle_files) {
    std::error_code error;
    require(!std::filesystem::equivalent(input / name, destination, error),
            "export destination overwrites bundle artifact");
    std::ifstream stream(input / name, std::ios::binary);
    const auto size = std::filesystem::file_size(input / name);
    require(size <= artifact_limit && stream.good(), "invalid profile export artifact");
    std::unique_ptr<archive_entry, decltype(&archive_entry_free)> entry(archive_entry_new(),
                                                                        archive_entry_free);
    archive_entry_set_pathname(entry.get(), name);
    archive_entry_set_filetype(entry.get(), AE_IFREG);
    archive_entry_set_perm(entry.get(), 0644);
    archive_entry_set_size(entry.get(), static_cast<la_int64_t>(size));
    require(archive_write_header(writer.get(), entry.get()) == ARCHIVE_OK,
            "cannot write archive header");
    std::array<char, 65536> buffer{};
    while (stream.read(buffer.data(), buffer.size()) || stream.gcount())
      require(archive_write_data(writer.get(), buffer.data(),
                                 static_cast<std::size_t>(stream.gcount())) == stream.gcount(),
              "cannot write archive artifact");
    require(stream.eof(), "cannot read profile artifact");
  }
  require(archive_write_close(writer.get()) == ARCHIVE_OK, "cannot finish profile export");
  std::filesystem::rename(pending, destination);
}

Json select_profile(int ordinal) {
  Json diagnostics = {{"source", "official"},
                      {"identity", nullptr},
                      {"target", nullptr},
                      {"kernels", Json::array()},
                      {"dft_schedules", Json::array()},
                      {"rejected", Json::array()}};
  try {
    const auto probe = probe_device(ordinal);
    diagnostics["source"] = probe["device"]["portable"] == 0 ? "official" : "portable";
    diagnostics["identity"] = probe["device"]["official_profile"];
    const char* explicit_path = std::getenv("GENERATIVEQC_PROFILE");
    if (explicit_path && std::string_view(explicit_path) == "off") {
      diagnostics["rejected"].push_back("local profiles explicitly disabled");
      return diagnostics;
    }
    std::filesystem::path directory;
    if (explicit_path && *explicit_path) {
      directory = explicit_path;
      if (std::string_view(explicit_path).starts_with("~/")) {
        const char* home = std::getenv("HOME");
        require(home != nullptr, "cannot expand profile path without HOME");
        directory = std::filesystem::path(home) / (explicit_path + 2);
      }
    } else {
      const auto root = profile_cache_root();
      const auto index = index_at(root);
      const auto key = canonical_hash(compatible_identity(probe));
      if (!index.contains(key)) {
        if (!index.empty())
          diagnostics["rejected"].push_back(
              "cached profiles have incompatible hardware/source/runtime identities");
        return diagnostics;
      }
      const std::string id = index.at(key);
      require(is_digest(id), "invalid active bundle identifier");
      directory = root / "bundles" / id;
    }
    const auto profile = validate_profile_bundle(directory);
    require(profile.at("identity") == compatible_identity(probe),
            "GPU/runtime/driver/source/ABI identity changed");
    check_toolchain(profile);
    verify_library(directory, probe, ordinal, !profile.value("kernels", Json::array()).empty());
    diagnostics.update({{"source", "local"},
                        {"identity", canonical_hash(profile)},
                        {"target", profile.at("identity")},
                        {"kernels", profile.value("kernels", Json::array())},
                        {"dft_schedules", profile.value("dft_schedules", Json::array())},
                        {"directory", std::filesystem::absolute(directory).string()}});
  } catch (const std::exception& error) {
    diagnostics["rejected"].push_back(error.what());
  }
  return diagnostics;
}

}  // namespace

std::filesystem::path profile_cache_root() {
  if (const char* configured = std::getenv("GENERATIVEQC_PROFILE_CACHE");
      configured && *configured) {
    const std::string value(configured);
    if (value == "~" || value.starts_with("~/")) {
      const char* home = std::getenv("HOME");
      if (!home || !*home)
        throw UsageError("cannot expand GENERATIVEQC_PROFILE_CACHE without HOME");
      return value == "~" ? std::filesystem::path(home)
                          : std::filesystem::path(home) / value.substr(2);
    }
    return value;
  }
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
    return std::filesystem::path(xdg) / "generativeqc/profiles";
  const char* home = std::getenv("HOME");
  if (!home || !*home)
    throw UsageError("cannot resolve profile cache without HOME or XDG_CACHE_HOME");
  return std::filesystem::path(home) / ".cache/generativeqc/profiles";
}

Json validate_profile_bundle(const std::filesystem::path& directory) {
  const auto profile = read_json(directory / "profile.json");
  require(profile.at("schema") == "generativeqc.local_profile" && profile.at("schema_version") == 1,
          "unsupported local profile schema");
  const auto& identity = profile.at("identity");
  require(identity.at("policy") == policy && identity.at("schema") == 1,
          "incompatible codegen/consumer policy");
  require(identity.at("device").at("release_build") == 1 &&
              identity.at("device").at("fast_compile") == 0,
          "local profiles require release builds without fast compilation");
  const auto& toolchain = profile.at("toolchain");
  require(toolchain.is_object() && toolchain.size() == 2 && toolchain.at("nvcc").is_string() &&
              toolchain.at("ptxas").is_string() &&
              !toolchain.at("nvcc").get<std::string>().empty() &&
              !toolchain.at("ptxas").get<std::string>().empty(),
          "missing compiler identity");
  const auto& artifacts = profile.at("artifacts");
  require(artifacts.is_object() && artifacts.size() == 3, "incomplete local profile artifacts");
  for (const auto name : {"libgenerativeqc.so", "manifest.json", "evidence.json"}) {
    require(artifacts.at(name).is_string() && is_digest(artifacts.at(name).get<std::string>()) &&
                file_hash(directory / name) == artifacts.at(name),
            std::string("local profile artifact hash differs: ") + name);
  }
  const auto kernels = profile.value("kernels", Json::array());
  const auto dft = profile.value("dft_schedules", Json::array());
  require(kernels.is_array() && dft.is_array() && (!kernels.empty() || !dft.empty()),
          "profile requires validated winner arrays");
  const auto evidence = read_json(directory / "evidence.json");
  std::set<std::pair<std::string, std::string>> seen, accepted;
  if (!kernels.empty())
    require(evidence.at("endpoint").value("passed", false),
            "complete endpoint acceptance is missing");
  for (const auto& row : evidence.value("candidates", Json::array()))
    if (row.value("accepted", false)) accepted.emplace(row.at("shell_class"), row.at("consumer"));
  for (const auto& kernel : kernels) {
    const std::string shell = kernel.at("shell_class"), consumer = kernel.at("consumer");
    require(shell.size() == 4 && shell.find_first_not_of("spdf") == shell.npos &&
                (consumer == "fock" || consumer == "force") && seen.emplace(shell, consumer).second,
            "duplicate or unsupported kernel consumer");
    const std::string hash = kernel.at("schedule_hash"), source = kernel.at("source_hash");
    require(hash == canonical_hash(kernel.at("schedule")) && is_digest(source),
            "invalid kernel schedule/source hash");
    require(kernel.at("gates") ==
                gates({"target", "resources", "numerical", "performance", "endpoint"}),
            "candidate did not pass all promotion gates");
    Json matched;
    for (const auto& row : evidence.value("candidates", Json::array()))
      if (row.value("accepted", false) && row.at("shell_class") == shell &&
          row.at("consumer") == consumer)
        matched = row;
    const auto& isolated = matched.at("isolated");
    const auto& measured = isolated.at("device");
    const auto& actual = identity.at("device");
    for (const auto& fields :
         std::array<std::pair<const char*, const char*>, 5>{{{"name", "name"},
                                                             {"major", "major"},
                                                             {"minor", "minor"},
                                                             {"driver", "driver_version"},
                                                             {"runtime", "runtime_version"}}})
      require(measured.at(fields.first) == actual.at(fields.second),
              "numerical evidence used a different CUDA target/runtime");
    Json winner;
    for (const auto& trial : matched.at("tuning").at("winners"))
      if (trial.at("trial_key") == matched.at("selected_trial")) winner = trial;
    require(isolated.value("passed", false) && isolated.at("source_hash") == source &&
                isolated.at("schedule_hash") == hash &&
                winner.at("schedule") == kernel.at("schedule") &&
                winner.at("production_validation").value("accepted", false) &&
                isolated.at("errors").is_object() && !isolated.at("errors").empty() &&
                matched.at("endpoint").value("passed", false),
            "kernel lacks matching numerical/endpoint evidence");
    for (const auto& error : isolated.at("errors"))
      require(error.value("passed", false), "kernel numerical acceptance failed");
  }
  require(seen == accepted, "profile winners differ from accepted evidence");
  if (!dft.empty()) validate_dft(dft, evidence, identity);
  return profile;
}

int profile_bundle_command(int argc, char** argv) {
  const std::string_view operation = argc > 2 ? argv[2] : "";
  if (operation == "export") {
    if (argc != 5) throw UsageError("profile export requires DIRECTORY DESTINATION.zip");
    export_archive(argv[3], argv[4]);
    std::cout << argv[4] << '\n';
    return 0;
  }
  if (operation == "validate") {
    if (argc != 4) throw UsageError("profile validate requires DIRECTORY");
    const auto profile = validate_profile_bundle(argv[3]);
    std::cout << Json({{"valid", true}, {"identity", canonical_hash(profile)}}).dump(2) << '\n';
    return 0;
  }
  if (operation == "diagnose") {
    if (argc != 3) throw UsageError("profile diagnose accepts no extra arguments");
    std::cout << select_profile(0).dump(2) << '\n';
    return 0;
  }
  if (operation == "install") {
    if (argc != 4) throw UsageError("profile install requires ARCHIVE.zip");
    TemporaryDirectory staging(std::filesystem::temp_directory_path());
    extract_archive(argv[3], staging.path);
    const auto profile = validate_profile_bundle(staging.path);
    const auto probe = probe_device(0);
    require(profile.at("identity") == compatible_identity(probe),
            "GPU/runtime/driver/source/ABI identity changed");
    check_toolchain(profile);
    verify_library(staging.path, probe, 0, !profile.value("kernels", Json::array()).empty());
    const auto root = profile_cache_root();
    std::filesystem::create_directories(root / "bundles");
    CacheLock lock(root);
    const auto destination = root / "bundles" / canonical_hash(profile);
    if (!std::filesystem::exists(destination)) {
      TemporaryDirectory pending(root / "bundles");
      const auto bundle = pending.path / "bundle";
      std::filesystem::create_directory(bundle);
      for (const auto name : bundle_files)
        std::filesystem::copy_file(staging.path / name, bundle / name);
      require(validate_profile_bundle(bundle) == profile, "staged profile changed");
      std::filesystem::rename(bundle, destination);
    } else
      require(validate_profile_bundle(destination) == profile, "immutable profile changed");
    auto index = index_at(root);
    index[canonical_hash(profile.at("identity"))] = canonical_hash(profile);
    atomic_json(root / "active.json", index);
    std::cout << destination.string() << '\n';
    return 0;
  }
  throw UsageError("unknown profile operation: " + std::string(operation));
}

void activate_local_profile(int argc, char** argv, int device_id) {
  (void)argc;
  const auto diagnostics = select_profile(device_id);
  if (diagnostics.at("source") != "local") return;
  // DFT schedule-only bundles require workload-specific admission in the KS
  // prepared owner. Do not pretend loading their binary applies that schedule.
  if (diagnostics.at("kernels").empty()) return;
#if defined(__linux__)
  const auto library =
      std::filesystem::path(diagnostics.at("directory").get<std::string>()) / "libgenerativeqc.so";
  Dl_info loaded{};
  if (::dladdr(reinterpret_cast<void*>(generativeqc_get_source_identity), &loaded) &&
      loaded.dli_fname && std::filesystem::equivalent(library, loaded.dli_fname))
    return;
  require(std::getenv("GENERATIVEQC_NATIVE_PROFILE_REEXEC") == nullptr,
          "profile library was not selected after native re-execution");
  require(library.string().find_first_of(" :\t\n") == std::string::npos,
          "profile library path is not safe for native loader selection");
  std::string preload = library.string();
  if (const char* existing = std::getenv("LD_PRELOAD"); existing && *existing)
    preload += ':' + std::string(existing);
  require(::setenv("LD_PRELOAD", preload.c_str(), 1) == 0 &&
              ::setenv("GENERATIVEQC_NATIVE_PROFILE_REEXEC", "1", 1) == 0,
          "cannot select native profile environment");
  ::execv("/proc/self/exe", argv);
  throw std::runtime_error("cannot re-execute native CLI with selected profile");
#else
  throw std::runtime_error("native profile binary selection currently requires Linux");
#endif
}

}  // namespace generativeqc::cli
