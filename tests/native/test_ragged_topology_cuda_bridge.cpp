// CPU-host call traces of the actual unchanged production CUDA adapter.
// Device definitions are excluded. Only launch syntax is replaced, and the
// copy boundary supplies a chosen diagnostic rather than executing a kernel.
#include <sys/mman.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "backends/cuda/gfn2_plan_schema.cuh"

using namespace generativeqc::xtb::detail;
namespace adapter = generativeqc::xtb::detail::cuda;

constexpr std::size_t kArenaBytes = 64 * 4096;
constexpr std::uint64_t kToken = 0x5abc;
std::uintptr_t arena = 0;
const void* host_diagnostic = nullptr;
std::vector<std::string> events;
std::vector<std::string> observations;
const std::int64_t* watched_index = nullptr;
const std::uint64_t* watched_output_token = nullptr;
int attribute_index = 0;
int fail_attribute = -1;
int host_attribute = -1;
bool managed_memory = false;
int launch_status = 0;
int copy_status = 0;
int sync_status = 0;
int last_error = 0;
Gfn2PlanSchemaDiagnostic copied_diagnostic{};

void reset() {
  events.clear();
  observations.clear();
  watched_index = nullptr;
  watched_output_token = nullptr;
  host_diagnostic = nullptr;
  attribute_index = 0;
  fail_attribute = host_attribute = -1;
  managed_memory = false;
  launch_status = copy_status = sync_status = last_error = 0;
  copied_diagnostic = {};
}

template <typename T>
T* metadata(std::size_t slot) {
  return reinterpret_cast<T*>(arena + slot * 4096);
}

template <typename T, std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>, int> = 0>
std::string describe(T value) {
  if constexpr (std::is_enum_v<T>)
    return std::to_string(static_cast<std::uint32_t>(value));
  else
    return std::to_string(value);
}

template <typename T>
std::string describe(T* value) {
  if (value == nullptr) return "null";
  if (static_cast<const void*>(value) == host_diagnostic) return "host-diagnostic";
  const auto address = reinterpret_cast<std::uintptr_t>(value);
  const auto diagnostic_address = reinterpret_cast<std::uintptr_t>(host_diagnostic);
  if (host_diagnostic != nullptr && address > diagnostic_address &&
      address < diagnostic_address + sizeof(Gfn2PlanSchemaDiagnostic)) {
    return "host-diagnostic+" + std::to_string(address - diagnostic_address);
  }
  if (address >= arena && address < arena + kArenaBytes) {
    return "meta+" + std::to_string(address - arena);
  }
  return "ptr+" + std::to_string(address);
}

template <typename T>
void poison_field(T& value) {
  if constexpr (std::is_pointer_v<T>)
    value = reinterpret_cast<T>(arena + 31 * 4096);
  else
    value = static_cast<T>(99);
}

#include "ragged_topology_cuda_serialization.inc"

void observe(const char* stage) {
  // Only explicit alias cases watch real, valid scalar subobjects. Every
  // other metadata pointer remains PROT_NONE and is never read by the harness.
  if (watched_index != nullptr) {
    observations.push_back(std::string(stage) + "|index=" + describe(*watched_index) +
                           "|output-token=" + describe(*watched_output_token));
  }
}

// A PROT_NONE mapping makes an accidental metadata read in the production
// adapter, native wrapper, or shared validator fail the ordinary probe cases.
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* pointer) {
  observe("attributes");
  const int index = attribute_index++;
  const int status = index == fail_attribute ? 700 : 0;
  attributes->type = index == host_attribute ? cudaMemoryTypeHost
                     : managed_memory        ? cudaMemoryTypeManaged
                                             : cudaMemoryTypeDevice;
  events.push_back("attributes|" + describe(pointer) + "|" + describe(status) + "|" +
                   describe(attributes->type));
  if (status != 0) last_error = status;
  return status;
}

cudaError_t cudaGetLastError() {
  observe("last-error");
  const int status = last_error;
  last_error = 0;
  events.push_back("last-error|" + describe(status));
  return status;
}

cudaError_t cudaMemcpyAsync(void* destination, const void* source, std::size_t bytes,
                            cudaMemcpyKind kind, cudaStream_t stream) {
  observe("copy-enter");
  events.push_back("copy|" + describe(destination) + "|" + describe(source) + "|" +
                   describe(bytes) + "|" + describe(kind) + "|" + describe(stream) + "|" +
                   describe(copy_status));
  if (destination != host_diagnostic || bytes != sizeof(Gfn2PlanSchemaDiagnostic) ||
      kind != cudaMemcpyDeviceToHost)
    std::abort();
  if (copy_status == 0) {
    *static_cast<Gfn2PlanSchemaDiagnostic*>(destination) = copied_diagnostic;
  }
  observe("copy-leave");
  return copy_status;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  observe("sync");
  events.push_back("sync|" + describe(stream) + "|" + describe(sync_status));
  return sync_status;
}

struct KernelBoundary {
  std::string prefix;
  template <typename... Args>
  void operator()(const Args&... arguments) const {
    observe("launch");
    std::string event = prefix;
    ((event += "|" + describe(arguments)), ...);
    events.push_back(event);
    last_error = launch_status;
  }
};

template <typename... Args>
KernelBoundary trace_kernel(const char* name, const Args&... configuration) {
  std::string event = "launch|" + std::string(name);
  ((event += "|" + describe(configuration)), ...);
  return {event};
}

#include "ragged_topology_cuda_host.inc"

Gfn2RaggedTopologyView topology() {
  Gfn2RaggedTopologyView t{};
  t.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  t.pair_map_kind = Gfn2PairMapKind::kExplicit;
  t.plan_token = kToken;
  t.batch_size = 2;
  t.total_atoms = 3;
  t.total_shells = 4;
  t.total_orbitals = 5;
  t.total_matrix_elements = 13;
  t.total_pairs = 1;
  t.bucket_count = 2;
  t.atom_offset_count = t.batch_shell_offset_count = t.batch_orbital_offset_count = 3;
  t.matrix_offset_count = 3;
  t.atom_shell_offset_count = 4;
  t.shell_orbital_offset_count = 5;
  t.shell_to_atom_count = 4;
  t.orbital_to_shell_count = t.orbital_to_atom_count = 5;
  t.pair_offset_count = 3;
  t.atom_pair_count = 1;
  t.bucket_offset_count = 3;
  t.bucket_system_count = t.bucket_orbital_count = 2;
  t.atom_offsets = metadata<std::int64_t>(1);
  t.batch_shell_offsets = metadata<std::int64_t>(2);
  t.batch_orbital_offsets = metadata<std::int64_t>(3);
  t.matrix_offsets = metadata<std::int64_t>(4);
  t.atom_shell_offsets = metadata<std::int64_t>(5);
  t.shell_orbital_offsets = metadata<std::int64_t>(6);
  t.shell_to_atom = metadata<std::int64_t>(7);
  t.orbital_to_shell = metadata<std::int64_t>(8);
  t.orbital_to_atom = metadata<std::int64_t>(9);
  t.pair_offsets = metadata<std::int64_t>(10);
  t.atom_pairs = metadata<Gfn2AtomPair>(11);
  t.bucket_offsets = metadata<std::int64_t>(12);
  t.bucket_systems = metadata<std::int32_t>(13);
  t.bucket_orbital_counts = metadata<std::int32_t>(14);
  return t;
}

void packed_pairs(Gfn2RaggedTopologyView& t) {
  t.pair_map_kind = Gfn2PairMapKind::kPackedLowerTriangle;
  t.atom_pair_count = 0;
  t.atom_pairs = nullptr;
}

void no_pairs(Gfn2RaggedTopologyView& t) {
  t.pair_map_kind = Gfn2PairMapKind::kNone;
  t.total_pairs = t.pair_offset_count = t.atom_pair_count = 0;
  t.pair_offsets = nullptr;
  t.atom_pairs = nullptr;
}

void no_buckets(Gfn2RaggedTopologyView& t) {
  t.bucket_count = t.bucket_offset_count = t.bucket_system_count = t.bucket_orbital_count = 0;
  t.bucket_offsets = nullptr;
  t.bucket_systems = t.bucket_orbital_counts = nullptr;
}

void set_pointer(Gfn2RaggedTopologyView& t, int index, std::uintptr_t address) {
#define POINTER_CASE(number, field)                         \
  case number:                                              \
    t.field = reinterpret_cast<decltype(t.field)>(address); \
    break
  switch (index) {
    POINTER_CASE(0, atom_offsets);
    POINTER_CASE(1, batch_shell_offsets);
    POINTER_CASE(2, batch_orbital_offsets);
    POINTER_CASE(3, matrix_offsets);
    POINTER_CASE(4, atom_shell_offsets);
    POINTER_CASE(5, shell_orbital_offsets);
    POINTER_CASE(6, shell_to_atom);
    POINTER_CASE(7, orbital_to_shell);
    POINTER_CASE(8, orbital_to_atom);
    POINTER_CASE(9, pair_offsets);
    POINTER_CASE(10, atom_pairs);
    POINTER_CASE(11, bucket_offsets);
    POINTER_CASE(12, bucket_systems);
    POINTER_CASE(13, bucket_orbital_counts);
    default:
      std::abort();
  }
#undef POINTER_CASE
}

const std::array<std::int64_t Gfn2RaggedTopologyView::*, 14> counts = {
    &Gfn2RaggedTopologyView::atom_offset_count,
    &Gfn2RaggedTopologyView::batch_shell_offset_count,
    &Gfn2RaggedTopologyView::batch_orbital_offset_count,
    &Gfn2RaggedTopologyView::matrix_offset_count,
    &Gfn2RaggedTopologyView::atom_shell_offset_count,
    &Gfn2RaggedTopologyView::shell_orbital_offset_count,
    &Gfn2RaggedTopologyView::shell_to_atom_count,
    &Gfn2RaggedTopologyView::orbital_to_shell_count,
    &Gfn2RaggedTopologyView::orbital_to_atom_count,
    &Gfn2RaggedTopologyView::pair_offset_count,
    &Gfn2RaggedTopologyView::atom_pair_count,
    &Gfn2RaggedTopologyView::bucket_offset_count,
    &Gfn2RaggedTopologyView::bucket_system_count,
    &Gfn2RaggedTopologyView::bucket_orbital_count,
};

bool starts(const std::string& value, const std::string& prefix) {
  return value.compare(0, prefix.size(), prefix) == 0;
}

void mutate(Gfn2RaggedTopologyView& t, const std::string& name) {
  if (name == "wrong-space")
    t.memory_space = Gfn2PlanMemorySpace::kHost;
  else if (name == "unknown-space")
    t.memory_space = static_cast<Gfn2PlanMemorySpace>(99);
  else if (name == "zero-token")
    t.plan_token = 0;
  else if (name == "bad-shape")
    --t.atom_offset_count;
  else if (name == "negative-batch")
    t.batch_size = -1;
  else if (name == "zero-batch")
    t.batch_size = 0;
  else if (name == "negative-atoms")
    t.total_atoms = -1;
  else if (name == "negative-shells")
    t.total_shells = -1;
  else if (name == "negative-orbitals")
    t.total_orbitals = -1;
  else if (name == "negative-matrices")
    t.total_matrix_elements = -1;
  else if (name == "negative-pairs")
    t.total_pairs = -1;
  else if (name == "negative-buckets")
    t.bucket_count = -1;
  else if (name == "overflow")
    t.batch_size = std::numeric_limits<std::int64_t>::max();
  else if (name == "atom-overflow")
    t.total_atoms = std::numeric_limits<std::int64_t>::max();
  else if (name == "shell-overflow")
    t.total_shells = std::numeric_limits<std::int64_t>::max();
  else if (name == "bucket-overflow")
    t.bucket_count = std::numeric_limits<std::int64_t>::max();
  else if (name == "alias")
    t.matrix_offsets = t.atom_offsets + 1;
  else if (name == "address-overflow") {
    t.atom_offsets =
        reinterpret_cast<const std::int64_t*>(std::numeric_limits<std::uintptr_t>::max() - 7);
  } else if (name == "byte-count-overflow") {
    no_buckets(t);
    t.total_orbitals = t.orbital_to_shell_count = t.orbital_to_atom_count =
        std::numeric_limits<std::int64_t>::max();
  } else if (name == "packed")
    packed_pairs(t);
  else if (name == "explicit") {
    t.pair_map_kind = Gfn2PairMapKind::kExplicit;
    t.atom_pair_count = t.total_pairs;
    t.atom_pairs = metadata<Gfn2AtomPair>(11);
  } else if (name == "no-pairs")
    no_pairs(t);
  else if (name == "no-buckets")
    no_buckets(t);
  else if (name == "unknown-pairs")
    t.pair_map_kind = static_cast<Gfn2PairMapKind>(99);
  else if (name == "empty-systems") {
    no_pairs(t);
    no_buckets(t);
    t.total_atoms = t.total_shells = t.total_orbitals = t.total_matrix_elements = 0;
    t.atom_shell_offset_count = t.shell_orbital_offset_count = 1;
    t.shell_to_atom_count = t.orbital_to_shell_count = t.orbital_to_atom_count = 0;
    t.shell_to_atom = t.orbital_to_shell = t.orbital_to_atom = nullptr;
  } else if (starts(name, "count-")) {
    ++(t.*counts.at(std::stoi(name.substr(6))));
  } else if (starts(name, "null-")) {
    set_pointer(t, std::stoi(name.substr(5)), 0);
  } else if (starts(name, "misaligned-")) {
    const int index = std::stoi(name.substr(11));
    set_pointer(t, index, arena + (index + 1) * 4096 + 1);
  } else if (starts(name, "alias-")) {
    const int index = std::stoi(name.substr(6));
    set_pointer(t, index, arena + (index == 0 ? 2 : 1) * 4096);
  }
}

std::vector<std::string> topology_mutations() {
  std::vector<std::string> names = {
      "valid",
      "wrong-space",
      "unknown-space",
      "zero-token",
      "bad-shape",
      "negative-batch",
      "zero-batch",
      "negative-atoms",
      "negative-shells",
      "negative-orbitals",
      "negative-matrices",
      "negative-pairs",
      "negative-buckets",
      "overflow",
      "atom-overflow",
      "shell-overflow",
      "bucket-overflow",
      "alias",
      "address-overflow",
      "byte-count-overflow",
      "packed",
      "explicit",
      "no-pairs",
      "no-buckets",
      "unknown-pairs",
      "empty-systems",
  };
  for (const std::string prefix : {"count-", "null-", "misaligned-", "alias-"}) {
    for (int index = 0; index < 14; ++index) names.push_back(prefix + std::to_string(index));
  }
  return names;
}

Gfn2PlanSchemaDiagnostic sentinel() {
  return {Gfn2PlanSchemaError::kElementCountMismatch, Gfn2PlanSchemaField::kElementIdentity, 777};
}

template <typename Input, typename Output>
void record(const std::string& name, int status, const Input& input, const Output& output,
            const Gfn2PlanSchemaDiagnostic& diagnostic) {
  // All descriptions are generated from scalar names/numbers, without quotes
  // or backslashes, so this bounded JSON encoding needs no string escaping.
  std::cout << "{\"case\":\"" << name << "\",\"status\":" << status << ",\"input\":\""
            << describe(input) << "\",\"binding\":\"" << describe(output)
            << "\",\"cleared\":" << (describe(output) == describe(Output{}) ? "true" : "false")
            << ",\"published\":" << (output.plan_token == kToken ? "true" : "false")
            << ",\"diagnostic\":\"" << describe(diagnostic) << "\",\"events\":[";
  for (std::size_t index = 0; index < events.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << '"' << events[index] << '"';
  }
  std::cout << "],\"observations\":[";
  for (std::size_t index = 0; index < observations.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << '"' << observations[index] << '"';
  }
  std::cout << "]}\n";
}

Gfn2PlanSchemaDiagnostic* configure_boundary(const std::string& mutation,
                                             const Gfn2RaggedTopologyView& t) {
  auto* device = metadata<Gfn2PlanSchemaDiagnostic>(15);
  if (mutation == "diagnostic-null")
    device = nullptr;
  else if (mutation == "diagnostic-misaligned") {
    device = reinterpret_cast<Gfn2PlanSchemaDiagnostic*>(arena + 15 * 4096 + 1);
  } else if (mutation == "diagnostic-overflow") {
    device =
        reinterpret_cast<Gfn2PlanSchemaDiagnostic*>(std::numeric_limits<std::uintptr_t>::max() - 7);
  } else if (mutation == "diagnostic-alias") {
    device = const_cast<Gfn2PlanSchemaDiagnostic*>(
        reinterpret_cast<const Gfn2PlanSchemaDiagnostic*>(t.atom_offsets));
  } else if (starts(mutation, "diagnostic-alias-")) {
    device = metadata<Gfn2PlanSchemaDiagnostic>(std::stoi(mutation.substr(17)) + 1);
  } else if (starts(mutation, "attribute-error-")) {
    fail_attribute = std::stoi(mutation.substr(16));
  } else if (starts(mutation, "host-pointer-")) {
    host_attribute = std::stoi(mutation.substr(13));
  } else if (mutation == "managed")
    managed_memory = true;
  else if (mutation == "launch-error")
    launch_status = 701;
  else if (mutation == "copy-error")
    copy_status = 702;
  else if (mutation == "sync-error")
    sync_status = 703;
  else if (mutation == "semantic-error") {
    copied_diagnostic = {Gfn2PlanSchemaError::kInvalidOffsets, Gfn2PlanSchemaField::kAtomOffsets,
                         1};
  }
  return device;
}

void run_topology(const std::string& mutation, bool asynchronous) {
  reset();
  auto t = topology();
  mutate(t, mutation);
  auto* device = configure_boundary(mutation, t);
  auto diagnostic = sentinel();
  host_diagnostic = &diagnostic;
  Gfn2RaggedTopologyView output{};
  if (!asynchronous) poison(output);
  const auto stream = reinterpret_cast<cudaStream_t>(0x1234);
  const int status = asynchronous
                         ? adapter::validate_gfn2_topology_cuda_async(t, device, stream)
                         : adapter::bind_gfn2_topology_cuda(t, output, device, diagnostic, stream);
  record(std::string(asynchronous ? "topology-async/" : "topology-bind/") + mutation, status, t,
         output, diagnostic);
}

template <typename View, typename Binder>
void run_projection(const std::string& name, const std::string& mutation, Binder binder) {
  reset();
  auto t = topology();
  if (name == "packed-pair") packed_pairs(t);
  mutate(t, mutation);
  View output{};
  poison(output);
  const int status = binder(t, output);
  record(name + "/" + mutation, status, t, output, sentinel());
}

void run_provenance(const std::string& mutation, bool asynchronous) {
  reset();
  auto t = topology();
  mutate(t, mutation);
  Gfn2GeometryCacheProvenanceView input{};
  input.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  input.generation_scope = Gfn2GenerationScope::kPerSystem;
  input.plan_token = kToken;
  input.batch_size = 2;
  input.system_generation_count = 2;
  input.system_geometry_generations = metadata<std::uint64_t>(16);
  const std::uint8_t* active = metadata<std::uint8_t>(17);
  std::int64_t active_count = 2;
  std::uint64_t expected_generation = 37;
  // Boundary mutations whose suffixes are not numeric belong to provenance.
  auto* device = metadata<Gfn2PlanSchemaDiagnostic>(15);
  if (!starts(mutation, "diagnostic-alias-generation") &&
      !starts(mutation, "diagnostic-alias-active"))
    device = configure_boundary(mutation, t);
  if (mutation == "cross-plan")
    ++input.plan_token;
  else if (mutation == "provenance-space")
    input.memory_space = Gfn2PlanMemorySpace::kHost;
  else if (mutation == "generation-count")
    --input.system_generation_count;
  else if (mutation == "generation-null")
    input.system_geometry_generations = nullptr;
  else if (mutation == "generation-alias") {
    input.system_geometry_generations = reinterpret_cast<const std::uint64_t*>(t.atom_offsets);
  } else if (mutation == "batch-generation") {
    input.generation_scope = Gfn2GenerationScope::kBatch;
    input.geometry_generation = 37;
    input.system_generation_count = 0;
    input.system_geometry_generations = nullptr;
  } else if (mutation == "zero-generation")
    expected_generation = 0;
  else if (mutation == "active-null") {
    active = nullptr;
    active_count = 0;
  } else if (mutation == "active-null-count")
    active = nullptr;
  else if (mutation == "active-count")
    --active_count;
  else if (mutation == "active-alias-topology")
    active = reinterpret_cast<const std::uint8_t*>(t.atom_offsets);
  else if (mutation == "active-alias-generation")
    active = reinterpret_cast<const std::uint8_t*>(input.system_geometry_generations);
  else if (mutation == "diagnostic-alias-generation")
    device = metadata<Gfn2PlanSchemaDiagnostic>(16);
  else if (mutation == "diagnostic-alias-active")
    device = metadata<Gfn2PlanSchemaDiagnostic>(17);
  auto diagnostic = sentinel();
  host_diagnostic = &diagnostic;
  Gfn2GeometryCacheProvenanceView output{};
  if (!asynchronous) poison(output);
  const auto stream = reinterpret_cast<cudaStream_t>(0x1234);
  const int status =
      asynchronous ? adapter::validate_gfn2_geometry_provenance_cuda_async(
                         t, input, expected_generation, active, active_count, device, stream)
                   : adapter::bind_gfn2_geometry_provenance_cuda(t, input, expected_generation,
                                                                 active, active_count, output,
                                                                 device, diagnostic, stream);
  record(std::string(asynchronous ? "provenance-async/" : "provenance-bind/") + mutation, status,
         input, output, diagnostic);
}

void run_element(const std::string& mutation) {
  reset();
  Gfn2ElementIdentityProjectionView input{};
  input.plan_token = kToken;
  input.total_atoms = input.atomic_number_count = 3;
  input.element_fingerprint = 0x12345678;
  input.atomic_numbers = metadata<std::int32_t>(18);
  const std::int32_t* device = metadata<std::int32_t>(19);
  if (mutation == "zero-token")
    input.plan_token = 0;
  else if (mutation == "zero-fingerprint")
    input.element_fingerprint = 0;
  else if (mutation == "bad-count")
    ++input.atomic_number_count;
  else if (mutation == "negative-count")
    input.total_atoms = input.atomic_number_count = -1;
  else if (mutation == "attribute-error")
    fail_attribute = 0;
  else if (mutation == "host-pointer")
    host_attribute = 0;
  else if (mutation == "managed")
    managed_memory = true;
  else if (mutation == "null")
    device = nullptr;
  else if (mutation == "misaligned")
    device = reinterpret_cast<const std::int32_t*>(arena + 19 * 4096 + 1);
  else if (mutation == "address-overflow") {
    device = reinterpret_cast<const std::int32_t*>(std::numeric_limits<std::uintptr_t>::max() - 3);
  } else if (mutation == "empty" || mutation == "empty-nonnull") {
    input.total_atoms = input.atomic_number_count = 0;
    input.atomic_numbers = nullptr;
    if (mutation == "empty") device = nullptr;
  }
  Gfn2ElementIdentityProjectionView output{};
  poison(output);
  const int status = adapter::bind_gfn2_element_identity_projection_cuda(input, device, output);
  record("element/" + mutation, status, input, output, sentinel());
}

void run_self_binding() {
  const auto stream = reinterpret_cast<cudaStream_t>(0x1234);
  auto* device = metadata<Gfn2PlanSchemaDiagnostic>(15);
  {
    reset();
    auto input = topology();
    const auto before = input;
    auto diagnostic = sentinel();
    host_diagnostic = &diagnostic;
    const int status = adapter::bind_gfn2_topology_cuda(input, input, device, diagnostic, stream);
    record("topology-bind/self-binding", status, before, input, diagnostic);
  }
  {
    reset();
    const auto t = topology();
    Gfn2GeometryCacheProvenanceView input{};
    input.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    input.plan_token = kToken;
    input.batch_size = 2;
    input.geometry_generation = 37;
    const auto before = input;
    auto diagnostic = sentinel();
    host_diagnostic = &diagnostic;
    const int status = adapter::bind_gfn2_geometry_provenance_cuda(t, input, 37, nullptr, 0, input,
                                                                   device, diagnostic, stream);
    record("provenance-bind/self-binding", status, before, input, diagnostic);
  }
  {
    reset();
    Gfn2ElementIdentityProjectionView input{};
    input.plan_token = kToken;
    input.element_fingerprint = 37;
    input.total_atoms = input.atomic_number_count = 3;
    input.atomic_numbers = metadata<std::int32_t>(18);
    const auto before = input;
    const int status = adapter::bind_gfn2_element_identity_projection_cuda(
        input, metadata<std::int32_t>(19), input);
    record("element/self-binding", status, before, input, sentinel());
  }
}

void run_host_diagnostic_alias(const std::string& mutation) {
  reset();
  auto t = topology();
  mutate(t, "empty-systems");
  auto diagnostic = sentinel();
  host_diagnostic = &diagnostic;
  // A one-element int64 array can legally be the diagnostic.index subobject.
  // No incompatible struct casts or out-of-bounds array extents are involved.
  t.atom_shell_offsets = &diagnostic.index;
  if (mutation == "invalid")
    --t.atom_offset_count;
  else if (mutation == "copy-error")
    copy_status = 702;
  else if (mutation == "sync-error")
    sync_status = 703;
  else if (mutation == "semantic-error") {
    copied_diagnostic = {Gfn2PlanSchemaError::kInvalidOffsets, Gfn2PlanSchemaField::kAtomOffsets,
                         1};
  }
  Gfn2RaggedTopologyView output{};
  poison(output);
  watched_index = &diagnostic.index;
  watched_output_token = &output.plan_token;
  observe("before");
  const int status =
      adapter::bind_gfn2_topology_cuda(t, output, metadata<Gfn2PlanSchemaDiagnostic>(15),
                                       diagnostic, reinterpret_cast<cudaStream_t>(0x1234));
  observe("after");
  record("topology-bind/host-diagnostic-index-" + mutation, status, t, output, diagnostic);
}

int main() {
  void* mapping = mmap(nullptr, kArenaBytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED) return 2;
  arena = reinterpret_cast<std::uintptr_t>(mapping);
  run_self_binding();
  for (const std::string mutation :
       {"valid", "invalid", "copy-error", "sync-error", "semantic-error"}) {
    run_host_diagnostic_alias(mutation);
  }
  const auto mutations = topology_mutations();
  for (const auto& mutation : mutations) {
    run_topology(mutation, false);
    run_topology(mutation, true);
    run_projection<Gfn2AtomProjectionView>("atom", mutation,
                                           adapter::bind_gfn2_atom_projection_cuda);
    run_projection<Gfn2ShellOwnershipProjectionView>(
        "shell", mutation, adapter::bind_gfn2_shell_ownership_projection_cuda);
    run_projection<Gfn2AOMatrixProjectionView>("ao-matrix", mutation,
                                               adapter::bind_gfn2_ao_matrix_projection_cuda);
    run_projection<Gfn2PackedAllPairProjectionView>(
        "packed-pair", mutation, adapter::bind_gfn2_packed_all_pair_projection_cuda);
    run_projection<Gfn2AOBucketProjectionView>("ao-bucket", mutation,
                                               adapter::bind_gfn2_ao_bucket_projection_cuda);
  }
  for (const std::string mutation :
       {"diagnostic-null", "diagnostic-misaligned", "diagnostic-overflow", "diagnostic-alias",
        "managed", "launch-error", "copy-error", "sync-error", "semantic-error"}) {
    run_topology(mutation, false);
    run_topology(mutation, true);
  }
  for (int index = 0; index < 15; ++index) {
    run_topology("attribute-error-" + std::to_string(index), false);
    run_topology("host-pointer-" + std::to_string(index), false);
  }
  for (int index = 0; index < 14; ++index) {
    run_topology("diagnostic-alias-" + std::to_string(index), false);
    run_topology("diagnostic-alias-" + std::to_string(index), true);
  }
  for (const std::string mutation : {"valid",
                                     "wrong-space",
                                     "zero-token",
                                     "bad-shape",
                                     "cross-plan",
                                     "provenance-space",
                                     "generation-count",
                                     "generation-null",
                                     "generation-alias",
                                     "batch-generation",
                                     "zero-generation",
                                     "active-null",
                                     "active-null-count",
                                     "active-count",
                                     "active-alias-topology",
                                     "active-alias-generation",
                                     "diagnostic-alias-generation",
                                     "diagnostic-alias-active",
                                     "diagnostic-null",
                                     "diagnostic-misaligned",
                                     "diagnostic-overflow",
                                     "diagnostic-alias",
                                     "managed",
                                     "launch-error",
                                     "copy-error",
                                     "sync-error",
                                     "semantic-error",
                                     "attribute-error-0",
                                     "attribute-error-1",
                                     "attribute-error-2",
                                     "host-pointer-0",
                                     "host-pointer-1",
                                     "host-pointer-2"}) {
    run_provenance(mutation, false);
    run_provenance(mutation, true);
  }
  for (const std::string mutation :
       {"valid", "zero-token", "zero-fingerprint", "bad-count", "negative-count", "attribute-error",
        "host-pointer", "managed", "null", "misaligned", "address-overflow", "empty",
        "empty-nonnull"})
    run_element(mutation);
  return munmap(mapping, kArenaBytes) == 0 ? 0 : 3;
}
