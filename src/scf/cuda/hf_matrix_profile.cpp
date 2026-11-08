#include "scf/cuda/hf_matrix_profile.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

namespace generativeqc::scf::cuda_policy {
namespace {
constexpr std::size_t kMaximumProfileBytes = 8192;

bool hex_identity(std::string_view value, std::size_t minimum, std::size_t maximum) {
  return value.size() >= minimum && value.size() <= maximum &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
         });
}

template <std::size_t Capacity>
bool read_identity(std::istream& stream, std::array<char, Capacity>& output, std::size_t minimum) {
  std::string token;
  if (!(stream >> token) || !hex_identity(token, minimum, Capacity - 1)) return false;
  std::copy(token.begin(), token.end(), output.begin());
  return true;
}

bool key(std::istream& stream, std::string_view expected) {
  std::string token;
  return (stream >> token) && token == expected;
}

template <std::size_t Capacity>
bool stored_identity(const std::array<char, Capacity>& value, std::size_t minimum) {
  const auto end = std::find(value.begin(), value.end(), '\0');
  return end != value.end() &&
         hex_identity({value.data(), static_cast<std::size_t>(end - value.begin())}, minimum,
                      Capacity - 1);
}

bool complete_context(const HfMatrixProfileContext& context) {
  const auto& target = context.target;
  return context.provider == runtime::CudaProviderKind::Nvidia &&
         stored_identity(context.source_identity, 64) &&
         stored_identity(context.native_build_id, 1) &&
         stored_identity(context.runtime_build_id, 1) &&
         stored_identity(context.driver_build_id, 1) &&
         stored_identity(context.cublas_build_id, 1) && context.driver_version > 0 &&
         context.runtime_version > 0 && context.toolkit_version > 0 &&
         context.cublas_version[0] > 0 && context.cublas_version[1] >= 0 &&
         context.cublas_version[2] >= 0 && target.compute_capability_major > 0 &&
         target.compute_capability_minor >= 0 && target.warp_size > 0 &&
         target.maximum_threads_per_block > 0 && target.maximum_threads_per_sm > 0 &&
         target.maximum_blocks_per_sm > 0 && target.registers_per_sm > 0 &&
         target.shared_memory_per_block > 0 && target.shared_memory_per_block_optin > 0 &&
         target.shared_memory_per_sm > 0 && target.multiprocessor_count > 0 &&
         target.total_global_memory > 0 && target.fp32_fp64_ratio > 0;
}
}  // namespace

HfMatrixProfileAdmission admit_hf_matrix_profile(std::string_view payload,
                                                 const HfMatrixProfileContext& context) {
  const HfMatrixProfileAdmission invalid{HfMatrixProfileStatus::Invalid, {}};
  if (payload.empty() || payload.size() > kMaximumProfileBytes) return invalid;
  std::istringstream stream{std::string(payload)};
  stream.imbue(std::locale::classic());
  // Formatted unsigned extraction accepts negative spellings modulo 2^N.
  // Disallow signed tokens before extracting any dimensions or resource facts.
  std::string token;
  while (stream >> token)
    if (token.front() == '-' || token.front() == '+') return invalid;
  stream.clear();
  stream.seekg(0);
  HfMatrixProfile record;
  auto& stack = record.context;
  auto& target = stack.target;
  int release = 0, fast = 0;
  if (!key(stream, "generativeqc-hf-matrix-profile-v1") || !key(stream, "source_identity") ||
      !read_identity(stream, stack.source_identity, 64) || !key(stream, "provider") ||
      !key(stream, "nvidia") || !key(stream, "versions") ||
      !(stream >> stack.driver_version >> stack.runtime_version >> stack.toolkit_version >>
        stack.cublas_version[0] >> stack.cublas_version[1] >> stack.cublas_version[2]) ||
      !key(stream, "build_mode") || !(stream >> release >> fast) ||
      (release != 0 && release != 1) || (fast != 0 && fast != 1) ||
      !key(stream, "native_build_id") || !read_identity(stream, stack.native_build_id, 1) ||
      !key(stream, "runtime_build_id") || !read_identity(stream, stack.runtime_build_id, 1) ||
      !key(stream, "driver_build_id") || !read_identity(stream, stack.driver_build_id, 1) ||
      !key(stream, "cublas_build_id") || !read_identity(stream, stack.cublas_build_id, 1) ||
      !key(stream, "target") ||
      !(stream >> target.compute_capability_major >> target.compute_capability_minor >>
        target.warp_size >> target.maximum_threads_per_block >> target.maximum_threads_per_sm >>
        target.maximum_blocks_per_sm >> target.registers_per_sm >> target.shared_memory_per_block >>
        target.shared_memory_per_block_optin >> target.shared_memory_per_sm >>
        target.multiprocessor_count >> target.total_global_memory >> target.fp32_fp64_ratio >>
        target.clock_rate_khz >> target.memory_clock_rate_khz >> target.memory_bus_width_bits) ||
      !key(stream, "bounds") ||
      !(stream >> record.minimum_dimension >> record.maximum_dimension >> record.minimum_matrices >>
        record.maximum_matrices) ||
      !key(stream, "evidence_sha256") || !read_identity(stream, record.evidence_sha256, 64) ||
      !key(stream, "native") ||
      !(stream >> record.calibration.native_launch_nanoseconds >>
        record.calibration.native_fp64_flops_per_nanosecond >>
        record.calibration.native_bytes_per_nanosecond) ||
      !key(stream, "cublas") ||
      !(stream >> record.calibration.cublas_launch_nanoseconds >>
        record.calibration.cublas_fp64_flops_per_nanosecond >>
        record.calibration.cublas_bytes_per_nanosecond) ||
      !key(stream, "provider_setup_nanoseconds") ||
      !(stream >> record.provider_setup_nanoseconds) || !key(stream, "end"))
    return invalid;
  stream >> std::ws;
  if (!stream.eof()) return invalid;
  stack.provider = runtime::CudaProviderKind::Nvidia;
  stack.release_build = release != 0;
  stack.fast_compile = fast != 0;
  if (!complete_context(stack) || !record.minimum_dimension || !record.minimum_matrices ||
      record.minimum_dimension > record.maximum_dimension ||
      record.minimum_matrices > record.maximum_matrices ||
      record.maximum_dimension > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      record.maximum_matrices > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      !tensor::CudaMatrixProductAdmission::resolve(stack.target, record.maximum_dimension,
                                                   record.maximum_matrices, record.matrix_profile())
           .calibrated())
    return invalid;
  if (!complete_context(context) || stack != context)
    return {HfMatrixProfileStatus::Incompatible, {}};
  return {HfMatrixProfileStatus::Admitted, record};
}

HfMatrixProfileAdmission load_hf_matrix_profile(const char* path,
                                                const HfMatrixProfileContext& context) {
  if (!path || !*path) return {};
  const HfMatrixProfileAdmission unreadable{HfMatrixProfileStatus::Unreadable, {}};
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error) || error) return unreadable;
  std::ifstream stream(path, std::ios::binary);
  if (!stream) return unreadable;
  std::array<char, kMaximumProfileBytes + 1> contents{};
  stream.read(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (stream.bad()) return unreadable;
  return admit_hf_matrix_profile({contents.data(), static_cast<std::size_t>(stream.gcount())},
                                 context);
}

}  // namespace generativeqc::scf::cuda_policy
