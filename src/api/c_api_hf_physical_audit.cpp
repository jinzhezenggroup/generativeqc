#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "runtime/bounded_workspace.hpp"
#include "scf/hf_physical_audit.hpp"

namespace {
namespace audit = generativeqc::scf::physical_audit;

audit::Controls controls(const generativeqc_hf_physical_audit_controls_v1& input) {
  if (input.unrestricted > 1 || input.density_fitted > 1 || input.reserved ||
      input.maximum_host_bytes > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("invalid physical HF audit metadata");
  return {bool(input.unrestricted), bool(input.density_fitted), input.diis_history,
          input.maximum_host_bytes, input.screening_tolerance,  input.metric_relative_threshold};
}

void disjoint(const void* first, std::size_t first_bytes, const void* second,
              std::size_t second_bytes) {
  if (generativeqc::runtime::ranges_overlap(first, first_bytes, second, second_bytes))
    throw std::invalid_argument("physical HF audit borrowed/publication ranges overlap");
}

generativeqc_status failure(std::string* detail = nullptr) {
  try {
    throw;
  } catch (const audit::Unsupported& error) {
    if (detail) *detail = error.what();
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  } catch (const std::overflow_error& error) {
    if (detail) *detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    return generativeqc::api::map_exception(detail);
  }
}
}  // namespace

extern "C" generativeqc_status generativeqc_hf_physical_audit_cpu_query_v1(
    const generativeqc_system* system, const generativeqc_system* auxiliary,
    const generativeqc_hf_physical_audit_controls_v1* input,
    generativeqc_hf_physical_audit_capacity_v1* output) {
  if (!system || !input || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(input) || !generativeqc::api::valid_descriptor(output))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  try {
    disjoint(input, sizeof(*input), output, sizeof(*output));
    const auto result =
        audit::capacity(system->data, auxiliary ? &auxiliary->data : nullptr, controls(*input));
    *output = {sizeof(*output),       GENERATIVEQC_ABI_VERSION, result.ao_count,
               result.spin_count,     result.density_count,     result.overlap_count,
               result.provider_bytes, result.workspace_bytes,   result.required_host_bytes};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return failure();
  }
}

extern "C" generativeqc_status generativeqc_hf_physical_audit_cpu_v1(
    generativeqc_context* context, const generativeqc_system* system,
    const generativeqc_system* auxiliary, const generativeqc_hf_physical_audit_controls_v1* input,
    const double* density, std::uint64_t density_count, double target_energy,
    generativeqc_hf_physical_audit_result_v1* output) {
  if (!context || !system || !input || !output || !density)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(input) || !generativeqc::api::valid_descriptor(output))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  context->last_detail.clear();
  try {
    if (context->state.executed_backend != generativeqc::core::kCpuReferenceBackend)
      throw audit::Unsupported("physical HF CPU audit requires an explicit CPU context");
    const auto selected = controls(*input);
    const auto required =
        audit::capacity(system->data, auxiliary ? &auxiliary->data : nullptr, selected);
    if (density_count != required.density_count || !output->overlap ||
        output->overlap_capacity < required.overlap_count)
      throw std::invalid_argument("physical HF audit density or overlap count mismatch");
    const auto maximum_count = std::numeric_limits<std::size_t>::max() / sizeof(double);
    if (density_count > maximum_count || output->overlap_capacity > maximum_count ||
        reinterpret_cast<std::uintptr_t>(density) % alignof(double) ||
        reinterpret_cast<std::uintptr_t>(output->overlap) % alignof(double))
      throw std::invalid_argument("physical HF audit array range overflow or misalignment");
    const auto density_bytes = density_count * sizeof(double);
    const auto overlap_bytes = required.overlap_count * sizeof(double);
    disjoint(input, sizeof(*input), output, sizeof(*output));
    disjoint(density, density_bytes, input, sizeof(*input));
    disjoint(density, density_bytes, output, sizeof(*output));
    disjoint(output->overlap, overlap_bytes, input, sizeof(*input));
    disjoint(output->overlap, overlap_bytes, output, sizeof(*output));
    disjoint(output->overlap, overlap_bytes, density, density_bytes);
    const auto result =
        audit::evaluate(system->data, auxiliary ? &auxiliary->data : nullptr, selected,
                        {density, static_cast<std::size_t>(density_count)}, target_energy);
    auto published = *output;
    published.energy = result.energy;
    published.energy_difference = result.energy_difference;
    published.maximum_commutator = result.maximum_commutator;
    published.rms_commutator = result.rms_commutator;
    published.fock_builds = result.fock_builds;
    published.auxiliary_rank = result.auxiliary_rank;
    std::copy(result.overlap.begin(), result.overlap.end(), output->overlap);
    *output = published;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return failure(&context->last_detail);
  }
}
