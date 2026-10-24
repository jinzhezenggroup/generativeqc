#include <limits>
#include <memory>
#include <stdexcept>

#include "api/error.hpp"
#include "generativeqc/generativeqc.h"
#include "response/native_gmres.hpp"
#include "response/sequence.hpp"

struct generativeqc_response_result {
  std::size_t dimension;
  generativeqc::response::GmresResult data;
};

namespace {

struct CallbackFailure {
  generativeqc_status status;
};

std::size_t native_count(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("response count exceeds native size_t");
  return static_cast<std::size_t>(value);
}

generativeqc::response::GmresPlan plan_for(
    const generativeqc_response_gmres_descriptor_v1& descriptor) {
  constexpr auto flags = GENERATIVEQC_RESPONSE_COLLECT_HISTORY |
                         GENERATIVEQC_RESPONSE_COLLECT_BASIS | GENERATIVEQC_RESPONSE_COLLECT_TIMING;
  if (descriptor.flags & ~flags) throw std::invalid_argument("unknown response diagnostic flag");
  generativeqc::response::GmresOptions options;
  options.relative_tolerance = descriptor.rtol;
  options.absolute_tolerance = descriptor.atol;
  options.restart = native_count(descriptor.restart);
  options.max_iterations = native_count(descriptor.max_iterations);
  options.max_workspace_bytes = native_count(descriptor.max_workspace_bytes);
  options.reorthogonalize = descriptor.reorthogonalize;
  options.breakdown_tolerance = descriptor.breakdown_tolerance;
  options.true_residual_every = native_count(descriptor.true_residual_every);
  options.stagnation_window = native_count(descriptor.stagnation_window);
  options.stagnation_tolerance = descriptor.stagnation_tolerance;
  options.collect_history = descriptor.flags & GENERATIVEQC_RESPONSE_COLLECT_HISTORY;
  options.collect_basis = descriptor.flags & GENERATIVEQC_RESPONSE_COLLECT_BASIS;
  options.collect_timing = descriptor.flags & GENERATIVEQC_RESPONSE_COLLECT_TIMING;
  return generativeqc::response::prepare_gmres(native_count(descriptor.dimension), options);
}

bool abi_valid(const generativeqc_response_gmres_descriptor_v1& descriptor) {
  return descriptor.struct_size >= sizeof(descriptor) &&
         descriptor.abi_version == GENERATIVEQC_ABI_VERSION;
}

generativeqc::response::LinearOperator action_for(generativeqc_response_action_v1 action,
                                                  void* user_data) {
  if (!action) return {};
  return [action, user_data](std::span<const double> input, std::span<double> output) {
    const auto status = action(user_data, input.data(), output.data(), input.size());
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw CallbackFailure{status};
  };
}

std::int32_t reason_for(generativeqc::response::GmresStatus status) {
  using generativeqc::response::GmresStatus;
  switch (status) {
    case GmresStatus::initial_residual:
      return GENERATIVEQC_RESPONSE_INITIAL_RESIDUAL;
    case GmresStatus::converged:
      return GENERATIVEQC_RESPONSE_CONVERGED;
    case GmresStatus::workspace_limit:
      return GENERATIVEQC_RESPONSE_WORKSPACE_LIMIT;
    case GmresStatus::max_iterations:
      return GENERATIVEQC_RESPONSE_MAX_ITERATIONS;
    case GmresStatus::breakdown:
      return GENERATIVEQC_RESPONSE_BREAKDOWN;
    case GmresStatus::stagnation:
      return GENERATIVEQC_RESPONSE_STAGNATION;
    case GmresStatus::nonfinite_input:
      return GENERATIVEQC_RESPONSE_NONFINITE_INPUT;
    case GmresStatus::nonfinite_operator:
      return GENERATIVEQC_RESPONSE_NONFINITE_OPERATOR;
    case GmresStatus::nonfinite_preconditioner:
      return GENERATIVEQC_RESPONSE_NONFINITE_PRECONDITIONER;
  }
  throw std::logic_error("unrecognized response solve status");
}

generativeqc_response_result_view_v1 view_for(const generativeqc::response::GmresResult& data,
                                              std::size_t dimension) {
  return {sizeof(generativeqc_response_result_view_v1),
          GENERATIVEQC_ABI_VERSION,
          reason_for(data.status),
          data.converged(),
          dimension,
          data.solution.empty() ? nullptr : data.solution.data(),
          data.solution.size(),
          data.history.empty() ? nullptr : data.history.data(),
          data.history.size(),
          data.basis.empty() ? nullptr : data.basis.data(),
          data.basis_columns,
          data.residual_norm,
          data.relative_residual,
          data.iterations,
          data.restarts,
          data.operator_actions,
          data.preconditioner_actions,
          data.workspace_bytes,
          data.measured_workspace_peak_bytes,
          data.workspace_allocation_count,
          data.orthogonalization_seconds,
          data.operator_seconds};
}

bool abi_valid(const generativeqc_response_sequence_descriptor_v1& descriptor) {
  return descriptor.struct_size >= sizeof(descriptor) &&
         descriptor.abi_version == GENERATIVEQC_ABI_VERSION && abi_valid(descriptor.gmres);
}

generativeqc::response::ResponseSequencePlan sequence_plan_for(
    const generativeqc_response_sequence_descriptor_v1& descriptor) {
  const auto gmres = plan_for(descriptor.gmres);
  return generativeqc::response::prepare_response_sequence(
      gmres.dimension, gmres.options, native_count(descriptor.rhs_count),
      native_count(descriptor.row_stride), native_count(descriptor.column_stride),
      native_count(descriptor.rhs_element_count), native_count(descriptor.max_numeric_bytes));
}

void validate_sequence(const generativeqc_response_sequence_descriptor_v1& descriptor) {
  if (descriptor.validate) {
    const auto status = descriptor.validate(descriptor.validate_data);
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw CallbackFailure{status};
  }
}

}  // namespace

extern "C" {

generativeqc_status generativeqc_response_gmres_workspace_v1(
    const generativeqc_response_gmres_descriptor_v1* descriptor, std::uint64_t* output) {
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!abi_valid(*descriptor)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  try {
    const auto plan = plan_for(*descriptor);
    *output = plan.workspace_bytes;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::overflow_error&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    return generativeqc::api::map_exception();
  }
}

generativeqc_status generativeqc_response_solve_v1(
    const generativeqc_response_gmres_descriptor_v1* descriptor,
    generativeqc_response_result** result) {
  if (!descriptor || !result) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *result = nullptr;
  if (!abi_valid(*descriptor)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (!descriptor->rhs || !descriptor->apply) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    const auto plan = plan_for(*descriptor);
    const auto guess = descriptor->initial_guess
                           ? std::span(descriptor->initial_guess, plan.dimension)
                           : std::span<const double>{};
    auto data = generativeqc::response::solve_gmres(
        plan, action_for(descriptor->apply, descriptor->apply_data),
        std::span(descriptor->rhs, plan.dimension), guess, {},
        action_for(descriptor->precondition, descriptor->precondition_data));
    auto owned = std::make_unique<generativeqc_response_result>(
        generativeqc_response_result{plan.dimension, std::move(data)});
    *result = owned.release();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const CallbackFailure& failure) {
    return failure.status;
  } catch (const std::overflow_error&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    return generativeqc::api::map_exception();
  }
}

generativeqc_status generativeqc_response_result_view_v1_get(
    const generativeqc_response_result* result, generativeqc_response_result_view_v1* output) {
  if (!result || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (output->struct_size < sizeof(*output) || output->abi_version != GENERATIVEQC_ABI_VERSION)
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  *output = view_for(result->data, result->dimension);
  return GENERATIVEQC_STATUS_SUCCESS;
}

void generativeqc_response_result_destroy(generativeqc_response_result* result) { delete result; }

generativeqc_status generativeqc_response_sequence_workspace_v1(
    const generativeqc_response_sequence_descriptor_v1* descriptor, std::uint64_t* first,
    std::uint64_t* complete) {
  if (!descriptor || !first || !complete) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!abi_valid(*descriptor)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  try {
    const auto plan = sequence_plan_for(*descriptor);
    *first = plan.first_workspace_bytes;
    *complete = plan.complete_workspace_bytes;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::overflow_error&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    return generativeqc::api::map_exception();
  }
}

generativeqc_status generativeqc_response_sequence_solve_v1(
    const generativeqc_response_sequence_descriptor_v1* descriptor,
    generativeqc_response_sequence_summary_v1* output) {
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!abi_valid(*descriptor) || output->struct_size < sizeof(*output) ||
      output->abi_version != GENERATIVEQC_ABI_VERSION)
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (descriptor->gmres.initial_guess ||
      (descriptor->rhs_count &&
       (!descriptor->gmres.rhs || !descriptor->gmres.apply || !descriptor->publish)))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    const auto plan = sequence_plan_for(*descriptor);
    generativeqc::response::ResponseSequenceResult result;
    if (!plan.request_count) {
      validate_sequence(*descriptor);
    } else {
      const generativeqc::response::LinearResponseProblem problem(
          plan.solve_plan.dimension,
          action_for(descriptor->gmres.apply, descriptor->gmres.apply_data));
      result = generativeqc::response::solve_response_sequence(
          plan, problem, std::span(descriptor->gmres.rhs, plan.storage_count),
          [&](std::size_t index, const generativeqc::response::GmresResult& data) {
            const auto view = view_for(data, problem.dimension());
            const auto status = descriptor->publish(descriptor->publish_data, index, &view);
            if (status != GENERATIVEQC_STATUS_SUCCESS) throw CallbackFailure{status};
          },
          action_for(descriptor->gmres.precondition, descriptor->gmres.precondition_data),
          [&] { validate_sequence(*descriptor); });
    }
    *output = {sizeof(*output),
               GENERATIVEQC_ABI_VERSION,
               result.workspace_limited,
               result.published_count,
               result.converged_count,
               result.operator_actions,
               result.preconditioner_actions,
               result.peak_workspace_bytes,
               result.retained_solution_basis_bytes,
               result.staging_peak_bytes};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const CallbackFailure& failure) {
    return failure.status;
  } catch (const std::overflow_error&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    return generativeqc::api::map_exception();
  }
}

}  // extern "C"
