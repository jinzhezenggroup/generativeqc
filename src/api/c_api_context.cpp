#include <cstddef>
#include <memory>
#include <utility>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "generativeqc/generativeqc.h"
#include "molecule/basis.hpp"
#include "runtime/context.hpp"

extern "C" {

generativeqc_status generativeqc_context_create(const generativeqc_context_descriptor* descriptor,
                                                generativeqc_context** context) {
  if (context == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *context = nullptr;
  if (descriptor == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (descriptor->abi_version != GENERATIVEQC_ABI_VERSION ||
      descriptor->struct_size < sizeof(generativeqc_context_descriptor)) {
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  }
  try {
    auto candidate = std::make_unique<generativeqc_context>();
    candidate->state.device_id = descriptor->device_id;
    candidate->state.requested_backend = descriptor->backend;
    const generativeqc_status status =
        generativeqc::runtime::initialize_context(candidate->state, candidate->last_detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    *context = candidate.release();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception();
  }
}

void generativeqc_context_destroy(generativeqc_context* context) { delete context; }

const char* generativeqc_context_get_last_detail(const generativeqc_context* context) {
  if (context == nullptr) return "invalid context";
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  // Failure storage belongs to the context and is only replaced on failure.
  // Successful operations and queries must preserve every borrowed pointer.
  return context->last_detail.c_str();
}

const char* generativeqc_context_last_error(const generativeqc_context* context) {
  return generativeqc_context_get_last_detail(context);
}

generativeqc_status generativeqc_system_create(generativeqc_context* context,
                                               const generativeqc_system_descriptor* descriptor,
                                               generativeqc_system** system) {
  if (context == nullptr || system == nullptr || descriptor == nullptr) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  *system = nullptr;
  if (descriptor->abi_version != GENERATIVEQC_ABI_VERSION ||
      descriptor->struct_size < offsetof(generativeqc_system_descriptor, basis_representation)) {
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  }
  if (descriptor->atoms == nullptr || descriptor->atom_count == 0) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  const bool atom_only = descriptor->shell_count == 0 && descriptor->primitive_count == 0;
  if (!atom_only && (descriptor->shells == nullptr || descriptor->primitives == nullptr ||
                     descriptor->shell_count == 0 || descriptor->primitive_count == 0)) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  if (atom_only && (descriptor->shells != nullptr || descriptor->primitives != nullptr)) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  std::lock_guard<std::recursive_mutex> context_lock(context->mutex);
  try {
    auto candidate = std::make_unique<generativeqc_system>();
    candidate->data.charge = descriptor->charge;
    candidate->data.multiplicity = descriptor->multiplicity;
    candidate->data.basis_representation =
        descriptor->struct_size >= sizeof(generativeqc_system_descriptor)
            ? descriptor->basis_representation
            : GENERATIVEQC_BASIS_CARTESIAN;
    candidate->data.atoms.reserve(descriptor->atom_count);
    for (std::uint32_t i = 0; i < descriptor->atom_count; ++i) {
      const generativeqc_atom& atom = descriptor->atoms[i];
      candidate->data.atoms.push_back({atom.atomic_number, {atom.x, atom.y, atom.z}});
    }
    candidate->data.shells.reserve(descriptor->shell_count);
    for (std::uint32_t i = 0; i < descriptor->shell_count; ++i) {
      const generativeqc_shell& shell = descriptor->shells[i];
      if (context->state.executed_backend == GENERATIVEQC_BACKEND_CUDA &&
          shell.angular_momentum > 3) {
        context->last_detail = "CUDA basis execution supports l<=3; g shells require CPU reference";
        return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
      }
      if (shell.primitive_count == 0 || shell.primitive_offset > descriptor->primitive_count ||
          shell.primitive_count > descriptor->primitive_count - shell.primitive_offset) {
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      }
      generativeqc::core::Shell native_shell;
      native_shell.atom_index = shell.atom_index;
      native_shell.angular_momentum = shell.angular_momentum;
      native_shell.primitives.reserve(shell.primitive_count);
      for (std::uint32_t p = 0; p < shell.primitive_count; ++p) {
        const generativeqc_primitive& primitive =
            descriptor->primitives[shell.primitive_offset + p];
        native_shell.primitives.push_back({primitive.exponent, primitive.coefficient});
      }
      candidate->data.shells.push_back(std::move(native_shell));
    }
    const generativeqc_status status =
        generativeqc::molecule::validate_and_normalize(candidate->data, context->last_detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    *system = candidate.release();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

void generativeqc_system_destroy(generativeqc_system* system) { delete system; }

}  // extern "C"
