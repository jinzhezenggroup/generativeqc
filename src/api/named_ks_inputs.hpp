#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "api/error.hpp"
#include "dft/application_precision.hpp"
#include "methods/application_precision.hpp"
#include "methods/named_ks_catalog.hpp"

namespace generativeqc::api {

/** Copy a recognized size into zero-filled current storage before reading any
 * optional field. Old bindings need not allocate the appended suffix. */
inline generativeqc_named_ks_descriptor copy_named_ks_descriptor(const void* input) {
  std::array<std::uint32_t, 2> header{};
  std::memcpy(header.data(), input, sizeof(header));
  constexpr auto prefix = offsetof(generativeqc_named_ks_descriptor, cpu_packed_eri_maximum_bytes);
  constexpr auto packed = offsetof(generativeqc_named_ks_descriptor, cpu_df_host_maximum_bytes);
  constexpr auto local_host = offsetof(generativeqc_named_ks_descriptor, host_resources);
  if (header[1] != GENERATIVEQC_ABI_VERSION ||
      (header[0] != prefix && header[0] != packed && header[0] != local_host &&
       header[0] != sizeof(generativeqc_named_ks_descriptor)))
    throw methods::MethodError(GENERATIVEQC_STATUS_ABI_MISMATCH,
                               "named KS descriptor ABI mismatch");
  generativeqc_named_ks_descriptor result{};
  std::memcpy(&result, input, header[0]);
  if (result.reserved || (!result.cpu_df_host_properties &&
                          (result.cpu_df_host_maximum_bytes || result.host_resources)))
    throw methods::MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                               "invalid CPU DF host admission controls");
  return result;
}

/** Resolve exact native program identity before constructing any owner. */
inline const methods::named_ks::Method& named_ks_program(
    const generativeqc_named_ks_descriptor& named, generativeqc_backend backend) {
  if (!named.name || !*named.name)
    throw methods::MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                               "a named KS selector is required");
  const auto* program = methods::named_ks::find_method(named.name);
  if (!program || !(backend == GENERATIVEQC_BACKEND_CUDA ? program->cuda : program->cpu))
    throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                               "named KS program is unavailable on this backend");
  if (named.identity && program->identity != named.identity)
    throw methods::MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "named KS identity mismatch");
  return *program;
}

/** Synchronous named-program transport. Scientific arrays borrow immutable
 * library storage; execution controls retain their existing size-tagged ABI.
 * No pointer in this temporary survives the native prepared owner's copy. */
struct NamedKsInputs {
  generativeqc_ks_nonlocal_grid_options storage{};
  generativeqc_ks_options nonlocal_grid{};
  methods::detail::ApplicationPrecisionMethodOptions method{};
  const generativeqc_named_ks_descriptor selection;
  const methods::named_ks::Method* program{};

  NamedKsInputs(const generativeqc_method_descriptor& input,
                const generativeqc_named_ks_descriptor* named, generativeqc_backend backend)
      : selection(copy_named_ks_descriptor(named)) {
    if (input.struct_size != sizeof(input) && input.struct_size != sizeof(method))
      throw methods::MethodError(GENERATIVEQC_STATUS_ABI_MISMATCH,
                                 "unknown named method controls ABI");
    std::memcpy(&method, &input, input.struct_size);
    program = &named_ks_program(selection, backend);
    if (method.base.method != program->carrier)
      throw methods::MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                                 "named KS identity or spin carrier mismatch");
    auto& ks = storage.base.base;
    if (input.ks_options) {
      const auto size = input.ks_options->struct_size;
      if (input.ks_options->abi_version != GENERATIVEQC_ABI_VERSION ||
          (size != sizeof(generativeqc_ks_options) &&
           size != sizeof(generativeqc_ks_precision_options) && size != sizeof(storage)))
        throw methods::MethodError(GENERATIVEQC_STATUS_ABI_MISMATCH,
                                   "unknown named KS controls ABI");
      std::memcpy(&storage, input.ks_options, size);
    } else {
      ks.struct_size = sizeof(ks);
      ks.abi_version = GENERATIVEQC_ABI_VERSION;
      ks.grid_version = 4;
      ks.radial_points = 3;
      ks.partition_iterations = 3;
      ks.coincident_tolerance = 1e-12;
      ks.tile_points = 0;
      ks.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
      ks.nonlocal_maximum_bytes = 256ull * 1024 * 1024;
    }
    methods::named_ks::electronic_options(*program, ks);
    if (ks.struct_size == sizeof(storage) && storage.grid) {
      if (storage.grid->struct_size != sizeof(nonlocal_grid) ||
          storage.grid->abi_version != GENERATIVEQC_ABI_VERSION)
        throw methods::MethodError(GENERATIVEQC_STATUS_ABI_MISMATCH,
                                   "unknown named nonlocal grid controls ABI");
      // The nested descriptor also contributes execution controls only. In
      // particular, callers need not duplicate the named model's SCF domain.
      nonlocal_grid = *storage.grid;
      methods::named_ks::electronic_options(*program, nonlocal_grid);
      storage.grid = &nonlocal_grid;
    }
    method.base.ks_options = &ks;
  }
};

}  // namespace generativeqc::api
