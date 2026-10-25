#pragma once

#include <cstring>
#include <string_view>

#include "api/error.hpp"
#include "dft/application_precision.hpp"
#include "methods/application_precision.hpp"
#include "methods/named_ks_catalog.hpp"

namespace generativeqc::api {

/** Synchronous named-program transport. Scientific arrays borrow immutable
 * library storage; execution controls retain their existing size-tagged ABI.
 * No pointer in this temporary survives the native prepared owner's copy. */
struct NamedKsInputs {
  dft::NonlocalGridKsOptions storage{};
  generativeqc_ks_options nonlocal_grid{};
  methods::detail::ApplicationPrecisionMethodOptions method{};
  const methods::named_ks::Method* program{};

  NamedKsInputs(const generativeqc_method_descriptor& input,
                const generativeqc_named_ks_descriptor& named, generativeqc_backend backend) {
    if (input.struct_size != sizeof(input) && input.struct_size != sizeof(method))
      throw methods::MethodError(GENERATIVEQC_STATUS_ABI_MISMATCH,
                                 "unknown named method controls ABI");
    std::memcpy(&method, &input, input.struct_size);
    if (!valid_descriptor(&named))
      throw methods::MethodError(GENERATIVEQC_STATUS_ABI_MISMATCH,
                                 "named KS descriptor ABI mismatch");
    if (!named.name || !*named.name)
      throw methods::MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                                 "a named KS selector is required");
    program = methods::named_ks::find_method(named.name);
    if (!program || !(backend == GENERATIVEQC_BACKEND_CUDA ? program->cuda : program->cpu))
      throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                                 "named KS program is unavailable on this backend");
    if ((named.identity && program->identity != named.identity) ||
        method.base.method != program->carrier)
      throw methods::MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                                 "named KS identity or spin carrier mismatch");
    auto& ks = storage.base.base;
    if (input.ks_options) {
      const auto size = input.ks_options->struct_size;
      if (input.ks_options->abi_version != GENERATIVEQC_ABI_VERSION ||
          (size != sizeof(generativeqc_ks_options) &&
           size != sizeof(dft::ApplicationPrecisionKsOptions) && size != sizeof(storage)))
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
      ks.tile_points = 256;
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
