// Validation-only adapter. Publish scalar diagnostics only after the native
// owner's complete stream drain and all numerical checks have succeeded.
#include <algorithm>
#include <cstdio>
#include <exception>

#include "cc/df_triples.hpp"

// Optional test-library hook: production libraries need not export test controls.
extern "C" void df_triples_reject_library_for_test_v1(bool) __attribute__((weak));
extern "C" bool tensor_cutensor_qualification_for_test_v1(bool, int) __attribute__((weak));

// Version the validation ABI: older adapters had neither precision admission
// nor provider diagnostics. A mismatched caller must fail symbol lookup rather
// than shift pointer arguments or overrun its diagnostic buffer.
extern "C" int df_triples_probe_v2(std::size_t o, std::size_t v, std::size_t q,
                                   const double* const* inputs, double threshold,
                                   std::size_t budget, std::size_t panels, int precision,
                                   double* values, std::size_t* counts, std::size_t counts_size,
                                   char* error, std::size_t error_size) noexcept {
  try {
    if (!values || !counts || counts_size < 26)
      throw std::invalid_argument("insufficient triples diagnostic buffer");
    if (precision != 0 && precision != 1 && precision != 2 && precision != 3 && precision != 4 &&
        precision != 5 && precision != 12 && precision != 13)
      throw std::invalid_argument("invalid precision admission");
    struct Reset {
      ~Reset() {
        if (df_triples_reject_library_for_test_v1) df_triples_reject_library_for_test_v1(false);
        if (tensor_cutensor_qualification_for_test_v1)
          (void)tensor_cutensor_qualification_for_test_v1(false, -1);
      }
    } reset;
    if (precision & 2) {
      if (!df_triples_reject_library_for_test_v1) throw std::runtime_error("test hook unavailable");
      df_triples_reject_library_for_test_v1(true);
    }
    if (precision & 4) {
      if (!tensor_cutensor_qualification_for_test_v1 ||
          !tensor_cutensor_qualification_for_test_v1(true, (precision & 8) ? 2 : -1))
        throw std::runtime_error("cuTENSOR test hook unavailable");
    }
    const auto mode = precision % 2 == 0 ? generativeqc::runtime::strict_fp64_precision()
                                         : generativeqc::runtime::PrecisionDirective{
                                               generativeqc::runtime::PrecisionDtype::Fp32,
                                               generativeqc::runtime::PrecisionDtype::Fp32,
                                               generativeqc::runtime::PrecisionDtype::Fp32,
                                               "issue1764/df-triples-w-fp32-candidate-v1"};
    const auto result = generativeqc::cc::triples::evaluate_df_cuda(
        o, v, q, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6],
        inputs[7], inputs[8], threshold, budget, 0, panels, mode);
    const double scalars[] = {result.energy, result.minimum_absolute_denominator, result.seconds};
    const std::size_t work[] = {result.virtual_triples,
                                result.occupied_tiles,
                                result.workspace_bytes,
                                result.arena_bytes,
                                result.provider_retained_bytes,
                                result.panel_capacity,
                                result.panel_gemms,
                                result.moment_gemms,
                                result.epilogue_kernels,
                                result.reduction_kernels,
                                result.epilogue_points,
                                result.contraction_summands,
                                result.h2d_bytes,
                                result.d2h_bytes,
                                result.fp64_gemms,
                                result.fp32_gemms,
                                result.precision_cast_elements,
                                result.w_contraction_storage_bits,
                                result.w_contraction_compute_bits,
                                result.w_contraction_accumulation_bits,
                                std::size_t(result.w_provider == "generated.cuda"),
                                std::size_t(result.retained_incumbent),
                                std::size_t(result.resource_fallback),
                                result.host_binding_bytes,
                                std::size_t(result.w_provider == "cutensor"),
                                std::size_t(result.w_provider_version)};
    std::copy(std::begin(scalars), std::end(scalars), values);
    std::copy(std::begin(work), std::end(work), counts);
    return 0;
  } catch (const std::exception& e) {
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return 1;
  }
}
