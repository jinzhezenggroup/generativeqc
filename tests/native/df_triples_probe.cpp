// Validation-only adapter. Publish scalar diagnostics only after the native
// owner's complete stream drain and all numerical checks have succeeded.
#include <algorithm>
#include <cstdio>
#include <exception>

#include "cc/df_triples.hpp"

extern "C" int df_triples_probe(std::size_t o, std::size_t v, std::size_t q,
                                const double* const* inputs, double threshold, std::size_t budget,
                                std::size_t panels, int precision, double* values,
                                std::size_t* counts, char* error, std::size_t error_size) noexcept {
  try {
    const auto mode = precision == 0
                          ? generativeqc::cc::triples::DFTriplesPrecision::Fp64
                          : generativeqc::cc::triples::DFTriplesPrecision::WFp32;
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
                                result.w_contraction_accumulation_bits};
    std::copy(std::begin(scalars), std::end(scalars), values);
    std::copy(std::begin(work), std::end(work), counts);
    return 0;
  } catch (const std::exception& e) {
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return 1;
  }
}
