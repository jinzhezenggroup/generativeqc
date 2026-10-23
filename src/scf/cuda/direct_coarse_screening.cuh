#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

namespace generativeqc::scf::cuda_execution {

/** Only Refine requires reading the original FP64 metadata. */
enum class DirectBlockDecision { Refine, Reject, Accept };

// The proof relies on NVIDIA's directed-rounding intrinsics. Compatibility
// compilers keep the FP64 gate until their rounding semantics are qualified.
#if defined(__NVCC__)

/** Upper enclosure of a nonnegative FP64 bound, including FTZ builds.
 * Invalid/negative inputs disable early rejection. Positive subnormals are
 * promoted to FLT_MIN before any FP32 arithmetic: directed rounding alone
 * would not protect a bound when the compiler flushes subnormal operands.
 */
__device__ __forceinline__ float direct_screening_upper_float(double value) {
  const auto bits = static_cast<unsigned long long>(__double_as_longlong(value));
  if ((bits >> 63U) != 0U || (bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL)
    return CUDART_INF_F;
  if (value == 0.0) return 0.0F;
  return value < 0x1p-126 ? 0x1p-126F : __double2float_ru(value);
}

/** Round a product upward and keep positive underflow conservative under FTZ. */
__device__ __forceinline__ float direct_screening_upper_product(float first, float second) {
  const float product = __fmul_ru(first, second);
  return product < 0x1p-126F && first > 0.0F && second > 0.0F ? 0x1p-126F : product;
}

/** Recover a lower endpoint from one stored upward-rounded FP32 bound.
 * A predecessor encloses the original FP64 value even when it was exactly
 * representable. The FLT_MIN floor used by preparation needs a zero lower
 * endpoint: its original value may have been a binary64 subnormal.
 */
__device__ __forceinline__ float direct_screening_lower_float(float upper) {
  return upper <= 0x1p-126F ? 0.0F : __uint_as_float(__float_as_uint(upper) - 1U);
}

/** Classify the incumbent sequential FP64 block predicate using prepared bounds.
 * Lower products first certify the common all-survive case without FP64 loads
 * or upper products. Upper products can certify rejection; overlapping intervals
 * fall back. Invalid/overflowed prepared inputs use infinity as a sentinel.
 * No fast-math contraction is allowed: directed rounding at every multiplication
 * encloses the corresponding FP64 rounding boundary, including FTZ underflow.
 */
template <bool Force>
__device__ __forceinline__ DirectBlockDecision direct_fp32_block_decision(float first, float second,
                                                                          float density,
                                                                          double tolerance,
                                                                          double force_tolerance) {
  if (!(first < CUDART_INF_F && second < CUDART_INF_F && density < CUDART_INF_F))
    return DirectBlockDecision::Refine;
  const float density_lower = direct_screening_lower_float(density);
  const float quartet_lower =
      __fmul_rd(direct_screening_lower_float(first), direct_screening_lower_float(second));
  const float contribution_lower = __fmul_rd(quartet_lower, density_lower);
  // An upward threshold must remain an upper bound when FTZ comparisons flush
  // subnormals. Use the same positive-underflow floor as prepared metadata.
  const float threshold_upper = direct_screening_upper_float(tolerance);
  if (quartet_lower >= threshold_upper && contribution_lower >= threshold_upper) {
    if constexpr (!Force) return DirectBlockDecision::Accept;
    if (__fmul_rd(contribution_lower, density_lower) >=
        direct_screening_upper_float(force_tolerance))
      return DirectBlockDecision::Accept;
  }
  const float quartet_upper = direct_screening_upper_product(first, second);
  const float threshold_lower = __double2float_rd(tolerance);
  if (quartet_upper < threshold_lower) return DirectBlockDecision::Reject;
  const float contribution_upper = direct_screening_upper_product(quartet_upper, density);
  if (contribution_upper < threshold_lower) return DirectBlockDecision::Reject;
  if constexpr (Force)
    if (direct_screening_upper_product(contribution_upper, density) <
        __double2float_rd(force_tolerance))
      return DirectBlockDecision::Reject;
  return DirectBlockDecision::Refine;
}

/** A sufficient rejection only; survivors MUST run the incumbent FP64 gate.
 * Each operation encloses the corresponding sequential FP64 operation, not
 * merely its unrounded real product. Preserve 0*infinity as unordered and
 * round the threshold down. Thus uncertain/overflowed bounds fall through.
 * This helper neither changes stored bounds nor evaluates any integral.
 */
template <bool Force>
__device__ __forceinline__ bool direct_fp32_block_rejects(double first, double second,
                                                          double density, double tolerance,
                                                          double force_tolerance) {
  const float threshold = __double2float_rd(tolerance);
  const float quartet = direct_screening_upper_product(direct_screening_upper_float(first),
                                                       direct_screening_upper_float(second));
  if (quartet < threshold) return true;
  const float density_upper = direct_screening_upper_float(density);
  const float contribution = direct_screening_upper_product(quartet, density_upper);
  if (contribution < threshold) return true;
  if constexpr (Force)
    return direct_screening_upper_product(contribution, density_upper) <
           __double2float_rd(force_tolerance);
  return false;
}

#else
// Preparation may run with an unqualified compatibility compiler as well.
__device__ __forceinline__ float direct_screening_upper_float(double) { return CUDART_INF_F; }
template <bool Force>
__device__ __forceinline__ DirectBlockDecision direct_fp32_block_decision(float, float, float,
                                                                          double, double) {
  return DirectBlockDecision::Refine;
}
template <bool Force>
__device__ __forceinline__ bool direct_fp32_block_rejects(double, double, double, double, double) {
  return false;
}
#endif

}  // namespace generativeqc::scf::cuda_execution
