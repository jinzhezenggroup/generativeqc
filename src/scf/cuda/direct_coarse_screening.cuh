#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

namespace generativeqc::scf::cuda_execution {

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
template <bool Force>
__device__ __forceinline__ bool direct_fp32_block_rejects(double, double, double, double, double) {
  return false;
}
#endif

}  // namespace generativeqc::scf::cuda_execution
