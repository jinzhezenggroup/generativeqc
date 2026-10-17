#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace generativeqc::dft::hessian_aot {

/** Build-time compiler-AD leaves, not a complete stationary Hessian capability.
 *
 * Every callback is linked native CUDA code. There is no runtime compiler,
 * interpreter, allocator, synchronization, device selection or CPU fallback.
 * The enclosing numerical owner must retain the input/output buffers and stream,
 * initialize its device error flag and discard ALL outputs if that flag is set.
 * Input names and component indices describe the exact scientific packing;
 * missing programs are a package-admission failure, never a request for JIT.
 */
enum class Kind { array, ao, integral };

using ArrayLaunch = int (*)(const double*, std::size_t, double*, int*, cudaStream_t);
using AoLaunch = int (*)(const double*, std::size_t, std::size_t, std::size_t, const double*,
                         std::size_t, std::size_t, double*, int*, cudaStream_t);
using IntegralLaunch = int (*)(const double*, std::size_t, const long long*, const double*, double*,
                               int*, cudaStream_t);
struct FactorizedIntegralBatch;
using IntegralBatchLaunch = int (*)(const FactorizedIntegralBatch*, std::size_t, const double*,
                                    int*, cudaStream_t);

/** Scientific selectors for integral closure, independent of program names.
 * Components use the flattened CCA Cartesian product of these shell orders.
 * Empty selectors belong to non-integral programs. No runtime code discovery
 * or compilation is authorized by this metadata. */
struct IntegralSignature {
  std::string_view family;
  std::span<const unsigned> angular;
  std::string_view radial;
  double omega{};
};

struct Program {
  std::string_view name;
  std::string_view identity;
  Kind kind;
  std::span<const std::string_view> inputs;
  std::size_t outputs{};
  std::span<const std::uint32_t> components;
  std::size_t centers{};
  std::size_t record_stride{};
  ArrayLaunch array{};
  AoLaunch ao{};
  IntegralLaunch integral{};
  IntegralSignature signature{};
  IntegralBatchLaunch integral_batch{};
};

/** Immutable, explicitly selected build inventory; no runtime discovery. */
std::span<const Program> programs() noexcept;
const Program* find(std::string_view name) noexcept;

/** Scalar arrays use input-major/output-major [field,lane] FP64 panels.
 * A zero-lane launch is a no-op. Nonempty launches require disjoint panels.
 */
cudaError_t launch_array(std::string_view name, const double* inputs, std::size_t lanes,
                         double* outputs, int* status, cudaStream_t stream) noexcept;

/** AO input is the authoritative normalized 16-double record layout.
 * Jets are complete order-0/1/2/3 prefixes (1/4/10/20), including real-spherical
 * records. Output is [jet,point,AO]. Basis metadata must already be validated by
 * the native basis owner; this leaf does not dereference device metadata on CPU.
 */
cudaError_t launch_ao(const double* basis, std::size_t atoms, std::size_t primitives,
                      std::size_t aos, const double* points, std::size_t point_count,
                      std::size_t jets, double* outputs, int* status, cudaStream_t stream) noexcept;

/** Integral records are [exponents,center_xyz,primitive_scale].
 * The fixed record scale includes radial contraction weights and, for
 * attraction, the physical nuclear charge (the program uses unit Z). Component
 * normalization must be applied exactly once by the enclosing public projection
 * or, for a single component, by the record scale; it is not built into the leaf.
 * Output is component-major [value,gradient(3*C),gradient-JVP(3*C)]. Atoms map
 * mathematical centers to the caller's molecular direction; coincidence does
 * not remove centers. All raw component weights are unity before compiler AD.
 */
cudaError_t launch_integral(std::string_view name, const double* records, std::size_t count,
                            const long long* atoms, const double* direction, double* outputs,
                            int* status, cudaStream_t stream) noexcept;

}  // namespace generativeqc::dft::hessian_aot
