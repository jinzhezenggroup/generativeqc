#pragma once

// Host-only queued CUDA/provider doubles. The physical controller and all of
// its generated affine arithmetic are compiled unchanged apart from launch syntax.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorNotSupported = 2,
              cudaErrorInvalidResourceHandle = 3, cudaErrorStreamCaptureUnsupported = 4;
constexpr int cudaMemoryTypeDevice = 2, cudaMemcpyDeviceToHost = 1, cudaMemcpyDeviceToDevice = 2,
              cudaMemcpyHostToDevice = 3;
using cudaStreamCaptureStatus = int;
constexpr int cudaStreamCaptureStatusNone = 0;
struct cudaPointerAttributes {
  int type, device;
};
#define __global__
struct QueueDimension {
  unsigned x;
};
inline QueueDimension blockIdx{0}, threadIdx{0}, blockDim{128}, gridDim{1};
using std::isfinite;

namespace queue_test {
inline std::vector<std::function<void()>> pending;
inline int launches{}, fail_launch{}, fences{}, late_reads{};
inline bool returned{}, capturing{}, revoked{};
inline unsigned spins = 1;
inline bool fitted{};
inline cudaStream_t stream = reinterpret_cast<void*>(8);
inline std::vector<double> metadata(256);
inline void reset_action(int failure = 0) {
  if (!pending.empty()) throw std::runtime_error("undrained preparation");
  launches = fences = late_reads = 0;
  fail_launch = failure;
  returned = false;
}
}  // namespace queue_test

inline cudaError_t cudaGetLastError() {
  return ++queue_test::launches == queue_test::fail_launch ? cudaErrorInvalidValue : cudaSuccess;
}
inline cudaError_t cudaGetDevice(int* device) {
  *device = 0;
  return cudaSuccess;
}
inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void*) {
  *attributes = {cudaMemoryTypeDevice, 0};
  return cudaSuccess;
}
inline cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* capture) {
  *capture = queue_test::capturing ? 1 : cudaStreamCaptureStatusNone;
  return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* destination, int value, std::size_t bytes, cudaStream_t) {
  queue_test::pending.emplace_back([=] { std::memset(destination, value, bytes); });
  return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* destination, const void* source, std::size_t bytes, int,
                                   cudaStream_t) {
  queue_test::pending.emplace_back([=] { std::memcpy(destination, source, bytes); });
  return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) {
  ++queue_test::fences;
  auto work = std::move(queue_test::pending);
  queue_test::pending.clear();
  for (auto& operation : work) operation();
  return cudaSuccess;
}
inline cudaError_t cudaMalloc(void** pointer, std::size_t bytes) {
  *pointer = std::malloc(bytes);
  return *pointer ? cudaSuccess : cudaErrorInvalidValue;
}
inline cudaError_t cudaFree(void* pointer) {
  std::free(pointer);
  return cudaSuccess;
}
inline const char* cudaGetErrorString(cudaError_t) { return "injected CUDA launch rejection"; }
inline int atomicAdd(int* pointer, int value) {
  const int old = *pointer;
  *pointer += value;
  return old;
}
inline int atomicExch(int* pointer, int value) {
  const int old = *pointer;
  *pointer = value;
  return old;
}
template <class Function>
void queue_kernel(unsigned blocks, unsigned threads, Function function) {
  queue_test::pending.emplace_back([=] {
    gridDim.x = blocks;
    blockDim.x = threads;
    for (blockIdx.x = 0; blockIdx.x < blocks; ++blockIdx.x)
      for (threadIdx.x = 0; threadIdx.x < threads; ++threadIdx.x) function();
  });
}

using generativeqc_status = int;
constexpr int GENERATIVEQC_STATUS_SUCCESS = 0, GENERATIVEQC_STATUS_INVALID_ARGUMENT = 1,
              GENERATIVEQC_STATUS_NUMERICAL_FAILURE = 2;
namespace generativeqc {
namespace core {
struct System {};
}  // namespace core
namespace molecule {
inline std::size_t cartesian_ao_count(const core::System&) { return 2; }
}  // namespace molecule
namespace nlc {
enum class Vv10Variant { vv10 };
enum class Vv10DensityDomain { MolecularV1 };
struct Correlation {
  Vv10Variant variant{Vv10Variant::vv10};
  double b{6}, c{0.01}, coefficient{1};
};
}  // namespace nlc
namespace scf {
enum class FockBackend { Cuda };
enum class FockSpin { Restricted, Unrestricted };
enum class FockOperator { FullRange, LongRange };
enum class FockApproximation { Exact, DensityFitted };
struct Term {
  bool present{true};
  double coefficient{1};
  FockOperator op{FockOperator::FullRange};
  double omega{};
  FockApproximation approximation{FockApproximation::Exact};
  bool operator==(const Term&) const = default;
};
struct FockBuildSpec {
  unsigned version{1}, derivative_order{};
  FockSpin spin{FockSpin::Restricted};
  Term coulomb{}, exchange{};
  bool operator==(const FockBuildSpec&) const = default;
};
struct ResolvedFockBuild {
  FockBackend backend{FockBackend::Cuda};
  FockBuildSpec spec{};
  bool operator==(const ResolvedFockBuild&) const = default;
};
struct CudaDensityFittingIntegralSource {};
struct CudaDensityFittingJkPlan {};
struct PreparedFockPlan {
  ResolvedFockBuild model;
  const ResolvedFockBuild& strategy() const { return model; }
  const core::System& system() const {
    static core::System system;
    return system;
  }
  const CudaDensityFittingJkPlan* cuda_fitted_source() const {
    static CudaDensityFittingJkPlan plan;
    return &plan;
  }
};
inline PreparedFockPlan primary;
inline ResolvedFockBuild correction;
struct PreparedBinding {
  int device_id{};
  cudaStream_t stream{queue_test::stream};
  std::size_t nbf{2};
  explicit operator bool() const { return true; }
};
inline PreparedBinding prepared_cuda_fock_binding(const PreparedFockPlan&) { return {}; }
struct CudaDfReadonlySource {
  int device_id{};
  cudaStream_t stream{queue_test::stream};
  CudaDensityFittingIntegralSource* integrals{};
  const double* whitening{};
  std::size_t nao{2}, naux{3};
  std::uint64_t identity{1};
  bool operator==(const CudaDfReadonlySource&) const = default;
  explicit operator bool() const { return integrals != nullptr; }
};
inline CudaDfReadonlySource cuda_density_fitting_readonly_source(const CudaDensityFittingJkPlan*) {
  static CudaDensityFittingIntegralSource integral;
  return {0, queue_test::stream, &integral, queue_test::metadata.data() + 160, 2, 3, 1};
}
inline generativeqc_status enqueue_cuda_density_fitting_raw_values(
    CudaDensityFittingIntegralSource*, std::size_t, std::size_t pairs, void* stream, double* output,
    std::string&) {
  return cudaMemsetAsync(output, 0, pairs * 3 * sizeof(double), stream);
}
struct CudaDirectJkPlan {};
struct CudaDirectJkDiagnostic {
  std::size_t device_bytes{128};
};
inline generativeqc_status create_cuda_direct_jk_resident_plan_on_stream(
    int, const std::vector<core::System>&, double, std::size_t, cudaStream_t,
    CudaDirectJkPlan** plan, CudaDirectJkDiagnostic&, std::string&) {
  *plan = new CudaDirectJkPlan;
  return GENERATIVEQC_STATUS_SUCCESS;
}
inline void destroy_cuda_direct_jk_plan(CudaDirectJkPlan* plan) { delete plan; }
inline bool cuda_direct_jk_linear_available(const CudaDirectJkPlan*) { return true; }
inline generativeqc_status enqueue_cuda_direct_jk_compensated_device(
    CudaDirectJkPlan*, FockBuildSpec, const double* density, std::size_t count, double* coulomb,
    double* exchange, double*, std::size_t, int*, double, std::uint64_t* census, std::string&, bool,
    const double* beta, double* beta_exchange, bool) {
  queue_test::pending.emplace_back([=] {
    if (queue_test::returned) ++queue_test::late_reads;
    double alpha{}, minority{};
    for (std::size_t index = 0; index < count; ++index) {
      alpha += density[index];
      if (beta) minority += beta[index];
    }
    for (std::size_t index = 0; index < count; ++index) {
      if (coulomb) coulomb[index] = alpha + minority;
      if (exchange) exchange[index] = alpha;
      if (beta_exchange) beta_exchange[index] = minority;
    }
    census[0] = census[1] = 1;
  });
  return GENERATIVEQC_STATUS_SUCCESS;
}
}  // namespace scf
namespace dft {
enum class SemilocalFamily { Wb97mv };
inline int semilocal_family_domain_version(SemilocalFamily) { return 1; }
inline int semilocal_family_code(SemilocalFamily) { return 1; }
struct Model {
  int version{1}, scf_domain_version{1}, functional{1};
  unsigned spins{1};
  int device{};
  double semilocal_exchange_scale{1}, semilocal_correlation_scale{1};
  std::optional<nlc::Correlation> nonlocal_correlation{nlc::Correlation{}};
  nlc::Vv10DensityDomain nonlocal_density_domain{nlc::Vv10DensityDomain::MolecularV1};
  std::optional<scf::ResolvedFockBuild> range_correction;
};
struct CudaKsFinalStateToken {
  struct Identity {
    Model model;
    struct Determinant {
      scf::ResolvedFockBuild model;
    } determinant;
  } identity;
};
struct CudaKsResidentFockBinding {
  int device_id{};
  const scf::PreparedFockPlan* primary{&scf::primary};
  const scf::ResolvedFockBuild* correction{&scf::correction};
  const double* hcore{queue_test::metadata.data()};
  const double* fock{queue_test::metadata.data() + 8};
  std::size_t nao{2};
  unsigned spins{queue_test::spins};
  void* stream{queue_test::stream};
  bool operator==(const CudaKsResidentFockBinding&) const = default;
};
struct CudaKsResidentOrbitalFrameBinding {
  const double* coefficients{queue_test::metadata.data() + 16};
  const double* orbital_energies{queue_test::metadata.data() + 24};
  const double* density{queue_test::metadata.data() + 32};
  const double* overlap{queue_test::metadata.data() + 40};
};
struct CudaWb97mvGridOptions {
  double symmetry_absolute_tolerance{1e-12}, symmetry_relative_tolerance{1e-10};
};
struct CudaWb97mvGridWork {};
struct CudaWb97mvGridShape {
  std::size_t atoms{1}, primitives{1}, nao{2}, points{1};
  unsigned spins{queue_test::spins};
};
struct CudaWb97mvGridReference {
  const double* basis{queue_test::metadata.data() + 48};
  const double* points{queue_test::metadata.data() + 96};
  const double* weights{queue_test::metadata.data() + 104};
  const double* density{queue_test::metadata.data() + 32};
  int device_id{};
  cudaStream_t stream{queue_test::stream};
  cudaError_t (*authenticate)(void*) = [](void*) {
    return queue_test::revoked ? cudaErrorInvalidResourceHandle : cudaSuccess;
  };
  void* owner{};
};
inline cudaError_t cuda_wb97mv_grid_workspace(const CudaWb97mvGridShape&,
                                              const CudaWb97mvGridOptions&, std::size_t& bytes) {
  bytes = 64;
  return cudaSuccess;
}
struct CudaWb97mvGridResponse {
  cudaError_t prepare(const CudaWb97mvGridShape&, const CudaWb97mvGridOptions&,
                      const CudaWb97mvGridReference&, void*, std::size_t) {
    return cudaSuccess;
  }
  cudaError_t reference_potential(const double*& pointer) {
    pointer = queue_test::metadata.data() + 112;
    return cudaSuccess;
  }
  cudaError_t apply(const double*, double* output, cudaStream_t) {
    std::fill(output, output + 4 * queue_test::spins, 0);
    return cudaSuccess;
  }
  CudaWb97mvGridWork work() const { return {}; }
};
}  // namespace dft
namespace methods {
struct PreparedBatch {};
namespace detail {
inline generativeqc_status dft_cuda_resident_fock_sources(PreparedBatch&, std::size_t,
                                                          const dft::CudaKsFinalStateToken&,
                                                          dft::CudaKsResidentFockBinding& binding,
                                                          std::string&) {
  binding = {};
  return queue_test::revoked ? GENERATIVEQC_STATUS_INVALID_ARGUMENT : GENERATIVEQC_STATUS_SUCCESS;
}
inline generativeqc_status dft_cuda_resident_orbital_frame(
    PreparedBatch&, std::size_t, const dft::CudaKsFinalStateToken&,
    dft::CudaKsResidentOrbitalFrameBinding& binding, std::string&) {
  binding = {};
  return GENERATIVEQC_STATUS_SUCCESS;
}
}  // namespace detail
}  // namespace methods
namespace dft {
struct CudaKsGridReference {
  CudaKsGridReference(methods::PreparedBatch&, std::size_t, CudaKsFinalStateToken) {}
  generativeqc_status prepare(std::string&) { return GENERATIVEQC_STATUS_SUCCESS; }
  CudaWb97mvGridReference reference() const { return {}; }
  CudaWb97mvGridShape shape() const { return {}; }
};
}  // namespace dft
namespace response {
struct CudaCpksShape {
  std::size_t nao{};
  unsigned spins{};
};
}  // namespace response
}  // namespace generativeqc
