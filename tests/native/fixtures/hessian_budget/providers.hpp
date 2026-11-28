#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dft/cuda_ks_resident_grid.hpp"
#include "dft/cuda_ks_wb97mv_hessian.hpp"
#include "dft/dispersion/d3_response.hpp"
#include "dft/dispersion/d3_runtime.hpp"
#include "dft/dispersion/d4_runtime.hpp"
#include "dft/dispersion/d4_second_order.hpp"
#include "dft/hessian_aot.hpp"
#include "generated_dispersion_scalar.hpp"

namespace generativeqc::runtime {
/** Route the owner's ledger-aware allocation calls through the same instrumented
 * host CUDA double as its mocked providers. Real ledger admission is tested
 * separately at the public CUDA endpoint. */
template <class T>
inline cudaError_t resource_cuda_malloc(T** output, std::size_t bytes) {
  return cudaMalloc(reinterpret_cast<void**>(output), bytes);
}
inline cudaError_t resource_cuda_free(void* pointer) { return cudaFree(pointer); }
}  // namespace generativeqc::runtime

namespace stub {
inline std::size_t points = 8;
inline unsigned spins = 1;
inline bool df = false, range_df = false;
inline bool canonical_source_limit = false;
inline bool retain_df_tape = false;
inline constexpr std::size_t df_tape_bytes = 256;
inline std::size_t screened_exact_preparations = 0;
enum class Provider { none, core, induced, exact, raw_df, fitted, raw_df_range, fitted_range, ecp };
inline Provider reject_provider = Provider::none;
inline std::size_t grid_scalar() { return 21600 + (points - 8) * 128; }
inline std::size_t grid_parallel() { return grid_scalar() + 21 * points * sizeof(double); }
inline constexpr std::size_t core_scalar = 272, core_parallel = 528;
struct Storage {
  void* arena{};
  ~Storage() { cudaFree(arena); }
  generativeqc_status allocate(std::size_t bytes, std::size_t cap, Provider provider) {
    if (reject_provider == provider || bytes > cap) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    return cudaMalloc(&arena, bytes) == cudaSuccess ? GENERATIVEQC_STATUS_SUCCESS
                                                    : GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  }
};
}  // namespace stub
namespace generativeqc::methods {
class PreparedBatch {};
}  // namespace generativeqc::methods
namespace generativeqc::scf {
struct Source {
  std::size_t nao = 2, naux = 3;
  explicit operator bool() const { return true; }
};
inline Source cuda_density_fitting_readonly_source(const void*) { return {}; }
}  // namespace generativeqc::scf
namespace generativeqc::dft {
struct FockPlanStub {
  core::System data;
  FockPlanStub() {
    data.atoms.resize(2);
    data.shells.resize(2);
    for (unsigned index = 0; index < 2; ++index) {
      data.atoms[index].atomic_number = 1;
      data.shells[index].atom_index = index;
      data.shells[index].primitives.resize(2);
    }
  }
  const core::System& system() const { return data; }
  struct Strategy {
    scf::FockBuildSpec spec;
  };
  Strategy strategy() const {
    Strategy result;
    result.spec.coulomb.approximation =
        stub::df ? scf::FockApproximation::DensityFitted : scf::FockApproximation::Exact;
    return result;
  }
  const void* cuda_fitted_source() const { return this; }
};
}  // namespace generativeqc::dft
namespace generativeqc::scf {
using PreparedFockPlan = dft::FockPlanStub;
}  // namespace generativeqc::scf
namespace generativeqc::dft {
inline FockPlanStub primary, fitted_range;
struct CudaKsResidentFockBinding {
  FockPlanStub* primary = &dft::primary;
  FockPlanStub::Strategy correction_model{};
  const FockPlanStub::Strategy* correction{};
  FockPlanStub* fitted_range{};
};
struct CudaKsResidentOrbitalFrameBinding {
  std::size_t nao = 2;
  unsigned spins = stub::spins;
  const double* density{};
  int device_id{};
  void* stream{};
};
inline bool cuda_ks_hessian_model_matches(const CudaKsFinalStateToken&,
                                          const CudaKsResidentFockBinding&,
                                          CudaWb97mvGridOptions*) {
  return true;
}
class CudaKsGridReference {
 public:
  CudaKsGridReference(methods::PreparedBatch&, std::size_t, const CudaKsFinalStateToken&,
                      CudaKsGridRole = CudaKsGridRole::primary) {}
  generativeqc_status prepare(std::string&) { return GENERATIVEQC_STATUS_SUCCESS; }
  CudaWb97mvGridShape shape() const { return {2, 4, 2, stub::points, stub::spins}; }
  CudaWb97mvGridReference reference() const {
    CudaWb97mvGridReference value;
    value.device_id = 0;
    return value;
  }
  CudaGridGeometryShape geometry_shape() const { return {2, 4, 2, stub::points, stub::points / 2}; }
  CudaGridGeometryOptions geometry_options() const { return {}; }
  CudaGridGeometryReference geometry_reference() const { return {}; }
};
cudaError_t cuda_wb97mv_grid_workspace(const CudaWb97mvGridShape&, const CudaWb97mvGridOptions& o,
                                       std::size_t& bytes) noexcept {
  if (o.max_nuclear_affine_terms == 1 || o.max_workspace_bytes < stub::grid_scalar())
    return cudaErrorInvalidValue;
  bytes = o.replay_nuclear_maps && o.max_workspace_bytes >= stub::grid_parallel()
              ? stub::grid_parallel()
              : stub::grid_scalar();
  return cudaSuccess;
}
CudaWb97mvGridResponse::~CudaWb97mvGridResponse() = default;
// Host resource doubles do not support the optional numerical weight fold.
bool CudaWb97mvGridResponse::admits_cartesian_mixed(std::size_t) const noexcept { return false; }
const double* CudaWb97mvGridResponse::private_becke_energy() const noexcept { return nullptr; }
std::size_t CudaGridGeometryResponse::cartesian_weight_fold_workspace() const noexcept { return 0; }
std::size_t CudaGridGeometryResponse::cartesian_weight_fold_provider_allowance() noexcept {
  return 0;
}
cudaError_t CudaGridGeometryResponse::prepare_cartesian_weight_fold(const double*, void*,
                                                                    std::size_t,
                                                                    cudaStream_t) noexcept {
  return cudaErrorNotSupported;
}
cudaError_t CudaWb97mvGridResponse::prepare(const CudaWb97mvGridShape& shape,
                                            const CudaWb97mvGridOptions& options,
                                            const CudaWb97mvGridReference&, void*,
                                            std::size_t bytes) noexcept {
  std::size_t required{};
  auto error = cuda_wb97mv_grid_workspace(shape, options, required);
  if (error != cudaSuccess || bytes != required) return cudaErrorInvalidValue;
  work_.workspace_bytes = bytes;
  return cudaSuccess;
}
cudaError_t cuda_grid_geometry_workspace(const CudaGridGeometryShape& shape,
                                         const CudaGridGeometryOptions& options, std::size_t& bytes,
                                         std::size_t* minimum) noexcept {
  bytes = 4096 + shape.points * shape.nao * 160;
  if (minimum) *minimum = bytes;
  return bytes <= options.max_workspace_bytes ? cudaSuccess : cudaErrorInvalidValue;
}
cudaError_t CudaGridGeometryResponse::prepare(const CudaGridGeometryShape& shape,
                                              const CudaGridGeometryOptions& options,
                                              const CudaGridGeometryReference&, void*,
                                              std::size_t bytes) noexcept {
  std::size_t required{};
  auto error = cuda_grid_geometry_workspace(shape, options, required);
  if (error != cudaSuccess || required != bytes) return cudaErrorInvalidValue;
  work_.workspace_bytes = bytes;
  return cudaSuccess;
}
std::size_t cuda_ks_one_electron_workspace(const core::System&,
                                           const integrals::CudaOneElectronShape&,
                                           const integrals::CudaOneElectronOptions& options) {
  if (options.max_workspace_bytes < stub::core_scalar || options.max_jobs == 1)
    throw std::invalid_argument("one-electron admission");
  return options.parallel_streams && options.max_workspace_bytes >= stub::core_parallel
             ? stub::core_parallel
             : stub::core_scalar;
}
struct CudaKsOneElectronResponse::Impl : stub::Storage {
  CudaKsOneElectronWork work;
};
CudaKsOneElectronResponse::CudaKsOneElectronResponse() = default;
CudaKsOneElectronResponse::~CudaKsOneElectronResponse() = default;
generativeqc_status CudaKsOneElectronResponse::prepare(
    methods::PreparedBatch&, std::size_t, const CudaKsFinalStateToken&,
    const integrals::CudaOneElectronOptions& options, std::string&) {
  impl_ = std::make_unique<Impl>();
  const auto bytes = cuda_ks_one_electron_workspace(primary.data, {}, options);
  impl_->work.device_bytes = bytes;
  return impl_->allocate(bytes, options.max_workspace_bytes, stub::Provider::core);
}
CudaKsOneElectronWork CudaKsOneElectronResponse::work() const noexcept { return impl_->work; }
struct CudaKsWb97mvResponse::Impl : stub::Storage {
  CudaKsWb97mvResponseWork work;
};
CudaKsWb97mvResponse::CudaKsWb97mvResponse() = default;
CudaKsWb97mvResponse::~CudaKsWb97mvResponse() = default;
generativeqc_status CudaKsWb97mvResponse::prepare(methods::PreparedBatch&, std::size_t,
                                                  const CudaKsFinalStateToken&,
                                                  const CudaKsWb97mvResponseOptions& options,
                                                  std::string&) {
  assert(options.defer_resident_integrals);
  impl_ = std::make_unique<Impl>();
  impl_->work.device_bytes = 1200;
  return impl_->allocate(1200, options.max_device_bytes, stub::Provider::induced);
}
CudaKsWb97mvResponseWork CudaKsWb97mvResponse::work() const noexcept { return impl_->work; }
generativeqc_status CudaKsWb97mvResponse::retain_integrals(std::size_t, bool, std::string&) {
  return GENERATIVEQC_STATUS_SUCCESS;
}
cudaError_t CudaKsWb97mvResponse::retain_fixed_grid_points(std::size_t) noexcept {
  return cudaSuccess;  // Decline optional residency without charging the mandatory budget.
}
cudaError_t CudaKsWb97mvResponse::retain_transition_builder(const response::CudaCpksShape&,
                                                            std::size_t) noexcept {
  return cudaSuccess;
}
cudaError_t CudaKsWb97mvResponse::seed_local_xc(void*, const response::CudaCpksShape&,
                                                const response::CudaCpksReference&, double*,
                                                cudaStream_t) noexcept {
  return cudaErrorNotSupported;
}
cudaError_t CudaKsWb97mvResponse::induced_nonlocal_fock(void*, const double*, double*,
                                                        const response::CudaCpksShape&,
                                                        cudaStream_t) noexcept {
  return cudaErrorNotSupported;
}
struct CudaKsWb97mvNuclearJk::Impl : stub::Storage {
  CudaWb97mvJkWork work;
};
CudaKsWb97mvNuclearJk::CudaKsWb97mvNuclearJk() = default;
CudaKsWb97mvNuclearJk::~CudaKsWb97mvNuclearJk() = default;
generativeqc_status CudaKsWb97mvNuclearJk::prepare(methods::PreparedBatch&, std::size_t,
                                                   const CudaKsFinalStateToken&,
                                                   const CudaWb97mvJkOptions& options, std::string&,
                                                   bool screened_frozen,
                                                   CudaWb97mvJkAdmissionFailure* failure) {
  if (failure) *failure = {};
  if (stub::canonical_source_limit && !screened_frozen) {
    if (failure) *failure = CudaWb97mvJkAdmissionFailure::canonical_source_limit;
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  assert(!screened_frozen ||
         (stub::canonical_source_limit && options.prefer_fresh_direction_batches));
  stub::screened_exact_preparations += screened_frozen;
  impl_ = std::make_unique<Impl>();
  impl_->work.omitted_frozen_source_groups = screened_frozen ? 1 : 0;
  impl_->work.workspace_bytes = 1700;
  return impl_->allocate(1700, options.max_workspace_bytes, stub::Provider::exact);
}
const CudaWb97mvJkWork& CudaKsWb97mvNuclearJk::work() const noexcept { return impl_->work; }
#define STUB_PROVIDER(CLASS, OPTIONS, WORK, BYTES, ID, REF)                                \
  struct CLASS::Impl : stub::Storage {                                                     \
    WORK work;                                                                             \
  };                                                                                       \
  CLASS::CLASS() = default;                                                                \
  CLASS::~CLASS() = default;                                                               \
  generativeqc_status CLASS::prepare(methods::PreparedBatch&, std::size_t,                 \
                                     const CudaKsFinalStateToken&, const OPTIONS& options, \
                                     std::string&) {                                       \
    impl_ = std::make_unique<Impl>();                                                      \
    impl_->work.workspace_bytes = BYTES;                                                   \
    return impl_->allocate(BYTES, options.max_workspace_bytes, ID);                        \
  }                                                                                        \
  WORK REF CLASS::work() const noexcept { return impl_->work; }
struct CudaKsWb97mvDfNuclear::Impl : stub::Storage {
  integrals::CudaDfNuclearWork work;
  void* tape{};
  ~Impl() { cudaFree(tape); }
};
CudaKsWb97mvDfNuclear::CudaKsWb97mvDfNuclear() = default;
CudaKsWb97mvDfNuclear::~CudaKsWb97mvDfNuclear() = default;
generativeqc_status CudaKsWb97mvDfNuclear::prepare(methods::PreparedBatch&, std::size_t,
                                                   const CudaKsFinalStateToken&,
                                                   const integrals::CudaDfNuclearOptions& options,
                                                   std::string&) {
  impl_ = std::make_unique<Impl>();
  // Model the real raw provider: an optional contiguous tape fits whenever it
  // falls inside this provider's cap, even if the owner needs the bytes later.
  const auto bytes = 700 + (stub::retain_df_tape && options.retain_directional_source &&
                                    options.max_workspace_bytes >= 700 + stub::df_tape_bytes
                                ? stub::df_tape_bytes
                                : 0);
  impl_->work.workspace_bytes = bytes;
  return impl_->allocate(
      bytes, options.max_workspace_bytes,
      options.range_omega ? stub::Provider::raw_df_range : stub::Provider::raw_df);
}
cudaError_t CudaKsWb97mvDfNuclear::retain_directional_source(std::size_t available) {
  if (!stub::retain_df_tape || available < stub::df_tape_bytes) return cudaSuccess;
  const auto status = cudaMalloc(&impl_->tape, stub::df_tape_bytes);
  if (status == cudaSuccess) impl_->work.workspace_bytes += stub::df_tape_bytes;
  return status == cudaErrorMemoryAllocation ? cudaSuccess : status;
}
const integrals::CudaDfNuclearWork& CudaKsWb97mvDfNuclear::work() const noexcept {
  return impl_->work;
}
STUB_PROVIDER(CudaKsWb97mvDfResponse, CudaWb97mvDfOptions, CudaWb97mvDfWork, 900,
              options.coulomb ? stub::Provider::fitted : stub::Provider::fitted_range, )
STUB_PROVIDER(CudaKsWb97mvEcpNuclear, integrals::CudaEcpNuclearOptions,
              integrals::CudaEcpNuclearWork, 1100, stub::Provider::ecp, const&)
#undef STUB_PROVIDER
std::size_t CudaKsWb97mvEcpNuclear::mandatory_workspace(const core::System&,
                                                        const CudaWb97mvGridShape&,
                                                        integrals::CudaEcpNuclearOptions) {
  return 1100;
}
std::size_t CudaKsWb97mvEcpNuclear::device_bytes() const noexcept {
  return impl_->work.workspace_bytes;
}
}  // namespace generativeqc::dft
namespace generativeqc::dft::dispersion {
class GcpCudaPlan;
inline D3CudaDirectionSource D3Plan::cuda_direction_source() const noexcept {
  std::abort();  // D3 numerics are not exercised by the electronic resource controls.
}
inline D4Tables eeq_d4_host_tables(D4EEQProfile) {
  throw std::logic_error("unexpected D4 table load in electronic resource controls");
}
inline EEQTables eeq2019_host_tables() {
  throw std::logic_error("unexpected EEQ table load in electronic resource controls");
}
}  // namespace generativeqc::dft::dispersion

namespace generativeqc::methods::detail {
struct KsHessianCorrections {
  const dft::dispersion::D3Plan* d3{};
  const dft::dispersion::D4Plan* d4{};
  const dft::dispersion::GcpCudaPlan* gcp{};
};
inline generativeqc_status borrow_dft_hessian_corrections(PreparedBatch&, std::size_t,
                                                          const dft::CudaKsFinalStateToken&,
                                                          KsHessianCorrections& out, std::string&) {
  out = {};
  return GENERATIVEQC_STATUS_SUCCESS;
}
inline generativeqc_status dft_cuda_resident_fock_sources(PreparedBatch&, std::size_t,
                                                          const dft::CudaKsFinalStateToken&,
                                                          dft::CudaKsResidentFockBinding& out,
                                                          std::string&) {
  out = {};
  if (stub::range_df) {
    out.fitted_range = &dft::fitted_range;
    out.correction_model.spec.exchange.omega = 0.3;
    out.correction = &out.correction_model;
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}
inline generativeqc_status dft_cuda_resident_orbital_frame(
    PreparedBatch&, std::size_t, const dft::CudaKsFinalStateToken&,
    dft::CudaKsResidentOrbitalFrameBinding& out, std::string&) {
  out = {};
  return GENERATIVEQC_STATUS_SUCCESS;
}
}  // namespace generativeqc::methods::detail
namespace generativeqc::response {
class CudaKsCpksReference {
 public:
  CudaKsCpksReference(methods::PreparedBatch&, std::size_t, const dft::CudaKsFinalStateToken&) {}
  generativeqc_status prepare(std::string&) { return GENERATIVEQC_STATUS_SUCCESS; }
  CudaCpksShape shape() const { return {2, stub::spins, {1, 1}}; }
  CudaCpksReference reference() const { return {}; }
};
cudaError_t cuda_cpks_workspace(const CudaCpksShape&, const CudaCpksOptions& options,
                                std::size_t& bytes) noexcept {
  bytes = 3072;
  return bytes <= options.max_workspace_bytes ? cudaSuccess : cudaErrorInvalidValue;
}
cudaError_t cuda_cpks_operator_workspace(const CudaCpksShape&, const CudaCpksOptions&,
                                         std::size_t& bytes) noexcept {
  bytes = 128;
  return cudaSuccess;
}
cudaError_t cuda_cpks_reference_weights(const CudaCpksShape&, const CudaCpksOptions&,
                                        const CudaCpksReference&, double*, void*, std::size_t,
                                        cudaStream_t) noexcept {
  return cudaSuccess;
}
}  // namespace generativeqc::response
namespace generativeqc::dft::hessian_aot {
const Program* find(std::string_view) noexcept {
  static constexpr std::string_view inputs[7]{};
  static const Program program{"pair", "pair", Kind::array, inputs, 3};
  return &program;
}
}  // namespace generativeqc::dft::hessian_aot
namespace generativeqc_wb97mv_stationary_maps {
struct Extents {
  std::size_t axes, coordinates, nao, neighbors, pairs, spins;
};
#define STUB_MAP(NAME)                                               \
  inline cudaError_t workspace_##NAME(Extents, std::size_t& bytes) { \
    bytes = 512;                                                     \
    return cudaSuccess;                                              \
  }                                                                  \
  inline bool work_##NAME(Extents, std::size_t& terms) {             \
    terms = 32;                                                      \
    return true;                                                     \
  }
STUB_MAP(weights)
STUB_MAP(rhs) STUB_MAP(hvp) STUB_MAP(nuclear_inputs) STUB_MAP(nuclear_second) STUB_MAP(nuclear_atom)
#undef STUB_MAP
}  // namespace generativeqc_wb97mv_stationary_maps
