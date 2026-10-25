#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dft/cuda_ks_wb97mv_hessian.hpp"
#include "dft/hessian_aot.hpp"

namespace stub {
inline std::size_t points = 8;
inline unsigned spins = 1;
inline bool df = false, ecp = false;
inline int reject_provider = 0;
inline std::size_t grid_scalar() { return 21600 + (points - 8) * 128; }
inline std::size_t grid_parallel() { return grid_scalar() + 21 * points * sizeof(double); }
inline constexpr std::size_t core_scalar = 272, core_parallel = 528;
struct Storage {
  void* arena{};
  ~Storage() { cudaFree(arena); }
  generativeqc_status allocate(std::size_t bytes, std::size_t cap, int provider) {
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
inline Source cuda_density_fitting_readonly_source(int) { return {}; }
}  // namespace generativeqc::scf
namespace generativeqc::dft {
struct Primary {
  core::System data;
  Primary() {
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
    struct Spec {
      struct Coulomb {
        scf::FockApproximation approximation;
      } coulomb;
    } spec;
  };
  Strategy strategy() const {
    return {{{stub::df ? scf::FockApproximation::DensityFitted : scf::FockApproximation::Exact}}};
  }
  int cuda_fitted_source() const { return 0; }
};
inline Primary primary;
struct CudaKsResidentFockBinding {
  Primary* primary = &dft::primary;
};
struct CudaKsResidentOrbitalFrameBinding {
  std::size_t nao = 2;
  unsigned spins = stub::spins;
  const double* density{};
  int device_id{};
  void* stream{};
};
inline bool cuda_ks_wb97mv_model_matches(const CudaKsFinalStateToken&,
                                         const CudaKsResidentFockBinding&) {
  return true;
}
class CudaKsGridReference {
 public:
  CudaKsGridReference(methods::PreparedBatch&, std::size_t, const CudaKsFinalStateToken&) {}
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
                                         const CudaGridGeometryOptions& options,
                                         std::size_t& bytes) noexcept {
  bytes = 4096 + shape.points * shape.nao * 160;
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
  return impl_->allocate(bytes, options.max_workspace_bytes, 1);
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
  return impl_->allocate(1200, options.max_device_bytes, 2);
}
CudaKsWb97mvResponseWork CudaKsWb97mvResponse::work() const noexcept { return impl_->work; }
generativeqc_status CudaKsWb97mvResponse::retain_integrals(std::size_t, std::string&) {
  return GENERATIVEQC_STATUS_SUCCESS;
}
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
STUB_PROVIDER(CudaKsWb97mvNuclearJk, CudaWb97mvJkOptions, CudaWb97mvJkWork, 1700, 3, const&)
STUB_PROVIDER(CudaKsWb97mvDfNuclear, integrals::CudaDfNuclearOptions, integrals::CudaDfNuclearWork,
              700, 4, const&)
STUB_PROVIDER(CudaKsWb97mvDfResponse, CudaWb97mvDfOptions, CudaWb97mvDfWork, 900, 5, )
STUB_PROVIDER(CudaKsWb97mvEcpNuclear, integrals::CudaEcpNuclearOptions,
              integrals::CudaEcpNuclearWork, 1100, 6, const&)
#undef STUB_PROVIDER
std::size_t CudaKsWb97mvEcpNuclear::device_bytes() const noexcept {
  return impl_->work.workspace_bytes;
}
}  // namespace generativeqc::dft
namespace generativeqc::methods::detail {
inline generativeqc_status dft_cuda_resident_fock_sources(PreparedBatch&, std::size_t,
                                                          const dft::CudaKsFinalStateToken&,
                                                          dft::CudaKsResidentFockBinding& out,
                                                          std::string&) {
  out = {};
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
  bytes = 0;
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
