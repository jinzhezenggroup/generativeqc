#include <stdexcept>
#include <vector>

#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_model.hpp"
#include "dft/cuda_ks_wb97mv_nuclear_jk.hpp"
#include "methods/dft_method.hpp"
#include "scf/fock_prepared.hpp"

namespace generativeqc::dft {
namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void require(bool valid, const char* message) {
  if (!valid) throw std::invalid_argument(message);
}
}  // namespace
struct CudaKsWb97mvNuclearJk::Impl {
  methods::PreparedBatch& batch;
  std::size_t item;
  CudaKsFinalStateToken expected;
  CudaKsGridReference grid;
  CudaKsResidentFockBinding fock;
  CudaWb97mvNuclearJk response;
  void* arena{};
  cudaStream_t stream{};
  Impl(methods::PreparedBatch& owner, std::size_t index, CudaKsFinalStateToken token)
      : batch(owner), item(index), expected(std::move(token)), grid(batch, item, expected) {}
  ~Impl() {
    if (stream) (void)cudaStreamSynchronize(stream);
    if (arena) (void)cudaFree(arena);
  }
  static cudaError_t authenticate(void* context) noexcept {
    auto& self = *static_cast<Impl*>(context);
    try {
      CudaKsResidentFockBinding current;
      std::string detail;
      if (methods::detail::dft_cuda_resident_fock_sources(self.batch, self.item, self.expected,
                                                          current,
                                                          detail) != GENERATIVEQC_STATUS_SUCCESS ||
          current != self.fock)
        return cudaErrorInvalidResourceHandle;
      const auto lease = self.grid.reference();
      return lease.authenticate ? lease.authenticate(lease.owner) : cudaErrorInvalidResourceHandle;
    } catch (...) {
      return cudaErrorUnknown;
    }
  }
};
CudaKsWb97mvNuclearJk::CudaKsWb97mvNuclearJk() = default;
CudaKsWb97mvNuclearJk::~CudaKsWb97mvNuclearJk() = default;
generativeqc_status CudaKsWb97mvNuclearJk::prepare(methods::PreparedBatch& batch, std::size_t item,
                                                   const CudaKsFinalStateToken& expected,
                                                   const CudaWb97mvJkOptions& options,
                                                   std::string& detail) {
  impl_.reset();
  try {
    auto candidate = std::make_unique<Impl>(batch, item, expected);
    auto status = candidate->grid.prepare(detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    status = methods::detail::dft_cuda_resident_fock_sources(batch, item, expected, candidate->fock,
                                                             detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    require(cuda_ks_wb97mv_model_matches(expected, candidate->fock),
            "J/K nuclear response requires the complete native WB97M-V model");
    require(
        candidate->fock.primary->strategy().spec.coulomb.approximation !=
                scf::FockApproximation::DensityFitted ||
            options.part == CudaWb97mvJkPart::correction,
        "DF nuclear owner admits only exact LR correction; fitted primary response is separate");
    const auto shape = candidate->grid.shape();
    const auto lease = candidate->grid.reference();
    const auto& system = candidate->fock.primary->system();
    int device{-1};
    check(cudaGetDevice(&device));
    require(candidate->fock.device_id == device && lease.device_id == device &&
                lease.stream == reinterpret_cast<cudaStream_t>(candidate->fock.stream) &&
                shape.nao == candidate->fock.nao && shape.spins == candidate->fock.spins &&
                shape.atoms == system.atoms.size(),
            "J/K source differs from its canonical native KS owner");
    candidate->stream = lease.stream;
    cudaStreamCaptureStatus capture{};
    check(cudaStreamIsCapturing(candidate->stream, &capture));
    require(capture == cudaStreamCaptureStatusNone, "J/K source preparation cannot capture");
    std::vector<integrals::CudaOneElectronShell> shells;
    std::size_t primitive_begin{}, ao_begin{};
    for (const auto& shell : system.shells) {
      const auto angular = static_cast<std::size_t>(shell.angular_momentum);
      const auto count = system.basis_representation == core::kSphericalBasis
                             ? 2 * angular + 1
                             : (angular + 1) * (angular + 2) / 2;
      shells.push_back(
          {shell.atom_index, angular, primitive_begin, shell.primitives.size(), ao_begin, count});
      primitive_begin += shell.primitives.size();
      ao_begin += count;
    }
    require(primitive_begin == shape.primitives && ao_begin == shape.nao,
            "J/K topology differs from normalized native basis");
    const CudaWb97mvJkShape integral_shape{shape.atoms, shape.primitives, shape.nao, shape.spins};
    std::size_t bytes{};
    const auto admission =
        cuda_wb97mv_jk_workspace(integral_shape, shells, options, bytes, &detail);
    if (admission != cudaSuccess) throw std::runtime_error("J/K workspace admission: " + detail);
    check(cudaMalloc(&candidate->arena, bytes));
    const CudaWb97mvJkReference reference{lease.basis, device, candidate->stream,
                                          Impl::authenticate, candidate.get()};
    check(candidate->response.prepare(integral_shape, shells, options, reference, candidate->arena,
                                      bytes));
    impl_ = std::move(candidate);
    detail.clear();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "native J/K source allocation failed";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
}
cudaError_t CudaKsWb97mvNuclearJk::apply(const double* direction, const double* delta_density,
                                         const CudaWb97mvJkOutput& output,
                                         cudaStream_t stream) noexcept {
  if (!impl_) return cudaErrorInvalidResourceHandle;
  const auto lease = impl_->grid.reference();
  if (!lease.density) return cudaErrorInvalidResourceHandle;
  return impl_->response.apply(direction, lease.density, delta_density, output, stream);
}
const CudaWb97mvJkWork& CudaKsWb97mvNuclearJk::work() const noexcept {
  static const CudaWb97mvJkWork empty;
  return impl_ ? impl_->response.work() : empty;
}
}  // namespace generativeqc::dft
