#include <stdexcept>
#include <vector>

#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_df_nuclear.hpp"
#include "dft/cuda_ks_wb97mv_model.hpp"
#include "scf/cuda_density_fitting_device.hpp"
#include "scf/fock_prepared.hpp"

namespace generativeqc::dft {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}  // namespace
struct CudaKsWb97mvDfNuclear::Impl {
  methods::PreparedBatch& batch;
  std::size_t item;
  CudaKsFinalStateToken expected;
  CudaKsGridReference grid;
  CudaKsResidentFockBinding fock;
  scf::CudaDfReadonlySource fitted;
  scf::CudaDfReadonlyBasis basis;
  integrals::CudaDfNuclearShape shape{};
  integrals::CudaDfNuclearResponse response;
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
          current != self.fock || !cuda_ks_wb97mv_model_matches(self.expected, current) ||
          scf::cuda_density_fitting_readonly_source(current.primary->cuda_fitted_source()) !=
              self.fitted ||
          scf::cuda_density_fitting_readonly_basis(self.fitted.integrals) != self.basis)
        return cudaErrorInvalidResourceHandle;
      const auto reference = self.grid.reference();
      return reference.authenticate ? reference.authenticate(reference.owner)
                                    : cudaErrorInvalidResourceHandle;
    } catch (...) {
      return cudaErrorUnknown;
    }
  }
};
CudaKsWb97mvDfNuclear::CudaKsWb97mvDfNuclear() = default;
CudaKsWb97mvDfNuclear::~CudaKsWb97mvDfNuclear() = default;
generativeqc_status CudaKsWb97mvDfNuclear::prepare(methods::PreparedBatch& batch, std::size_t item,
                                                   const CudaKsFinalStateToken& expected,
                                                   const integrals::CudaDfNuclearOptions& options,
                                                   std::string& detail) {
  impl_.reset();
  detail.clear();
  try {
    auto owner = std::make_unique<Impl>(batch, item, expected);
    auto status = owner->grid.prepare(detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    status =
        methods::detail::dft_cuda_resident_fock_sources(batch, item, expected, owner->fock, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    require(cuda_ks_wb97mv_model_matches(expected, owner->fock) &&
                owner->fock.primary->strategy().spec.coulomb.approximation ==
                    scf::FockApproximation::DensityFitted,
            "raw DF nuclear source requires the complete native WB97M-V fitted model");
    owner->fitted =
        scf::cuda_density_fitting_readonly_source(owner->fock.primary->cuda_fitted_source());
    require(static_cast<bool>(owner->fitted), "raw DF nuclear source requires live fitted sources");
    owner->basis = scf::cuda_density_fitting_readonly_basis(owner->fitted.integrals);
    const auto lease = owner->grid.reference();
    const auto grid_shape = owner->grid.shape();
    require(
        owner->basis && owner->basis.nao == grid_shape.nao &&
            owner->basis.atoms == grid_shape.atoms && owner->basis.nao == owner->fitted.nao &&
            owner->basis.naux == owner->fitted.naux && owner->basis.device_id == lease.device_id &&
            owner->fitted.device_id == lease.device_id && owner->fitted.stream == lease.stream &&
            owner->fock.stream == reinterpret_cast<void*>(lease.stream),
        "raw DF nuclear source differs from its actual canonical KS owners");
    owner->stream = lease.stream;
    cudaStreamCaptureStatus capture{};
    check(cudaStreamIsCapturing(owner->stream, &capture));
    require(capture == cudaStreamCaptureStatusNone, "raw DF nuclear preparation cannot capture");
    const auto& orbital_system = owner->fock.primary->system();
    const auto* auxiliary_system = owner->fock.primary->auxiliary_system();
    require(auxiliary_system && scf::cuda_density_fitting_integral_source_geometry_matches(
                                    owner->fitted.integrals, 0, orbital_system, *auxiliary_system),
            "DF topology lacks its actual normalized source identity");
    std::vector<integrals::CudaDfNuclearShell> orbital, auxiliary;
    std::size_t primitive_begin{};
    const auto topology = [&](const core::System& system, auto& shells) {
      std::size_t ao_begin{}, cartesian_begin{};
      for (const auto& shell : system.shells) {
        const auto angular = static_cast<std::size_t>(shell.angular_momentum);
        const auto cartesian_count = (angular + 1) * (angular + 2) / 2;
        const auto public_count = system.basis_representation == core::kSphericalBasis
                                      ? 2 * angular + 1
                                      : cartesian_count;
        shells.push_back({shell.atom_index, angular, primitive_begin, shell.primitives.size(),
                          ao_begin, public_count, cartesian_begin});
        primitive_begin += shell.primitives.size();
        ao_begin += public_count;
        cartesian_begin += cartesian_count;
      }
    };
    topology(orbital_system, orbital);
    topology(*auxiliary_system, auxiliary);
    owner->shape = {owner->basis.atoms, owner->basis.primitives,    owner->basis.nao,
                    owner->basis.naux,  owner->basis.cartesian_nao, owner->basis.cartesian_naux};
    std::size_t bytes{};
    check(integrals::cuda_df_nuclear_workspace(owner->shape, orbital, auxiliary, options, bytes));
    check(cudaMalloc(&owner->arena, bytes));
    const integrals::CudaDfNuclearReference reference{owner->basis, owner->stream,
                                                      Impl::authenticate, owner.get()};
    check(owner->response.prepare(owner->shape, orbital, auxiliary, options, reference,
                                  owner->arena, bytes));
    impl_ = std::move(owner);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "native raw DF nuclear source allocation failed";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
}
cudaError_t CudaKsWb97mvDfNuclear::values(const double* direction,
                                          const integrals::CudaDfNuclearValues& output,
                                          cudaStream_t stream) noexcept {
  return impl_ ? impl_->response.values(direction, output, stream) : cudaErrorInvalidResourceHandle;
}
cudaError_t CudaKsWb97mvDfNuclear::contract(const double* direction,
                                            const integrals::CudaDfNuclearWeights& weights,
                                            const integrals::CudaDfNuclearVectors& output,
                                            cudaStream_t stream) noexcept {
  return impl_ ? impl_->response.contract(direction, weights, output, stream)
               : cudaErrorInvalidResourceHandle;
}
integrals::CudaDfNuclearShape CudaKsWb97mvDfNuclear::shape() const noexcept {
  return impl_ ? impl_->shape : integrals::CudaDfNuclearShape{};
}
const integrals::CudaDfNuclearWork& CudaKsWb97mvDfNuclear::work() const noexcept {
  static const integrals::CudaDfNuclearWork empty;
  return impl_ ? impl_->response.work() : empty;
}
}  // namespace generativeqc::dft
