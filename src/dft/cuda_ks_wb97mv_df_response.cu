#include <limits>
#include <stdexcept>

#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_df_response.hpp"
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
std::size_t product(std::size_t first, std::size_t second) {
  require(!second || first <= SIZE_MAX / second, "fitted-response extent overflow");
  return first * second;
}
}  // namespace
struct CudaKsWb97mvDfResponse::Impl {
  methods::PreparedBatch& batch;
  std::size_t item;
  CudaKsFinalStateToken expected;
  CudaKsGridReference grid;
  CudaKsResidentFockBinding fock;
  scf::CudaDfReadonlySource fitted;
  scf::CudaDfReadonlyMetric metric;
  CudaWb97mvDfShape shape{};
  CudaWb97mvDfResponse response;
  void* arena{};
  std::size_t raw_bytes{};
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
          scf::cuda_density_fitting_readonly_metric(current.primary->cuda_fitted_source()) !=
              self.metric)
        return cudaErrorInvalidResourceHandle;
      const auto lease = self.grid.reference();
      return lease.authenticate ? lease.authenticate(lease.owner) : cudaErrorInvalidResourceHandle;
    } catch (...) {
      return cudaErrorUnknown;
    }
  }
};
CudaKsWb97mvDfResponse::CudaKsWb97mvDfResponse() = default;
CudaKsWb97mvDfResponse::~CudaKsWb97mvDfResponse() = default;

generativeqc_status CudaKsWb97mvDfResponse::prepare(methods::PreparedBatch& batch, std::size_t item,
                                                    const CudaKsFinalStateToken& expected,
                                                    const CudaWb97mvDfOptions& options,
                                                    std::string& detail) {
  impl_.reset();
  detail.clear();
  try {
    auto owner = std::make_unique<Impl>(batch, item, expected);
    auto result = owner->grid.prepare(detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    result =
        methods::detail::dft_cuda_resident_fock_sources(batch, item, expected, owner->fock, detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    require(cuda_ks_wb97mv_model_matches(expected, owner->fock) &&
                owner->fock.primary->strategy().spec.coulomb.approximation ==
                    scf::FockApproximation::DensityFitted,
            "fitted response requires the complete native WB97M-V DF model");
    const auto* plan = owner->fock.primary->cuda_fitted_source();
    owner->fitted = scf::cuda_density_fitting_readonly_source(plan);
    owner->metric = scf::cuda_density_fitting_readonly_metric(plan);
    const auto lease = owner->grid.reference();
    const auto grid_shape = owner->grid.shape();
    require(owner->fitted && owner->metric && owner->fitted.identity == owner->metric.identity &&
                owner->fitted.naux == owner->metric.naux && owner->fitted.nao == grid_shape.nao &&
                owner->fock.nao == grid_shape.nao && owner->fock.spins == grid_shape.spins &&
                owner->fitted.device_id == lease.device_id &&
                owner->fock.device_id == lease.device_id && owner->fitted.stream == lease.stream &&
                owner->fock.stream == reinterpret_cast<void*>(lease.stream),
            "fitted response requires live, matching full-rank raw and metric sources");
    owner->stream = lease.stream;
    cudaStreamCaptureStatus capture{};
    check(cudaStreamIsCapturing(owner->stream, &capture));
    require(capture == cudaStreamCaptureStatusNone, "fitted response preparation cannot capture");
    owner->shape = {grid_shape.nao, owner->fitted.naux, grid_shape.spins};
    std::size_t bytes{};
    check(cuda_wb97mv_df_workspace(owner->shape, options, bytes));
    const auto pairs = product(grid_shape.nao, grid_shape.nao);
    owner->raw_bytes = product(product(pairs, owner->fitted.naux), sizeof(double));
    require(owner->raw_bytes <= SIZE_MAX - 255, "fitted response raw-source alignment overflow");
    owner->raw_bytes = (owner->raw_bytes + 255) / 256 * 256;
    require(bytes <= options.max_workspace_bytes &&
                owner->raw_bytes <= options.max_workspace_bytes - bytes,
            "fitted response retained raw source exceeds its explicit device budget");
    check(cudaMalloc(&owner->arena, owner->raw_bytes + bytes));
    auto* raw = static_cast<double*>(owner->arena);
    result = scf::enqueue_cuda_density_fitting_raw_values(
        owner->fitted.integrals, 0, pairs, reinterpret_cast<void*>(owner->stream), raw, detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    const CudaWb97mvDfReference reference{lease.density,
                                          raw,
                                          owner->fitted.whitening,
                                          owner->metric.eigenvalues,
                                          owner->metric.relative_threshold,
                                          lease.device_id,
                                          lease.stream,
                                          Impl::authenticate,
                                          owner.get()};
    check(owner->response.prepare(owner->shape, options, reference,
                                  static_cast<char*>(owner->arena) + owner->raw_bytes, bytes));
    impl_ = std::move(owner);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "native fitted-response allocation failed";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
}
cudaError_t CudaKsWb97mvDfResponse::apply(const double* delta_density,
                                          const double* delta_three_center,
                                          const double* delta_metric,
                                          const CudaWb97mvDfOutput& output,
                                          cudaStream_t stream) noexcept {
  return impl_ ? impl_->response.apply(delta_density, delta_three_center, delta_metric, output,
                                       stream)
               : cudaErrorInvalidResourceHandle;
}
CudaWb97mvDfShape CudaKsWb97mvDfResponse::shape() const noexcept {
  return impl_ ? impl_->shape : CudaWb97mvDfShape{};
}
CudaWb97mvDfWork CudaKsWb97mvDfResponse::work() const noexcept {
  auto receipt = impl_ ? impl_->response.work() : CudaWb97mvDfWork{};
  if (impl_) receipt.workspace_bytes += impl_->raw_bytes;
  return receipt;
}
}  // namespace generativeqc::dft
