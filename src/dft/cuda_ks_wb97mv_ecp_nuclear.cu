#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_ecp_nuclear.hpp"
#include "dft/cuda_ks_wb97mv_model.hpp"
#include "response/ecp_density.cuh"
#include "scf/fock_prepared.hpp"

namespace generativeqc::dft {
namespace {
namespace maps = generativeqc_ecp_density_maps;
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
std::size_t product(std::size_t first, std::size_t second) {
  require(!second || first <= SIZE_MAX / second, "ECP source extent overflow");
  return first * second;
}
bool disjoint(const void* first, std::size_t first_bytes, const void* second,
              std::size_t second_bytes) {
  const auto left = reinterpret_cast<std::uintptr_t>(first),
             right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left >= first_bytes : left - right >= second_bytes;
}
struct Range {
  const double* pointer{};
  std::size_t bytes{};
};
__global__ void symmetric_spins(const double* density, std::size_t nao, unsigned spins,
                                double absolute, double relative, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < spins * nao * nao;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto value = density[index];
    const auto other =
        density[index / (nao * nao) * nao * nao + index % nao * nao + index / nao % nao];
    if (!isfinite(value) || !isfinite(other) ||
        fabs(value - other) > absolute + relative * fmax(fabs(value), fabs(other)))
      atomicExch(status, 1);
  }
}
}  // namespace
struct CudaKsWb97mvEcpNuclear::Impl {
  methods::PreparedBatch& batch;
  std::size_t item;
  CudaKsFinalStateToken expected;
  CudaKsGridReference grid;
  CudaKsResidentFockBinding fock;
  CudaKsResidentOrbitalFrameBinding frame;
  integrals::CudaEcpNuclearShape shape{};
  integrals::CudaEcpNuclearResponse response;
  const core::EcpTerm* host_terms{};
  void* arena{};
  std::size_t bytes{}, matrix{}, spin_bytes{}, basis_bytes{}, points{}, sum_offset{}, sum_bytes{},
      total_offset{}, status_offset{}, sum_terms{};
  unsigned spins{};
  double symmetry_absolute{}, symmetry_relative{};
  cudaStream_t stream{};

  Impl(methods::PreparedBatch& owner, std::size_t index, CudaKsFinalStateToken token)
      : batch(owner), item(index), expected(std::move(token)), grid(batch, item, expected) {}
  ~Impl() {
    if (stream) (void)cudaStreamSynchronize(stream);
    if (arena) (void)cudaFree(arena);
  }
  template <class Value = double>
  Value* at(std::size_t offset) {
    return reinterpret_cast<Value*>(static_cast<char*>(arena) + offset);
  }
  static cudaError_t authenticate(void* context) noexcept {
    auto& self = *static_cast<Impl*>(context);
    try {
      CudaKsResidentFockBinding current;
      CudaKsResidentOrbitalFrameBinding current_frame;
      std::string detail;
      if (methods::detail::dft_cuda_resident_fock_sources(self.batch, self.item, self.expected,
                                                          current,
                                                          detail) != GENERATIVEQC_STATUS_SUCCESS ||
          current != self.fock || !cuda_ks_wb97mv_model_matches(self.expected, current) ||
          methods::detail::dft_cuda_resident_orbital_frame(self.batch, self.item, self.expected,
                                                           current_frame,
                                                           detail) != GENERATIVEQC_STATUS_SUCCESS ||
          current_frame != self.frame)
        return cudaErrorInvalidResourceHandle;
      const auto& terms = current.primary->system().ecp_terms;
      if (terms.data() != self.host_terms || terms.size() != self.shape.terms)
        return cudaErrorInvalidResourceHandle;
      const auto lease = self.grid.reference();
      return lease.authenticate ? lease.authenticate(lease.owner) : cudaErrorInvalidResourceHandle;
    } catch (...) {
      return cudaErrorUnknown;
    }
  }
  cudaError_t admit(const double* direction, const double* spin_direction,
                    const std::array<Range, 4>& output, cudaStream_t requested) {
    if (requested != stream) return cudaErrorInvalidResourceHandle;
    cudaStreamCaptureStatus capture{};
    auto error = cudaStreamIsCapturing(stream, &capture);
    if (error != cudaSuccess) return error;
    if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    if ((error = authenticate(this)) != cudaSuccess) return error;
    const auto lease = grid.reference();
    const Range sources[]{{direction, 3 * shape.atoms * sizeof(double)},
                          {spin_direction, spin_direction ? spin_bytes : 0}};
    const auto device_range = [&](Range range) {
      if (!range.pointer || reinterpret_cast<std::uintptr_t>(range.pointer) % alignof(double))
        return cudaErrorInvalidValue;
      cudaPointerAttributes attributes{};
      const auto result = cudaPointerGetAttributes(&attributes, range.pointer);
      if (result != cudaSuccess) return result;
      return attributes.type == cudaMemoryTypeDevice && attributes.device == lease.device_id
                 ? cudaSuccess
                 : cudaErrorInvalidValue;
    };
    for (const auto source : sources) {
      if (!source.bytes) continue;
      if ((error = device_range(source)) != cudaSuccess) return error;
      if (!disjoint(source.pointer, source.bytes, arena, bytes)) return cudaErrorInvalidValue;
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
      const auto range = output[index];
      if ((error = device_range(range)) != cudaSuccess) return error;
      if (!disjoint(range.pointer, range.bytes, arena, bytes) ||
          !disjoint(range.pointer, range.bytes, lease.basis, basis_bytes) ||
          !disjoint(range.pointer, range.bytes, lease.points, 3 * points * sizeof(double)) ||
          !disjoint(range.pointer, range.bytes, lease.weights, points * sizeof(double)) ||
          !disjoint(range.pointer, range.bytes, lease.density, spin_bytes) ||
          !disjoint(range.pointer, range.bytes, frame.coefficients, spin_bytes) ||
          !disjoint(range.pointer, range.bytes, frame.orbital_energies,
                    spins * shape.nao * sizeof(double)) ||
          !disjoint(range.pointer, range.bytes, frame.overlap, matrix * sizeof(double)) ||
          !disjoint(range.pointer, range.bytes, fock.hcore, matrix * sizeof(double)) ||
          !disjoint(range.pointer, range.bytes, fock.fock, spin_bytes))
        return cudaErrorInvalidValue;
      for (const auto source : sources)
        if (source.bytes && !disjoint(range.pointer, range.bytes, source.pointer, source.bytes))
          return cudaErrorInvalidValue;
      for (std::size_t earlier = 0; earlier < index; ++earlier)
        if (!disjoint(range.pointer, range.bytes, output[earlier].pointer, output[earlier].bytes))
          return cudaErrorInvalidValue;
    }
    return cudaSuccess;
  }
};
CudaKsWb97mvEcpNuclear::CudaKsWb97mvEcpNuclear() = default;
CudaKsWb97mvEcpNuclear::~CudaKsWb97mvEcpNuclear() = default;

generativeqc_status CudaKsWb97mvEcpNuclear::prepare(methods::PreparedBatch& batch, std::size_t item,
                                                    const CudaKsFinalStateToken& expected,
                                                    const integrals::CudaEcpNuclearOptions& options,
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
    require(cuda_ks_wb97mv_model_matches(expected, owner->fock),
            "ECP source requires the complete native WB97M-V model");
    status = methods::detail::dft_cuda_resident_orbital_frame(batch, item, expected, owner->frame,
                                                              detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    const auto lease = owner->grid.reference();
    const auto shape = owner->grid.shape();
    const auto& system = owner->fock.primary->system();
    require(shape.nao == owner->fock.nao && shape.spins == owner->fock.spins &&
                shape.atoms == system.atoms.size() && owner->fock.device_id == lease.device_id &&
                owner->fock.stream == reinterpret_cast<void*>(lease.stream),
            "ECP source differs from its actual canonical native KS owners");
    require(!system.ecp_terms.empty(), "ECP source requires residual Gaussian terms");
    owner->stream = lease.stream;
    cudaStreamCaptureStatus capture{};
    check(cudaStreamIsCapturing(owner->stream, &capture));
    require(capture == cudaStreamCaptureStatusNone, "ECP source preparation cannot capture");
    std::vector<unsigned> centers;
    for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
      if (system.atoms[atom].ecp_core) centers.push_back(static_cast<unsigned>(atom));
    owner->shape = {shape.atoms, shape.primitives, shape.nao, system.ecp_terms.size()};
    owner->host_terms = system.ecp_terms.data();
    owner->spins = shape.spins;
    owner->points = shape.points;
    owner->symmetry_absolute = options.symmetry_absolute_tolerance;
    owner->symmetry_relative = options.symmetry_relative_tolerance;
    owner->matrix = product(shape.nao, shape.nao);
    owner->spin_bytes = product(product(owner->matrix, shape.spins), sizeof(double));
    owner->basis_bytes =
        product(3 * shape.atoms + 2 * shape.primitives + 16 * shape.nao, sizeof(double));
    std::size_t bytes{};
    check(integrals::cuda_ecp_nuclear_workspace(owner->shape, centers, options, bytes));
    const auto reserve = [&](std::size_t count, std::size_t size) {
      const auto offset = bytes, extra = product(count, size);
      require(extra <= SIZE_MAX - 255 && bytes <= SIZE_MAX - extra - 255,
              "ECP staging alignment overflow");
      bytes = (bytes + extra + 255) / 256 * 256;
      return offset;
    };
    const auto workspace_bytes = bytes;
    const auto terms_offset = reserve(system.ecp_terms.size(), sizeof(core::EcpTerm));
    owner->total_offset = reserve(product(2, owner->matrix), sizeof(double));
    const maps::Extents extents{shape.nao, shape.spins};
    check(maps::workspace_total(extents, owner->sum_bytes));
    require(
        maps::work_total(extents, owner->sum_terms) && owner->sum_terms <= options.max_affine_terms,
        "ECP physical spin sum exceeds affine work budget");
    owner->sum_offset = reserve(owner->sum_bytes, 1);
    owner->status_offset = reserve(1, sizeof(int));
    require(bytes <= options.max_workspace_bytes, "ECP source plus staging exceeds device budget");
    owner->bytes = bytes;
    check(cudaMalloc(&owner->arena, bytes));
    auto* terms = owner->at<core::EcpTerm>(terms_offset);
    check(cudaMemcpyAsync(terms, system.ecp_terms.data(),
                          system.ecp_terms.size() * sizeof(core::EcpTerm), cudaMemcpyHostToDevice,
                          owner->stream));
    check(cudaStreamSynchronize(owner->stream));
    const integrals::CudaEcpNuclearReference reference{
        lease.basis, terms, lease.device_id, owner->stream, Impl::authenticate, owner.get()};
    check(owner->response.prepare(owner->shape, centers, options, reference, owner->arena,
                                  workspace_bytes));
    require(owner->response.work().contraction_affine_terms <=
                options.max_affine_terms - owner->sum_terms,
            "ECP contraction plus spin sum exceeds affine work budget");
    impl_ = std::move(owner);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "native ECP source allocation failed";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
}
cudaError_t CudaKsWb97mvEcpNuclear::values(const double* direction,
                                           const integrals::CudaEcpNuclearValues& output,
                                           cudaStream_t stream) noexcept {
  if (!impl_) return cudaErrorInvalidResourceHandle;
  const auto bytes = impl_->matrix * sizeof(double);
  const std::array<Range, 4> ranges{{{output.local, bytes},
                                     {output.nonlocal, bytes},
                                     {output.direction_local, bytes},
                                     {output.direction_nonlocal, bytes}}};
  const auto error = impl_->admit(direction, nullptr, ranges, stream);
  return error == cudaSuccess ? impl_->response.values(direction, output, stream) : error;
}
cudaError_t CudaKsWb97mvEcpNuclear::contract(const double* direction,
                                             const double* spin_direction_density,
                                             const integrals::CudaEcpNuclearVectors& output,
                                             cudaStream_t stream) noexcept {
  if (!impl_) return cudaErrorInvalidResourceHandle;
  if (!spin_direction_density) return cudaErrorInvalidValue;
  const auto bytes = 3 * impl_->shape.atoms * sizeof(double);
  const std::array<Range, 4> ranges{{{output.gradient_local, bytes},
                                     {output.gradient_nonlocal, bytes},
                                     {output.hvp_local, bytes},
                                     {output.hvp_nonlocal, bytes}}};
  auto error = impl_->admit(direction, spin_direction_density, ranges, stream);
  if (error != cudaSuccess) return error;
  auto* status = impl_->at<int>(impl_->status_offset);
  auto* total = impl_->at(impl_->total_offset);
  error = cudaMemsetAsync(status, 0, sizeof(int), stream);
  if (error == cudaSuccess) {
    symmetric_spins<<<maps::blocks(impl_->spins * impl_->matrix), 128, 0, stream>>>(
        spin_direction_density, impl_->shape.nao, impl_->spins, impl_->symmetry_absolute,
        impl_->symmetry_relative, status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess)
    error = maps::launch_total({impl_->shape.nao, impl_->spins}, spin_direction_density,
                               impl_->grid.reference().density, total + impl_->matrix, total,
                               impl_->at(impl_->sum_offset), impl_->sum_bytes, status, stream);
  int failed{};
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(&failed, status, sizeof(failed), cudaMemcpyDeviceToHost, stream);
  const auto completed = cudaStreamSynchronize(stream);
  if (error == cudaSuccess) error = completed;
  if (error != cudaSuccess) return error;
  if (failed) return cudaErrorInvalidValue;
  return impl_->response.contract(direction, {total, total + impl_->matrix}, output, stream);
}
integrals::CudaEcpNuclearShape CudaKsWb97mvEcpNuclear::shape() const noexcept {
  return impl_ ? impl_->shape : integrals::CudaEcpNuclearShape{};
}
const integrals::CudaEcpNuclearWork& CudaKsWb97mvEcpNuclear::work() const noexcept {
  static const integrals::CudaEcpNuclearWork empty;
  return impl_ ? impl_->response.work() : empty;
}
std::size_t CudaKsWb97mvEcpNuclear::device_bytes() const noexcept {
  return impl_ ? impl_->bytes : 0;
}
std::uint64_t CudaKsWb97mvEcpNuclear::spin_sum_terms_per_contraction() const noexcept {
  return impl_ ? impl_->sum_terms : 0;
}
}  // namespace generativeqc::dft
