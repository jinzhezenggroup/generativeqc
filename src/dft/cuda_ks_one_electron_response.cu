#include <stdexcept>
#include <vector>

#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_one_electron_response.hpp"
#include "dft/cuda_ks_resident_fock.hpp"
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
// Shared host-only plan: aggregate owners reserve the exact scalar baseline
// through the same topology, charge accounting and admission as preparation.
struct Plan {
  std::vector<integrals::CudaOneElectronShell> shells;
  integrals::CudaOneElectronOptions options;
  std::size_t bytes{}, device_bytes{};
  Plan(const core::System& system, const integrals::CudaOneElectronShape& shape,
       const integrals::CudaOneElectronOptions& requested)
      : options(requested) {
    require(shape.atoms == system.atoms.size(), "one-electron atom count differs from basis");
    std::size_t primitive_begin{}, ao_begin{};
    for (const auto& shell : system.shells) {
      const auto angular = static_cast<std::size_t>(shell.angular_momentum);
      require(angular <= 3, "one-electron angular momentum exceeds the linked source limit");
      const auto count = system.basis_representation == core::kSphericalBasis
                             ? 2 * angular + 1
                             : (angular + 1) * (angular + 2) / 2;
      shells.push_back(
          {shell.atom_index, angular, primitive_begin, shell.primitives.size(), ao_begin, count});
      require(shell.primitives.size() <= SIZE_MAX - primitive_begin && count <= SIZE_MAX - ao_begin,
              "one-electron topology overflow");
      primitive_begin += shell.primitives.size();
      ao_begin += count;
    }
    require(primitive_begin == shape.primitives && ao_begin == shape.nao,
            "one-electron topology differs from the native normalized basis");
    require(shape.atoms <= options.max_workspace_bytes / sizeof(double),
            "one-electron effective charges exceed the device budget");
    options.max_workspace_bytes -= shape.atoms * sizeof(double);
    check(integrals::cuda_one_electron_workspace(shape, shells, options, bytes));
    require(bytes <= options.max_workspace_bytes,
            "one-electron arena plus effective charges exceed the device budget");
    device_bytes = bytes + shape.atoms * sizeof(double);
  }
};
}  // namespace
std::size_t cuda_ks_one_electron_workspace(const core::System& system,
                                           const integrals::CudaOneElectronShape& shape,
                                           const integrals::CudaOneElectronOptions& options) {
  return Plan(system, shape, options).device_bytes;
}
struct CudaKsOneElectronResponse::Impl {
  methods::PreparedBatch& batch;
  std::size_t item;
  CudaKsFinalStateToken expected;
  CudaKsGridReference grid;
  CudaKsResidentFockBinding fock;
  integrals::CudaOneElectronResponse response;
  void* arena{};
  std::size_t device_bytes{};
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
CudaKsOneElectronResponse::CudaKsOneElectronResponse() = default;
CudaKsOneElectronResponse::~CudaKsOneElectronResponse() = default;

generativeqc_status CudaKsOneElectronResponse::prepare(
    methods::PreparedBatch& batch, std::size_t item, const CudaKsFinalStateToken& expected,
    const integrals::CudaOneElectronOptions& options, std::string& detail) {
  impl_.reset();
  try {
    auto candidate = std::make_unique<Impl>(batch, item, expected);
    auto status = candidate->grid.prepare(detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    status = methods::detail::dft_cuda_resident_fock_sources(batch, item, expected, candidate->fock,
                                                             detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    require(static_cast<bool>(candidate->fock), "native KS Fock source is empty");
    const auto shape = candidate->grid.shape();
    const auto lease = candidate->grid.reference();
    const auto& system = candidate->fock.primary->system();
    int device{-1};
    check(cudaGetDevice(&device));
    require(candidate->fock && candidate->fock.device_id == device && lease.device_id == device &&
                lease.stream == reinterpret_cast<cudaStream_t>(candidate->fock.stream) &&
                shape.nao == candidate->fock.nao && shape.atoms == system.atoms.size(),
            "one-electron source does not match its actual canonical native KS owner");
    candidate->stream = lease.stream;
    cudaStreamCaptureStatus capture{};
    check(cudaStreamIsCapturing(candidate->stream, &capture));
    require(capture == cudaStreamCaptureStatusNone, "one-electron preparation cannot capture");
    const integrals::CudaOneElectronShape integral_shape{shape.atoms, shape.primitives, shape.nao};
    const Plan plan(system, integral_shape, options);
    const auto bytes = plan.bytes;
    std::vector<double> charges;
    for (const auto& atom : system.atoms) charges.push_back(atom.ionic_charge());
    candidate->device_bytes = plan.device_bytes;
    check(cudaMalloc(&candidate->arena, candidate->device_bytes));
    auto* device_charges = reinterpret_cast<double*>(static_cast<char*>(candidate->arena) + bytes);
    // This is native immutable input packing from integer ionic charges, not a
    // host numerical integral or derivative contraction.
    const auto uploaded =
        cudaMemcpyAsync(device_charges, charges.data(), shape.atoms * sizeof(double),
                        cudaMemcpyHostToDevice, candidate->stream);
    const auto completed = cudaStreamSynchronize(candidate->stream);
    check(uploaded);
    check(completed);
    const integrals::CudaOneElectronReference reference{
        lease.basis,       device_charges,     device,
        candidate->stream, Impl::authenticate, candidate.get()};
    check(candidate->response.prepare(integral_shape, plan.shells, plan.options, reference,
                                      candidate->arena, bytes));
    impl_ = std::move(candidate);
    detail.clear();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "native one-electron source allocation failed";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
}
cudaError_t CudaKsOneElectronResponse::apply(const double* direction,
                                             const integrals::CudaOneElectronWeights& weights,
                                             const integrals::CudaOneElectronOutput& output,
                                             cudaStream_t stream) noexcept {
  return impl_ ? impl_->response.apply(direction, weights, output, stream)
               : cudaErrorInvalidResourceHandle;
}
CudaKsOneElectronWork CudaKsOneElectronResponse::work() const noexcept {
  return impl_ ? CudaKsOneElectronWork{impl_->device_bytes, impl_->response.work()}
               : CudaKsOneElectronWork{};
}
}  // namespace generativeqc::dft
