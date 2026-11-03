#include <cusolverDn.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/cuda_grid_api.hpp"
#include "dft/electronic_properties_cuda.hpp"
#include "methods/method.hpp"
#include "runtime/bounded_workspace.hpp"
#include "scf/cuda/one_electron_native_overlap.cuh"
#include "solver/cuda/symmetric_eigen_handles.hpp"
#include "tensor/cuda_runtime.cuh"
#include "tensor/cuda_square_linalg.hpp"

namespace generativeqc::dft {
namespace {
using generativeqc_tensor::blas_check;
using generativeqc_tensor::cuda_check;
using runtime::OwnedCudaBuffer;
using namespace scf::cuda_execution;

/** Drain asynchronous provider work before borrowed host workspace and caller
 * output buffers can leave scope, including allocation/submission failures. */
struct StreamCompletion {
  cudaStream_t stream;
  ~StreamCompletion() { (void)cudaStreamSynchronize(stream); }
};

unsigned blocks(std::size_t count) {
  return static_cast<unsigned>(std::min<std::size_t>((count + 127) / 128, 65535));
}

/** The packed AO weights already contain public Cartesian/spherical
 * normalization. Position raises only the ket polynomial: r=(r-B)+B.
 * Reuse the force path's shared overlap recurrence, which admits f->g;
 * renormalizing the raised component would change the operator. */
__global__ void position_overlap(const double* basis, std::size_t natom, std::size_t nprimitive,
                                 std::size_t nao, bool position, double* overlap, double* moments) {
  const auto* primitives = basis + 3 * natom;
  const auto* aos = primitives + 2 * nprimitive;
  const auto square = nao * nao;
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < square;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const auto row = index / nao, column = index % nao;
    if (row < column) continue;
    const auto* bra = aos + 16 * row;
    const auto* ket = aos + 16 * column;
    const auto* first_center = basis + 3 * static_cast<std::size_t>(bra[0]);
    const auto* second_center = basis + 3 * static_cast<std::size_t>(ket[0]);
    const Vec3<double> first{first_center[0], first_center[1], first_center[2]};
    const Vec3<double> second{second_center[0], second_center[1], second_center[2]};
    double value = 0, axes[3]{};
    for (std::size_t first_primitive = bra[1]; first_primitive < bra[1] + bra[2];
         ++first_primitive) {
      const double alpha = primitives[2 * first_primitive];
      for (std::size_t second_primitive = ket[1]; second_primitive < ket[1] + ket[2];
           ++second_primitive) {
        const double beta = primitives[2 * second_primitive];
        const double primitive_weight =
            primitives[2 * first_primitive + 1] * primitives[2 * second_primitive + 1];
        for (unsigned first_term = 0; first_term < static_cast<unsigned>(bra[3]); ++first_term) {
          const auto* bra_term = bra + 4 + 4 * first_term;
          const Angular first_angular{unsigned(bra_term[0]), unsigned(bra_term[1]),
                                      unsigned(bra_term[2])};
          for (unsigned second_term = 0; second_term < static_cast<unsigned>(ket[3]);
               ++second_term) {
            const auto* ket_term = ket + 4 + 4 * second_term;
            const Angular second_angular{unsigned(ket_term[0]), unsigned(ket_term[1]),
                                         unsigned(ket_term[2])};
            const double weight = primitive_weight * bra_term[3] * ket_term[3];
            const double base = primitive_overlap_cartesian_compact(alpha, first, first_angular,
                                                                    beta, second, second_angular);
            value += weight * base;
            if (position)
              for (int axis = 0; axis < 3; ++axis) {
                auto raised = second_angular;
                add_angular_axis(raised, axis, 1);
                axes[axis] += weight * (primitive_overlap_cartesian_compact(
                                            alpha, first, first_angular, beta, second, raised) +
                                        second_center[axis] * base);
              }
          }
        }
      }
    }
    overlap[index] = overlap[column * nao + row] = value;
    if (position)
      for (unsigned axis = 0; axis < 3; ++axis)
        moments[axis * square + index] = moments[axis * square + column * nao + row] = axes[axis];
  }
}

__global__ void sum_density(const double* source, std::size_t square, unsigned spins,
                            double* total) {
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < square;
       index += std::size_t(blockDim.x) * gridDim.x)
    total[index] = source[index] + (spins == 2 ? source[square + index] : 0.0);
}

__global__ void dipole_reduce(const double* density, const double* moments, const double* basis,
                              const double* nuclei, std::size_t natom, std::size_t square,
                              double* output) {
  __shared__ double partial[128];
  const unsigned axis = blockIdx.x;
  double value = 0;
  for (std::size_t index = threadIdx.x; index < square; index += blockDim.x)
    value -= density[index] * moments[axis * square + index];
  for (std::size_t atom = threadIdx.x; atom < natom; atom += blockDim.x)
    value += nuclei[atom] * basis[3 * atom + axis];
  partial[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = 64; stride; stride /= 2) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (!threadIdx.x) output[axis] = partial[0];
}

/** cuSOLVER overwrites column-major eigenvectors. Scale columns by sqrt(s)
 * before V sqrt(s) V^T; keep the original AO normalization and full rank. */
__global__ void scale_vectors(const double* vectors, const double* values, std::size_t nao,
                              double* scaled, int* error) {
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < nao * nao;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const double eigenvalue = values[index / nao];
    if (!(eigenvalue > 0) || !isfinite(eigenvalue)) atomicExch(error, 1);
    scaled[index] = vectors[index] * sqrt(eigenvalue);
  }
}

__global__ void atomic_charges(const double* basis, const double* nuclei, std::size_t natom,
                               std::size_t nprimitive, std::size_t nao, const double* density,
                               const double* overlap, const double* lowdin_matrix, double* output) {
  const auto* aos = basis + 3 * natom + 2 * nprimitive;
  for (std::size_t atom = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; atom < natom;
       atom += std::size_t(blockDim.x) * gridDim.x) {
    double mulliken = 0, lowdin = 0;
    for (std::size_t row = 0; row < nao; ++row) {
      if (static_cast<std::size_t>(aos[16 * row]) != atom) continue;
      lowdin += lowdin_matrix[row * nao + row];
      for (std::size_t column = 0; column < nao; ++column)
        mulliken += density[row * nao + column] * overlap[row * nao + column];
    }
    output[atom] = nuclei[atom] - mulliken;
    output[natom + atom] = nuclei[atom] - lowdin;
  }
}

__global__ void combine_features(const double* features, std::size_t count, bool spin,
                                 double* output, int* error) {
  for (std::size_t point = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; point < count;
       point += std::size_t(blockDim.x) * gridDim.x) {
    const double value = features[point] + (spin ? -1.0 : 1.0) * features[5 * count + point];
    if (!isfinite(value)) atomicExch(error, 1);
    output[point] = value;
  }
}

void solver_check(std::uint32_t status) {
  if (status == CUSOLVER_STATUS_ALLOC_FAILED) throw std::bad_alloc();
  if (status != 0)
    throw methods::MethodError(GENERATIVEQC_STATUS_CUDA_ERROR,
                               "electronic-property cuSOLVER status " + std::to_string(status));
}

void grid_check(int status, const char* error) {
  if (status != 0) throw methods::MethodError(static_cast<generativeqc_status>(status), error);
}

class PropertyState final : public CudaElectronicProperties {
 public:
  PropertyState(const core::System& system, int device, std::size_t capacity)
      : basis_(system), device_(device), capacity_(capacity) {}

  void initialize(const core::System& system, const double* density, std::size_t count) {
    const auto square = runtime::size_mul(basis_.nao, basis_.nao);
    if (count != square && count != runtime::size_mul(2, square))
      throw std::invalid_argument("CUDA property density size mismatch");
    spins_ = count == square ? 1 : 2;
    for (std::size_t index = 0; index < count; ++index) {
      const double value = density[index];
      const double transpose =
          density[(index / square * basis_.nao + index % basis_.nao) * basis_.nao +
                  index / basis_.nao % basis_.nao];
      if (!std::isfinite(value) ||
          std::abs(value - transpose) >
              1e-12 + 1e-10 * std::max(std::abs(value), std::abs(transpose)))
        throw std::invalid_argument("CUDA property density must be finite and symmetric");
    }
    runtime::CudaDeviceScope guard(device_, cuda_check);
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device_));
    major_ = properties.major;
    minor_ = properties.minor;
    context_.prepare(device_, major_, minor_, 256 + (4U << 20), 0, 256, 4U << 20, 96U << 20, true);
    packed_.allocate(device_, basis_.packed.size());
    density_.allocate(device_, count);
    total_.allocate(device_, square);
    nuclei_.allocate(device_, basis_.natom);
    std::vector<double> nuclei;
    for (const auto& atom : system.atoms) nuclei.push_back(atom.atomic_number);
    StreamCompletion completion{context_.stream};
    upload(packed_.get(), basis_.packed.data(), basis_.packed.size());
    upload(density_.get(), density, count);
    upload(nuclei_.get(), nuclei.data(), nuclei.size());
    sum_density<<<blocks(square), 128, 0, context_.stream>>>(density_.get(), square, spins_,
                                                             total_.get());
    cuda_check(cudaGetLastError());
    cuda_check(cudaStreamSynchronize(context_.stream));
  }

  ~PropertyState() override {
    if (grid_) {
      runtime::CudaDeviceScope guard(device_, std::nothrow);
      grid_cuda_destroy_v1(grid_);
    }
  }

  std::size_t atoms() const noexcept override { return basis_.natom; }
  int device() const noexcept override { return device_; }

  void analyze(double* dipole, double* mulliken, double* lowdin) override {
    runtime::CudaDeviceScope guard(device_, cuda_check);
    const auto square = runtime::size_mul(basis_.nao, basis_.nao);
    OwnedCudaBuffer<double> overlap(device_, square), moments,
        result(device_, 3 + 2 * basis_.natom);
    if (dipole) moments.allocate(device_, runtime::size_mul(3, square));
    StreamCompletion completion{context_.stream};
    position_overlap<<<blocks(square), 128, 0, context_.stream>>>(
        packed_.get(), basis_.natom, basis_.nprimitive, basis_.nao, dipole != nullptr,
        overlap.get(), moments.get());
    cuda_check(cudaGetLastError());
    if (dipole) {
      dipole_reduce<<<3, 128, 0, context_.stream>>>(total_.get(), moments.get(), packed_.get(),
                                                    nuclei_.get(), basis_.natom, square,
                                                    result.get());
      cuda_check(cudaGetLastError());
    }
    if (mulliken) {
      using namespace solver::cuda;
      PreparedSymmetricEigenHandles solver;
      solver_check(solver.create());
      solver_check(solver.bind_stream(context_.stream));
      solver_check(solver.create_parameters());
      OwnedCudaBuffer<double> vectors(device_, square), values(device_, basis_.nao),
          scaled(device_, square), root(device_, square), product(device_, square),
          lowdin_matrix(device_, square);
      OwnedCudaBuffer<int> info(device_, 1);
      cuda_check(cudaMemcpyAsync(vectors.get(), overlap.get(), square * sizeof(double),
                                 cudaMemcpyDeviceToDevice, context_.stream));
      auto resources = solver.view();
      const SymmetricEigenProblem problem{static_cast<std::int64_t>(basis_.nao), 1,
                                          Eigenvectors::values_and_vectors};
      SymmetricEigenWorkspace sizes{};
      solver_check(query_symmetric_eigen(resources, SymmetricEigenFamily::xsyevd, problem,
                                         vectors.get(), values.get(), sizes));
      OwnedCudaBuffer<unsigned char> workspace;
      if (sizes.device_bytes) workspace.allocate(device_, sizes.device_bytes);
      std::vector<unsigned char> host_workspace(sizes.host_bytes);
      StreamCompletion completion{context_.stream};
      resources.device_workspace = workspace.get();
      resources.device_workspace_bytes = sizes.device_bytes;
      resources.host_workspace = host_workspace.data();
      resources.host_workspace_bytes = sizes.host_bytes;
      solver_check(launch_symmetric_eigen(resources, SymmetricEigenFamily::xsyevd, problem,
                                          vectors.get(), values.get(), info.get(), 0));
      int solver_info = 0;
      cuda_check(cudaMemcpyAsync(&solver_info, info.get(), sizeof(int), cudaMemcpyDeviceToHost,
                                 context_.stream));
      cuda_check(cudaStreamSynchronize(context_.stream));
      if (solver_info) throw std::runtime_error("Lowdin overlap eigensolver did not converge");
      cuda_check(cudaMemsetAsync(context_.error, 0, sizeof(int), context_.stream));
      scale_vectors<<<blocks(square), 128, 0, context_.stream>>>(
          vectors.get(), values.get(), basis_.nao, scaled.get(), context_.error);
      cuda_check(cudaGetLastError());
      int failure = 0;
      cuda_check(cudaMemcpyAsync(&failure, context_.error, sizeof(int), cudaMemcpyDeviceToHost,
                                 context_.stream));
      cuda_check(cudaStreamSynchronize(context_.stream));
      if (failure) throw std::runtime_error("Lowdin charges require a positive-definite overlap");
      const int dimension = static_cast<int>(basis_.nao);
      blas_check(tensor::cuda::square_panel_product(context_.handle, dimension, dimension,
                                                    scaled.get(), false, vectors.get(), true,
                                                    root.get()));
      blas_check(tensor::cuda::square_gemm(context_.handle, false, dimension, 1, root.get(),
                                           total_.get(), product.get()));
      blas_check(tensor::cuda::square_gemm(context_.handle, false, dimension, 1, product.get(),
                                           root.get(), lowdin_matrix.get()));
      atomic_charges<<<blocks(basis_.natom), 128, 0, context_.stream>>>(
          packed_.get(), nuclei_.get(), basis_.natom, basis_.nprimitive, basis_.nao, total_.get(),
          overlap.get(), lowdin_matrix.get(), result.get() + 3);
      cuda_check(cudaGetLastError());
      download(mulliken, result.get() + 3, basis_.natom);
      download(lowdin, result.get() + 3 + basis_.natom, basis_.natom);
      cuda_check(cudaStreamSynchronize(context_.stream));
    }
    if (dipole) download(dipole, result.get(), 3);
    cuda_check(cudaStreamSynchronize(context_.stream));
    const auto finite = [](const double* values, std::size_t count) {
      for (std::size_t index = 0; index < count; ++index)
        if (!std::isfinite(values[index]))
          throw std::runtime_error("nonfinite CUDA property output");
    };
    if (dipole) finite(dipole, 3);
    if (mulliken) {
      finite(mulliken, basis_.natom);
      finite(lowdin, basis_.natom);
    }
  }

  void evaluate(const double* points, std::size_t count, bool spin, double* output) override {
    const auto coordinates = runtime::size_mul(3, count);
    for (std::size_t index = 0; index < coordinates; ++index)
      if (!std::isfinite(points[index]))
        throw std::invalid_argument("nonfinite CUDA density point");
    if (!count) return;
    runtime::CudaDeviceScope guard(device_, cuda_check);
    char error[1024]{};
    if (!grid_) {
      if (!sampled_) sampled_.allocate(device_, capacity_);
      const std::size_t dimensions[]{basis_.natom, basis_.nprimitive, basis_.nao};
      void* candidate = nullptr;
      grid_check(
          grid_cuda_create_v3(device_, major_, minor_, dimensions, basis_.packed.data(), capacity_,
                              0, 0, 0, nullptr, 0, 1, &candidate, error, sizeof(error)),
          error);
      const auto status = grid_cuda_density_device_v1(
          candidate, density_.get(),
          spins_ == 2 ? density_.get() + basis_.nao * basis_.nao : nullptr, basis_.nao * basis_.nao,
          spins_, context_.stream, error, sizeof(error));
      if (status) grid_cuda_destroy_v1(candidate);
      grid_check(status, error);
      grid_ = candidate;
    }
    GridBasisView basis_view{};
    grid_check(grid_cuda_basis_v1(grid_, &basis_view, error, sizeof(error)), error);
    StreamCompletion completion{basis_view.stream};
    for (std::size_t begin = 0; begin < count; begin += std::min(capacity_, count - begin)) {
      const auto tile = std::min(capacity_, count - begin);
      grid_check(grid_cuda_run_selected_deferred_v1(grid_, points + 3 * begin, tile, 1, nullptr, 0,
                                                    nullptr, nullptr, error, sizeof(error)),
                 error);
      GridTaskView view{};
      grid_check(grid_cuda_view_v1(grid_, &view, error, sizeof(error)), error);
      combine_features<<<blocks(tile), 128, 0, view.stream>>>(view.features, tile, spin,
                                                              sampled_.get(), view.error);
      cuda_check(cudaGetLastError());
      int failure = 0;
      cuda_check(cudaMemcpyAsync(output + begin, sampled_.get(), tile * sizeof(double),
                                 cudaMemcpyDeviceToHost, view.stream));
      cuda_check(
          cudaMemcpyAsync(&failure, view.error, sizeof(int), cudaMemcpyDeviceToHost, view.stream));
      cuda_check(cudaStreamSynchronize(view.stream));
      if (failure) throw std::runtime_error("nonfinite CUDA AO/density output");
    }
  }

 private:
  void upload(double* target, const double* source, std::size_t count) {
    cuda_check(cudaMemcpyAsync(target, source, count * sizeof(double), cudaMemcpyHostToDevice,
                               context_.stream));
  }
  void download(double* target, const double* source, std::size_t count) {
    cuda_check(cudaMemcpyAsync(target, source, count * sizeof(double), cudaMemcpyDeviceToHost,
                               context_.stream));
  }
  AoBasis basis_;
  int device_{}, major_{}, minor_{};
  unsigned spins_{};
  std::size_t capacity_{};
  generativeqc_tensor::Context context_;
  OwnedCudaBuffer<double> packed_, density_, total_, nuclei_, sampled_;
  void* grid_{};
};
}  // namespace

std::unique_ptr<CudaElectronicProperties> make_cuda_electronic_properties(
    const core::System& system, int device, const double* density, std::size_t count,
    std::size_t point_capacity) {
  if (!point_capacity || point_capacity > INT_MAX)
    throw std::invalid_argument("CUDA density point capacity must be in [1, INT_MAX]");
  auto state = std::make_unique<PropertyState>(system, device, point_capacity);
  state->initialize(system, density, count);
  return state;
}
}  // namespace generativeqc::dft
