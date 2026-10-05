// Isolated scheduling probe. Mathematics is supplied by the compiler emitter.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cuda/atomic>
#include <stdexcept>
#include <vector>

using namespace generativeqc_grid_adjoint;
using namespace generativeqc_grid_phased;
using Geometry = PreparedCenterGeometry<decltype(&local_ratio_prepared)>;

__device__ bool failed(int* error) {
  return cuda::atomic_ref<int, cuda::thread_scope_device>(*error).load(
             cuda::memory_order_relaxed) != 0;
}

static void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class Value>
struct Device {
  Value* data = nullptr;
  explicit Device(size_t count) {
    check(cudaMalloc(&data, std::max(size_t{1}, count) * sizeof(Value)));
  }
  ~Device() { cudaFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};
struct Event {
  cudaEvent_t value;
  Event() { check(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
};
struct Team {
  int* valid;
  __device__ size_t rank() const { return threadIdx.x; }
  __device__ size_t size() const { return blockDim.x; }
  __device__ void sync() const { __syncthreads(); }
  __device__ bool all(bool vote) const {
    if (!vote) atomicExch(valid, 0);
    sync();
    const bool result = *valid;
    sync();
    return result;
  }
};

__global__ void prepare(const double* centers, size_t atoms, CenterPair* pairs, int* error) {
  if (!prepare_center_geometry(centers, atoms, 1e-12, pairs, local_norm, local_ratio_geometry)) {
    *error = 1;
    printf("center preparation failed: atoms=%llu first=(%.17g,%.17g,%.17g)\n",
           static_cast<unsigned long long>(atoms), centers[0], centers[1], centers[2]);
    for (size_t first = 0; first < atoms; ++first)
      for (size_t second = 0; second < first; ++second) {
        bool valid = true;
        const auto values = distance(centers + 3 * first, centers + 3 * second, local_norm, valid);
        if (!valid || values[0] <= 1e-12)
          printf("invalid pair %llu %llu: %.17g %.17g %.17g %.17g\n",
                 static_cast<unsigned long long>(first), static_cast<unsigned long long>(second),
                 values[0], values[1], values[2], values[3]);
      }
  }
}
__global__ void cooperative(const double* points, const double* centers, size_t atoms,
                            const int64_t* owners, const double* seeds, const CenterPair* pairs,
                            double* scratch, double* output, int* error) {
  const size_t point = blockIdx.x;
  double* work = scratch + point * 9 * atoms;
  double* gradient = output + point * 3 * atoms;
  extern __shared__ double storage[];
  auto* states = reinterpret_cast<PointPair*>(storage);
  __shared__ int valid;
  if (threadIdx.x == 0) valid = 1;
  for (size_t index = threadIdx.x; index < 3 * atoms; index += blockDim.x) gradient[index] = 0;
  __syncthreads();
  auto* distances = reinterpret_cast<std::array<double, 4>*>(work + 5 * atoms);
  auto* zeros = reinterpret_cast<size_t*>(work + 4 * atoms);
  const bool success =
      atoms <= 32 ? contract_point_cooperative(points + 3 * point, centers, atoms, owners[point],
                                               seeds[point], gradient, work, work + atoms,
                                               work + 2 * atoms, work + 3 * atoms, zeros, distances,
                                               states, Team{&valid}, local_norm, local_ratio,
                                               local_log, local_becke, pairs, local_ratio_prepared)
                  : contract_point_tiled_cooperative(
                        points + 3 * point, centers, atoms, owners[point], seeds[point], gradient,
                        work, work + atoms, work + 2 * atoms, work + 3 * atoms, zeros, distances,
                        states, 4, Team{&valid}, local_norm, local_ratio, local_log, local_becke,
                        pairs, local_ratio_prepared);
  if (!success && threadIdx.x == 0) atomicExch(error, 1);
}

template <int Phase>
__global__ void atom_phase(Workspace work, const double* points, const double* centers,
                           const int64_t* owners, int* error) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t atom = blockIdx.y;
  if (point >= work.points || failed(error)) return;
  bool valid = true;
  if constexpr (Phase == 0) valid = distance_phase(work, point, atom, points, centers, local_norm);
  if constexpr (Phase == 1) atom_logs_phase(work, point, atom);
  if constexpr (Phase == 2) atom_gather_phase(work, point, atom);
  if constexpr (Phase == 3) valid = point_motion_phase(work, point, atom, owners[point]);
  if (!valid) atomicExch(error, 1);
}
template <bool Reverse>
__global__ void pair_phase(Workspace work, const uint2* indices, const CenterPair* centers,
                           int* error) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= work.points || failed(error)) return;
  const auto pair = indices[blockIdx.y];
  const Geometry geometry{centers, local_ratio_prepared};
  bool valid;
  if constexpr (Reverse)
    valid = pair_reverse_phase(work, point, pair.x, pair.y, geometry, local_log);
  else
    valid = pair_primal_phase(work, point, pair.x, pair.y, geometry, local_log, local_becke);
  if (!valid) atomicExch(error, 1);
}
__global__ void normalize(Workspace work, const int64_t* owners, const double* seeds, int* error) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= work.points || failed(error)) return;
  if (!point_normalize_phase(work, point, owners[point], seeds[point], local_ratio))
    atomicExch(error, 1);
}
__global__ void publish(Workspace work, double* output, int* error) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t atom = blockIdx.y;
  if (point >= work.points || failed(error)) return;
  for (size_t axis = 0; axis < 3; ++axis)
    output[(point * work.atoms + atom) * 3 + axis] = work.field(8 + axis, point)[atom];
}

// Allocation, input transfer, center preparation, and final readback are outside
// the event interval. Each interval includes every sampled tile and publication.
// This deliberately is NOT an XC geometry stage or an energy/force endpoint.
extern "C" int probe(size_t atoms, size_t point_count, size_t tile_points,
                     const double* host_centers, const double* host_points,
                     const int64_t* host_owners, const double* host_seeds, double* old_output,
                     double* new_output, double* milliseconds) {
  try {
    if (!atoms || atoms > 128 || !tile_points || !point_count) return -2;
    const size_t pair_count = atoms * (atoms - 1) / 2;
    Device<double> centers(3 * atoms), points(3 * point_count), seeds(point_count);
    Device<int64_t> owners(point_count);
    Device<CenterPair> geometry(pair_count);
    Device<uint2> indices(pair_count);
    Device<int> error(1);
    Device<double> old_scratch(9 * atoms * tile_points), old_result(3 * atoms * point_count);
    Device<double> pair_storage(4 * pair_count * tile_points), fields(11 * atoms * tile_points);
    Device<size_t> zeros(atoms * tile_points);
    Device<double> maximum(tile_points), new_result(3 * atoms * point_count);
    check(
        cudaMemcpy(centers.data, host_centers, 3 * atoms * sizeof(double), cudaMemcpyHostToDevice));
    check(cudaMemcpy(points.data, host_points, 3 * point_count * sizeof(double),
                     cudaMemcpyHostToDevice));
    check(cudaMemcpy(owners.data, host_owners, point_count * sizeof(int64_t),
                     cudaMemcpyHostToDevice));
    check(cudaMemcpy(seeds.data, host_seeds, point_count * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<uint2> descriptors;
    for (size_t first = 0; first < atoms; ++first)
      for (size_t second = 0; second < first; ++second)
        descriptors.push_back(make_uint2(first, second));
    check(cudaMemcpy(indices.data, descriptors.data(), pair_count * sizeof(uint2),
                     cudaMemcpyHostToDevice));
    check(cudaMemset(error.data, 0, sizeof(int)));
    prepare<<<1, 1>>>(centers.data, atoms, geometry.data, error.data);
    int status = 0;
    check(cudaMemcpy(&status, error.data, sizeof(int), cudaMemcpyDeviceToHost));
    if (status) return -3;
    const size_t rows = std::min(size_t{4}, atoms - 1);
    const size_t shared_pairs = atoms <= 32 ? pair_count : rows * (2 * atoms - rows - 1) / 2;
    auto launch = [&](bool phased) {
      for (size_t begin = 0; begin < point_count; begin += tile_points) {
        const size_t count = std::min(tile_points, point_count - begin);
        if (!phased) {
          cooperative<<<count, 32, shared_pairs * sizeof(PointPair)>>>(
              points.data + 3 * begin, centers.data, atoms, owners.data + begin, seeds.data + begin,
              geometry.data, old_scratch.data, old_result.data + 3 * atoms * begin, error.data);
        } else {
          Workspace work{atoms, count, pair_storage.data, fields.data, zeros.data, maximum.data};
          const dim3 atom_blocks((count + 127) / 128, atoms);
          const dim3 pair_blocks((count + 127) / 128, pair_count);
          atom_phase<0><<<atom_blocks, 128>>>(work, points.data + 3 * begin, centers.data,
                                              owners.data + begin, error.data);
          if (pair_count)
            pair_phase<false><<<pair_blocks, 128>>>(work, indices.data, geometry.data, error.data);
          atom_phase<1><<<atom_blocks, 128>>>(work, nullptr, nullptr, nullptr, error.data);
          normalize<<<(count + 127) / 128, 128>>>(work, owners.data + begin, seeds.data + begin,
                                                  error.data);
          if (pair_count)
            pair_phase<true><<<pair_blocks, 128>>>(work, indices.data, geometry.data, error.data);
          atom_phase<2><<<atom_blocks, 128>>>(work, nullptr, nullptr, nullptr, error.data);
          atom_phase<3>
              <<<atom_blocks, 128>>>(work, nullptr, nullptr, owners.data + begin, error.data);
          publish<<<atom_blocks, 128>>>(work, new_result.data + 3 * atoms * begin, error.data);
        }
      }
      check(cudaGetLastError());
    };
    launch(false);
    launch(true);
    check(cudaDeviceSynchronize());
    Event start, stop;
    for (size_t repeat = 0; repeat < 8; ++repeat) {
      const bool phased = repeat % 4 == 1 || repeat % 4 == 2;
      check(cudaEventRecord(start.value));
      launch(phased);
      check(cudaEventRecord(stop.value));
      check(cudaEventSynchronize(stop.value));
      float elapsed = 0;
      check(cudaEventElapsedTime(&elapsed, start.value, stop.value));
      milliseconds[repeat] = elapsed;
    }
    check(cudaMemcpy(&status, error.data, sizeof(int), cudaMemcpyDeviceToHost));
    if (status) return -4;
    check(cudaMemcpy(old_output, old_result.data, 3 * atoms * point_count * sizeof(double),
                     cudaMemcpyDeviceToHost));
    check(cudaMemcpy(new_output, new_result.data, 3 * atoms * point_count * sizeof(double),
                     cudaMemcpyDeviceToHost));
    return 0;
  } catch (const std::exception& failure) {
    std::fprintf(stderr, "%s\n", failure.what());
    return -1;
  }
}
