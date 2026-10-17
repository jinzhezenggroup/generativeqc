#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "dft/hessian_aot.hpp"
#include "integrals/cuda_one_electron_response.hpp"
#include "response/nuclear_integral_response.cuh"

namespace generativeqc::integrals {
namespace {
namespace aot = dft::hessian_aot;
namespace ad = generativeqc_nuclear_integral_ad;

struct DeviceJob {
  CudaOneElectronShell first, second;
  long long atoms[3]{};
  unsigned powers[16][2][3]{};
  unsigned components{}, centers{}, family{};
};
struct Job {
  const aot::Program* program{};
  DeviceJob device;
  std::size_t radials{};
};
struct ParallelTask {
  std::size_t job{}, begin{}, count{}, records{}, raw{}, projected{}, candidate{};
};
struct Layout {
  std::size_t bytes{}, matrix{}, basis_elements{}, descriptors{}, shells{}, records{}, raw{},
      projected{}, candidates{}, status{}, radial_records{};
  std::uint64_t components{}, projections{};
  std::size_t parallel_begin{}, parallel_descriptors{}, parallel_direction{}, parallel_weights{},
      parallel_candidates{}, parallel_candidate_bytes{};
  std::vector<ParallelTask> tasks;
};

bool multiply(std::size_t first, std::size_t second, std::size_t& result) {
  if (second && first > SIZE_MAX / second) return false;
  result = first * second;
  return true;
}
bool add(std::size_t& value, std::size_t term) {
  if (term > SIZE_MAX - value) return false;
  value += term;
  return true;
}
bool append(Layout& plan, std::size_t count, std::size_t width, std::size_t& offset) {
  std::size_t bytes{};
  if (!multiply(count, width, bytes) || bytes > SIZE_MAX - 255 ||
      plan.bytes > SIZE_MAX - bytes - 255)
    return false;
  offset = plan.bytes;
  plan.bytes = (plan.bytes + bytes + 255) / 256 * 256;
  return true;
}
unsigned cartesian_count(unsigned angular) { return (angular + 1) * (angular + 2) / 2; }
std::array<unsigned, 3> powers(unsigned angular, unsigned component) {
  for (unsigned first = angular + 1; first-- > 0;)
    for (unsigned second = angular - first + 1; second-- > 0;)
      if (!component--) return {first, second, angular - first - second};
  return {};
}

// This is integer package admission, not numerical integral construction. Every
// ordered shell pair is explicit so no bra/ket or coincident-center convention
// is inferred by the native scheduler.
cudaError_t schedule(const CudaOneElectronShape& shape,
                     std::span<const CudaOneElectronShell> shells,
                     const CudaOneElectronOptions& options, Layout& plan, std::vector<Job>& jobs) {
  if (!shape.atoms || !shape.primitives || !shape.nao || shells.empty() ||
      shape.atoms > static_cast<std::size_t>(std::numeric_limits<long long>::max()) / 3 ||
      !options.radial_tile || options.radial_tile > 64 || !options.max_jobs ||
      options.parallel_streams > 32 || !std::isfinite(options.symmetry_absolute_tolerance) ||
      options.symmetry_absolute_tolerance < 0 ||
      !std::isfinite(options.symmetry_relative_tolerance) ||
      options.symmetry_relative_tolerance < 0 || !multiply(shape.nao, shape.nao, plan.matrix) ||
      !multiply(shape.atoms, 3, plan.basis_elements))
    return cudaErrorInvalidValue;
  std::size_t count{};
  if (!multiply(shape.primitives, 2, count) || !add(plan.basis_elements, count) ||
      !multiply(shape.nao, 16, count) || !add(plan.basis_elements, count) ||
      plan.basis_elements > SIZE_MAX / sizeof(double))
    return cudaErrorInvalidValue;
  std::size_t primitive_begin{}, ao_begin{};
  std::array<bool, 4> needed{};
  for (const auto& shell : shells) {
    if (shell.atom >= shape.atoms || shell.angular > 3 || !shell.primitive_count ||
        shell.primitive_begin != primitive_begin || shell.ao_begin != ao_begin ||
        (shell.ao_count != cartesian_count(shell.angular) &&
         shell.ao_count != 2 * shell.angular + 1) ||
        !add(primitive_begin, shell.primitive_count) || !add(ao_begin, shell.ao_count) ||
        primitive_begin > shape.primitives || ao_begin > shape.nao)
      return cudaErrorInvalidValue;
    needed[shell.angular] = true;
  }
  if (primitive_begin != shape.primitives || ao_begin != shape.nao) return cudaErrorInvalidValue;

  constexpr std::array families{"overlap", "kinetic", "nuclear_attraction"};
  std::array<std::vector<const aot::Program*>, 48> closure;
  for (unsigned family = 0; family < 3; ++family)
    for (unsigned first = 0; first < 4; ++first)
      for (unsigned second = 0; second < 4; ++second) {
        if (!needed[first] || !needed[second]) continue;
        std::array<bool, 100> covered{};
        const auto components = cartesian_count(first) * cartesian_count(second);
        auto& selected = closure[16 * family + 4 * first + second];
        for (const auto& program : aot::programs()) {
          const auto& signature = program.signature;
          if (program.kind != aot::Kind::integral || signature.family != families[family] ||
              signature.angular.size() != 2 || signature.angular[0] != first ||
              signature.angular[1] != second || signature.radial != "full_range")
            continue;
          const auto centers = family == 2 ? 3U : 2U;
          if (signature.omega != 0 || !program.integral || program.centers != centers ||
              program.record_stride != 3 + 3 * centers || program.outputs != 1 + 6 * centers ||
              program.components.empty() || program.components.size() > 16)
            return cudaErrorNotSupported;
          for (const auto component : program.components) {
            if (component >= components || covered[component]) return cudaErrorNotSupported;
            covered[component] = true;
          }
          selected.push_back(&program);
        }
        for (unsigned component = 0; component < components; ++component)
          if (!covered[component]) return cudaErrorNotSupported;
      }
  for (unsigned family = 0; family < 3; ++family)
    for (const auto& first : shells)
      for (const auto& second : shells) {
        std::size_t radials{};
        if (!multiply(first.primitive_count, second.primitive_count, radials))
          return cudaErrorInvalidValue;
        for (const auto* program : closure[16 * family + 4 * first.angular + second.angular])
          for (std::size_t atom = 0; atom < (family == 2 ? shape.atoms : 1); ++atom) {
            if (jobs.size() >= options.max_jobs) return cudaErrorInvalidValue;
            Job job{program, {}, radials};
            job.device.first = first;
            job.device.second = second;
            job.device.atoms[0] = first.atom;
            job.device.atoms[1] = second.atom;
            job.device.atoms[2] = atom;
            job.device.components = program->components.size();
            job.device.centers = program->centers;
            job.device.family = family;
            for (unsigned component = 0; component < job.device.components; ++component) {
              const auto flat = program->components[component];
              const auto width = cartesian_count(second.angular);
              const auto left = powers(first.angular, flat / width);
              const auto right = powers(second.angular, flat % width);
              for (unsigned axis = 0; axis < 3; ++axis) {
                job.device.powers[component][0][axis] = left[axis];
                job.device.powers[component][1][axis] = right[axis];
              }
            }
            if (!multiply(radials, program->components.size(), count) ||
                !add(plan.components, count) ||
                !multiply(first.ao_count * second.ao_count, program->components.size(), count) ||
                !multiply(count, (radials - 1) / options.radial_tile + 1, count) ||
                !multiply(count, 9, count) || !add(plan.projections, count) ||
                !add(plan.radial_records, radials) ||
                plan.components > options.max_primitive_component_evaluations ||
                plan.projections > options.max_projection_terms)
              return cudaErrorInvalidValue;
            jobs.push_back(job);
          }
      }
  std::size_t candidates{};
  if (!multiply(plan.matrix, 4, candidates) || !multiply(shape.atoms, 6, count) ||
      !add(candidates, count) || !append(plan, jobs.size(), sizeof(DeviceJob), plan.descriptors) ||
      !append(plan, shells.size(), sizeof(CudaOneElectronShell), plan.shells) ||
      !append(plan, options.radial_tile * 12, sizeof(double), plan.records) ||
      !append(plan, 16 * 19, sizeof(double), plan.raw) ||
      !append(plan, 2 * 100 * 9, sizeof(double), plan.projected) ||
      !append(plan, candidates, sizeof(double), plan.candidates) ||
      !append(plan, 1, sizeof(int), plan.status) || plan.bytes > options.max_workspace_bytes)
    return cudaErrorInvalidValue;
  if (!options.parallel_streams || !options.max_parallel_workspace_bytes ||
      !options.max_parallel_tasks)
    return cudaSuccess;
  Layout parallel = plan;
  parallel.parallel_begin = plan.bytes;
  for (std::size_t index = 0; index < jobs.size(); ++index) {
    const auto& job = jobs[index];
    for (std::size_t begin = 0; begin < job.radials;) {
      if (parallel.tasks.size() == options.max_parallel_tasks) return cudaSuccess;
      ParallelTask task{index, begin, std::min(options.radial_tile, job.radials - begin)};
      if (!append(parallel, task.count * job.program->record_stride, sizeof(double),
                  task.records) ||
          !append(parallel, job.program->components.size() * job.program->outputs, sizeof(double),
                  task.raw) ||
          !append(
              parallel,
              2 * job.device.first.ao_count * job.device.second.ao_count * 3 * job.device.centers,
              sizeof(double), task.projected))
        return cudaSuccess;
      parallel.tasks.push_back(task);
      begin += task.count;
      if (parallel.bytes - plan.bytes > options.max_parallel_workspace_bytes ||
          parallel.bytes > options.max_workspace_bytes)
        return cudaSuccess;
    }
  }
  if (!multiply(parallel.tasks.size(), 4 * plan.matrix, count) ||
      !multiply(count, sizeof(double), parallel.parallel_candidate_bytes) ||
      !append(parallel, count, sizeof(double), parallel.parallel_candidates) ||
      !append(parallel, parallel.tasks.size(), sizeof(ParallelTask),
              parallel.parallel_descriptors) ||
      !append(parallel, 3 * shape.atoms, sizeof(double), parallel.parallel_direction) ||
      !append(parallel, 4 * plan.matrix, sizeof(double), parallel.parallel_weights) ||
      parallel.bytes - plan.bytes > options.max_parallel_workspace_bytes ||
      parallel.bytes > options.max_workspace_bytes)
    return cudaSuccess;
  for (std::size_t index = 0; index < parallel.tasks.size(); ++index)
    parallel.tasks[index].candidate =
        parallel.parallel_candidates + index * 4 * plan.matrix * sizeof(double);
  plan = std::move(parallel);
  return cudaSuccess;
}

unsigned blocks(std::size_t count) { return std::min<std::size_t>((count - 1) / 128 + 1, 65535); }
bool disjoint(const void* first, std::size_t first_bytes, const void* second,
              std::size_t second_bytes) {
  if (!first || !second) return false;
  const auto left = reinterpret_cast<std::uintptr_t>(first);
  const auto right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left >= first_bytes : left - right >= second_bytes;
}
cudaError_t device_buffer(const void* pointer, int device) {
  if (!pointer || reinterpret_cast<std::uintptr_t>(pointer) % alignof(double))
    return cudaErrorInvalidValue;
  cudaPointerAttributes attributes{};
  const auto error = cudaPointerGetAttributes(&attributes, pointer);
  if (error != cudaSuccess) return error;
  return attributes.type == cudaMemoryTypeDevice && attributes.device == device
             ? cudaSuccess
             : cudaErrorInvalidValue;
}
cudaError_t drain(cudaError_t error, cudaStream_t stream) {
  const auto completed = cudaStreamSynchronize(stream);
  return error == cudaSuccess ? completed : error;
}
cudaError_t gate(int* status, cudaStream_t stream) {
  int failed{};
  auto error = cudaMemcpyAsync(&failed, status, sizeof(failed), cudaMemcpyDeviceToHost, stream);
  error = drain(error, stream);
  return error == cudaSuccess && failed ? cudaErrorInvalidValue : error;
}
__global__ void finite_values(const double* values, std::size_t count, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!isfinite(values[index])) atomicExch(status, 1);
}
__global__ void validate_basis(CudaOneElectronShape shape, const CudaOneElectronShell* shells,
                               std::size_t count, const double* basis, const double* charges,
                               int* status) {
  const auto* primitives = basis + 3 * shape.atoms;
  const auto* records = primitives + 2 * shape.primitives;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < max(max(count, shape.primitives), shape.atoms);
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    if (index < shape.atoms && !(charges[index] >= 0)) atomicExch(status, 1);
    if (index < shape.primitives && !(primitives[2 * index] > 0)) atomicExch(status, 1);
    if (index >= count) continue;
    const auto shell = shells[index];
    for (std::size_t orbital = 0; orbital < shell.ao_count; ++orbital) {
      const auto* record = records + 16 * (shell.ao_begin + orbital);
      if (record[0] != shell.atom || record[1] != shell.primitive_begin ||
          record[2] != shell.primitive_count || !isfinite(record[3]) || record[3] < 1 ||
          record[3] > 3 || floor(record[3]) != record[3]) {
        atomicExch(status, 1);
        continue;
      }
      for (unsigned term = 0; term < static_cast<unsigned>(record[3]); ++term) {
        double degree{};
        for (unsigned axis = 0; axis < 3; ++axis) {
          const auto power = record[4 + 4 * term + axis];
          if (!isfinite(power) || power < 0 || power > 3 || floor(power) != power)
            atomicExch(status, 1);
          degree += power;
        }
        if (degree != shell.angular) atomicExch(status, 1);
      }
    }
  }
}
__global__ void symmetric(const double* values, std::size_t nao, double absolute, double relative,
                          int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nao * nao;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto value = values[index], other = values[(index % nao) * nao + index / nao];
    if (!isfinite(value) || !isfinite(other) ||
        fabs(value - other) > absolute + relative * fmax(fabs(value), fabs(other)))
      atomicExch(status, 1);
  }
}
__global__ void build_records(CudaOneElectronShape shape, const DeviceJob* job, const double* basis,
                              const double* charges, std::size_t begin, std::size_t count,
                              double* records, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto* primitives = basis + 3 * shape.atoms;
  const auto stride = 3 + 3 * job->centers;
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    const auto radial = begin + index;
    const auto first = job->first.primitive_begin + radial / job->second.primitive_count;
    const auto second = job->second.primitive_begin + radial % job->second.primitive_count;
    auto* record = records + index * stride;
    record[0] = primitives[2 * first];
    record[1] = primitives[2 * second];
    for (unsigned center = 0; center < job->centers; ++center)
      for (unsigned axis = 0; axis < 3; ++axis)
        record[2 + 3 * center + axis] = basis[3 * job->atoms[center] + axis];
    auto scale = primitives[2 * first + 1] * primitives[2 * second + 1];
    if (job->family == 2) scale *= charges[job->atoms[2]];
    if (!isfinite(scale)) {
      atomicExch(status, 1);
      scale = 0;
    }
    record[stride - 1] = scale;
  }
}
__device__ double coefficient(const double* record, const unsigned* powers) {
  double result{};
  for (unsigned term = 0; term < static_cast<unsigned>(record[3]); ++term)
    if (record[4 + 4 * term] == powers[0] && record[5 + 4 * term] == powers[1] &&
        record[6 + 4 * term] == powers[2])
      result += record[7 + 4 * term];
  return result;
}
__global__ void project(CudaOneElectronShape shape, const DeviceJob* job, const double* basis,
                        const double* direction, CudaOneElectronWeights weights, const double* raw,
                        double* candidate, double* projected, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto pair = threadIdx.x;
  const auto pairs = job->first.ao_count * job->second.ao_count;
  if (pair >= pairs) return;
  const auto first = job->first.ao_begin + pair / job->second.ao_count;
  const auto second = job->second.ao_begin + pair % job->second.ao_count;
  const auto index = first * shape.nao + second;
  const auto* aos = basis + 3 * shape.atoms + 2 * shape.primitives;
  const auto weight = job->family ? weights.core[index] : weights.overlap[index];
  const auto delta_weight =
      job->family ? weights.core_direction[index] : weights.overlap_direction[index];
  const auto coordinates = 3 * job->centers;
  double value{}, motion{}, gradient[9]{}, hvp[9]{};
  for (unsigned component = 0; component < job->components; ++component) {
    // AoBasis owns both Cartesian normalization and spherical expansion. Raw
    // primitive coefficients therefore must not apply either a second time.
    const auto projection = coefficient(aos + 16 * first, job->powers[component][0]) *
                            coefficient(aos + 16 * second, job->powers[component][1]);
    const auto* integral = raw + component * (1 + 2 * coordinates);
    for (unsigned coordinate = 0; coordinate < coordinates; ++coordinate) {
      double result[4];
      if (!ad::contract(integral[0], integral[1 + coordinate],
                        integral[1 + coordinates + coordinate], weight, delta_weight, projection,
                        direction[3 * job->atoms[coordinate / 3] + coordinate % 3], result))
        atomicExch(status, 1);
      if (!coordinate) value += result[0];
      motion += result[1];
      gradient[coordinate] += result[2];
      hvp[coordinate] += result[3];
    }
  }
  const auto matrix = shape.nao * shape.nao;
  const auto panel = job->family ? 0 : 1;
  candidate[panel * matrix + index] += value;
  candidate[(panel + 2) * matrix + index] += motion;
  for (unsigned coordinate = 0; coordinate < coordinates; ++coordinate) {
    projected[pair * coordinates + coordinate] = gradient[coordinate];
    projected[(pairs + pair) * coordinates + coordinate] = hvp[coordinate];
  }
}
__global__ void scatter(const DeviceJob* job, std::size_t atoms, const double* projected,
                        double* vectors, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto axis = threadIdx.x;
  if (axis >= 3) return;
  const auto pairs = job->first.ao_count * job->second.ao_count;
  const auto coordinates = 3 * job->centers;
  // One lane owns each Cartesian axis. Fixed-order pair/center reductions also
  // retain repeated mathematical centers mapping to the same physical atom.
  for (unsigned center = 0; center < job->centers; ++center) {
    double gradient{}, hvp{};
    for (std::size_t pair = 0; pair < pairs; ++pair) {
      gradient += projected[pair * coordinates + 3 * center + axis];
      hvp += projected[(pairs + pair) * coordinates + 3 * center + axis];
    }
    vectors[3 * job->atoms[center] + axis] += gradient;
    vectors[3 * (atoms + job->atoms[center]) + axis] += hvp;
  }
}

__global__ void collect_parallel(CudaOneElectronShape shape, const DeviceJob* jobs,
                                 const ParallelTask* tasks, std::size_t count, const char* base,
                                 double* candidate, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto matrix = shape.nao * shape.nao;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < 4 * matrix + 3 * shape.atoms;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    double value{}, motion{};
    if (index < 4 * matrix) {
      for (std::size_t task = 0; task < count; ++task)
        value += reinterpret_cast<const double*>(base + tasks[task].candidate)[index];
    } else {
      const auto coordinate = index - 4 * matrix;
      for (std::size_t task = 0; task < count; ++task) {
        const auto& job = jobs[tasks[task].job];
        const auto pairs = job.first.ao_count * job.second.ao_count;
        const auto coordinates = 3 * job.centers;
        const auto* projected = reinterpret_cast<const double*>(base + tasks[task].projected);
        for (unsigned center = 0; center < job.centers; ++center) {
          if (static_cast<std::size_t>(job.atoms[center]) != coordinate / 3) continue;
          double gradient{}, hvp{};
          for (std::size_t pair = 0; pair < pairs; ++pair) {
            gradient += projected[pair * coordinates + 3 * center + coordinate % 3];
            hvp += projected[(pairs + pair) * coordinates + 3 * center + coordinate % 3];
          }
          value += gradient;
          motion += hvp;
        }
      }
      candidate[index + 3 * shape.atoms] = motion;
    }
    candidate[index] = value;
  }
}
}  // namespace

struct CudaOneElectronResponse::Impl {
  CudaOneElectronShape shape;
  CudaOneElectronOptions options;
  CudaOneElectronReference reference;
  CudaOneElectronWork work;
  Layout plan;
  std::vector<Job> jobs;
  void* workspace{};
  bool ready{};
  cudaGraph_t graph{};
  cudaGraphExec_t replay{};
  std::vector<cudaStream_t> workers;
  std::vector<cudaEvent_t> events;

  ~Impl() { release_graph(); }
  void release_graph() noexcept {
    if (replay) (void)cudaGraphExecDestroy(replay);
    if (graph) (void)cudaGraphDestroy(graph);
    for (auto event : events) (void)cudaEventDestroy(event);
    for (auto worker : workers) (void)cudaStreamDestroy(worker);
    replay = nullptr;
    graph = nullptr;
    events.clear();
    workers.clear();
  }
  cudaError_t prepare_graph() {
    if (plan.tasks.empty()) return cudaSuccess;
    workers.reserve(options.parallel_streams);
    events.reserve(options.parallel_streams + 1);
    const auto stream = reference.stream;
    for (unsigned index = 0; index < options.parallel_streams; ++index) {
      cudaStream_t worker{};
      auto error = cudaStreamCreateWithFlags(&worker, cudaStreamNonBlocking);
      if (error != cudaSuccess) return error;
      workers.push_back(worker);
    }
    for (unsigned index = 0; index <= options.parallel_streams; ++index) {
      cudaEvent_t event{};
      auto error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
      if (error != cudaSuccess) return error;
      events.push_back(event);
    }
    auto* base = static_cast<char*>(workspace);
    const auto pointer = [&](std::size_t offset) {
      return reinterpret_cast<double*>(base + offset);
    };
    auto* status = reinterpret_cast<int*>(base + plan.status);
    const auto* descriptors = reinterpret_cast<const DeviceJob*>(base + plan.descriptors);
    const CudaOneElectronWeights weights{pointer(plan.parallel_weights),
                                         pointer(plan.parallel_weights) + plan.matrix,
                                         pointer(plan.parallel_weights) + 2 * plan.matrix,
                                         pointer(plan.parallel_weights) + 3 * plan.matrix};
    auto error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (error != cudaSuccess) return error;
    error = cudaEventRecord(events[0], stream);
    for (auto worker : workers)
      if (error == cudaSuccess) error = cudaStreamWaitEvent(worker, events[0], 0);
    for (std::size_t index = 0; index < plan.tasks.size() && error == cudaSuccess; ++index) {
      const auto& task = plan.tasks[index];
      const auto& job = jobs[task.job];
      const auto worker = workers[index % workers.size()];
      build_records<<<1, 128, 0, worker>>>(shape, descriptors + task.job, reference.basis,
                                           reference.charges, task.begin, task.count,
                                           pointer(task.records), status);
      error = cudaGetLastError();
      if (error == cudaSuccess)
        error = static_cast<cudaError_t>(job.program->integral(
            pointer(task.records), task.count, descriptors[task.job].atoms,
            pointer(plan.parallel_direction), pointer(task.raw), status, worker));
      if (error == cudaSuccess) {
        project<<<1, 128, 0, worker>>>(shape, descriptors + task.job, reference.basis,
                                       pointer(plan.parallel_direction), weights, pointer(task.raw),
                                       pointer(task.candidate), pointer(task.projected), status);
        error = cudaGetLastError();
      }
    }
    for (std::size_t index = 0; index < workers.size(); ++index) {
      const auto recorded = cudaEventRecord(events[index + 1], workers[index]);
      const auto joined = cudaStreamWaitEvent(stream, events[index + 1], 0);
      if (error == cudaSuccess) error = recorded != cudaSuccess ? recorded : joined;
    }
    const auto ended = cudaStreamEndCapture(stream, &graph);
    if (error != cudaSuccess) return error;
    if (ended != cudaSuccess) return ended;
    return cudaGraphInstantiate(&replay, graph, nullptr, nullptr, 0);
  }

  cudaError_t authenticate(cudaStream_t stream) noexcept {
    if (!ready || stream != reference.stream) return cudaErrorInvalidResourceHandle;
    int device{-1};
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    if (device != reference.device_id) return cudaErrorInvalidResourceHandle;
    cudaStreamCaptureStatus capture{};
    error = cudaStreamIsCapturing(stream, &capture);
    if (error != cudaSuccess) return error;
    if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    error = reference.authenticate(reference.owner);
    if (error != cudaSuccess) ready = false;
    return error;
  }
};

cudaError_t cuda_one_electron_workspace(const CudaOneElectronShape& shape,
                                        std::span<const CudaOneElectronShell> shells,
                                        const CudaOneElectronOptions& options,
                                        std::size_t& bytes) noexcept {
  bytes = 0;
  try {
    Layout plan;
    std::vector<Job> jobs;
    const auto error = schedule(shape, shells, options, plan, jobs);
    if (error == cudaSuccess) bytes = plan.bytes;
    return error;
  } catch (...) {
    return cudaErrorMemoryAllocation;
  }
}
CudaOneElectronResponse::CudaOneElectronResponse() : impl_(std::make_unique<Impl>()) {}
CudaOneElectronResponse::~CudaOneElectronResponse() = default;
const CudaOneElectronWork& CudaOneElectronResponse::work() const noexcept { return impl_->work; }

cudaError_t CudaOneElectronResponse::prepare(const CudaOneElectronShape& shape,
                                             std::span<const CudaOneElectronShell> shells,
                                             const CudaOneElectronOptions& options,
                                             const CudaOneElectronReference& reference,
                                             void* workspace,
                                             std::size_t workspace_bytes) noexcept {
  impl_->release_graph();
  impl_->ready = false;
  impl_->work = {};
  try {
    Layout plan;
    std::vector<Job> jobs;
    auto error = schedule(shape, shells, options, plan, jobs);
    if (error != cudaSuccess) return error;
    if (workspace_bytes < plan.bytes || !reference.stream || !reference.authenticate ||
        !reference.owner ||
        !disjoint(workspace, plan.bytes, reference.basis, plan.basis_elements * sizeof(double)) ||
        !disjoint(workspace, plan.bytes, reference.charges, shape.atoms * sizeof(double)) ||
        !disjoint(reference.basis, plan.basis_elements * sizeof(double), reference.charges,
                  shape.atoms * sizeof(double)))
      return cudaErrorInvalidValue;
    for (const auto* pointer :
         {reference.basis, reference.charges, static_cast<const double*>(workspace)}) {
      error = device_buffer(pointer, reference.device_id);
      if (error != cudaSuccess) return error;
    }
    auto& state = *impl_;
    state.shape = shape;
    state.options = options;
    state.reference = reference;
    state.plan = plan;
    state.jobs = std::move(jobs);
    state.workspace = workspace;
    state.ready = true;
    if ((error = state.authenticate(reference.stream)) != cudaSuccess) {
      state.ready = false;
      return error;
    }
    const auto stream = reference.stream;
    auto* base = static_cast<char*>(workspace);
    auto* status = reinterpret_cast<int*>(base + plan.status);
    auto* device_jobs = reinterpret_cast<DeviceJob*>(base + plan.descriptors);
    auto* device_shells = reinterpret_cast<CudaOneElectronShell*>(base + plan.shells);
    error = cudaMemsetAsync(status, 0, sizeof(int), stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(device_shells, shells.data(), shells.size_bytes(),
                              cudaMemcpyHostToDevice, stream);
    for (std::size_t index = 0; index < state.jobs.size() && error == cudaSuccess; ++index)
      error = cudaMemcpyAsync(device_jobs + index, &state.jobs[index].device, sizeof(DeviceJob),
                              cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess && !plan.tasks.empty())
      error =
          cudaMemcpyAsync(base + plan.parallel_descriptors, plan.tasks.data(),
                          plan.tasks.size() * sizeof(ParallelTask), cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess) {
      finite_values<<<blocks(plan.basis_elements), 128, 0, stream>>>(reference.basis,
                                                                     plan.basis_elements, status);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      finite_values<<<blocks(shape.atoms), 128, 0, stream>>>(reference.charges, shape.atoms,
                                                             status);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      validate_basis<<<blocks(std::max({shells.size(), shape.primitives, shape.atoms})), 128, 0,
                       stream>>>(shape, device_shells, shells.size(), reference.basis,
                                 reference.charges, status);
      error = cudaGetLastError();
    }
    error = error == cudaSuccess ? gate(status, stream) : drain(error, stream);
    if (error == cudaSuccess) {
      error = state.prepare_graph();
      if (error == cudaErrorMemoryAllocation || error == cudaErrorNotSupported ||
          error == cudaErrorStreamCaptureUnsupported) {
        state.release_graph();
        (void)cudaGetLastError();
        error = cudaSuccess;
      }
    }
    if (error == cudaSuccess) error = state.authenticate(stream);
    state.ready = error == cudaSuccess;
    if (state.ready) {
      state.work = {plan.bytes, state.jobs.size(), plan.radial_records, plan.components,
                    plan.projections};
      if (state.replay) {
        state.work.parallel_workspace_bytes = plan.bytes - plan.parallel_begin;
        state.work.parallel_tasks = plan.tasks.size();
      }
    }
    return error;
  } catch (...) {
    impl_->ready = false;
    return cudaErrorMemoryAllocation;
  }
}

cudaError_t CudaOneElectronResponse::apply(const double* direction,
                                           const CudaOneElectronWeights& weights,
                                           const CudaOneElectronOutput& output,
                                           cudaStream_t stream) noexcept {
  auto& state = *impl_;
  if (state.work.attempted_actions == UINT64_MAX) return cudaErrorInvalidValue;
  ++state.work.attempted_actions;
  auto error = state.authenticate(stream);
  if (error != cudaSuccess) return error;
  const auto& plan = state.plan;
  const auto& shape = state.shape;
  const auto& reference = state.reference;
  const double* inputs[] = {
      reference.basis, reference.charges,        direction, weights.core, weights.core_direction,
      weights.overlap, weights.overlap_direction};
  const std::size_t input_counts[] = {plan.basis_elements, shape.atoms, 3 * shape.atoms,
                                      plan.matrix,         plan.matrix, plan.matrix,
                                      plan.matrix};
  double* outputs[] = {output.core,           output.overlap,
                       output.core_direction, output.overlap_direction,
                       output.gradient,       output.hvp};
  const std::size_t output_counts[] = {plan.matrix, plan.matrix,     plan.matrix,
                                       plan.matrix, 3 * shape.atoms, 3 * shape.atoms};
  for (unsigned index = 0; index < 7; ++index) {
    if (!disjoint(state.workspace, plan.bytes, inputs[index], input_counts[index] * sizeof(double)))
      return cudaErrorInvalidValue;
    if ((error = device_buffer(inputs[index], reference.device_id)) != cudaSuccess) return error;
  }
  for (unsigned index = 0; index < 6; ++index) {
    if (!disjoint(state.workspace, plan.bytes, outputs[index],
                  output_counts[index] * sizeof(double)))
      return cudaErrorInvalidValue;
    for (unsigned other = 0; other < 7; ++other)
      if (!disjoint(outputs[index], output_counts[index] * sizeof(double), inputs[other],
                    input_counts[other] * sizeof(double)))
        return cudaErrorInvalidValue;
    for (unsigned other = 0; other < index; ++other)
      if (!disjoint(outputs[index], output_counts[index] * sizeof(double), outputs[other],
                    output_counts[other] * sizeof(double)))
        return cudaErrorInvalidValue;
    if ((error = device_buffer(outputs[index], reference.device_id)) != cudaSuccess) return error;
  }
  if (state.work.component_evaluations > UINT64_MAX - plan.components ||
      state.work.projection_terms > UINT64_MAX - plan.projections)
    return cudaErrorInvalidValue;
  auto* base = static_cast<char*>(state.workspace);
  auto* status = reinterpret_cast<int*>(base + plan.status);
  auto* records = reinterpret_cast<double*>(base + plan.records);
  auto* raw = reinterpret_cast<double*>(base + plan.raw);
  auto* projected = reinterpret_cast<double*>(base + plan.projected);
  auto* candidate = reinterpret_cast<double*>(base + plan.candidates);
  const auto* jobs = reinterpret_cast<DeviceJob*>(base + plan.descriptors);
  error = cudaMemsetAsync(status, 0, sizeof(int), stream);
  if (error == cudaSuccess) {
    finite_values<<<blocks(3 * shape.atoms), 128, 0, stream>>>(direction, 3 * shape.atoms, status);
    error = cudaGetLastError();
  }
  for (unsigned index = 3; index < 7 && error == cudaSuccess; ++index) {
    symmetric<<<blocks(plan.matrix), 128, 0, stream>>>(
        inputs[index], shape.nao, state.options.symmetry_absolute_tolerance,
        state.options.symmetry_relative_tolerance, status);
    error = cudaGetLastError();
  }
  error = error == cudaSuccess ? gate(status, stream) : drain(error, stream);
  if (error != cudaSuccess) return error;
  error =
      cudaMemsetAsync(candidate, 0, (4 * plan.matrix + 6 * shape.atoms) * sizeof(double), stream);
  if (state.replay && error == cudaSuccess) {
    error = cudaMemcpyAsync(base + plan.parallel_direction, direction,
                            3 * shape.atoms * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    const double* panels[]{weights.core, weights.core_direction, weights.overlap,
                           weights.overlap_direction};
    for (unsigned index = 0; index < 4 && error == cudaSuccess; ++index)
      error = cudaMemcpyAsync(base + plan.parallel_weights + index * plan.matrix * sizeof(double),
                              panels[index], plan.matrix * sizeof(double), cudaMemcpyDeviceToDevice,
                              stream);
    if (error == cudaSuccess)
      error = cudaMemsetAsync(base + plan.parallel_candidates, 0, plan.parallel_candidate_bytes,
                              stream);
    if (error == cudaSuccess) error = cudaGraphLaunch(state.replay, stream);
    if (error == cudaSuccess) {
      ++state.work.parallel_actions;
      state.work.component_evaluations += plan.components;
      state.work.projection_terms += plan.projections;
      collect_parallel<<<blocks(4 * plan.matrix + 3 * shape.atoms), 128, 0, stream>>>(
          shape, jobs, reinterpret_cast<const ParallelTask*>(base + plan.parallel_descriptors),
          plan.tasks.size(), base, candidate, status);
      error = cudaGetLastError();
    }
  }
  for (std::size_t index = 0; !state.replay && index < state.jobs.size() && error == cudaSuccess;
       ++index) {
    const auto& job = state.jobs[index];
    for (std::size_t begin = 0; begin < job.radials && error == cudaSuccess;) {
      const auto count = std::min(state.options.radial_tile, job.radials - begin);
      build_records<<<1, 128, 0, stream>>>(shape, jobs + index, reference.basis, reference.charges,
                                           begin, count, records, status);
      error = cudaGetLastError();
      if (error == cudaSuccess)
        error = static_cast<cudaError_t>(job.program->integral(records, count, jobs[index].atoms,
                                                               direction, raw, status, stream));
      if (error == cudaSuccess) {
        state.work.component_evaluations += count * job.device.components;
        state.work.projection_terms +=
            job.device.first.ao_count * job.device.second.ao_count * job.device.components * 9;
        project<<<1, 128, 0, stream>>>(shape, jobs + index, reference.basis, direction, weights,
                                       raw, candidate, projected, status);
        error = cudaGetLastError();
      }
      if (error == cudaSuccess) {
        scatter<<<1, 32, 0, stream>>>(jobs + index, shape.atoms, projected,
                                      candidate + 4 * plan.matrix, status);
        error = cudaGetLastError();
      }
      begin += count;
    }
  }
  for (unsigned panel = 0; panel < 4 && error == cudaSuccess; ++panel) {
    symmetric<<<blocks(plan.matrix), 128, 0, stream>>>(
        candidate + panel * plan.matrix, shape.nao, state.options.symmetry_absolute_tolerance,
        state.options.symmetry_relative_tolerance, status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    finite_values<<<blocks(6 * shape.atoms), 128, 0, stream>>>(candidate + 4 * plan.matrix,
                                                               6 * shape.atoms, status);
    error = cudaGetLastError();
  }
  error = error == cudaSuccess ? gate(status, stream) : drain(error, stream);
  if (error == cudaSuccess) error = state.authenticate(stream);
  if (error != cudaSuccess) return error;
  std::size_t offset{};
  for (unsigned index = 0; index < 6 && error == cudaSuccess; ++index) {
    error =
        cudaMemcpyAsync(outputs[index], candidate + offset, output_counts[index] * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream);
    offset += output_counts[index];
  }
  error = drain(error, stream);
  if (error == cudaSuccess) ++state.work.completed_actions;
  return error;
}
}  // namespace generativeqc::integrals
