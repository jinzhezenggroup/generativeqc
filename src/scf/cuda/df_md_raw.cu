#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "runtime/resource_cuda.cuh"
#include "scf/cuda/df_md_layout.hpp"
#include "scf/cuda/df_md_raw.hpp"
#include "scf/cuda/df_md_transforms.cuh"

namespace generativeqc::scf::cuda_execution {
namespace {
/** One public component and the contiguous primitive run of its shell pair.
 * Sorting by angular class and public index makes panel lookup logarithmic. */
struct Component {
  std::size_t index, begin;
  unsigned count, component;
};
struct Components {
  std::vector<Component> host;
  Component* device{};
  std::array<std::size_t, 8> offsets{};
};

Components components(const DfMdLayout& layout) {
  Components result;
  const auto n = layout.aos.size();
  result.host.reserve(layout.auxiliary ? n : n * (n + 1) / 2);
  for (unsigned l = 0; l < 7; ++l) {
    result.offsets[l] = result.host.size();
    for (auto begin = layout.offsets[l]; begin < layout.offsets[l + 1];) {
      const auto& p = layout.primitives[begin];
      auto end = begin + 1;
      while (end < layout.offsets[l + 1] && layout.primitives[end].first_ao == p.first_ao &&
             layout.primitives[end].second_ao == p.second_ao)
        ++end;
      for (unsigned i = 0; i < p.first_count; ++i)
        for (unsigned j = 0; j < p.second_count; ++j) {
          const auto a = p.first_ao + i, b = p.second_ao + j;
          if (!layout.auxiliary && a < b) continue;
          result.host.push_back({layout.auxiliary ? b : a * (a + 1) / 2 + b, begin,
                                 static_cast<unsigned>(end - begin), i * p.second_count + j});
        }
      begin = end;
    }
    std::sort(result.host.begin() + result.offsets[l], result.host.end(),
              [](const auto& a, const auto& b) { return a.index < b.index; });
  }
  result.offsets[7] = result.host.size();
  return result;
}

template <unsigned P, unsigned A>
/** One writer per public value, with a fixed ordered-primitive traversal.
 * Prepared E already contains public-component and radial normalization.
 * The compiler's Coulomb matrix and auxiliary parity are shared with the
 * charge/J consumer. Exact E zeros skip multiplications, never a tolerance. */
__global__ void raw_tile(DfMdView pairs, DfMdView auxiliary, const Component* pair_components,
                         std::size_t pair_components_count, const Component* aux_components,
                         std::size_t aux_components_count, std::size_t pair_begin,
                         std::size_t aux_begin, std::size_t aux_count, double* output) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pair_components_count * aux_components_count) return;
  const auto pc = pair_components[index / aux_components_count];
  const auto ac = aux_components[index % aux_components_count];
  double value = 0;
  for (unsigned i = 0; i < pc.count; ++i) {
    const auto pair = pairs.primitives[pc.begin + i];
    const auto* e = pairs.transforms + pair.transform_offset + pc.component * df_md_count(P);
    for (unsigned j = 0; j < ac.count; ++j) {
      const auto aux = auxiliary.primitives[ac.begin + j];
      const auto* f = auxiliary.transforms + aux.transform_offset + ac.component * df_md_count(A);
      const double sum = pair.exponent + aux.exponent;
      const double rho = pair.exponent * (aux.exponent / sum);
      double radial[df_md_count(P + A)];
      df_md_radial<P + A>(rho, pair.product, aux.product, radial);
      double product = 0;
      for (unsigned h = 0; h < df_md_count(P); ++h)
        if (e[h] != 0) product += e[h] * df_md_pair_potential<P, A>(radial, f, h);
      value += (34.986836655249725 / (pair.exponent * aux.exponent * sqrt(sum))) * product;
    }
  }
  output[(pc.index - pair_begin) * aux_count + ac.index - aux_begin] = value;
}
}  // namespace

struct CudaDfMdRaw {
  int device{-1};
  cudaStream_t preparation{};
  DfMdView pair{}, auxiliary{};
  Components pairs, auxiliaries;
  std::array<void*, 9> allocations{};
  std::size_t allocation_count{};
  CudaDfMdResources resources{};

  ~CudaDfMdRaw() {
    int previous = -1;
    (void)cudaGetDevice(&previous);
    if (device >= 0 && cudaSetDevice(device) == cudaSuccess) {
      if (preparation) (void)cudaStreamSynchronize(preparation);
      // resource_cuda_malloc uses synchronous CUDA allocation. Its matching
      // free retires outstanding readers, including a failed setup upload.
      for (auto i = allocation_count; i; --i) (void)runtime::resource_cuda_free(allocations[i - 1]);
      if (preparation) (void)cudaStreamDestroy(preparation);
    }
    if (previous >= 0 && previous != device) (void)cudaSetDevice(previous);
  }

  template <class T>
  cudaError_t allocate(std::size_t count, T*& output, const T* host = nullptr) {
    void* pointer{};
    if (!count || count > std::numeric_limits<std::size_t>::max() / sizeof(T) ||
        allocation_count == allocations.size())
      return cudaErrorInvalidValue;
    auto status = runtime::resource_cuda_malloc(&pointer, count * sizeof(T));
    if (status != cudaSuccess) return status;
    allocations[allocation_count++] = pointer;
    output = static_cast<T*>(pointer);
    if (host)
      status =
          cudaMemcpyAsync(output, host, count * sizeof(T), cudaMemcpyHostToDevice, preparation);
    return status;
  }

  cudaError_t prepare(const DfMdLayout& layout, DfMdView& view, Components& outputs) {
    view.count = layout.primitives.size();
    view.nbf = layout.aos.size();
    view.auxiliary = layout.auxiliary;
    auto status = allocate(view.count, view.primitives, layout.primitives.data());
    if (status != cudaSuccess) return status;
    DfMdAo* aos{};
    status = allocate(layout.aos.size(), aos, layout.aos.data());
    if (status != cudaSuccess) return status;
    view.first_aos = view.second_aos = aos;
    if (layout.auxiliary) {
      DfMdAo unit{};
      unit.count = 1;
      unit.coefficients[0] = 1;
      status = allocate(1, aos, &unit);
      if (status != cudaSuccess) return status;
      view.first_aos = aos;
      status = cudaStreamSynchronize(preparation);
      if (status != cudaSuccess) return status;
    }
    status = allocate(layout.transform_elements, view.transforms);
    if (status != cudaSuccess) return status;
    status = allocate(outputs.host.size(), outputs.device, outputs.host.data());
    if (status != cudaSuccess) return status;
    df_md_prepare_transforms<<<view.count, 128, 0, preparation>>>(view);
    return cudaPeekAtLastError();
  }
};

generativeqc_status create_cuda_df_md_raw(int device, const core::System& orbital,
                                          const core::System& auxiliary, std::size_t device_budget,
                                          std::size_t host_budget, CudaDfMdRaw** owner,
                                          CudaDfMdResources& resources, std::string& detail) {
  resources = {};
  detail.clear();
  if (!owner) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *owner = nullptr;
  const auto fail = [&](cudaError_t error) {
    detail = std::string("DF raw Hermite CUDA operation failed: ") + cudaGetErrorString(error);
    return error == cudaErrorMemoryAllocation ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                              : GENERATIVEQC_STATUS_CUDA_ERROR;
  };
  int current{-1};
  auto error = cudaGetDevice(&current);
  if (error != cudaSuccess) return fail(error);
  if (device < 0 || current != device) {
    detail = "raw Hermite preparation requires the caller's current device";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    using namespace df_md_layout_detail;
    if (orbital.atoms.size() != auxiliary.atoms.size())
      throw std::invalid_argument("DF raw Hermite geometries differ");
    for (std::size_t i = 0; i < orbital.atoms.size(); ++i)
      if (orbital.atoms[i].atomic_number != auxiliary.atoms[i].atomic_number ||
          orbital.atoms[i].position != auxiliary.atoms[i].position)
        throw std::invalid_argument("DF raw Hermite geometries differ");
    const auto n = molecule::ao_count(orbital), a = molecule::ao_count(auxiliary);
    const auto components_bytes = multiply(add(multiply(n, add(n, 1)) / 2, a), sizeof(Component));
    const auto host_fixed = add(sizeof(CudaDfMdRaw), multiply(2, components_bytes));
    if (device_budget < components_bytes || host_budget < host_fixed)
      throw std::length_error("DF raw Hermite component metadata exceeds budget");
    const auto pairs = layout(orbital, false, device_budget - components_bytes,
                              host_budget - host_fixed, true, true);
    const auto auxiliaries =
        layout(auxiliary, true, device_budget - components_bytes - pairs.device_bytes,
               host_budget - host_fixed - pairs.host_peak_bytes, true, true);
    // Layouts outlive the candidate so every failed upload drains before their
    // pageable source storage is destroyed.
    auto candidate = std::make_unique<CudaDfMdRaw>();
    candidate->device = device;
    candidate->pairs = components(pairs);
    candidate->auxiliaries = components(auxiliaries);
    const auto component_capacity =
        multiply(add(candidate->pairs.host.capacity(), candidate->auxiliaries.host.capacity()),
                 sizeof(Component));
    if (component_capacity > multiply(2, components_bytes))
      throw std::length_error("DF raw Hermite component capacity exceeds its host envelope");
    error = cudaStreamCreateWithFlags(&candidate->preparation, cudaStreamNonBlocking);
    if (error == cudaSuccess) error = candidate->prepare(pairs, candidate->pair, candidate->pairs);
    if (error == cudaSuccess)
      error = candidate->prepare(auxiliaries, candidate->auxiliary, candidate->auxiliaries);
    if (error == cudaSuccess) error = cudaStreamSynchronize(candidate->preparation);
    if (error != cudaSuccess) return fail(error);
    candidate->resources = {components_bytes + pairs.device_bytes + auxiliaries.device_bytes,
                            sizeof(CudaDfMdRaw) + component_capacity,
                            host_fixed + pairs.host_peak_bytes + auxiliaries.host_peak_bytes,
                            pairs.primitives.size(),
                            auxiliaries.primitives.size(),
                            n,
                            a};
    resources = candidate->resources;
    *owner = candidate.release();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::length_error& e) {
    detail = e.what();
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::bad_alloc&) {
    detail = "DF raw Hermite host allocation failed";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument& e) {
    detail = e.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& e) {
    detail = e.what();
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}

void destroy_cuda_df_md_raw(CudaDfMdRaw* owner) noexcept { delete owner; }

namespace {
std::pair<std::size_t, std::size_t> interval(const Components& values, unsigned angular,
                                             std::size_t begin, std::size_t count) {
  const auto start = values.host.begin() + values.offsets[angular];
  const auto end = values.host.begin() + values.offsets[angular + 1];
  const auto cmp = [](const Component& value, std::size_t index) { return value.index < index; };
  const auto first = std::lower_bound(start, end, begin, cmp);
  const auto last = std::lower_bound(first, end, begin + count, cmp);
  return {first - values.host.begin(), last - first};
}
template <unsigned P, unsigned A>
cudaError_t launch(const CudaDfMdRaw& owner, cudaStream_t stream, std::size_t pb, std::size_t np,
                   std::size_t ab, std::size_t na, double* output) {
  const auto [p, pc] = interval(owner.pairs, P, pb, np);
  const auto [a, ac] = interval(owner.auxiliaries, A, ab, na);
  if (!pc || !ac) return cudaSuccess;
  if (pc > (std::numeric_limits<std::size_t>::max() - 127) / ac) return cudaErrorInvalidValue;
  const auto total = pc * ac;
  if ((total + 127) / 128 > std::numeric_limits<int>::max()) return cudaErrorInvalidValue;
  raw_tile<P, A><<<(total + 127) / 128, 128, 0, stream>>>(
      owner.pair, owner.auxiliary, owner.pairs.device + p, pc, owner.auxiliaries.device + a, ac, pb,
      ab, na, output);
  return cudaPeekAtLastError();
}
}  // namespace

cudaError_t cuda_df_md_raw_tile(const CudaDfMdRaw* owner, cudaStream_t stream,
                                std::size_t pair_begin, std::size_t pair_count,
                                std::size_t aux_begin, std::size_t aux_count, double* output) {
  if (!owner || !output || pair_begin > owner->pairs.host.size() ||
      pair_count > owner->pairs.host.size() - pair_begin || aux_begin > owner->resources.naux ||
      aux_count > owner->resources.naux - aux_begin)
    return cudaErrorInvalidValue;
  int device{-1};
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device != owner->device) return cudaErrorInvalidDevice;
  if (!pair_count || !aux_count) return cudaSuccess;
  if (pair_count > std::numeric_limits<std::size_t>::max() / sizeof(double) / aux_count)
    return cudaErrorInvalidValue;
#define CLASS(P, A)                                                                            \
  status = launch<P, A>(*owner, stream, pair_begin, pair_count, aux_begin, aux_count, output); \
  if (status != cudaSuccess) return status;
#define ROW(P) \
  CLASS(P, 0); \
  CLASS(P, 1); \
  CLASS(P, 2); \
  CLASS(P, 3); \
  CLASS(P, 4)
  ROW(0);
  ROW(1);
  ROW(2);
  ROW(3);
  ROW(4);
  ROW(5);
  ROW(6);
#undef ROW
#undef CLASS
  return cudaSuccess;
}
}  // namespace generativeqc::scf::cuda_execution
