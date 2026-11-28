#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <vector>

#include "dft/contracted_coulomb.hpp"
#include "dft/cuda_wb97mv_df_sweep.hpp"
#include "runtime/cuda_component_trace.hpp"
#include "runtime/resource_cuda.cuh"

namespace generativeqc::dft {
namespace {
namespace jets = hessian_aot::contracted;
using Shell = integrals::CudaDfNuclearShell;
using Pair = std::array<unsigned, 2>;
constexpr unsigned absent = std::numeric_limits<unsigned>::max();
constexpr unsigned window = 32768;
constexpr unsigned max_fields = jets::hessian_fields + 9 * (9 + 3);
constexpr unsigned max_incidence = 16 * window;
constexpr unsigned components[] = {1, 3, 9};
constexpr unsigned states[] = {1, 4, 10};
constexpr jets::PairLaunch orbital_prepare[]{jets::prepare_0_0, jets::prepare_0_1,
                                             jets::prepare_1_1};
constexpr jets::PairLaunch auxiliary_prepare[]{jets::prepare_aux_0_0, jets::prepare_aux_1_0};
constexpr jets::QuartetLaunch three_center[][2]{
    {jets::launch_fitted3_0_0, jets::launch_fitted3_0_1},
    {jets::launch_fitted3_1_0, jets::launch_fitted3_1_1},
    {jets::launch_fitted3_2_0, jets::launch_fitted3_2_1}};
constexpr jets::QuartetConfigure configure_three[][2]{
    {jets::configure_fitted3_0_0, jets::configure_fitted3_0_1},
    {jets::configure_fitted3_1_0, jets::configure_fitted3_1_1},
    {jets::configure_fitted3_2_0, jets::configure_fitted3_2_1}};
constexpr jets::QuartetLaunch metric[][2]{{jets::launch_fitted2_0_0},
                                          {jets::launch_fitted2_1_0, jets::launch_fitted2_1_1}};
constexpr jets::QuartetConfigure configure_metric[][2]{
    {jets::configure_fitted2_0_0}, {jets::configure_fitted2_1_0, jets::configure_fitted2_1_1}};

struct Pool {
  std::vector<Pair> pairs;
  jets::PairJets jet;
  Pair* device{};
};

__device__ double projection(scf::CudaDfReadonlyBasis basis, Shell shell, unsigned component,
                             bool auxiliary, int* status) {
  const auto ao = shell.ao_begin + component;
  const auto& expansion = (auxiliary ? basis.auxiliary_expansions : basis.orbital_expansions)[ao];
  const auto cartesian = shell.cartesian_begin + component;
  if (expansion.count != 1 || expansion.cartesian[0] != cartesian) {
    atomicExch(status, 1);
    return 0;
  }
  return expansion.coefficients[0] *
         basis.component_coefficients[cartesian + (auxiliary ? basis.cartesian_nao : 0)];
}

__global__ void pack(scf::CudaDfReadonlyBasis basis, const Shell* shells, const Pair* pairs,
                     unsigned count, const double* density, const double* charge, double* records,
                     int* status, bool auxiliary) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const auto* pair = reinterpret_cast<const unsigned*>(pairs) + 2 * index;
  const auto first = shells[pair[0]], second = shells[auxiliary ? pair[0] : pair[1]];
  auto* record = records + 27 * std::size_t(index);
  for (unsigned axis = 0; axis < 3; ++axis) {
    record[axis] = basis.positions[3 * first.atom + axis];
    record[axis + 3] = basis.positions[3 * second.atom + axis];
    record[axis + 6] = basis.exponents[first.primitive_begin + axis];
    record[axis + 9] = basis.exponents[second.primitive_begin + axis];
    record[axis + 12] = basis.coefficients[first.primitive_begin + axis];
    record[axis + 15] = basis.coefficients[second.primitive_begin + axis];
  }
  for (unsigned row = 0; row < first.ao_count; ++row) {
    const auto ao = first.ao_begin + row;
    const auto scale = projection(basis, first, row, auxiliary, status);
    if (auxiliary) {
      record[18 + row] = scale * charge[ao];
      continue;
    }
    for (unsigned column = 0; column < second.ao_count; ++column) {
      const auto other = second.ao_begin + column;
      double weight = density[ao * basis.nao + other];
      if (pair[0] != pair[1]) weight += density[other * basis.nao + ao];
      record[18 + row * second.ao_count + column] =
          scale * projection(basis, second, column, false, status) * weight;
    }
  }
}

// The generated relative jet omits slot 3. Its derivative is minus the sum
// of slots 0..2 by translation; the constant auxiliaries have zero motion.
__device__ double derivative(const double* jet, unsigned slot, unsigned axis) {
  return slot < 3 ? jet[3 * slot + axis] : -(jet[axis] + jet[axis + 3] + jet[axis + 6]);
}
__device__ double mixed(const double* jet, unsigned left, unsigned right, unsigned row,
                        unsigned column) {
  double value = 0;
  for (unsigned first = 0; first < 3; ++first) {
    if (left < 3 && first != left) continue;
    for (unsigned second = 0; second < 3; ++second) {
      if (right < 3 && second != right) continue;
      const auto i = 3 * first + row, j = 3 * second + column;
      const auto high = max(i, j), low = min(i, j);
      value += jet[high * (high + 1) / 2 + low];
    }
  }
  return ((left == 3) != (right == 3)) ? -value : value;
}

enum class Fold { hessian, fock, rhs };

/** Each dense destination has one block per window. Incidences for that
 * destination retain source order; atom coincidences do not introduce races. */
template <Fold Kind>
__global__ void fold(scf::CudaDfReadonlyBasis basis, const Shell* shells, const Pair* left,
                     bool same_pool, const Pair* groups, const unsigned* owners,
                     const unsigned* entries, unsigned left_components, unsigned right_components,
                     bool is_metric, const double* local, double* output, int* status) {
  const unsigned owner = owners[2 * blockIdx.x], begin = owners[2 * blockIdx.x + 1];
  const unsigned end = owners[2 * blockIdx.x + 3];
  const unsigned lane = threadIdx.x, axis = lane % 3, component = lane / 3;
  const unsigned stride = jets::hessian_fields + 9 * (left_components + right_components);
  if constexpr (Kind == Fold::hessian) {
    if (lane >= 9) return;
    const auto coordinates = 3 * basis.atoms;
    double value = 0;
    for (unsigned index = begin; index < end; ++index) {
      const auto entry = entries[index], group = entry / 16;
      const auto* ids = reinterpret_cast<const unsigned*>(groups) + 2 * group;
      const double scale = !is_metric ? 1.0 : ids[0] == ids[1] && same_pool ? -0.5 : -1.0;
      value += scale * mixed(local + std::size_t(group) * stride, entry / 4 % 4, entry % 4,
                             component, axis);
    }
    const auto row = 3 * (owner / basis.atoms) + component;
    const auto col = 3 * (owner % basis.atoms) + axis;
    output[col * coordinates + row] += value;
    if (!isfinite(output[col * coordinates + row])) atomicExch(status, 1);
  } else if constexpr (Kind == Fold::rhs) {
    const unsigned shell = owner / basis.atoms, atom = owner % basis.atoms;
    const auto target = shells[shell];
    if (component >= target.ao_count) return;
    double value = 0;
    for (unsigned index = begin; index < end; ++index) {
      const auto entry = entries[index], group = entry / 16;
      const unsigned side = entry / 4 % 2;
      const auto* jet = local + std::size_t(group) * stride + jets::hessian_fields +
                        9 * (component + (side ? left_components : 0));
      value += derivative(jet, entry % 4, axis);
    }
    auto& dest = output[(3 * atom + axis) * basis.naux + target.ao_begin + component];
    dest += (is_metric ? -1.0 : 1.0) * projection(basis, target, component, true, status) * value;
    if (!isfinite(dest)) atomicExch(status, 1);
  } else {
    if (component >= left_components) return;
    const auto pair = owner / basis.atoms, atom = owner % basis.atoms;
    const auto* ids = reinterpret_cast<const unsigned*>(left) + 2 * pair;
    const auto first = shells[ids[0]], second = shells[ids[1]];
    const auto row = component / second.ao_count, col = component % second.ao_count;
    double value = 0;
    for (unsigned index = begin; index < end; ++index) {
      const auto entry = entries[index];
      const auto* jet =
          local + std::size_t(entry / 16) * stride + jets::hessian_fields + 9 * component;
      value += derivative(jet, entry % 4, axis);
    }
    value *= projection(basis, first, row, false, status) *
             projection(basis, second, col, false, status);
    const auto i = first.ao_begin + row, j = second.ao_begin + col;
    auto* panel = output + (3 * atom + axis) * basis.nao * basis.nao;
    panel[i * basis.nao + j] += value;
    if (ids[0] != ids[1]) panel[j * basis.nao + i] += value;
    if (!isfinite(panel[i * basis.nao + j])) atomicExch(status, 1);
  }
}

struct Owners {
  std::vector<Pair> ordered, keys;
  std::vector<unsigned> entries;
  void add(unsigned key, unsigned entry) { ordered.push_back({key, entry}); }
  void compress() {
    keys.clear();
    entries.clear();
    // The incidence index is a tie-breaker in source and local slot order.
    std::sort(ordered.begin(), ordered.end());
    for (const auto& entry : ordered) {
      if (keys.empty() || keys.back()[0] != entry[0])
        keys.push_back({entry[0], static_cast<unsigned>(entries.size())});
      entries.push_back(entry[1]);
    }
    keys.push_back({absent, static_cast<unsigned>(entries.size())});
    ordered.clear();
  }
};

struct Scratch {
  void* data{};
  cudaStream_t stream{};
  ~Scratch() {
    if (data) {
      (void)cudaStreamSynchronize(stream);
      (void)runtime::resource_cuda_free(data);
    }
  }
};
}  // namespace

cudaError_t cuda_wb97mv_df_frozen_sweep(const scf::CudaDfReadonlyBasis& basis,
                                        std::span<const Shell> orbital,
                                        std::span<const Shell> auxiliary, const double* density,
                                        const double* charge, CudaDfFrozenTables output,
                                        std::size_t max_scratch_bytes, int* status,
                                        cudaStream_t stream) noexcept {
  // Bound optional metadata and work before constructing any pair inventory.
  // This is not an endpoint cap: the bounded raw nuclear replay is unchanged.
  constexpr std::size_t max_source_groups = 16 * 1024 * 1024;
  if (!basis || basis.range_omega || orbital.empty() || auxiliary.empty() ||
      orbital.size() > 4096 || auxiliary.size() > 4096 || basis.atoms > 4096 ||
      orbital.size() * (orbital.size() + 1) / 2 * auxiliary.size() > max_source_groups)
    return cudaErrorNotSupported;
  const auto supported = [](Shell shell) {
    return shell.angular <= 1 && shell.primitive_count == 3 &&
           shell.ao_count == 1 + 2 * shell.angular;
  };
  if (!std::all_of(orbital.begin(), orbital.end(), supported) ||
      !std::all_of(auxiliary.begin(), auxiliary.end(), supported))
    return cudaErrorNotSupported;

  bool submitted = false;
  try {
    std::array<Pool, 3> bra;
    std::array<Pool, 2> ket;
    std::vector<Shell> shells(orbital.begin(), orbital.end());
    shells.insert(shells.end(), auxiliary.begin(), auxiliary.end());
    for (unsigned first = 0; first < orbital.size(); ++first) {
      for (unsigned second = 0; second <= first; ++second) {
        const auto l = orbital[first].angular, r = orbital[second].angular;
        bra[l + r].pairs.push_back(l <= r ? Pair{first, second} : Pair{second, first});
      }
    }
    for (unsigned aux = orbital.size(); aux < shells.size(); ++aux)
      ket[shells[aux].angular].pairs.push_back({aux, absent});

    std::size_t bytes = 0;
    const auto reserve = [&](std::size_t size) {
      const auto offset = bytes;
      bytes += (size + 255) / 256 * 256;
      return offset;
    };
    const auto upload = [&](auto* dest, const auto& host) {
      return cudaMemcpyAsync(dest, host.data(), host.size() * sizeof(*dest), cudaMemcpyHostToDevice,
                             stream);
    };
    const auto shell_offset = reserve(shells.size() * sizeof(Shell));
    const auto groups_offset = reserve(window * sizeof(Pair));
    const auto local_offset = reserve(window * max_fields * sizeof(double));
    const auto keys_offset = reserve((max_incidence + 1) * sizeof(Pair));
    const auto entries_offset = reserve(max_incidence * sizeof(unsigned));
    struct Layout {
      std::size_t pairs, geometry, weighted, unweighted;
    };
    unsigned max_pairs = 0;
    const auto layout = [&](const Pool& pool, unsigned degree, unsigned radials) {
      const unsigned count = pool.pairs.size();
      max_pairs = std::max(max_pairs, count);
      const auto primitive = std::size_t(count) * radials;
      return Layout{reserve(count * sizeof(Pair)), reserve(6 * primitive * sizeof(double)),
                    reserve(13 * primitive * states[degree] * sizeof(double)),
                    reserve(4 * primitive * states[degree] * components[degree] * sizeof(double))};
    };
    std::array<Layout, 3> orbital_layout;
    std::array<Layout, 2> auxiliary_layout;
    for (unsigned degree = 0; degree < bra.size(); ++degree)
      orbital_layout[degree] = layout(bra[degree], degree, 9);
    for (unsigned degree = 0; degree < ket.size(); ++degree)
      auxiliary_layout[degree] = layout(ket[degree], degree, 3);
    const auto record_offset = reserve(27 * max_pairs * sizeof(double));
    if (bytes > max_scratch_bytes) return cudaErrorNotSupported;
    Scratch scratch{nullptr, stream};
    auto error = runtime::resource_cuda_malloc(&scratch.data, bytes);
    if (error != cudaSuccess) return error;
    for (const auto& row : configure_three)
      for (const auto configure : row)
        if ((error = configure()) != cudaSuccess) return error;
    for (const auto& row : configure_metric)
      for (const auto configure : row)
        if (configure && (error = configure()) != cudaSuccess) return error;

    runtime::cuda_trace::TraceRegion trace("hessian_fitted_contracted_sources", stream);
    const auto at = [&]<class T>(std::size_t offset) {
      return reinterpret_cast<T*>(static_cast<char*>(scratch.data) + offset);
    };
    auto* device_shells = at.template operator()<Shell>(shell_offset);
    auto* device_groups = at.template operator()<Pair>(groups_offset);
    auto* local = at.template operator()<double>(local_offset);
    auto* keys = at.template operator()<Pair>(keys_offset);
    auto* entries = at.template operator()<unsigned>(entries_offset);
    auto* records = at.template operator()<double>(record_offset);
    const auto zero = [&](double* pointer, std::size_t count) {
      return cudaMemsetAsync(pointer, 0, count * sizeof(double), stream);
    };
    const auto extent = basis.atoms * 3;
    submitted = true;
    if ((error = upload(device_shells, shells)) != cudaSuccess ||
        (error = zero(output.hessian, extent * extent)) != cudaSuccess ||
        (error = zero(output.explicit_fock, extent * basis.nao * basis.nao)) != cudaSuccess ||
        (error = zero(output.auxiliary_rhs, extent * basis.naux)) != cudaSuccess)
      return error == cudaErrorMemoryAllocation ? cudaErrorUnknown : error;
    const auto prepare = [&](Pool& pool, const Layout& offsets, jets::PairLaunch launch,
                             bool auxiliary_side) {
      const unsigned count = pool.pairs.size();
      if (!count || error != cudaSuccess) return;
      pool.device = at.template operator()<Pair>(offsets.pairs);
      pool.jet = {at.template operator()<double>(offsets.geometry),
                  at.template operator()<double>(offsets.weighted),
                  at.template operator()<double>(offsets.unweighted)};
      error = upload(pool.device, pool.pairs);
      if (error != cudaSuccess) return;
      pack<<<(count + 127) / 128, 128, 0, stream>>>(basis, device_shells, pool.device, count,
                                                    density, charge, records, status,
                                                    auxiliary_side);
      if ((error = cudaGetLastError()) == cudaSuccess)
        error = launch(records, count, pool.jet, status, stream);
    };
    for (unsigned degree = 0; degree < bra.size(); ++degree)
      prepare(bra[degree], orbital_layout[degree], orbital_prepare[degree], false);
    for (unsigned degree = 0; degree < ket.size(); ++degree)
      prepare(ket[degree], auxiliary_layout[degree], auxiliary_prepare[degree], true);

    std::vector<Pair> groups;
    std::array<Owners, 3> tables;
    // Wait before reusing host-backed upload buffers, including error exits.
    struct Drain {
      cudaStream_t stream;
      ~Drain() { (void)cudaStreamSynchronize(stream); }
    } drain{stream};
    groups.reserve(window);
    std::size_t visited = 0;
    const auto sweep = [&](unsigned l, unsigned r, bool is_metric) {
      if (error != cudaSuccess) return;
      const auto& first = is_metric ? ket[l] : bra[l];
      const auto& second = ket[r];
      const auto flush = [&]() {
        if (error != cudaSuccess || groups.empty()) return;
        error = upload(device_groups, groups);
        if (error == cudaSuccess) {
          const auto launch = is_metric ? metric[l][r] : three_center[l][r];
          error = launch(first.jet, second.jet, reinterpret_cast<unsigned*>(device_groups),
                         groups.size(), local, status, stream);
        }
        const auto submit = [&]<Fold kind>(double* destination) {
          auto& owners = tables[static_cast<unsigned>(kind)];
          if (error != cudaSuccess || owners.ordered.empty()) return;
          owners.compress();
          if ((error = upload(keys, owners.keys)) != cudaSuccess ||
              (error = upload(entries, owners.entries)) != cudaSuccess)
            return;
          fold<kind><<<owners.keys.size() - 1, 32, 0, stream>>>(
              basis, device_shells, first.device, is_metric && l == r, device_groups,
              reinterpret_cast<unsigned*>(keys), entries, components[l], components[r], is_metric,
              local, destination, status);
          error = cudaGetLastError();
        };
        submit.template operator()<Fold::hessian>(output.hessian);
        submit.template operator()<Fold::fock>(output.explicit_fock);
        submit.template operator()<Fold::rhs>(output.auxiliary_rhs);
        if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
        visited += groups.size();
        groups.clear();
      };
      for (unsigned i = 0; i < first.pairs.size() && error == cudaSuccess; ++i) {
        const auto count = is_metric && l == r ? i + 1 : second.pairs.size();
        for (unsigned j = 0; j < count && error == cudaSuccess; ++j) {
          const auto base = static_cast<unsigned>(16 * groups.size());
          groups.push_back({i, j});
          const unsigned center[]{first.pairs[i][0], first.pairs[i][1], second.pairs[j][0], absent};
          for (unsigned a = 0; a < 3; ++a) {
            if (center[a] == absent) continue;
            const unsigned atom = shells[center[a]].atom;
            for (unsigned b = 0; b < 3; ++b) {
              if (center[b] == absent) continue;
              tables[0].add(atom * basis.atoms + shells[center[b]].atom, base + 4 * a + b);
            }
            if (!is_metric) tables[1].add(i * basis.atoms + atom, base + a);
            if (is_metric) tables[2].add(center[0] * basis.atoms + atom, base + a);
            if (!is_metric || l != r || i != j)
              tables[2].add(center[2] * basis.atoms + atom, base + 4 + a);
          }
          if (groups.size() == window) flush();
        }
      }
      flush();
    };
    for (unsigned l = 0; l < bra.size(); ++l)
      for (unsigned r = 0; r < ket.size(); ++r) sweep(l, r, false);
    for (unsigned l = 0; l < ket.size(); ++l)
      for (unsigned r = 0; r <= l; ++r) sweep(l, r, true);
    if (error == cudaSuccess) runtime::cuda_trace::trace_counter("df_contracted_groups", visited);
    return error == cudaErrorMemoryAllocation || error == cudaErrorNotSupported ? cudaErrorUnknown
                                                                                : error;
  } catch (const std::bad_alloc&) {
    return submitted ? cudaErrorUnknown : cudaErrorMemoryAllocation;
  } catch (...) {
    return cudaErrorUnknown;
  }
}
}  // namespace generativeqc::dft
