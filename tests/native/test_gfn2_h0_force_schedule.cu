// Qualify tiled H0/Pulay contraction against both the retained one-block path
// and an independent long-double analytic reference (no generated helpers).
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "backends/cuda/gfn2_h0_force.cuh"
#include "generated_gfn2_h0_native.hpp"

using namespace generativeqc::xtb::detail::cuda;
using I = std::int64_t;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct Storage {
  std::vector<void*> owners;
  ~Storage() {
    for (void* p : owners) cudaFree(p);
  }
  template <class T>
  T* make(I size) {
    T* result = nullptr;
    check(cudaMallocManaged(&result, std::max(I{1}, size) * sizeof(T)));
    owners.push_back(result);
    std::fill(result, result + size, T{});
    return result;
  }
};

enum class Fault {
  none,
  input,
  parameter,
  metadata,
  position,
  overflow,
  coincidence,
  difference,
  seed,
  mask,
  inactive,
  failed_scc,
  closed
};
struct Output {
  std::vector<double> matrix, atoms;
  std::vector<std::uint32_t> errors;
  std::uint32_t error;
};

void close(double actual, long double expected) {
  // Atomics already have an unspecified inter-pair order. This gate bounds
  // FP64 accumulation roundoff; it is not a change to production precision.
  const long double limit =
      256 * std::numeric_limits<double>::epsilon() * std::max(1.0L, std::abs(expected));
  require(std::isfinite(actual) && std::abs(actual - expected) <= limit,
          "H0/Pulay output exceeded the FP64 roundoff gate");
}

void compare(const Output& tiled, const Output& single) {
  require(tiled.errors == single.errors && tiled.error == single.error,
          "schedule changed diagnostics");
  require(tiled.matrix.size() == single.matrix.size() && tiled.atoms.size() == single.atoms.size(),
          "schedule changed output extents");
  require(std::memcmp(tiled.matrix.data(), single.matrix.data(),
                      tiled.matrix.size() * sizeof(double)) == 0,
          "schedule changed ordered AO contraction");
  for (std::size_t i = 0; i < tiled.atoms.size(); ++i) close(tiled.atoms[i], single.atoms[i]);
}

// Each original size is a shell count. Consecutive s/p/d shells share an atom,
// exercising same-atom off-diagonal pairs and nontrivial AO block reductions.
// Independent one-shell systems lower the mean to select the retained path;
// no original geometry, parameter, AO block or pair is changed by this padding.
Output run(std::vector<I> sizes, bool padded, bool graph, Fault fault) {
  const I original_systems = sizes.size();
  const I bad = std::max_element(sizes.begin(), sizes.end()) - sizes.begin();
  I pairs = std::accumulate(sizes.begin(), sizes.end(), I{0},
                            [](I sum, I size) { return sum + size * size; });
  if (padded) {
    while (generativeqc::xtb::generated::gfn2_h0_force_pair_tiles(pairs, sizes.size()) != 1) {
      sizes.push_back(1);
      ++pairs;
    }
  }
  const I systems = sizes.size();
  const I shells = std::accumulate(sizes.begin(), sizes.end(), I{0});
  Storage storage;
  auto* atoms = storage.make<I>(systems + 1);
  auto* shell_offsets = storage.make<I>(systems + 1);
  auto* orbital_offsets = storage.make<I>(systems + 1);
  auto* matrices = storage.make<I>(systems + 1);
  auto* pair_offsets = storage.make<I>(systems + 1);
  auto* shell_orbitals = storage.make<I>(shells + 1);
  auto* shell_atoms = storage.make<I>(shells);
  for (I system = 0; system < systems; ++system) {
    atoms[system + 1] = atoms[system] + (sizes[system] + 2) / 3;
    shell_offsets[system + 1] = shell_offsets[system] + sizes[system];
    for (I local = 0; local < sizes[system]; ++local) {
      const I shell = shell_offsets[system] + local;
      shell_atoms[shell] = atoms[system] + local / 3;
      shell_orbitals[shell + 1] = shell_orbitals[shell] + 2 * (local % 3) + 1;
    }
    orbital_offsets[system + 1] = shell_orbitals[shell_offsets[system + 1]];
    const I orbitals = orbital_offsets[system + 1] - orbital_offsets[system];
    matrices[system + 1] = matrices[system] + orbitals * orbitals;
    pair_offsets[system + 1] = pair_offsets[system] + sizes[system] * sizes[system];
  }
  const I n = atoms[systems], m = matrices[systems];
  auto* atom_shells = storage.make<I>(n + 1);
  for (I system = 0; system < systems; ++system) {
    for (I atom = atoms[system]; atom < atoms[system + 1]; ++atom)
      atom_shells[atom] = shell_offsets[system] + 3 * (atom - atoms[system]);
  }
  atom_shells[n] = shells;
  Gfn2IntegralDeviceBatch batch{};
  batch.batch_size = systems;
  batch.total_atoms = n;
  batch.total_shells = shells;
  batch.total_orbitals = orbital_offsets[systems];
  batch.total_matrix_elements = m;
  batch.total_shell_pair_elements = pairs;
  batch.plan_token = 23;
  batch.atom_offsets = atoms;
  batch.batch_shell_offsets = shell_offsets;
  batch.batch_orbital_offsets = orbital_offsets;
  batch.matrix_offsets = matrices;
  batch.shell_pair_offsets = pair_offsets;
  batch.atom_shell_offsets = atom_shells;
  batch.shell_orbital_offsets = shell_orbitals;
  batch.shell_to_atom = shell_atoms;
  batch.atom_offset_count = batch.batch_shell_offset_count = batch.batch_orbital_offset_count =
      batch.matrix_offset_count = batch.shell_pair_offset_count = systems + 1;
  batch.atom_shell_offset_count = n + 1;
  batch.shell_orbital_offset_count = shells + 1;
  batch.shell_to_atom_count = shells;
  auto* radius = storage.make<double>(n);
  auto* level = storage.make<double>(shells);
  auto* cn_scale = storage.make<double>(shells);
  auto* polynomial = storage.make<double>(shells);
  auto* pair_scale = storage.make<double>(pairs);
  Gfn2H0DevicePlan plan{n,      shells, shells,   shells,     pairs,     23,
                        radius, level,  cn_scale, polynomial, pair_scale};
  auto* requested = storage.make<std::uint8_t>(systems);
  auto* statuses = storage.make<generativeqc_xtb_status_t>(systems);
  std::fill(requested, requested + systems, 1);
  std::fill(statuses, statuses + systems, GENERATIVEQC_XTB_STATUS_SUCCESS);
  Gfn2ForceDeviceActivity activity{requested, statuses, systems, 23};
  auto* positions = storage.make<double>(3 * n);
  auto* cn = storage.make<double>(n);
  auto* overlap = storage.make<double>(m);
  auto* density = storage.make<double>(m);
  auto* weighted = storage.make<double>(m);
  Gfn2H0ForceDeviceInput input{positions, 3 * n, cn, n, overlap, m, density, m, weighted, m, 23};
  auto* overlap_out = storage.make<double>(m);
  auto* cn_out = storage.make<double>(n);
  auto* gradient = storage.make<double>(3 * n);
  Gfn2H0ForceDeviceOutput output{overlap_out, m, cn_out, n, gradient, 3 * n, 23};
  Gfn2H0ForceDeviceWorkspace work{storage.make<double>(m),
                                  m,
                                  storage.make<double>(n),
                                  n,
                                  storage.make<double>(3 * n),
                                  3 * n,
                                  storage.make<std::uint32_t>(1),
                                  1,
                                  23};
  auto* errors = storage.make<std::uint32_t>(systems);
  auto* error = storage.make<std::uint32_t>(1);
  for (I i = 0; i < n; ++i) {
    radius[i] = 1.1 + 0.03 * (i % 5);
    cn[i] = 0.7 + 0.07 * (i % 7);
    positions[3 * i] = 0.3 * (i % 7);
    positions[3 * i + 1] = 0.7 * (i / 7 % 5);
    positions[3 * i + 2] = 0.6 * (i / 35) + 0.03 * std::cos(i);
  }
  for (I i = 0; i < shells; ++i) {
    level[i] = -0.3 - 0.01 * (i % 7);
    cn_scale[i] = 0.01 * (i % 5 - 2);
    polynomial[i] = 0.02 * (i % 3 - 1);
  }
  for (I i = 0; i < pairs; ++i) pair_scale[i] = 0.9 + 0.01 * (i % 7);
  for (I i = 0; i < m; ++i) {
    overlap[i] = 0.02 * std::cos(i);
    density[i] = 0.005 * std::sin(i) + 0.003;
    weighted[i] = 0.01 * std::sin(0.3 * i);
  }
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const I last = atoms[bad + 1] - 1, last_shell = shell_offsets[bad + 1] - 1;
  const I last_matrix = matrices[bad + 1] - 1;
  if (fault == Fault::input) density[last_matrix] = nan;
  if (fault == Fault::parameter) pair_scale[pair_offsets[bad + 1] - 1] = nan;
  if (fault == Fault::metadata) shell_atoms[last_shell] = atoms[bad + 1];
  if (fault == Fault::position) positions[3 * last + 2] = nan;
  if (fault == Fault::overflow) overlap[last_matrix] = density[last_matrix] = 1e308;
  if (fault == Fault::coincidence)
    std::copy(positions + 3 * (last - 1), positions + 3 * last, positions + 3 * last);
  if (fault == Fault::difference) positions[3 * last + 2] = 1e308;
  if (fault == Fault::mask) requested[bad] = 2;
  if (fault == Fault::inactive) {
    requested[bad] = 0;
    density[last_matrix] = nan;
  }
  if (fault == Fault::failed_scc) {
    statuses[bad] = static_cast<generativeqc_xtb_status_t>(1);
    density[last_matrix] = nan;
  }
  auto initialize = [&] {
    std::fill(overlap_out, overlap_out + m, 0.11);
    std::fill(cn_out, cn_out + n, 0.13);
    std::fill(gradient, gradient + 3 * n, -0.17);
    std::fill(errors, errors + systems, 0);
    *error = fault == Fault::closed ? 9 : 0;
    if (fault == Fault::seed) overlap_out[last_matrix] = nan;
  };
  cudaStream_t stream;
  check(cudaStreamCreate(&stream));
  auto launch = [&] {
    check(add_gfn2_h0_pulay_gradient_cuda(batch, plan, activity, input, output, work, errors, error,
                                          stream));
  };
  initialize();
  if (graph) {
    cudaGraph_t captured;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    check(cudaStreamEndCapture(stream, &captured));
    check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) {
      initialize();
      check(cudaGraphLaunch(executable, stream));
      check(cudaStreamSynchronize(stream));
    }
    check(cudaGraphExecDestroy(executable));
    check(cudaGraphDestroy(captured));
  } else {
    launch();
    check(cudaStreamSynchronize(stream));
  }
  std::uint32_t expected = 0;
  if (fault == Fault::input) expected = 8;
  if (fault == Fault::parameter) expected = 4;
  if (fault == Fault::metadata) expected = 3;
  if (fault == Fault::position) expected = 5;
  if (fault == Fault::overflow) expected = 10;
  if (fault == Fault::coincidence) expected = 7;
  if (fault == Fault::difference) expected = 6;
  if (fault == Fault::seed) expected = 9;
  if (fault == Fault::mask) expected = 1;
  require(*error == (fault == Fault::closed ? 9 : expected), "wrong aggregate error");
  for (I system = 0; system < original_systems; ++system) {
    require(errors[system] == (system == bad ? expected : 0), "wrong system error");
    const bool preserved =
        fault == Fault::closed ||
        (system == bad && (expected || fault == Fault::inactive || fault == Fault::failed_scc));
    if (preserved) {
      for (I i = matrices[system]; i < matrices[system + 1]; ++i)
        require(fault == Fault::seed && i == last_matrix ? std::isnan(overlap_out[i])
                                                         : overlap_out[i] == 0.11,
                "failed or gated matrix published partial output");
      for (I i = atoms[system]; i < atoms[system + 1]; ++i) {
        require(cn_out[i] == 0.13, "failed or gated CN adjoint changed");
        for (int k = 0; k < 3; ++k)
          require(gradient[3 * i + k] == -0.17, "failed or gated gradient changed");
      }
      continue;
    }

    // Analytic host reference checks every healthy member, including peers of
    // a failed member. This intentionally does not use generated primal/AD code.
    std::vector<long double> cn_ref(n, 0.13), gradient_ref(3 * n, -0.17);
    const I begin = shell_offsets[system], count = sizes[system];
    const I orbital_begin = orbital_offsets[system];
    const I orbitals = orbital_offsets[system + 1] - orbital_begin;
    for (I first = begin; first < begin + count; ++first) {
      for (I second = begin; second < begin + count; ++second) {
        const I a = shell_atoms[first], b = shell_atoms[second];
        const long double average = 0.5L * (level[first] - (long double)cn_scale[first] * cn[a] +
                                            level[second] - (long double)cn_scale[second] * cn[b]);
        long double spatial = 1, radial = 0, distance = 0, delta[3]{};
        if (a != b) {
          for (int k = 0; k < 3; ++k) {
            delta[k] = (long double)positions[3 * a + k] - positions[3 * b + k];
            distance += delta[k] * delta[k];
          }
          distance = std::sqrt(distance);
          const long double reduced = std::sqrt(distance / (radius[a] + (long double)radius[b]));
          const long double x = 1 + polynomial[first] * reduced;
          const long double y = 1 + polynomial[second] * reduced;
          const long double scale =
              pair_scale[pair_offsets[system] + (first - begin) * count + second - begin];
          spatial = scale * x * y;
          radial = average * scale * (polynomial[first] * y + polynomial[second] * x) * reduced /
                   (2 * distance);
        }
        long double weight = 0;
        for (I row = shell_orbitals[first]; row < shell_orbitals[first + 1]; ++row) {
          for (I col = shell_orbitals[second]; col < shell_orbitals[second + 1]; ++col) {
            const I index =
                matrices[system] + (row - orbital_begin) * orbitals + col - orbital_begin;
            weight += (long double)density[index] * overlap[index];
            close(overlap_out[index], 0.11L - weighted[index] + density[index] * average * spatial);
          }
        }
        cn_ref[a] -= 0.5L * cn_scale[first] * spatial * weight;
        cn_ref[b] -= 0.5L * cn_scale[second] * spatial * weight;
        if (a != b)
          for (int k = 0; k < 3; ++k) {
            const long double contribution = weight * radial * delta[k] / distance;
            gradient_ref[3 * a + k] += contribution;
            gradient_ref[3 * b + k] -= contribution;
          }
      }
    }
    for (I atom = atoms[system]; atom < atoms[system + 1]; ++atom) {
      close(cn_out[atom], cn_ref[atom]);
      for (int k = 0; k < 3; ++k) close(gradient[3 * atom + k], gradient_ref[3 * atom + k]);
    }
  }
  Output result;
  result.matrix.assign(overlap_out, overlap_out + matrices[original_systems]);
  for (I i = 0; i < atoms[original_systems]; ++i) {
    result.atoms.push_back(cn_out[i]);
    for (int k = 0; k < 3; ++k) result.atoms.push_back(gradient[3 * i + k]);
  }
  result.errors.assign(errors, errors + original_systems);
  result.error = *error;
  check(cudaStreamDestroy(stream));
  return result;
}

int main() {
  try {
    for (bool graph : {false, true}) {
      for (const std::vector<I>& sizes : {std::vector<I>{1},
                                          {11},
                                          {12},
                                          {16},
                                          {17},
                                          {180},
                                          {181},
                                          {182},
                                          {257},
                                          {3, 193, 17},
                                          {1, 365, 2}})
        compare(run(sizes, false, graph, Fault::none), run(sizes, true, graph, Fault::none));
      for (Fault fault : {Fault::input, Fault::parameter, Fault::metadata, Fault::position,
                          Fault::overflow, Fault::coincidence, Fault::difference, Fault::seed,
                          Fault::mask, Fault::inactive, Fault::failed_scc, Fault::closed})
        compare(run({3, 193, 17}, false, graph, fault), run({3, 193, 17}, true, graph, fault));
    }
    std::puts(
        "H0/Pulay schedules: analytic oracle, ragged s/p/d, tails, grid strides, Graph and "
        "failures passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "H0/Pulay schedule failure: %s\n", error.what());
    return 1;
  }
}
