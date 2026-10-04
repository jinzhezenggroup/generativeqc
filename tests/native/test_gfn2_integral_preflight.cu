// Qualify admission/publication independently of the unchanged force algebra.
// Including the production owner exposes its private kernels without adding a
// production test API. Snapshot the seeded scratch, then replace it with a
// known downstream payload: healthy members publish it and failed members must
// preserve their original bytes. Public tblite tests gate the unchanged algebra.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "backends/cuda/gfn2_integrals.cu"

using namespace generativeqc::xtb::detail::cuda;
using I = std::int64_t;
using Error = Gfn2IntegralDeviceError;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct Storage {
  std::vector<void*> pointers;
  ~Storage() {
    for (auto p : pointers) cudaFree(p);
  }
  template <class T>
  T* make(I count) {
    T* p = nullptr;
    check(cudaMallocManaged(&p, std::max(I{1}, count) * sizeof(T)));
    pointers.push_back(p);
    std::fill(p, p + count, T{});
    return p;
  }
};
enum class Fault {
  none,
  overlap,
  dipole,
  quadrupole,
  offsets,
  shell,
  primitive,
  position,
  seed,
  mask,
  inactive,
  failed_scc,
  closed,
  prior_error
};

void run(std::vector<I> sizes, bool mixed, bool selected, bool graph, Fault fault) {
  Storage storage;
  const I systems = sizes.size();
  const I bad = std::max_element(sizes.begin(), sizes.end()) - sizes.begin();
  const I shells = std::accumulate(sizes.begin(), sizes.end(), I{0});
  auto* atoms = storage.make<I>(systems + 1);
  auto* shell_offsets = storage.make<I>(systems + 1);
  auto* orbital_offsets = storage.make<I>(systems + 1);
  auto* matrices = storage.make<I>(systems + 1);
  auto* pairs = storage.make<I>(systems + 1);
  auto* shell_orbitals = storage.make<I>(shells + 1);
  auto* primitives = storage.make<I>(shells + 1);
  auto* shell_atoms = storage.make<I>(shells);
  auto* angular = storage.make<std::uint8_t>(shells);
  auto* exponents = storage.make<double>(shells);
  auto* coefficients = storage.make<double>(shells);
  for (I s = 0; s < systems; ++s) {
    atoms[s + 1] = s + 1;
    shell_offsets[s + 1] = shell_offsets[s] + sizes[s];
    for (I shell = shell_offsets[s]; shell < shell_offsets[s + 1]; ++shell) {
      angular[shell] = mixed ? shell % 3 : 0;
      shell_orbitals[shell + 1] = shell_orbitals[shell] + 2 * angular[shell] + 1;
      primitives[shell + 1] = shell + 1;
      shell_atoms[shell] = s;
      exponents[shell] = 0.7;
      coefficients[shell] = 1.0;
    }
    orbital_offsets[s + 1] = shell_orbitals[shell_offsets[s + 1]];
    const I orbitals = orbital_offsets[s + 1] - orbital_offsets[s];
    matrices[s + 1] = matrices[s] + orbitals * orbitals;
    pairs[s + 1] = pairs[s] + sizes[s] * sizes[s];
  }
  const I matrix_count = matrices[systems];
  auto* positions = storage.make<double>(3 * systems);
  auto* overlap = storage.make<double>(matrix_count);
  auto* dipole = storage.make<double>(3 * matrix_count);
  auto* quadrupole = storage.make<double>(6 * matrix_count);
  // Finite, nonsymmetric component data exercises every scan's actual extent.
  for (I i = 0; i < matrix_count; ++i) overlap[i] = 0.001 * (i % 13 - 6);
  for (I i = 0; i < 3 * matrix_count; ++i) dipole[i] = 0.002 * (i % 17 - 8);
  for (I i = 0; i < 6 * matrix_count; ++i) quadrupole[i] = 0.003 * (i % 19 - 9);
  auto* gradients = storage.make<double>(3 * systems);
  auto* scratch = storage.make<double>(3 * systems);
  auto* seeded = storage.make<double>(3 * systems);
  auto* active = storage.make<std::uint32_t>(1);
  auto* errors = storage.make<std::uint32_t>(systems);
  auto* device_error = storage.make<std::uint32_t>(1);
  auto* requested = storage.make<std::uint8_t>(systems);
  auto* statuses = storage.make<generativeqc_xtb_status_t>(systems);
  std::fill(requested, requested + systems, 1);
  std::vector<double> seeds(3 * systems);
  for (I i = 0; i < 3 * systems; ++i) seeds[i] = 0.125 * (i + 1);
  const I last = matrices[bad + 1] - 1;
  const I last_shell = shell_offsets[bad + 1] - 1;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Error expected = Error::kSuccess;
  switch (fault) {
    case Fault::overlap:
      overlap[last] = nan;
      expected = Error::kNonfiniteAdjoint;
      break;
    case Fault::dipole:
      dipole[2 * matrix_count + last] = nan;
      expected = Error::kNonfiniteAdjoint;
      break;
    case Fault::quadrupole:
      quadrupole[5 * matrix_count + last] = nan;
      expected = Error::kNonfiniteAdjoint;
      break;
    case Fault::offsets:
      pairs[bad + 1] = std::numeric_limits<I>::max();
      expected = Error::kInvalidOffsets;
      break;
    case Fault::shell:
      angular[last_shell] = 3;
      expected = Error::kInvalidShellMetadata;
      break;
    case Fault::primitive:
      exponents[last_shell] = nan;
      expected = Error::kInvalidPrimitiveData;
      break;
    case Fault::position:
      positions[3 * bad + 2] = nan;
      expected = Error::kNonfinitePosition;
      break;
    case Fault::seed:
      seeds[3 * bad + 2] = nan;
      expected = Error::kNonfiniteGradientSeed;
      break;
    case Fault::mask:
      requested[bad] = 2;
      expected = Error::kInvalidActiveMask;
      break;
    case Fault::inactive:
      requested[bad] = 0;
      overlap[last] = nan;
      break;
    case Fault::failed_scc:
      statuses[bad] = GENERATIVEQC_XTB_STATUS_SCC_NOT_CONVERGED;
      overlap[last] = nan;
      break;
    case Fault::closed:
      overlap[last] = nan;
      break;
    case Fault::prior_error:
      overlap[last] = nan;
      expected = Error::kInvalidH0Parameter;
      break;
    case Fault::none:
      break;
  }
  Gfn2IntegralDeviceBatch batch{};
  batch.batch_size = systems;
  batch.total_atoms = systems;
  batch.total_shells = shells;
  batch.total_orbitals = orbital_offsets[systems];
  batch.total_primitives = shells;
  batch.total_matrix_elements = matrix_count;
  // Keep the true upper bound independent of the deliberately corrupted offset.
  batch.total_shell_pair_elements =
      std::accumulate(sizes.begin(), sizes.end(), I{0}, [](I sum, I n) { return sum + n * n; });
  batch.atom_offsets = atoms;
  batch.batch_shell_offsets = shell_offsets;
  batch.batch_orbital_offsets = orbital_offsets;
  batch.matrix_offsets = matrices;
  batch.shell_pair_offsets = pairs;
  batch.atom_shell_offsets = shell_offsets;
  batch.shell_orbital_offsets = shell_orbitals;
  batch.shell_primitive_offsets = primitives;
  batch.shell_to_atom = shell_atoms;
  batch.angular_momenta = angular;
  batch.primitive_exponents = exponents;
  batch.primitive_coefficients = coefficients;
  Gfn2ForceDeviceActivity activity{requested, statuses, systems, 1};
  Gfn2IntegralForceDeviceInput input{positions,    3 * systems,      overlap,
                                     matrix_count, dipole,           3 * matrix_count,
                                     quadrupole,   6 * matrix_count, 1};
  Gfn2IntegralForceDeviceOutput output{gradients, 3 * systems, 1};
  Gfn2IntegralForceDeviceWorkspace workspace{scratch, 3 * systems, active, 1, 1};
  const auto width =
      selected ? generativeqc::xtb::generated::gfn2_force_preflight_threads(matrix_count, systems)
               : 64U;
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto launch = [&] {
    capture_force_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace);
    integral_force_preflight_kernel<<<systems, width, 0, stream>>>(batch, activity, input, output,
                                                                   workspace, errors, device_error);
    check(cudaMemcpyAsync(seeded, scratch, 3 * systems * sizeof(double), cudaMemcpyDeviceToDevice,
                          stream));
    check(cudaMemsetAsync(scratch, 0, 3 * systems * sizeof(double), stream));
    publish_integral_force_kernel<<<systems, 64, 0, stream>>>(batch, activity, output, workspace,
                                                              errors);
    check(cudaGetLastError());
  };
  cudaGraph_t captured = nullptr;
  cudaGraphExec_t executable = nullptr;
  if (graph) {
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    check(cudaStreamEndCapture(stream, &captured));
    check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
  }
  for (int replay = 0; replay < 2; ++replay) {
    std::copy(seeds.begin(), seeds.end(), gradients);
    std::fill(scratch, scratch + 3 * systems, -321.0);
    std::fill(errors, errors + systems, 0u);
    if (fault == Fault::prior_error) errors[bad] = static_cast<std::uint32_t>(expected);
    *device_error = fault == Fault::closed ? 91 : 0;
    *active = 123;
    if (graph)
      check(cudaGraphLaunch(executable, stream));
    else
      launch();
    check(cudaStreamSynchronize(stream));
    for (I s = 0; s < systems; ++s) {
      const bool affected = s == bad || (fault == Fault::offsets && s == bad + 1);
      require(errors[s] == (affected ? static_cast<std::uint32_t>(expected) : 0u),
              "admission diagnostic differs from independent fixture expectation");
      const bool publishes = fault != Fault::closed && requested[s] == 1 &&
                             statuses[s] == GENERATIVEQC_XTB_STATUS_SUCCESS && errors[s] == 0;
      if (publishes) {
        for (int axis = 0; axis < 3; ++axis)
          require(gradients[3 * s + axis] == 0.0, "healthy member did not publish");
      } else {
        require(std::memcmp(gradients + 3 * s, seeds.data() + 3 * s, 3 * sizeof(double)) == 0,
                "failed or gated member published scratch");
      }
      if (!affected || fault == Fault::none) {
        if (fault != Fault::closed)
          require(std::memcmp(seeded + 3 * s, seeds.data() + 3 * s, 3 * sizeof(double)) == 0,
                  "healthy peer's complete seed was not initialized");
      }
    }
    const auto expected_device = fault == Fault::closed ? 91u
                                 : fault == Fault::prior_error
                                     ? 0u
                                     : static_cast<std::uint32_t>(expected);
    require(*device_error == expected_device, "sequence diagnostic changed");
  }
  if (executable) check(cudaGraphExecDestroy(executable));
  if (captured) check(cudaGraphDestroy(captured));
  check(cudaStreamDestroy(stream));
}

int main() {
  try {
    for (bool selected : {false, true})
      for (bool graph : {false, true}) {
        for (I n : {63, 64, 65, 257}) run({n}, false, selected, graph, Fault::none);
        run({3, 257, 17}, true, selected, graph, Fault::none);
        for (int fault = 0; fault <= static_cast<int>(Fault::prior_error); ++fault)
          run({3, 97, 17}, true, selected, graph, static_cast<Fault>(fault));
      }
    std::puts(
        "Integral preflight: widths, ragged s/p/d, failures, publication and Graph replay passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "integral preflight qualification failed: %s\n", error.what());
    return 1;
  }
}
