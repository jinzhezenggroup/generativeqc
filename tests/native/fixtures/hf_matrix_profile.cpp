#include "scf/cuda/hf_matrix_profile.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "molecule/basis.hpp"
#include "runtime/lowering_binding.hpp"
#include "scf/cuda/arena.hpp"
#include "scf/cuda/matrix_library.hpp"
#include "scf/cuda/topology.hpp"

namespace {
using namespace generativeqc;
using namespace scf::cuda_policy;
using namespace scf::cuda_execution;
std::optional<HostBatch> previous;
std::optional<tensor::CudaMatrixProductBinding> prepared_binding;
cublasHandle_t prepared_handle{};

/** Synthetic target facts qualify host contracts, never device timings. */
HfMatrixProfileContext context_for(int variant) {
  HfMatrixProfileContext context;
  context.provider = runtime::CudaProviderKind::Nvidia;
  context.source_identity.fill('a');
  context.source_identity.back() = '\0';
  std::copy_n("bb", 2, context.cublas_build_id.begin());
  std::copy_n("cc", 2, context.driver_build_id.begin());
  std::copy_n("dd", 2, context.runtime_build_id.begin());
  std::copy_n("ee", 2, context.native_build_id.begin());
  context.driver_version = 13000;
  context.runtime_version = context.toolkit_version = 12090;
  context.cublas_version = {12, 9, 1};
  context.release_build = true;
  context.target = {12,    0,       32,       1024,   2048, 32,
                    65536, 49152,   101376,   102400, 170,  std::size_t{32} << 30,
                    64,    2000000, 14000000, 512};
  if (variant == 1) context.target.multiprocessor_count = 16;
  if (variant == 2) ++context.driver_version;
  if (variant == 3) context.source_identity[0] = 'b';
  if (variant == 4) context.cublas_build_id[0] = 'e';
  if (variant == 5) context.target.total_global_memory = 4096;
  if (variant == 6) context.provider = runtime::CudaProviderKind::CuMetal;
  if (variant == 7) context.target = {};
  if (variant == 8) context.runtime_build_id[0] = 'e';
  if (variant == 9) context.driver_build_id[0] = 'e';
  if (variant == 10) context.fast_compile = true;
  if (variant == 11) context.release_build = false;
  if (variant == 12) context.cublas_build_id.fill('b');
  if (variant == 13) context.native_build_id[0] = 'a';
  return context;
}
}  // namespace

extern "C" {
void profile_forget_previous() {
  previous.reset();
  prepared_binding.reset();
}

int profile_prepare(const char* path, int dimension, int systems, int spins, int variant,
                    int provider_available, std::uint64_t* facts) {
  try {
    prepared_binding.reset();
    if (dimension <= 0 || systems <= 0 || (spins != 1 && spins != 2)) return 1;
    const auto context = context_for(variant);
    const auto profile = load_hf_matrix_profile(path, context);
    core::System system;
    system.atoms = {{2, {0., 0., 0.}}};
    for (int shell = 0; shell < dimension; ++shell)
      system.shells.push_back({0, 0, {{1. + 0.1 * shell, 1.}}});
    std::string detail;
    if (molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS) return 2;
    HostBatch host;
    if (!pack_host_batch(std::vector<core::System>(systems, system),
                         std::vector<const std::vector<double>*>(systems, nullptr), host,
                         spins == 2, false, false, ResidentPsssPolicy::Skip, HostBasisPacking::Scf,
                         context.target, {}, profile))
      return 3;
    const auto admitted = host.matrix_admission.available_if(provider_available != 0);
    const bool library = admitted.requires_provider_context();
    prepared_binding.emplace(reinterpret_cast<void*>(1),
                             library ? reinterpret_cast<void*>(2) : nullptr, dimension, systems,
                             spins);
    prepared_handle = library ? reinterpret_cast<void*>(2) : nullptr;
    const auto& binding = *prepared_binding;
    ArenaLayout layout;
    if (!make_layout(systems, dimension, dimension, systems, host.shell_atoms.size(),
                     host.shell_pair_first.size(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     host.primitive_exponents.size(), 2, 0, spins, host.persistent_eri, false,
                     false, false, false, false, false, false, layout, false, library))
      return 4;
    facts[0] = static_cast<unsigned>(profile.status);
    facts[1] = admitted.calibrated();
    facts[2] = binding.site(false, false, false, false).selected->provider == "cublas";
    facts[3] = host.persistent_eri;
    facts[4] = layout.bytes;
    facts[5] = binding.scratch_bytes();
    facts[6] = previous && same_topology(*previous, host);
    facts[7] = host.matrix_profile.has_value();
    facts[8] = admitted.requires_provider_context();
    previous = std::move(host);
    return 0;
  } catch (...) {
    return 5;
  }
}

int profile_submit(int systems, int spins, int dimension, int left_spin, int transpose,
                   int right_spin, const double* left, const double* right,
                   const std::uint8_t* active, double* output, double* scratch,
                   std::size_t capacity) {
  if (!prepared_binding) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const MatrixLibraryResources resources{reinterpret_cast<void*>(1), prepared_handle, scratch,
                                         capacity, &*prepared_binding};
  return launch_spin_matrix_product(resources, systems, spins, dimension, left, left_spin != 0,
                                    transpose != 0, right, right_spin != 0, active, output);
}
}
