/** Linux/ELF qualification adapter, never linked into the production library.
 * Only LR scheduling is ablated; the primary full-range indexed owner survives.
 * Clean arms forward to the unmodified library. Intrusive receipts instantiate
 * its existing bounded kernel with its existing post-screen shell-class ledger,
 * and must not be used as endpoint timing evidence.
 */
#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

#ifdef __CUDACC__
#include "lr-domain-kernel.cuh"
#endif
#include "scf/cuda/direct_coulomb.hpp"

namespace generativeqc::scf::cuda_execution {
namespace {
bool triangular_requested() {
  const char* mode = std::getenv("GENERATIVEQC_ACCEPTANCE_LR_DOMAIN");
  if (!mode || std::strcmp(mode, "indexed") == 0) return false;
  if (std::strcmp(mode, "triangular") == 0) return true;
  throw std::runtime_error("acceptance LR domain must be indexed or triangular");
}

const char* receipt_path() {
  const char* path = std::getenv("GENERATIVEQC_ACCEPTANCE_LR_RECEIPTS");
  return path && *path ? path : nullptr;
}

void checked(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename Function>
Function next_symbol(Function current) {
  Dl_info identity{};
  if (!dladdr(reinterpret_cast<void*>(current), &identity) || !identity.dli_sname)
    throw std::runtime_error("cannot identify the acceptance interposer symbol");
  auto next = reinterpret_cast<Function>(dlsym(RTLD_NEXT, identity.dli_sname));
  if (!next)
    throw std::runtime_error(std::string("production LR symbol is not dynamically interposable: ") +
                             identity.dli_sname);
  return next;
}

struct ProfileScratch {
  DeviceShellClassProfileEntry* device{};
  ~ProfileScratch() {
    if (device) (void)cudaFree(device);
  }
};
}  // namespace

__attribute__((visibility("default"))) void launch_bounded_shell_range_exchange_derivative(
    bool unrestricted, unsigned worker_blocks, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* class_state, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DirectCoulombRange range, double omega, double exchange_coefficient,
    detail::BoundedDirectBlockDomain block_domain) {
  const bool borrowed_prefix = block_domain.prefix != nullptr;
  if (range == DirectCoulombRange::Long && triangular_requested()) block_domain = {};
  static auto original = next_symbol(&launch_bounded_shell_range_exchange_derivative);
  const char* path = receipt_path();
  if (!path || range != DirectCoulombRange::Long) {
    original(unrestricted, worker_blocks, stream, batch, screening, shell_pair_bounds,
             shell_pair_density_bounds, pair_order, shell_pair_block_bounds, system_density_bounds,
             class_state, schwarz_bounds, density, active, output, cursor, range, omega,
             exchange_coefficient, block_domain);
    return;
  }

#ifndef __CUDACC__
  throw std::runtime_error("intrusive LR work receipts require the CUDA-built adapter");
#else
  // Observational scratch is separate from, and never retained by, the owner.
  constexpr std::size_t classes = detail::kDirectQuartetShellClassCount;
  constexpr auto scratch_bytes = classes * sizeof(DeviceShellClassProfileEntry);
  ProfileScratch scratch;
  checked(cudaMalloc(reinterpret_cast<void**>(&scratch.device), scratch_bytes));
  checked(cudaMemsetAsync(scratch.device, 0, scratch_bytes, stream));
  constexpr int long_operator = static_cast<int>(DirectRangeOperator::Long);
  auto launch = [&]<bool Unrestricted>() {
    // Keep the CUDA launch tokens intact in this dual C++/CUDA translation unit.
    // clang-format off
    bounded_direct_shell_quartet_kernel<Unrestricted, DirectScreeningPurpose::Force, true, -1,
                                        long_operator>
        <<<worker_blocks, kBoundedDirectThreads, 0, stream>>>(
            batch, screening, shell_pair_bounds, shell_pair_density_bounds, pair_order,
            shell_pair_block_bounds, system_density_bounds, nullptr, 0U, class_state,
            schwarz_bounds, density, active, output, cursor, scratch.device, 0.0,
            exchange_coefficient, DirectRangeOperator::Long, omega, 0.0, false, true, block_domain);
    // clang-format on
  };
  if (unrestricted)
    launch.template operator()<true>();
  else
    launch.template operator()<false>();
  checked(cudaGetLastError());
  checked(cudaStreamSynchronize(stream));
  std::array<DeviceShellClassProfileEntry, classes> ledger{};
  unsigned long long claims{};
  checked(cudaMemcpy(ledger.data(), scratch.device, scratch_bytes, cudaMemcpyDeviceToHost));
  checked(cudaMemcpy(&claims, cursor, sizeof(claims), cudaMemcpyDeviceToHost));
  const bool indexed = block_domain.prefix != nullptr;
  const auto products =
      indexed ? block_domain.quartet_count : batch.total_shell_pair_block_quartets;
  const auto pages = indexed ? detail::kBoundedDirectIndexedCandidatePages : 1U;
  if (claims != products * pages + worker_blocks)
    throw std::runtime_error("LR did not drain the selected domain exactly");

  std::ofstream receipt(path, std::ios::app);
  if (!receipt) throw std::runtime_error("cannot retain LR work receipt");
  receipt << "{\"kind\":\"lr_work\",\"indexed\":" << indexed
          << ",\"borrowed_prefix_available\":" << borrowed_prefix
          << ",\"unrestricted\":" << unrestricted << ",\"products\":" << products
          << ",\"triangle_products\":" << batch.total_shell_pair_block_quartets
          << ",\"candidate_pages\":" << pages << ",\"cursor_claims\":" << claims
          << ",\"launch_count\":1,\"blocks\":" << worker_blocks
          << ",\"threads\":" << kBoundedDirectThreads
          << ",\"observational_scratch_bytes\":" << scratch_bytes << ",\"classes\":[";
  for (std::size_t index = 0; index < classes; ++index) {
    const auto& entry = ledger[index];
    if (index) receipt << ',';
    receipt << '[' << entry.shell_quartets << ',' << entry.tiles << ',' << entry.ao_quartets << ','
            << entry.primitive_quartets << ']';
  }
  receipt << "]}\n";
  if (!receipt) throw std::runtime_error("cannot write LR work receipt");
#endif
}

__attribute__((visibility("default"))) cudaError_t execute_generated_rsh_energy_derivatives(
    GeneratedExchangePlan& plan, bool unrestricted, const double* alpha, const double* beta,
    double coulomb_coefficient, double short_exchange_coefficient, double long_exchange_coefficient,
    double omega, std::vector<double>& derivatives) {
  static auto original = next_symbol(&execute_generated_rsh_energy_derivatives);
  const auto owner_allocations = plan.allocations.size();
  const auto shared_allocations = plan.shared->allocations.size();
  const auto owner_bytes = plan.device_bytes;
  const auto shared_bytes = plan.shared->device_bytes;
  const auto prefix = plan.bounded_block_domain.prefix;
  const auto status =
      original(plan, unrestricted, alpha, beta, coulomb_coefficient, short_exchange_coefficient,
               long_exchange_coefficient, omega, derivatives);
  if (plan.allocations.size() != owner_allocations ||
      plan.shared->allocations.size() != shared_allocations || plan.device_bytes != owner_bytes ||
      plan.shared->device_bytes != shared_bytes || plan.bounded_block_domain.prefix != prefix)
    throw std::runtime_error("LR changed the retained primary owner allocation inventory");
  if (const char* path = receipt_path()) {
    std::ofstream receipt(path, std::ios::app);
    receipt << "{\"kind\":\"owner\",\"status\":" << static_cast<int>(status)
            << ",\"prefix_available\":" << (prefix != nullptr)
            << ",\"owner_allocations\":" << owner_allocations
            << ",\"shared_allocations\":" << shared_allocations
            << ",\"owner_device_bytes\":" << owner_bytes
            << ",\"shared_device_bytes\":" << shared_bytes
            << ",\"extra_retained_index_allocations\":0}\n";
    if (!receipt) throw std::runtime_error("cannot retain owner inventory receipt");
  }
  return status;
}
}  // namespace generativeqc::scf::cuda_execution
