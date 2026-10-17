#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "scf/df_value_storage.hpp"

namespace generativeqc::scf {

/** Preserve unsupported-domain diagnostics separately from invalid metadata or
 * an insufficient byte budget; neither permits frontend recomposition. */
class CudaDfResourceUnsupported : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

/** Expanded, scalar-only topology. Repetitions let the CLI describe a uniform
 * fleet without expanding its metadata. Python normally supplies one per row.
 * No coordinates, tensors, CUDA contexts or provider handles are borrowed. */
struct CudaDfResourceItem {
  std::uint64_t atoms, nbf, cartesian_nbf, shells, primitives;
  std::uint64_t naux, cartesian_naux, auxiliary_shells, auxiliary_primitives;
  std::uint64_t nalpha, nbeta, spins, repetitions;
};

inline constexpr std::size_t cuda_df_resource_item_width = 13;
inline constexpr std::array<const char*, 26> cuda_df_bucket_resource_keys{
    "batch",
    "nbf",
    "naux",
    "cartesian_nbf",
    "cartesian_naux",
    "coordinates",
    "source_bytes",
    "diis_device_bytes",
    "occupied",
    "rhf_occupied",
    "host_metadata",
    "preparation_minimum",
    "preparation_retained",
    "preparation_temporary",
    "response_minimum",
    "resident_response_capacity",
    "response_capacity",
    "response_host_capacity",
    "resident_host_bytes",
    "overlap_cache_host_bytes",
    "ordinary_eigen_device_bytes",
    "final_snapshot_device_bytes",
    "final_selection_host_bytes",
    "ordinary_eigen_host_workspace_bytes",
    "resident_device_bytes",
    "rhf_authorized"};

/** Numeric fields of DensityFittingResourceTile, excluding its frontend layout
 * identity and pair-storage label. Packed-single deliberately retains the
 * existing conservative two-factor shape-query bound. */
using CudaDfResourceTile = std::array<std::uint64_t, 20>;
struct CudaDfResourceBucket {
  std::array<std::uint64_t, cuda_df_bucket_resource_keys.size()> values;
  CudaDfResourceTile energy_tiles, force_tiles;
};

inline constexpr std::size_t cuda_df_resource_bucket_width =
    cuda_df_bucket_resource_keys.size() + 2 * CudaDfResourceTile{}.size();
inline constexpr std::size_t cuda_df_resource_candidate_width = 8;

struct CudaDfResourceCandidate {
  /** 0: resident, 1: source/recomputed, 2: explicit packed/resident. */
  std::uint64_t kind, sub_budget, relative_cost;
  std::uint64_t retained_host{}, workspace_host{}, retained_device{}, workspace_device{};
  std::uint64_t runtime_allowance_device{};
  std::vector<CudaDfResourceBucket> buckets{};

  std::array<std::uint64_t, cuda_df_resource_candidate_width> inventory() const noexcept {
    return {kind,           sub_budget,      relative_cost,    retained_host,
            workspace_host, retained_device, workspace_device, runtime_allowance_device};
  }
};

/** One authoritative CUDA DF bucket/fleet capacity policy for Python and CLI.
 * All retained buckets coexist; setup/force work is serialized, and a detached
 * singleton numerical retry coexists with the warm fleet. Native DF source,
 * DIIS and tile owners supply their own leaves. Metadata scratch is O(items +
 * buckets), not scientific storage. This is not complete provider/lease/RSS
 * accounting and does not change the explicit CPU numerical-recovery contract.
 * Invalid shapes, budgets and signed-int64 overflow throw before publication. */
std::vector<CudaDfResourceCandidate> cuda_df_fleet_resources(
    std::span<const CudaDfResourceItem> items, unsigned diis_history,
    std::uint64_t requested_budget, DfPairStorageRequest storage, bool occupied_exchange);

}  // namespace generativeqc::scf
