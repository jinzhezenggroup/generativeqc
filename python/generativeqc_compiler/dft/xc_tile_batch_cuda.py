"""Batch existing mapped FP64 bodies without changing AO or reduction domains.

Independent contractions write compact local matrices. A single CTA per spin
then visits tiles in original order: overlapping global entries never race and
no floating-point atomic, dense AO expansion, or reordered sum is introduced.
The small-domain admission bounds the deliberately serial scatter stage.
"""

from .xc_contraction_cuda import DEFAULT_XC_MATRIX_SCHEDULE, XcMatrixSchedule


def emit_native_xc_tile_batches(
    schedule: XcMatrixSchedule = DEFAULT_XC_MATRIX_SCHEDULE,
) -> str:
    """Emit wrappers around the incumbent scientific bodies and ordered scatter."""
    if not isinstance(schedule, XcMatrixSchedule):
        raise TypeError("XC tile batching requires XcMatrixSchedule")
    return _SOURCE.replace("@TILE@", str(schedule.tile))


_SOURCE = r"""
namespace generativeqc::dft::cuda_xc_detail {
namespace {
__device__ const size_t* compact_tile_ids(const CudaXcCompactTile& tile,
                                         const size_t* ids, I full_n) {
  return tile.active && tile.active < size_t(full_n) ? ids + tile.map_offset : nullptr;
}

__global__ void batch_density_products(const CudaXcCompactTile* tiles, const double* density,
    const double* ao, double* work, I full_n, I spins, I work_jets, const size_t* ids, int* error) {
  const auto& tile = tiles[blockIdx.z / (spins * work_jets)];
  if (!tile.active || blockIdx.x >= (tile.active + @TILE@ - 1) / @TILE@ ||
      blockIdx.y >= (tile.count + @TILE@ - 1) / @TILE@) return;
  tiled_density_product_body<false, true>(density, ao + tile.ao_offset, tile.active, tile.count,
      work_jets, work + tile.work_offset, error, compact_tile_ids(tile, ids, full_n),
      full_n, blockIdx.z % (spins * work_jets));
}

__global__ void batch_density_features(const CudaXcCompactTile* tiles, const double* ao,
    const double* work, double* features, I batch_begin, I spins, I jets, I work_jets,
    I terms, I functional, int* error) {
  const auto& tile = tiles[blockIdx.y];
  auto* target = features + (tile.begin - batch_begin) * spins * terms;
  if (tile.active >= 32)
    density_features_body<true>(ao + tile.ao_offset, work + tile.work_offset,
        tile.active, tile.count, spins, jets, work_jets, terms, functional, target, error);
  else
    density_features_body<false>(ao + tile.ao_offset, work + tile.work_offset,
        tile.active, tile.count, spins, jets, work_jets, terms, functional, target, error);
}

__global__ void batch_potential_panels(const CudaXcCompactTile* tiles, const double* ao,
    const double* coefficients, const double* weights, double* work, I batch_begin,
    I spins, I terms, I work_jets, int* error) {
  const auto& tile = tiles[blockIdx.y];
  compact_potential_panels_body(ao + tile.ao_offset,
      coefficients + (tile.begin - batch_begin) * spins * terms, weights + tile.begin,
      tile.active, tile.count, spins, terms, work_jets, work + tile.work_offset, error);
}

__global__ void batch_local_potentials(const CudaXcCompactTile* tiles, const double* ao,
    const double* work, double* potential, I spins, I work_jets, int* error) {
  const auto& tile = tiles[blockIdx.z / spins];
  const I columns = (tile.active + @TILE@ - 1) / @TILE@;
  if (!tile.active || blockIdx.x >= columns * (columns + 1) / 2) return;
  tiled_potential_body<true>(ao + tile.ao_offset, work + tile.work_offset, tile.active,
      tile.count, work_jets, nullptr, potential + tile.potential_offset, nullptr, false,
      error, nullptr, tile.active, blockIdx.z % spins);
}

__global__ void batch_ordered_scatter(const CudaXcCompactTile* tiles, I tile_count,
    const double* local, const double* point_totals, I batch_begin, const size_t* ids,
    I full_n, double* potential, double* totals, int* error) {
  const I spin = blockIdx.x;
  __shared__ I group_end;
  for (I tile_index = 0; tile_index < tile_count;) {
    const auto& tile = tiles[tile_index];
    const auto* map = compact_tile_ids(tile, ids, full_n);
    const I active = tile.active;
    if (threadIdx.x == 0) {
      group_end = tile_index + 1;
      for (; group_end < tile_count; ++group_end) {
        const auto& next = tiles[group_end];
        if (next.active != tile.active) break;
        const auto* next_map = compact_tile_ids(next, ids, full_n);
        bool identical = true;
        for (I column = 0; column < active && identical; ++column)
          identical = (map ? map[column] : size_t(column)) ==
                      (next_map ? next_map[column] : size_t(column));
        if (!identical) break;
      }
    }
    __syncthreads();
    // Snapshot before the final barrier: a faster lane may start discovering
    // the next group as soon as that barrier releases the current group.
    const I next_tile = group_end;
    // Only one CTA owns a spin output. The barrier separates overlapping maps
    // without serializing the independent density/potential contractions.
    for (I pair = threadIdx.x; pair < active * active; pair += blockDim.x) {
      const I mu = pair / active, nu = pair % active;
      if (mu > nu) continue;
      const I row = map ? map[mu] : mu, col = map ? map[nu] : nu;
      const I target = (spin * full_n + row) * full_n + col;
      double value = potential[target];
      // Register accumulation changes only global traffic for equal maps, not
      // any per-entry floating-point addition or the original tile ordering.
      for (I member = tile_index; member < next_tile; ++member)
        value = finite(value + local[tiles[member].potential_offset +
                                     spin * active * active + pair], error, 3);
      potential[target] = value;
      potential[(spin * full_n + col) * full_n + row] = value;
    }
    if (spin == 0 && threadIdx.x < 3) {
      const I channel = threadIdx.x;
      for (I member = tile_index; member < next_tile; ++member) {
        const auto& current = tiles[member];
        const auto* source = point_totals + 3 * (current.begin - batch_begin) + channel * current.count;
        double value = 0.0;
        for (I point = 0; point < I(current.count); ++point) value += source[point];
        totals[channel] = finite(totals[channel] + value, error, 3);
      }
    }
    __syncthreads();
    tile_index = next_tile;
  }
}

inline void launch_batch_density(cudaStream_t stream, const CudaXcLayout& layout,
    const CudaXcCompactTile* tiles, I tile_count, const double* density, const double* ao,
    double* work, double* features, I batch_begin, const size_t* ids, int* error) {
  batch_density_products<<<dim3((128 + @TILE@ - 1) / @TILE@,
      (layout.tile_points + @TILE@ - 1) / @TILE@, tile_count * layout.spins * layout.work_jets),
      dim3(@TILE@, @TILE@), 0, stream>>>(tiles, density, ao, work, layout.nao, layout.spins,
      layout.work_jets, ids, error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
  batch_density_features<<<dim3(generativeqc_tensor::blocks(layout.spins * layout.tile_points * 32,
      128), tile_count), 128, 0, stream>>>(tiles, ao, work, features, batch_begin, layout.spins,
      layout.jets, layout.work_jets, layout.feature_terms, layout.functional, error);
}

inline void launch_batch_potential(cudaStream_t stream, const CudaXcLayout& layout,
    const CudaXcCompactTile* tiles, I tile_count, const double* ao, const double* coefficients,
    const double* weights, double* work, double* local, const double* point_totals,
    I batch_begin, const size_t* ids, double* potential, double* totals, int* error) {
  batch_potential_panels<<<dim3(generativeqc_tensor::blocks(layout.spins * layout.tile_points * 128,
      128), tile_count), 128, 0, stream>>>(tiles, ao, coefficients, weights, work, batch_begin,
      layout.spins, layout.feature_terms, layout.work_jets, error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
  constexpr I columns = (128 + @TILE@ - 1) / @TILE@;
  batch_local_potentials<<<dim3(columns * (columns + 1) / 2, 1, tile_count * layout.spins),
      dim3(@TILE@, @TILE@), 0, stream>>>(tiles, ao, work, local, layout.spins, layout.work_jets, error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
  batch_ordered_scatter<<<layout.spins, 128, 0, stream>>>(tiles, tile_count, local, point_totals,
      batch_begin, ids, layout.nao, potential, totals, error);
}
}  // namespace
}  // namespace generativeqc::dft::cuda_xc_detail
"""
