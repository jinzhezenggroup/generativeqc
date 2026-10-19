"""Emit bounded source-strip reuse for exact directional residual Coulomb masks.

Primitive chunks bound shared storage independently of molecular size and
contraction length. Hermite potentials accumulate across the opposing primitive
chunk before AO back-projection; this changes FP64 summation order, not the mask.
The native callback retains the incumbent AO contraction for masks that cannot
profitably use the generated Hermite consumer.
"""

from __future__ import annotations

_SOURCE = r"""
namespace generativeqc::scf::cuda_execution {

struct MdMaskedTileProjection {
  CoulombAuxiliary<double, 3> auxiliary;
  double radial[40], bra_projected[2][40], ket_projected[2][16], factor;
};

/** The Hermite R matrix is invariant across cutoff groups and orientations.
 * Populate it with the shared native recurrence/index convention once, then
 * reuse it without expanding an AO four-center integral block. */
template <bool Reverse>
struct MdMaskedTileRadial {
  const double* values;
  __device__ double operator()(unsigned output, unsigned input) const {
    return values[Reverse ? input * 4 + output : output * 4 + input];
  }
};

union MdMaskedTileWorker {
  MdMaskedTileProjection projected;
  double incumbent[99];
};

/** Input strips own only directed entries. Their capacities reflect the two
 * different shell classes, without duplicating task-specific cutoff maps. */
struct MdMaskedTileInput {
  MdMaskedEntry* entries;
  unsigned count, capacity;
};

/** Directions/weights belong to the input strip, not to every task. Only
 * cutoff/group maps are task-specific; all four radial groups fit the worker. */
template <unsigned Capacity>
struct MdMaskedTileMask {
  static constexpr unsigned capacity = Capacity;
  const MdMaskedEntry* entries;
  unsigned count, groups, group[Capacity], cutoff[Capacity];
};

/** Two by four shell pairs share two-primitive density strips. Output values
 * survive every chunk, so primitive chunking adds no atomic publications. */
template <bool Primitive>
struct MdMaskedTileWorkspace {
  static constexpr unsigned chunk = Primitive ? 4 : 2;
  MdMaskedEntry input_entries[60];
  MdMaskedTileInput inputs[6];
  MdMaskedTileMask<18> bra[8];
  MdMaskedTileMask<6> ket[8];
  MdMaskedTileWorker workers[4];
  double bra_suffix[2][chunk][190], ket_suffix[4][chunk][28];
  double bra_potential[Primitive ? 8 : 1][Primitive ? 4 : 1][Primitive ? 40 : 1];
  double ket_potential[Primitive ? 8 : 1][Primitive ? 4 : 1][Primitive ? 16 : 1];
  double bra_values[8][18], ket_values[8][6];
  unsigned pairs[6], maximum_bra, maximum_ket;
  unsigned long long tile;
  bool active[8], admitted[8], valid[6], used[6], any_active, any_admitted;
};

static_assert(sizeof(MdMaskedTileWorkspace<true>) <= 48 * 1024);

/** Sixteen independent primitive products share each directed shell mask.
 * Shuffle reductions accumulate Hermite potentials before the separate AO
 * projection. Every half warp participates, including ragged/absent products;
 * only its source values are zeroed. The native recurrence remains the sole
 * owner of Coulomb algebra and the cutoff maps retain the FP64 AO predicate. */
__device__ __forceinline__ void md_masked_tile_primitives(
    MdMaskedTileWorkspace<true>& tile, MdJView md, unsigned bra_base, unsigned ket_base) {
  const unsigned task = threadIdx.x / 16, slot = threadIdx.x % 16;
  const unsigned bra_slot = slot / 4, ket_slot = slot % 4;
  const unsigned mask = 0xffffU << ((threadIdx.x % 32) / 16 * 16);
  MdJPair bra_pair{}, ket_pair{};
  MdJPrimitive bra{}, ket{};
  bool active = false;
  if (tile.admitted[task]) {
    bra_pair = md.pairs[tile.pairs[task / 4]];
    ket_pair = md.pairs[tile.pairs[2 + task % 4]];
    active = bra_pair.primitive_begin + bra_base + bra_slot < bra_pair.primitive_end &&
             ket_pair.primitive_begin + ket_base + ket_slot < ket_pair.primitive_end;
    if (active) {
      bra = md.primitives[bra_pair.primitive_begin + bra_base + bra_slot];
      ket = md.primitives[ket_pair.primitive_begin + ket_base + ket_slot];
    }
  }
  double radial[40];
  double factor = 0.0;
  if (active) {
    CoulombAuxiliary<double, 3> auxiliary;
    const double rho = bra.exponent * ket.exponent / (bra.exponent + ket.exponent);
    fill_coulomb<3>(rho, bra.product, ket.product, auxiliary);
    factor = 2.0 * pow(kPi, 2.5) /
        (bra.exponent * ket.exponent * sqrt(bra.exponent + ket.exponent));
    #pragma unroll
    for (unsigned output = 0; output < 10; ++output) {
      #pragma unroll
      for (unsigned input = 0; input < 4; ++input)
        radial[output * 4 + input] = MdMaskedRadial<2, 1, false>{auxiliary}(output, input);
    }
  }
  const unsigned bra_groups = tile.admitted[task] ? tile.bra[task].groups : 0;
  const unsigned ket_groups = tile.admitted[task] ? tile.ket[task].groups : 0;
  for (unsigned group = 0; group < bra_groups; ++group) {
    #pragma unroll
    for (unsigned derivative = 0; derivative < 10; ++derivative) {
      double value = 0.0;
      if (active) {
        const auto* suffix = tile.ket_suffix[task % 4][ket_slot] +
            tile.bra[task].cutoff[group] * 4;
        #pragma unroll
        for (unsigned input = 0; input < 4; ++input)
          value += radial[derivative * 4 + input] * suffix[input];
        value *= factor;
      }
      value += __shfl_down_sync(mask, value, 1, 4);
      value += __shfl_down_sync(mask, value, 2, 4);
      if (ket_slot == 0) tile.bra_potential[task][bra_slot][group * 10 + derivative] = value;
    }
  }
  for (unsigned group = 0; group < ket_groups; ++group) {
    #pragma unroll
    for (unsigned derivative = 0; derivative < 4; ++derivative) {
      double value = 0.0;
      if (active) {
        const auto* suffix = tile.bra_suffix[task / 4][bra_slot] +
            tile.ket[task].cutoff[group] * 10;
        #pragma unroll
        for (unsigned input = 0; input < 10; ++input)
          value += radial[input * 4 + derivative] * suffix[input];
        value *= factor;
      }
      value += __shfl_xor_sync(mask, value, 4, 16);
      value += __shfl_xor_sync(mask, value, 8, 16);
      if (bra_slot == 0) tile.ket_potential[task][ket_slot][group * 4 + derivative] = value;
    }
  }
  __syncthreads();
  if (tile.admitted[task]) {
    for (unsigned output = slot; output < tile.bra[task].count; output += 16) {
      double value = 0.0;
      for (unsigned primitive = 0; primitive < 4 &&
           bra_pair.primitive_begin + bra_base + primitive < bra_pair.primitive_end; ++primitive)
        value += md_masked_output<10>(tile.bra[task],
            MdMaskedTransform{md, md.primitives[bra_pair.primitive_begin + bra_base + primitive], bra_pair},
            tile.bra_potential[task][primitive], output);
      tile.bra_values[task][output] += value;
    }
    for (unsigned output = slot; output < tile.ket[task].count; output += 16) {
      double value = 0.0;
      for (unsigned primitive = 0; primitive < 4 &&
           ket_pair.primitive_begin + ket_base + primitive < ket_pair.primitive_end; ++primitive)
        value += md_masked_output<4>(tile.ket[task],
            MdMaskedTransform{md, md.primitives[ket_pair.primitive_begin + ket_base + primitive], ket_pair},
            tile.ket_potential[task][primitive], output);
      tile.ket_values[task][output] += value;
    }
  }
}

template <unsigned Capacity>
__device__ bool md_masked_tile_copy(MdMaskedTileMask<Capacity>& output,
                                    const MdMaskedTileInput& input, bool valid) {
  if (!valid || input.count > Capacity) return false;
  output.count = input.count;
  output.entries = input.entries;
  return true;
}

/** The native callback owns the unchanged AO fallback and its CUDA errors.
 * Every candidate is admitted with the incumbent directed-rounding density
 * budget. The tile is only a reuse scope, never a new screening predicate. */
template <bool CountWork, class Fallback, bool Primitive = false>
__global__ void md_masked_tiles(DeviceBatch batch, MdJView md,
                                std::size_t system_begin, std::size_t system_end,
                                bool unrestricted, double screening, const double* bounds,
                                const double* density, const double* beta, double* coulomb) {
  __shared__ __align__(8) unsigned char storage[sizeof(MdMaskedTileWorkspace<Primitive>)];
  auto& tile = *reinterpret_cast<MdMaskedTileWorkspace<Primitive>*>(storage);
  constexpr unsigned chunk = MdMaskedTileWorkspace<Primitive>::chunk;
  const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const auto bra_begin = md.active_pair_offsets[2], ket_begin = md.active_pair_offsets[1];
  const auto bra_count = md.active_pair_offsets[3] - bra_begin;
  const auto ket_count = md.active_pair_offsets[2] - ket_begin;
  const auto bra_tiles = (bra_count + 1ULL) / 2, ket_tiles = (ket_count + 3ULL) / 4;
  while (true) {
    if (threadIdx.x == 0) {
      tile.tile = atomicAdd(md.source_cursor, 1ULL);
      tile.any_active = false;
      tile.any_admitted = false;
      tile.maximum_bra = tile.maximum_ket = 0;
    }
    __syncthreads();
    if (tile.tile >= bra_tiles * ket_tiles) return;
    const auto bra_start = tile.tile / ket_tiles * 2;
    const auto ket_start = tile.tile % ket_tiles * 4;
    if (threadIdx.x < 6) {
      const unsigned index = threadIdx.x;
      tile.used[index] = false;
      tile.inputs[index].entries = tile.input_entries + (index < 2 ? index * 18 : 36 + (index - 2) * 6);
      tile.inputs[index].capacity = index < 2 ? 18 : 6;
      const auto position = index < 2 ? bra_start + index : ket_start + index - 2;
      const auto count = index < 2 ? bra_count : ket_count;
      tile.pairs[index] = position < count
          ? md.active_pairs[(index < 2 ? bra_begin : ket_begin) + position] : ~0U;
    }
    __syncthreads();
    if (threadIdx.x == 0)
      tile.any_active = Fallback::template tile_admitted<CountWork>(md, tile.pairs,
          static_cast<unsigned>(min(2ULL, bra_count - bra_start) * min(4ULL, ket_count - ket_start)));
    __syncthreads();
    const bool tile_active = tile.any_active;
    __syncthreads();
    if (!tile_active) continue;
    if (threadIdx.x == 0) tile.any_active = false;
    __syncthreads();
    if (threadIdx.x < 8) {
      const unsigned index = threadIdx.x;
      const auto bra = tile.pairs[index / 4], ket = tile.pairs[2 + index % 4];
      bool active = bra != ~0U && ket != ~0U;
      if (active) {
        active = Fallback::admitted(md, MdSourceTask{bra, ket}, system_begin, system_end, screening);
        if constexpr (CountWork) atomicAdd(&md.work_counts->residual_candidates[11], 1ULL);
      }
      tile.active[index] = active;
    }
    __syncthreads();
    if (threadIdx.x == 0)
      for (unsigned index = 0; index < 8; ++index)
        tile.any_active = tile.any_active || tile.active[index];
    __syncthreads();
    const bool any_active = tile.any_active;
    __syncthreads();
    if (!any_active) continue;
    for (unsigned index = warp; index < 6; index += 4) {
      if (lane == 0) {
        const auto pair_index = tile.pairs[index];
        tile.valid[index] = false;
        if (pair_index != ~0U) {
          const auto pair = md.pairs[pair_index];
          tile.valid[index] = md_masked_pair(tile.inputs[index], pair, batch.nbf,
              static_cast<std::size_t>(pair.system) * batch.nbf * batch.nbf,
              bounds, density, beta, unrestricted);
        }
      }
    }
    __syncthreads();
    for (unsigned index = warp; index < 8; index += 4) {
      if (lane == 0) {
        tile.admitted[index] = false;
        if (tile.active[index]) {
          const auto bra_pair = md.pairs[tile.pairs[index / 4]];
          const auto ket_pair = md.pairs[tile.pairs[2 + index % 4]];
          bool admitted = std::isfinite(screening) && screening >= 0.0 &&
              md_masked_tile_copy(tile.bra[index], tile.inputs[index / 4], tile.valid[index / 4]) &&
              md_masked_tile_copy(tile.ket[index], tile.inputs[2 + index % 4], tile.valid[2 + index % 4]);
          if (admitted) {
            md_masked_cutoffs(tile.bra[index], tile.ket[index], screening);
            md_masked_cutoffs(tile.ket[index], tile.bra[index], screening);
            admitted = md_masked_work_admitted<10, 4>(
                tile.bra[index].groups + tile.ket[index].groups,
                bra_pair.first_count * bra_pair.second_count,
                ket_pair.first_count * ket_pair.second_count, 4);
          }
          tile.admitted[index] = admitted;
          if constexpr (CountWork) {
            const auto products = static_cast<unsigned long long>(bra_pair.primitive_end - bra_pair.primitive_begin) *
                (ket_pair.primitive_end - ket_pair.primitive_begin);
            atomicAdd(&md.work_counts->residual_tasks[11], 1ULL);
            atomicAdd(&md.work_counts->residual_roots[11], products);
            if (admitted) {
              const auto groups = tile.bra[index].groups + tile.ket[index].groups;
              atomicAdd(&md.work_counts->masked_tasks[11], 1ULL);
              atomicAdd(&md.work_counts->masked_groups[11], static_cast<unsigned long long>(groups));
              atomicAdd(&md.work_counts->masked_summands[11], products * groups * 40);
              unsigned bra_outputs = 0, ket_outputs = 0;
              for (unsigned output = 0; output < tile.bra[index].count; ++output)
                bra_outputs += tile.bra[index].group[output] != 18;
              for (unsigned output = 0; output < tile.ket[index].count; ++output)
                ket_outputs += tile.ket[index].group[output] != 6;
              const auto bra_primitives = static_cast<unsigned long long>(bra_pair.primitive_end - bra_pair.primitive_begin);
              const auto ket_primitives = static_cast<unsigned long long>(ket_pair.primitive_end - ket_pair.primitive_begin);
              atomicAdd(&md.work_counts->masked_output_summands[11],
                  bra_primitives * ((ket_primitives + chunk - 1) / chunk) * bra_outputs * 10 +
                  ket_primitives * ((bra_primitives + chunk - 1) / chunk) * ket_outputs * 4);
              atomicAdd(&md.work_counts->masked_unfused_output_summands[11],
                  products * (bra_outputs * 10 + ket_outputs * 4));
            }
          }
        }
      }
      __syncwarp();
      if (!tile.active[index]) continue;
      if (!tile.admitted[index]) {
        Fallback{}(batch, md, MdSourceTask{tile.pairs[index / 4], tile.pairs[2 + index % 4]},
            unrestricted, screening, bounds, density, beta, coulomb,
            tile.workers[warp].incumbent);
      } else {
        if (lane < tile.bra[index].count) tile.bra_values[index][lane] = 0.0;
        if (lane < tile.ket[index].count) tile.ket_values[index][lane] = 0.0;
      }
      __syncwarp();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (unsigned index = 0; index < 8; ++index) {
        if (!tile.admitted[index]) continue;
        tile.any_admitted = true;
        tile.used[index / 4] = tile.used[2 + index % 4] = true;
      }
      for (unsigned index = 0; index < 6; ++index) {
        if (!tile.used[index]) continue;
        const auto pair = md.pairs[tile.pairs[index]];
        const auto primitives = pair.primitive_end - pair.primitive_begin;
        if (index < 2) tile.maximum_bra = max(tile.maximum_bra, primitives);
        else tile.maximum_ket = max(tile.maximum_ket, primitives);
      }
    }
    __syncthreads();
    const bool any_admitted = tile.any_admitted;
    __syncthreads();
    if (!any_admitted) continue;
    for (unsigned bra_base = 0; bra_base < tile.maximum_bra; bra_base += chunk) {
      if (warp < 2 && tile.used[warp]) {
        const auto pair = md.pairs[tile.pairs[warp]];
        for (unsigned slot = 0; slot < chunk && pair.primitive_begin + bra_base + slot < pair.primitive_end; ++slot) {
          const auto primitive = md.primitives[pair.primitive_begin + bra_base + slot];
          if (lane < 10) md_masked_suffix<10>(tile.inputs[warp], MdMaskedTransform{md, primitive, pair},
              lane, tile.bra_suffix[warp][slot]);
          if constexpr (CountWork) if (lane == 0) {
            atomicAdd(&md.work_counts->preproject_primitives[2], 1ULL);
            atomicAdd(&md.work_counts->preproject_summands[2], 10ULL * tile.inputs[warp].count);
          }
        }
      }
      __syncthreads();
      for (unsigned ket_base = 0; ket_base < tile.maximum_ket; ket_base += chunk) {
        if (tile.used[2 + warp]) {
          const auto pair = md.pairs[tile.pairs[2 + warp]];
          for (unsigned slot = 0; slot < chunk && pair.primitive_begin + ket_base + slot < pair.primitive_end; ++slot) {
            const auto primitive = md.primitives[pair.primitive_begin + ket_base + slot];
            if (lane < 4) md_masked_suffix<4>(tile.inputs[2 + warp], MdMaskedTransform{md, primitive, pair},
                lane, tile.ket_suffix[warp][slot]);
            if constexpr (CountWork) if (lane == 0) {
              atomicAdd(&md.work_counts->preproject_primitives[1], 1ULL);
              atomicAdd(&md.work_counts->preproject_summands[1], 4ULL * tile.inputs[2 + warp].count);
            }
          }
        }
        __syncthreads();
        if constexpr (Primitive) {
          md_masked_tile_primitives(tile, md, bra_base, ket_base);
        } else for (unsigned index = warp; index < 8; index += 4) {
          if (!tile.admitted[index]) continue;
          const auto bra_pair = md.pairs[tile.pairs[index / 4]];
          const auto ket_pair = md.pairs[tile.pairs[2 + index % 4]];
          if (bra_pair.primitive_begin + bra_base >= bra_pair.primitive_end ||
              ket_pair.primitive_begin + ket_base >= ket_pair.primitive_end) continue;
          auto& worker = tile.workers[warp].projected;
          for (unsigned slot = 0; slot < 2; ++slot) {
            for (unsigned element = lane; element < tile.bra[index].groups * 10; element += 32)
              worker.bra_projected[slot][element] = 0.0;
            for (unsigned element = lane; element < tile.ket[index].groups * 4; element += 32)
              worker.ket_projected[slot][element] = 0.0;
          }
          __syncwarp();
          for (unsigned bra_slot = 0; bra_slot < 2 && bra_pair.primitive_begin + bra_base + bra_slot < bra_pair.primitive_end; ++bra_slot) {
            const auto bra = md.primitives[bra_pair.primitive_begin + bra_base + bra_slot];
            for (unsigned ket_slot = 0; ket_slot < 2 && ket_pair.primitive_begin + ket_base + ket_slot < ket_pair.primitive_end; ++ket_slot) {
              const auto ket = md.primitives[ket_pair.primitive_begin + ket_base + ket_slot];
              if (lane == 0) {
                const double rho = bra.exponent * ket.exponent / (bra.exponent + ket.exponent);
                fill_coulomb<3>(rho, bra.product, ket.product, worker.auxiliary);
                worker.factor = 2.0 * pow(kPi, 2.5) /
                    (bra.exponent * ket.exponent * sqrt(bra.exponent + ket.exponent));
              }
              __syncwarp();
              for (unsigned element = lane; element < 40; element += 32)
                worker.radial[element] = MdMaskedRadial<2, 1, false>{worker.auxiliary}(element / 4, element % 4);
              __syncwarp();
              for (unsigned element = lane; element < tile.bra[index].groups * 10; element += 32)
                worker.bra_projected[bra_slot][element] += worker.factor * md_masked_project<10, 4>(tile.bra[index],
                    tile.ket_suffix[index % 4][ket_slot], element,
                    MdMaskedTileRadial<false>{worker.radial});
              for (unsigned element = lane; element < tile.ket[index].groups * 4; element += 32)
                worker.ket_projected[ket_slot][element] += worker.factor * md_masked_project<4, 10>(tile.ket[index],
                    tile.bra_suffix[index / 4][bra_slot], element,
                    MdMaskedTileRadial<true>{worker.radial});
              __syncwarp();
            }
          }
          for (unsigned slot = 0; slot < 2; ++slot) {
            if (lane < tile.bra[index].count && bra_pair.primitive_begin + bra_base + slot < bra_pair.primitive_end)
              tile.bra_values[index][lane] += md_masked_output<10>(tile.bra[index],
                  MdMaskedTransform{md, md.primitives[bra_pair.primitive_begin + bra_base + slot], bra_pair},
                  worker.bra_projected[slot], lane);
            if (lane < tile.ket[index].count && ket_pair.primitive_begin + ket_base + slot < ket_pair.primitive_end)
              tile.ket_values[index][lane] += md_masked_output<4>(tile.ket[index],
                  MdMaskedTransform{md, md.primitives[ket_pair.primitive_begin + ket_base + slot], ket_pair},
                  worker.ket_projected[slot], lane);
          }
          __syncwarp();
        }
        __syncthreads();
      }
    }
    for (unsigned index = warp; index < 8; index += 4) {
      if (!tile.admitted[index]) continue;
      const auto pair = md.pairs[tile.pairs[index / 4]];
      const auto offset = static_cast<std::size_t>(pair.system) * batch.nbf * batch.nbf;
      if (lane < tile.bra[index].count && tile.bra[index].group[lane] != 18) {
        const auto entry = tile.bra[index].entries[lane];
        atomicAdd(coulomb + offset + entry.first * batch.nbf + entry.second, tile.bra_values[index][lane]);
      }
      if (lane < tile.ket[index].count && tile.ket[index].group[lane] != 6) {
        const auto entry = tile.ket[index].entries[lane];
        atomicAdd(coulomb + offset + entry.first * batch.nbf + entry.second, tile.ket_values[index][lane]);
      }
      __syncwarp();
    }
    __syncthreads();
  }
}

}  // namespace generativeqc::scf::cuda_execution
"""


def emit_md_j_masked_tile() -> str:
    """Return the CUDA-only extension after the common masked math definitions."""
    return _SOURCE
