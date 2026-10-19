"""Emit a bounded compacted source stream for exact masked-Hermite Coulomb.

Candidate scanning fills a small shared queue across multiple admission rounds.
One thread then owns an admitted shell task, including all its primitive
products. Unlike a padded primitive rectangle, sparse admission and short
contractions do not reserve most lanes for absent work.
"""

from __future__ import annotations


def _emit_projections() -> str:
    """Specialize only scheduling/indexing; native radial algebra owns signs."""
    lines = [
        "namespace generativeqc::scf::cuda_execution {",
        "template <unsigned KetAngular> struct MdMaskedPackedProjection;",
    ]
    for angular, hermites in ((1, 4), (2, 10)):
        lines.append(f"template <> struct MdMaskedPackedProjection<{angular}> {{")
        for reverse in (False, True):
            name = "reverse" if reverse else "forward"
            lines.extend(
                (
                    "  template <class Auxiliary>",
                    f"  __device__ static void {name}(Auxiliary& auxiliary, const double* selected,",
                    "                                double* projected, double factor) {",
                    f"    const MdMaskedRadial<2, {angular}, false> radial{{auxiliary}};",
                )
            )
            for output in range(hermites if reverse else 10):
                lines.extend(("    {", "      double value = 0.0;"))
                for source in range(10 if reverse else hermites):
                    bra, ket = (source, output) if reverse else (output, source)
                    lines.append(
                        f"      value += radial({bra}, {ket}) * selected[{source}];"
                    )
                expression = "value" if reverse else "factor * value"
                operator = "=" if reverse else "+="
                lines.extend(
                    (f"      projected[{output}] {operator} {expression};", "    }")
                )
            lines.append("  }")
        lines.append("};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def emit_md_j_masked_packed() -> str:
    """Reuse the native recurrence and mask algebra in a thread-private schedule."""
    return (
        _emit_projections()
        + r"""
namespace generativeqc::scf::cuda_execution {

/** Private plans keep independent directed masks and output publications.
 * Only requested density tails are built. Bra potentials survive the entire
 * opposing contraction, rather than a fixed two/four-primitive chunk. */
template <unsigned KetAngular, bool CountWork>
__device__ bool md_masked_packed_contract(
    DeviceBatch batch, MdJView md, MdSourceTask task, bool unrestricted,
    double screening, const double* bounds, const double* density,
    const double* beta, double* coulomb) {
  static_assert(KetAngular == 1 || KetAngular == 2);
  const auto bra_pair = md.pairs[task.bra], ket_pair = md.pairs[task.ket];
  constexpr unsigned ket_hermites = md_j_hermite_count(KetAngular);
  constexpr unsigned ket_capacity = KetAngular == 1 ? 6 : 18;
  constexpr unsigned maximum_groups = KetAngular == 1 ? 4 : 8;
  constexpr unsigned angular = 10 + KetAngular;
  const bool reverse = task.bra != task.ket;
  const auto offset = static_cast<std::size_t>(bra_pair.system) * batch.nbf * batch.nbf;
  MdMaskedPlan<18> bra_plan;
  MdMaskedPlan<ket_capacity> ket_plan;
  if (!std::isfinite(screening) || screening < 0.0 ||
      !md_masked_pair(bra_plan, bra_pair, batch.nbf, offset, bounds, density, beta, unrestricted) ||
      !md_masked_pair(ket_plan, ket_pair, batch.nbf, offset, bounds, density, beta, unrestricted))
    return false;
  md_masked_cutoffs(bra_plan, ket_plan, screening);
  md_masked_cutoffs(ket_plan, bra_plan, screening);
  const unsigned groups = bra_plan.groups + (reverse ? ket_plan.groups : 0);
  if (!md_masked_work_admitted<10, ket_hermites>(groups,
          bra_pair.first_count * bra_pair.second_count,
          ket_pair.first_count * ket_pair.second_count, maximum_groups)) return false;
  double bra_values[18]{}, ket_values[ket_capacity]{};
  if constexpr (CountWork) {
    const auto bra_primitives = static_cast<unsigned long long>(bra_pair.primitive_end - bra_pair.primitive_begin);
    const auto ket_primitives = static_cast<unsigned long long>(ket_pair.primitive_end - ket_pair.primitive_begin);
    const auto products = bra_primitives * ket_primitives;
    atomicAdd(&md.work_counts->masked_tasks[angular], 1ULL);
    atomicAdd(&md.work_counts->masked_groups[angular], static_cast<unsigned long long>(groups));
    atomicAdd(&md.work_counts->masked_summands[angular], products * groups * 10 * ket_hermites);
    unsigned bra_outputs = 0, ket_outputs = 0;
    for (unsigned output = 0; output < bra_plan.count; ++output)
      bra_outputs += bra_plan.group[output] != 18;
    for (unsigned output = 0; output < ket_plan.count; ++output)
      ket_outputs += reverse && ket_plan.group[output] != ket_capacity;
    atomicAdd(&md.work_counts->masked_output_summands[angular],
        bra_primitives * bra_outputs * 10 + products * ket_outputs * ket_hermites);
    atomicAdd(&md.work_counts->masked_unfused_output_summands[angular],
        products * (bra_outputs * 10 + ket_outputs * ket_hermites));
  }
  for (auto bra_index = bra_pair.primitive_begin; bra_index < bra_pair.primitive_end; ++bra_index) {
    const auto bra = md.primitives[bra_index];
    const MdMaskedTransform bra_transform{md, bra, bra_pair};
    // Combined-group admission leaves room for reverse density channels in
    // this same bounded array; those channels do not receive bra potentials.
    double bra_projected[maximum_groups * 10]{};
    for (unsigned group = 0; reverse && group < ket_plan.groups; ++group) {
      #pragma unroll
      for (unsigned input = 0; input < 10; ++input)
        bra_projected[(bra_plan.groups + group) * 10 + input] =
            md_masked_tail<10>(bra_plan, bra_transform, input, ket_plan.cutoff[group]);
    }
    for (auto ket_index = ket_pair.primitive_begin; ket_index < ket_pair.primitive_end; ++ket_index) {
      const auto ket = md.primitives[ket_index];
      const MdMaskedTransform ket_transform{md, ket, ket_pair};
      CoulombAuxiliary<double, 2 + KetAngular> auxiliary;
      const double rho = bra.exponent * ket.exponent / (bra.exponent + ket.exponent);
      fill_coulomb<2 + KetAngular>(rho, bra.product, ket.product, auxiliary);
      const double factor = 2.0 * pow(kPi, 2.5) /
          (bra.exponent * ket.exponent * sqrt(bra.exponent + ket.exponent));
      for (unsigned group = 0; group < bra_plan.groups; ++group) {
        double selected[ket_hermites];
        #pragma unroll
        for (unsigned derivative = 0; derivative < ket_hermites; ++derivative)
          selected[derivative] = md_masked_tail<ket_hermites>(ket_plan, ket_transform,
              derivative, bra_plan.cutoff[group]);
        MdMaskedPackedProjection<KetAngular>::forward(auxiliary, selected,
            bra_projected + group * 10, factor);
      }
      for (unsigned group = 0; reverse && group < ket_plan.groups; ++group) {
        double projected[ket_hermites]{};
        MdMaskedPackedProjection<KetAngular>::reverse(auxiliary,
            bra_projected + (bra_plan.groups + group) * 10, projected, factor);
        for (unsigned output = 0; output < ket_plan.count; ++output) {
          if (ket_plan.group[output] != group) continue;
          double value = 0.0;
          #pragma unroll
          for (unsigned derivative = 0; derivative < ket_hermites; ++derivative)
            value += ket_transform(ket_plan.entries[output].component, derivative) * projected[derivative];
          ket_values[output] += factor * value;
        }
      }
    }
    for (unsigned output = 0; output < bra_plan.count; ++output)
      bra_values[output] += md_masked_output<10>(bra_plan, bra_transform, bra_projected, output);
  }
  for (unsigned output = 0; output < bra_plan.count; ++output) {
    if (bra_plan.group[output] == 18) continue;
    const auto entry = bra_plan.entries[output];
    atomicAdd(coulomb + offset + entry.first * batch.nbf + entry.second, bra_values[output]);
  }
  for (unsigned output = 0; reverse && output < ket_plan.count; ++output) {
    if (ket_plan.group[output] == ket_capacity) continue;
    const auto entry = ket_plan.entries[output];
    atomicAdd(coulomb + offset + entry.first * batch.nbf + entry.second, ket_values[output]);
  }
  return true;
}

/** Accumulate admitted tasks across candidate batches before consuming them.
 * The queue cannot exceed 255 entries: append at most 128 only below 128.
 * Leftovers remain local to the block; no molecular-size queue is retained.
 * Each admitted task is evaluated once, including scalar native fallbacks. */
template <unsigned KetAngular, bool CountWork, class Fallback>
__global__ void md_masked_packed_sources(
    DeviceBatch batch, MdJView md, std::size_t system_begin, std::size_t system_end,
    bool unrestricted, double screening, const double* bounds, const double* density,
    const double* beta, double* coulomb) {
  static_assert(KetAngular == 1 || KetAngular == 2);
  __shared__ MdSourceTask tasks[256];
  __shared__ unsigned queued;
  __shared__ unsigned long long candidate_base;
  __shared__ bool exhausted;
  constexpr unsigned angular = 10 + KetAngular;
  const auto bra_begin = md.active_pair_offsets[2], ket_begin = md.active_pair_offsets[KetAngular];
  const auto bra_count = md.active_pair_offsets[3] - bra_begin;
  const auto ket_count = md.active_pair_offsets[KetAngular + 1] - ket_begin;
  const auto candidates = KetAngular == 2
      ? static_cast<unsigned long long>(bra_count) * (bra_count + 1ULL) / 2
      : static_cast<unsigned long long>(bra_count) * ket_count;
  const auto units = md.residual_tiled
      ? md_source_tile_count(bra_count, ket_count, KetAngular == 2) : candidates;
  const unsigned stride = md.residual_tiled ? 2 : 128;
  const double allowance = __ddiv_rd(fmin(screening, kMdJDensityErrorCap), double(md.pair_count));
  const unsigned lane = threadIdx.x % 32;
  if (threadIdx.x == 0) { queued = 0; exhausted = false; }
  __syncthreads();
  while (true) {
    while (queued < 128 && !exhausted) {
      if (threadIdx.x == 0)
        candidate_base = atomicAdd(md.source_cursor, static_cast<unsigned long long>(stride));
      __syncthreads();
      MdSourceTile tile{};
      unsigned local_pair = ~0U;
      bool tile_kept = false;
      if (md.residual_tiled && threadIdx.x < 64 && candidate_base + threadIdx.x / 32 < units) {
        if (lane == 0)
          tile = md_source_tile(candidate_base + threadIdx.x / 32, ket_count, KetAngular == 2);
        tile.bra = __shfl_sync(0xffffffffU, tile.bra, 0);
        tile.ket = __shfl_sync(0xffffffffU, tile.ket, 0);
        const bool bra_side = lane < kMdSourceTileWidth;
        const auto position = bra_side ? tile.bra + lane : tile.ket + lane - kMdSourceTileWidth;
        const bool valid = lane < 2 * kMdSourceTileWidth && position < (bra_side ? bra_count : ket_count);
        double minimum = INFINITY, maximum = 0.0, density_maximum = 0.0;
        if (valid) {
          local_pair = md.active_pairs[(bra_side ? bra_begin : ket_begin) + position];
          minimum = md.minimum_bounds[local_pair];
          maximum = md.maximum_bounds[local_pair];
          density_maximum = md.density_bounds[local_pair];
        }
        for (unsigned step = kMdSourceTileWidth / 2; step; step /= 2) {
          minimum = fmin(minimum, __shfl_down_sync(0xffffffffU, minimum, step, kMdSourceTileWidth));
          maximum = fmax(maximum, __shfl_down_sync(0xffffffffU, maximum, step, kMdSourceTileWidth));
          density_maximum = fmax(density_maximum,
              __shfl_down_sync(0xffffffffU, density_maximum, step, kMdSourceTileWidth));
        }
        const double ket_minimum = __shfl_sync(0xffffffffU, minimum, kMdSourceTileWidth);
        const double ket_maximum = __shfl_sync(0xffffffffU, maximum, kMdSourceTileWidth);
        const double ket_density = __shfl_sync(0xffffffffU, density_maximum, kMdSourceTileWidth);
        if (lane == 0) {
          const double bound = __dmul_ru(maximum, ket_maximum);
          tile_kept = !md_source_tile_rejected(bound, __dmul_rd(minimum, ket_minimum),
              __dmul_ru(bound, density_maximum), __dmul_ru(bound, ket_density),
              density_maximum == 0.0 && ket_density == 0.0, screening, allowance);
          if constexpr (CountWork) {
            atomicAdd(&md.work_counts->residual_tiles[angular], 1ULL);
            if (!tile_kept) {
              const auto bra_size = min(kMdSourceTileWidth, bra_count - tile.bra);
              const auto ket_size = min(kMdSourceTileWidth, ket_count - tile.ket);
              const auto count = KetAngular == 2 && tile.bra == tile.ket
                  ? bra_size * (bra_size + 1) / 2 : bra_size * ket_size;
              atomicAdd(&md.work_counts->residual_screened_tiles[angular], 1ULL);
              atomicAdd(&md.work_counts->residual_screened_candidates[angular],
                  static_cast<unsigned long long>(count));
            }
          }
        }
        tile_kept = __shfl_sync(0xffffffffU, tile_kept, 0);
      }
      // A warp retains its 16 pair IDs in registers and scans both 32-task
      // halves. At most two warps append 64 tasks each before the next barrier.
      for (unsigned half = 0; half < (md.residual_tiled ? 2U : 1U); ++half) {
        MdSourceTask task{};
        bool valid = false;
        if (md.residual_tiled) {
          if (tile_kept) {
            const unsigned bra_slot = half * 4 + lane / kMdSourceTileWidth;
            const unsigned ket_slot = lane % kMdSourceTileWidth;
            const unsigned bra_pair = __shfl_sync(0xffffffffU, local_pair, bra_slot);
            const unsigned ket_pair = __shfl_sync(0xffffffffU, local_pair, kMdSourceTileWidth + ket_slot);
            const auto bra = tile.bra + bra_slot, ket = tile.ket + ket_slot;
            valid = bra < bra_count && ket < ket_count && (KetAngular != 2 || bra >= ket);
            if (valid) task = {bra_pair, ket_pair};
          }
        } else {
          const auto candidate = candidate_base + threadIdx.x;
          if (candidate < candidates) {
            valid = true;
            if constexpr (KetAngular == 2) {
              auto bra = static_cast<unsigned long long>((sqrt(8.0 * candidate + 1.0) - 1.0) * 0.5);
              while (bra * (bra + 1ULL) / 2 > candidate) --bra;
              while ((bra + 1ULL) * (bra + 2ULL) / 2 <= candidate) ++bra;
              const auto ket = candidate - bra * (bra + 1ULL) / 2;
              task = {md.active_pairs[bra_begin + bra], md.active_pairs[ket_begin + ket]};
            } else {
              task = {md.active_pairs[bra_begin + candidate / ket_count],
                      md.active_pairs[ket_begin + candidate % ket_count]};
            }
          }
        }
        bool accepted = false;
        if (valid) {
          accepted = Fallback::admitted(md, task, system_begin, system_end, screening);
          if constexpr (CountWork) atomicAdd(&md.work_counts->residual_candidates[angular], 1ULL);
        }
        const unsigned ballot = __ballot_sync(0xffffffffU, accepted);
        if (ballot) {
          const unsigned leader = __ffs(ballot) - 1;
          unsigned base = lane == leader ? atomicAdd(&queued, __popc(ballot)) : 0;
          base = __shfl_sync(0xffffffffU, base, leader);
          if (accepted) tasks[base + __popc(ballot & ((1U << lane) - 1))] = task;
        }
      }
      if (threadIdx.x == 0 && candidate_base + stride >= units) exhausted = true;
      __syncthreads();
    }
    const unsigned consuming = min(queued, 128U);
    if (threadIdx.x < consuming) {
      const auto task = tasks[threadIdx.x];
      if constexpr (CountWork) {
        const auto bra = md.pairs[task.bra], ket = md.pairs[task.ket];
        atomicAdd(&md.work_counts->residual_tasks[angular], 1ULL);
        atomicAdd(&md.work_counts->residual_roots[angular],
            static_cast<unsigned long long>(bra.primitive_end - bra.primitive_begin) *
                (ket.primitive_end - ket.primitive_begin));
      }
      if (!md_masked_packed_contract<KetAngular, CountWork>(batch, md, task, unrestricted,
              screening, bounds, density, beta, coulomb))
        Fallback::scalar(batch, md, task, unrestricted, screening, bounds, density, beta, coulomb);
    }
    __syncthreads();
    if (threadIdx.x < queued - consuming) tasks[threadIdx.x] = tasks[consuming + threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) queued -= consuming;
    __syncthreads();
    if (exhausted && queued == 0) return;
  }
}
}  // namespace generativeqc::scf::cuda_execution
"""
    )
