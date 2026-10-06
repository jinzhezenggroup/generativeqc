"""Lower the existing Coulomb DAG to a bounded cooperative CUDA schedule.

The compiler owns node dependencies and component/state mappings. CUDA lanes
execute independent nodes between publication barriers; no scalar recurrence
or derivative algebra is reimplemented for this schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from .shell_class import build_coulomb_derivative_algebra


@dataclass(frozen=True, slots=True)
class CooperativeCoulombSchedule:
    """Packed DAG instructions, level boundaries, leaves and spatial outputs.

    An instruction packs two 16-bit operands and a two-bit opcode. Operands
    address constants/leaves for those opcodes, otherwise prior DAG values.
    Root mappings pack x/y/z degrees in three nibbles and the value index above
    them. Bounds are checked before emission, never truncated by the backend.
    """

    instructions: tuple[int, ...]
    levels: tuple[int, ...]
    constants: tuple[float, ...]
    variables: tuple[str, ...]
    roots: tuple[int, ...]


@cache
def build_cooperative_coulomb_schedule() -> CooperativeCoulombSchedule:
    """Schedule the authoritative degree-eight spatial DAG for order 6/7 forces."""
    algebra = build_coulomb_derivative_algebra(8)
    graph = algebra.graph
    order = graph.topological_order(tuple(root for _, root in algebra.roots))
    depths: dict[int, int] = {}
    for identifier in order:
        depths[identifier] = max(
            (depths[argument] + 1 for argument in graph.nodes[identifier].arguments),
            default=0,
        )
    order.sort(key=lambda identifier: (depths[identifier], identifier))
    index = {identifier: position for position, identifier in enumerate(order)}
    variables = ("rho", "difference_x", "difference_y", "difference_z") + tuple(
        f"boys_{n}" for n in range(9)
    )
    constants: list[float] = []
    instructions: list[int] = []
    levels = [0]
    previous_depth = 0
    for identifier in order:
        node = graph.nodes[identifier]
        if depths[identifier] != previous_depth:
            levels.append(len(instructions))
            previous_depth = depths[identifier]
        if node.operation == "constant":
            opcode, first, second = 0, len(constants), 0
            if node.payload is None:
                raise ValueError("cooperative DAG constant is missing its value")
            constants.append(float(node.payload))
        elif node.operation == "variable":
            opcode, first, second = 1, variables.index(str(node.payload)), 0
        elif node.operation in ("multiply", "add"):
            opcode = 2 if node.operation == "multiply" else 3
            first, second = (index[argument] for argument in node.arguments)
        else:
            raise ValueError(f"unsupported cooperative DAG operation: {node.operation}")
        if not 0 <= first < 65536 or not 0 <= second < 65536:
            raise ValueError("cooperative DAG operand exceeds packed instruction")
        instructions.append(first | (second << 16) | (opcode << 32))
    levels.append(len(instructions))
    if len(instructions) >= 4096:
        raise ValueError("cooperative DAG exceeds packed spatial output mapping")
    roots = tuple(
        x | (y << 4) | (z << 8) | (index[root.identifier] << 12)
        for (x, y, z), root in algebra.roots
    )
    return CooperativeCoulombSchedule(
        tuple(instructions), tuple(levels), tuple(constants), variables, roots
    )


def emit_cooperative_direct_force_support() -> str:
    """Emit bounded shared execution; only order-six/seven s/p/d tasks are admitted."""
    schedule = build_cooperative_coulomb_schedule()
    instructions = ",\n".join(
        "  " + ", ".join(f"{value}ULL" for value in schedule.instructions[i : i + 8])
        for i in range(0, len(schedule.instructions), 8)
    )
    roots = ", ".join(f"{value}U" for value in schedule.roots)
    levels = ", ".join(f"{value}U" for value in schedule.levels)
    constants = ", ".join(repr(value) for value in schedule.constants)
    prefix = f"""
#if defined(__CUDACC__)
namespace generativeqc::scf::cuda_execution {{
// Lane-varying instruction/output tables use coalesced read-only global loads.
// Uniform level/constant reads use constant storage. The kernel does not carry
// scalar DAG roots or per-component recurrence arrays in private storage.
static __device__ const unsigned long long kCooperativeCoulombNodes[] = {{
{instructions}
}};
static __device__ __constant__ unsigned kCooperativeCoulombLevels[] = {{{levels}}};
static __device__ const unsigned kCooperativeCoulombRoots[] = {{{roots}}};
static __device__ __constant__ double kCooperativeCoulombConstants[] = {{{constants}}};
struct CooperativeDirectPairDerivativeRecurrence {{
  MaterializedDirectPairDerivativeRecurrence recurrence;
  double nodes[{len(schedule.instructions)}], boys[9], rho;
  Vec3<double> difference;
  double warp_sums[8];
}};
static_assert(sizeof(CooperativeDirectPairDerivativeRecurrence) <= (44U << 10));
"""
    return (
        prefix
        + _CONSUMER
        + "\n}  // namespace generativeqc::scf::cuda_execution\n#endif\n"
    )


_CONSUMER = r"""
/** Evaluate the existing spatial DAG once per primitive product. Independent
 * nodes are lane-strided within a level. A CTA barrier publishes every level,
 * including levels with fewer nodes than lanes. Only n=0 spatial states are
 * consumed by Cartesian contraction and its raised first-response view. */
__device__ inline void fill_cooperative_direct_coulomb(
    CooperativeDirectPairDerivativeRecurrence& shared) {
  constexpr unsigned Levels = sizeof(kCooperativeCoulombLevels) / sizeof(unsigned) - 1U;
  for (unsigned level = 0; level < Levels; ++level) {
    for (unsigned node = kCooperativeCoulombLevels[level] + threadIdx.x;
         node < kCooperativeCoulombLevels[level + 1]; node += blockDim.x) {
      const auto instruction = __ldg(kCooperativeCoulombNodes + node);
      const unsigned first = instruction & 65535U, second = (instruction >> 16) & 65535U;
      const unsigned opcode = instruction >> 32;
      double value;
      if (opcode == 0U) value = kCooperativeCoulombConstants[first];
      else if (opcode == 1U)
        value = first == 0U ? shared.rho
              : first < 4U ? vec_axis(shared.difference, first - 1U)
                           : shared.boys[first - 4U];
      else if (opcode == 2U) value = shared.nodes[first] * shared.nodes[second];
      else value = shared.nodes[first] + shared.nodes[second];
      shared.nodes[node] = value;
    }
    __syncthreads();
  }
  constexpr unsigned Roots = sizeof(kCooperativeCoulombRoots) / sizeof(unsigned);
  for (unsigned root = threadIdx.x; root < Roots; root += blockDim.x) {
    const unsigned mapping = __ldg(kCooperativeCoulombRoots + root);
    shared.recurrence.coulomb.at(0, mapping & 15U, (mapping >> 4) & 15U,
                               (mapping >> 8) & 15U) = shared.nodes[mapping >> 12];
  }
  __syncthreads();
}

/** All eight complete warps participate; scratch is retired before reuse.
 * Only CTA lane zero owns the returned sum and the global force scatter. */
__device__ inline double reduce_cooperative_direct_force(double value, double* scratch) {
  const unsigned lane = threadIdx.x % 32U, warp = threadIdx.x / 32U;
  for (unsigned offset = 16; offset; offset /= 2)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  if (lane == 0U) scratch[warp] = value;
  __syncthreads();
  value = threadIdx.x < blockDim.x / 32U ? scratch[lane] : 0.0;
  if (warp == 0U)
    for (unsigned offset = 16; offset; offset /= 2)
      value += __shfl_down_sync(0xffffffffU, value, offset);
  __syncthreads();
  return value;
}

/** One CTA owns a complete order-six/seven s/p/d shell quartet. Each lane
 * holds at most three density weights and accumulates weighted atom gradients,
 * never a private Coulomb recurrence or an array of raw component gradients.
 * Missing cache/schedule/basis preconditions are rejected by the launcher.
 * checked_derivatives is qualification-only; production passes nullptr. */
template <bool Unrestricted, DirectForceOutputMode Mode>
__device__ inline void contract_cooperative_direct_pair_force(
    DeviceBatch batch, const ActiveShellQuartetTile& task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active,
    double* forces, double coulomb_coefficient, double exchange_coefficient,
    CooperativeDirectPairDerivativeRecurrence& shared,
    MaterializedDirectPairWork* work = nullptr, double* checked_derivatives = nullptr) {
  constexpr unsigned Slots = 3, Sources = DirectForceSources<Mode>::count;
  const auto first_pair = task.first_pair, second_pair = task.second_pair;
  const auto system = batch.shell_pair_systems[first_pair];
  if ((active && !active[system]) ||
      (coulomb_coefficient == 0.0 && exchange_coefficient == 0.0)) return;
  const auto si = batch.shell_pair_first[first_pair], sj = batch.shell_pair_second[first_pair];
  const auto sk = batch.shell_pair_first[second_pair], sl = batch.shell_pair_second[second_pair];
  const std::int32_t atoms[4] = {batch.shell_atoms[si], batch.shell_atoms[sj],
                               batch.shell_atoms[sk], batch.shell_atoms[sl]};
  std::int32_t unique_atoms[4];
  const unsigned centers = direct_force_unique_center_atoms(atoms, unique_atoms);
  if (centers <= 1U) return;
  const auto li = batch.shell_angular[si], lj = batch.shell_angular[sj];
  const auto lk = batch.shell_angular[sk], ll = batch.shell_angular[sl];
  const std::size_t n = batch.direct_nbf, physical = std::size_t(system) * n * n;
  const std::size_t spin = 2 * physical, ao_begin = std::size_t(system) * n;
  const auto first_count = shell_ao_pair_count(batch, first_pair);
  const auto second_count = shell_ao_pair_count(batch, second_pair);
  const auto count = first_pair == second_pair ? first_count * (first_count + 1) / 2
                                              : first_count * second_count;
  std::size_t i[Slots]{}, j[Slots]{}, k[Slots]{}, l[Slots]{};
  bool admitted[Slots]{};
  double weights[Slots][Sources]{}, gradients[3][Sources][3]{};
  bool any_admitted = false;
  for (unsigned slot = 0; slot < Slots; ++slot) {
    const std::size_t ordinal = slot * detail::kDirectQuartetTileSize + threadIdx.x;
    if (ordinal >= count || !decode_direct_tile_ao_ordinal(batch, task, ordinal,
        first_count, second_count, ao_begin, n, i[slot], j[slot], k[slot], l[slot]) ||
        !direct_ao_quartet_survives_schwarz(schwarz_bounds, physical, n,
            i[slot], j[slot], k[slot], l[slot], screening_tolerance)) continue;
    for (unsigned source = 0; source < Sources; ++source) {
      const auto coefficients = DirectForceSources<Mode>::coefficients(
          source, coulomb_coefficient, exchange_coefficient);
      weights[slot][source] = direct_force_density_coefficient_scaled<Unrestricted>(
          n, physical, spin, density, i[slot], j[slot], k[slot], l[slot],
          coefficients.coulomb, coefficients.exchange);
      admitted[slot] |= weights[slot][source] != 0.0;
    }
    any_admitted |= admitted[slot];
  }
  if (!__syncthreads_or(any_admitted)) return;
  auto& recurrence = shared.recurrence;
  const auto bra_begin = batch.shell_pair_primitive_offsets[first_pair];
  const auto ket_begin = batch.shell_pair_primitive_offsets[second_pair];
  const auto nb = batch.shell_primitive_offsets[sj + 1] - batch.shell_primitive_offsets[sj];
  const auto nd = batch.shell_primitive_offsets[sl + 1] - batch.shell_primitive_offsets[sl];
  for (auto bra = bra_begin; bra < batch.shell_pair_primitive_offsets[first_pair + 1]; ++bra) {
    const auto& cached = batch.shell_primitive_pairs[bra];
    const auto a = batch.shell_primitive_offsets[si] + (bra - bra_begin) / nb;
    const auto b = batch.shell_primitive_offsets[sj] + (bra - bra_begin) % nb;
    if (threadIdx.x == 0U) {
      recurrence.p = cached.exponent_sum;
      recurrence.coefficients[0] = batch.primitive_coefficients[a];
      recurrence.coefficients[1] = batch.primitive_coefficients[b];
    }
    for (unsigned center = 0; center + 1U < centers; ++center) {
      if (threadIdx.x < 3U) {
        const auto coordinate = std::int64_t(unique_atoms[center]) * 3;
        const auto product = prepare_materialized_direct_pair_derivative(cached,
            batch.primitive_exponents[a], batch.primitive_exponents[b],
            atom_position<Dual3>(batch, atoms[0], coordinate),
            atom_position<Dual3>(batch, atoms[1], coordinate), recurrence.bra[center],
            li, lj, threadIdx.x, threadIdx.x + 1U);
        if (threadIdx.x == 0U) {
          recurrence.first_product[center] = product;
          if (work) atomicAdd(&work->bra_preparations, 1ULL);
        }
      }
    }
    __syncthreads();
    for (auto ket = ket_begin; ket < batch.shell_pair_primitive_offsets[second_pair + 1]; ++ket) {
      const auto& cached_ket = batch.shell_primitive_pairs[ket];
      const auto c = batch.shell_primitive_offsets[sk] + (ket - ket_begin) / nd;
      const auto d = batch.shell_primitive_offsets[sl] + (ket - ket_begin) % nd;
      if (threadIdx.x == 0U) {
        recurrence.q = cached_ket.exponent_sum;
        recurrence.coefficients[2] = batch.primitive_coefficients[c];
        recurrence.coefficients[3] = batch.primitive_coefficients[d];
        const Vec3<double> product{recurrence.first_product[0].x.value,
            recurrence.first_product[0].y.value, recurrence.first_product[0].z.value};
        shared.rho = recurrence.p * recurrence.q / (recurrence.p + recurrence.q);
        shared.difference = {product.x - cached_ket.product_center.x,
            product.y - cached_ket.product_center.y, product.z - cached_ket.product_center.z};
        boys_values<8>(shared.rho * distance_squared(product, cached_ket.product_center), shared.boys);
        if (work) atomicAdd(&work->coulomb_preparations, 1ULL);
      }
      __syncthreads();
      fill_cooperative_direct_coulomb(shared);
      for (unsigned center = 0; center + 1U < centers; ++center) {
        if (threadIdx.x < 3U) {
          const auto coordinate = std::int64_t(unique_atoms[center]) * 3;
          const auto product = prepare_materialized_direct_pair_derivative(cached_ket,
              batch.primitive_exponents[c], batch.primitive_exponents[d],
              atom_position<Dual3>(batch, atoms[2], coordinate),
              atom_position<Dual3>(batch, atoms[3], coordinate), recurrence.ket,
              lk, ll, threadIdx.x, threadIdx.x + 1U);
          if (threadIdx.x == 0U) {
            recurrence.second_product = product;
            if (work) atomicAdd(&work->ket_preparations, 1ULL);
          }
        }
        __syncthreads();
        const MaterializedCoulombSpatialResponse response{recurrence.coulomb,
            recurrence.first_product[center].x.derivative_x - recurrence.second_product.x.derivative_x};
        for (unsigned slot = 0; slot < Slots; ++slot) {
          if (!admitted[slot]) continue;
          const double coefficient = batch.direct_ao_coefficients[ao_begin + i[slot]] *
              batch.direct_ao_coefficients[ao_begin + j[slot]] *
              batch.direct_ao_coefficients[ao_begin + k[slot]] *
              batch.direct_ao_coefficients[ao_begin + l[slot]];
          const double weight = coefficient * recurrence.coefficients[0] * recurrence.coefficients[1] *
                                recurrence.coefficients[2] * recurrence.coefficients[3];
          const auto component = weight * consume_cartesian_coulomb_states<8, Dual3>(
              recurrence.p, recurrence.q, direct_ao_angular(batch, ao_begin + i[slot]),
              direct_ao_angular(batch, ao_begin + j[slot]), direct_ao_angular(batch, ao_begin + k[slot]),
              direct_ao_angular(batch, ao_begin + l[slot]), recurrence.bra[center], recurrence.ket, response);
          const double derivative[3] = {component.derivative_x, component.derivative_y,
                                        component.derivative_z};
          for (unsigned axis = 0; axis < 3U; ++axis) {
            for (unsigned source = 0; source < Sources; ++source)
              gradients[center][source][axis] += weights[slot][source] * derivative[axis];
            if (checked_derivatives) {
              const std::size_t ordinal = slot * detail::kDirectQuartetTileSize + threadIdx.x;
              checked_derivatives[(ordinal * 4 + center) * 3 + axis] += derivative[axis];
            }
          }
          if (work) atomicAdd(&work->component_contractions, 1ULL);
        }
        __syncthreads();
      }
    }
  }
  for (unsigned slot = 0; slot < Slots; ++slot) {
    if (!admitted[slot]) continue;
    if (checked_derivatives) {
      const std::size_t ordinal = slot * detail::kDirectQuartetTileSize + threadIdx.x;
      for (unsigned axis = 0; axis < 3U; ++axis) {
        double sum = 0.0;
        for (unsigned center = 0; center + 1U < centers; ++center)
          sum += checked_derivatives[(ordinal * 4 + center) * 3 + axis];
        checked_derivatives[(ordinal * 4 + centers - 1U) * 3 + axis] = -sum;
      }
    }
    if (work) atomicAdd(&work->published_components, 1ULL);
  }
  const std::size_t stride = std::size_t(batch.total_atoms) * 3;
  for (unsigned source = 0; source < Sources; ++source) {
    double total[3]{};
    for (unsigned center = 0; center + 1U < centers; ++center)
      for (unsigned axis = 0; axis < 3U; ++axis) {
        const double value = reduce_cooperative_direct_force(
            gradients[center][source][axis], shared.warp_sums);
        if (threadIdx.x == 0U) {
          total[axis] += value;
          if (value != 0.0)
            atomicAdd(forces + source * stride + std::int64_t(unique_atoms[center]) * 3 + axis, -value);
        }
      }
    if (threadIdx.x == 0U)
      for (unsigned axis = 0; axis < 3U; ++axis)
        if (total[axis] != 0.0)
          atomicAdd(forces + source * stride + std::int64_t(unique_atoms[centers - 1U]) * 3 + axis,
                    total[axis]);
  }
}
"""
