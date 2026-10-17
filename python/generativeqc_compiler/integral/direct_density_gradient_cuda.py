"""Factor exact pure-J force weights into shared density/Hermite projections.

Only a complete AO-Schwarz mask is separable. Partial masks, bilinear seeds,
exchange, unsupported angular domains and non-finite reordered intermediates
decline before publishing forces and retain the component consumer. The
Gaussian, relative-Hermite and Coulomb recurrences are supplied by their
existing cooperative lowering, not redefined here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .shell_spec import cartesian_components

if TYPE_CHECKING:
    from collections.abc import Callable


def density_hermite_states(maximum: int = 4) -> tuple[tuple[int, int, int], ...]:
    """Return all bounded Hermite states using the authoritative CCA ordering."""
    if not isinstance(maximum, int) or isinstance(maximum, bool):
        raise TypeError("maximum Hermite degree must be an integer")
    if not 0 <= maximum <= 4:
        raise ValueError("density projection supports s/p/d shell pairs only")
    return tuple(
        (component.count("x"), component.count("y"), component.count("z"))
        for degree in range(maximum + 1)
        for component in cartesian_components(degree)
    )


def emit_density_direct_force_support(
    traversal: Callable[..., str], scatter: str
) -> str:
    """Emit a bounded contraction around the common scientific traversal.

    For distinct shell pairs, the canonical J weight is cJ times the two
    symmetrized total-density pair weights. A repeated shell pair uses half
    the ordered product, recovering its triangular orbit exactly. Separate
    source mode writes only J; its K channel remains untouched.
    """
    mappings = ", ".join(
        f"{axis_x | (axis_y << 4) | (axis_z << 8)}U"
        for axis_x, axis_y, axis_z in density_hermite_states()
    )
    body = traversal(
        _CONTRACTION,
        bra_projection=_BRA_PROJECTION,
        ket_projection=_KET_PROJECTION,
    )
    return (
        _SUPPORT.replace("@STATES@", mappings)
        + _CONSUMER.replace("@TRAVERSAL@", body).replace("@FORCE_SCATTER@", scatter)
        + _DISPATCH
    )


_SUPPORT = r"""
struct DirectDensityForceInput {
  Angular first, second;
  double weight;
};
struct DirectDensityForceResponse {
  double value, derivative[3];
};
struct CooperativeDensityForceRecurrence {
  CooperativeDirectPairDerivativeRecurrence cooperative;
  DirectDensityForceInput pairs[2][36];
  DirectDensityForceResponse hermite[2][35];
};
static_assert(sizeof(CooperativeDensityForceRecurrence) <= (44U << 10));
static __device__ const unsigned kDirectDensityHermiteStates[] = {@STATES@};

__device__ inline unsigned direct_density_hermite_count(unsigned degree) {
  return (degree + 1U) * (degree + 2U) * (degree + 3U) / 6U;
}

template <bool Unrestricted>
__device__ inline bool prepare_direct_density_force_inputs(
    DeviceBatch batch, std::size_t pair, unsigned system, const double* density,
    double scale, DirectDensityForceInput* output) {
  const std::size_t count = shell_ao_pair_count(batch, pair);
  const std::size_t dimension = batch.direct_nbf, matrix = dimension * dimension;
  bool finite = true;
  for (std::size_t component = threadIdx.x; component < count; component += blockDim.x) {
    std::size_t first{}, second{};
    decode_shell_ao_pair(batch, pair, component, system * dimension, first, second);
    const auto direct = matrix_index(first, second, dimension);
    const auto transpose = matrix_index(second, first, dimension);
    double weight;
    if constexpr (Unrestricted) {
      weight = scale * (density[2 * system * matrix + direct] +
                        density[(2 * system + 1) * matrix + direct]);
      if (first != second)
        weight += scale * (density[2 * system * matrix + transpose] +
                           density[(2 * system + 1) * matrix + transpose]);
    } else {
      weight = scale * density[system * matrix + direct];
      if (first != second) weight += scale * density[system * matrix + transpose];
    }
    weight *= batch.direct_ao_coefficients[system * dimension + first] *
              batch.direct_ao_coefficients[system * dimension + second];
    finite &= isfinite(weight);
    output[component] = {direct_ao_angular(batch, system * dimension + first),
                        direct_ao_angular(batch, system * dimension + second), weight};
  }
  return finite;
}

__device__ inline bool project_direct_density_force_pair(
    const DirectRelativePairDerivativeRecurrence::Pair* pair,
    const DirectDensityForceInput* input, unsigned components, unsigned degree,
    DirectDensityForceResponse* output, DensityDirectPairWork* work) {
  bool finite = true;
  for (unsigned state = threadIdx.x; state < direct_density_hermite_count(degree);
       state += blockDim.x) {
    const auto packed = kDirectDensityHermiteStates[state];
    const unsigned axis_x = packed & 15U, axis_y = (packed >> 4) & 15U;
    const unsigned axis_z = (packed >> 8) & 15U;
    DirectDensityForceResponse sum{};
    unsigned terms = 0;
    for (unsigned component = 0; component < components; ++component) {
      const auto item = input[component];
      if (axis_x > item.first.x + item.second.x ||
          axis_y > item.first.y + item.second.y ||
          axis_z > item.first.z + item.second.z) continue;
      const auto first = pair[0].at(item.first.x, item.second.x, axis_x);
      const auto second = pair[1].at(item.first.y, item.second.y, axis_y);
      const auto third = pair[2].at(item.first.z, item.second.z, axis_z);
      sum.value += item.weight * first.value * second.value * third.value;
      sum.derivative[0] += item.weight * first.derivative * second.value * third.value;
      sum.derivative[1] += item.weight * first.value * second.derivative * third.value;
      sum.derivative[2] += item.weight * first.value * second.value * third.derivative;
      ++terms;
    }
    finite &= isfinite(sum.value) && isfinite(sum.derivative[0]) &&
              isfinite(sum.derivative[1]) && isfinite(sum.derivative[2]);
    output[state] = sum;
    if (work) atomicAdd(&work->density_terms, static_cast<unsigned long long>(terms));
  }
  return finite;
}
"""

_DISPATCH = r"""
template <bool Unrestricted, DirectForceOutputMode Mode, bool DensityContraction>
__device__ inline void contract_shared_direct_pair_force(
    DeviceBatch batch, const ActiveShellQuartetTile& task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active,
    double* forces, double coulomb_coefficient, double exchange_coefficient, void* storage) {
  if constexpr (DensityContraction) {
    auto& workspace = *reinterpret_cast<CooperativeDensityForceRecurrence*>(storage);
    if (try_contract_density_direct_pair_force<Unrestricted, Mode>(
            batch, task, screening_tolerance, schwarz_bounds, density, active, forces,
            coulomb_coefficient, exchange_coefficient, workspace)) return;
    contract_cooperative_direct_pair_force<Unrestricted, Mode>(
        batch, task, screening_tolerance, schwarz_bounds, density, active, forces,
        coulomb_coefficient, exchange_coefficient, workspace.cooperative);
  } else {
    auto& workspace = *reinterpret_cast<CooperativeDirectPairDerivativeRecurrence*>(storage);
    contract_cooperative_direct_pair_force<Unrestricted, Mode>(
        batch, task, screening_tolerance, schwarz_bounds, density, active, forces,
        coulomb_coefficient, exchange_coefficient, workspace);
  }
}
"""

_BRA_PROJECTION = r"""
    const bool finite_bra = project_direct_density_force_pair(recurrence.bra,
        density_shared.pairs[0], first_count, li + lj, density_shared.hermite[0], census);
    if (!__syncthreads_and(finite_bra)) {
      if (census && threadIdx.x == 0U) atomicAdd(&census->numerical_rejections, 1ULL);
      return false;
    }"""

_KET_PROJECTION = r"""      const bool finite_ket = project_direct_density_force_pair(recurrence.ket,
          density_shared.pairs[1], second_count, lk + ll, density_shared.hermite[1], census);
      if (!__syncthreads_and(finite_ket)) {
        if (census && threadIdx.x == 0U) atomicAdd(&census->numerical_rejections, 1ULL);
        return false;
      }
"""

_CONTRACTION = r"""      const unsigned bra_states = direct_density_hermite_count(li + lj);
      const unsigned ket_states = direct_density_hermite_count(lk + ll);
      const double prefactor = recurrence.coefficients[0] * recurrence.coefficients[1] *
          recurrence.coefficients[2] * recurrence.coefficients[3] * 2.0 * pow(kPi, 2.5) /
          (recurrence.p * recurrence.q * sqrt(recurrence.p + recurrence.q));
      for (unsigned state = threadIdx.x; state < bra_states * ket_states;
           state += blockDim.x) {
        const unsigned bra_state = state / ket_states, ket_state = state % ket_states;
        const auto bra_item = density_shared.hermite[0][bra_state];
        const auto ket_item = density_shared.hermite[1][ket_state];
        const auto bra_mapping = kDirectDensityHermiteStates[bra_state];
        const auto ket_mapping = kDirectDensityHermiteStates[ket_state];
        const unsigned axis_x = (bra_mapping & 15U) + (ket_mapping & 15U);
        const unsigned axis_y = ((bra_mapping >> 4) & 15U) + ((ket_mapping >> 4) & 15U);
        const unsigned axis_z = ((bra_mapping >> 8) & 15U) + ((ket_mapping >> 8) & 15U);
        const double sign = (((ket_mapping & 15U) + ((ket_mapping >> 4) & 15U) +
                             ((ket_mapping >> 8) & 15U)) & 1U) ? -1.0 : 1.0;
        const double coulomb = recurrence.coulomb.at(0, axis_x, axis_y, axis_z);
        for (unsigned axis = 0; axis < 3U; ++axis) {
          const double bra_response = bra_item.derivative[axis] * ket_item.value * coulomb;
          const double ket_response = bra_item.value * ket_item.derivative[axis] * coulomb;
          const double product_response = bra_item.value * ket_item.value *
              recurrence.coulomb.at(0, axis_x + (axis == 0U), axis_y + (axis == 1U),
                                    axis_z + (axis == 2U));
          for (unsigned center = 0; center + 1U < centers; ++center) {
            const auto* scales = recurrence.response_scales[center];
            gradients[center][0][axis] += sign * prefactor * (scales[0] * bra_response +
                scales[1] * ket_response + scales[2] * product_response);
          }
        }
        if (census) atomicAdd(&census->hermite_products, 1ULL);
      }"""

_CONSUMER = r"""
template <bool Unrestricted, DirectForceOutputMode Mode>
__device__ inline bool try_contract_density_direct_pair_force(
    DeviceBatch batch, const ActiveShellQuartetTile& task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active,
    double* forces, double coulomb_coefficient, double exchange_coefficient,
    CooperativeDensityForceRecurrence& density_shared) {
  constexpr unsigned Sources = DirectForceSources<Mode>::count;
  auto& shared = density_shared.cooperative;
  if (exchange_coefficient != 0.0 || batch.direct_force_seed) return false;
  const auto first_pair = task.first_pair, second_pair = task.second_pair;
  const auto system = batch.shell_pair_systems[first_pair];
  if ((active && !active[system]) || coulomb_coefficient == 0.0) return true;
  const auto si = batch.shell_pair_first[first_pair], sj = batch.shell_pair_second[first_pair];
  const auto sk = batch.shell_pair_first[second_pair], sl = batch.shell_pair_second[second_pair];
  const std::int32_t atoms[4] = {batch.shell_atoms[si], batch.shell_atoms[sj],
                               batch.shell_atoms[sk], batch.shell_atoms[sl]};
  std::int32_t unique_atoms[4];
  const unsigned centers = direct_force_unique_center_atoms(atoms, unique_atoms);
  if (centers <= 1U) return true;
  const auto li = batch.shell_angular[si], lj = batch.shell_angular[sj];
  const auto lk = batch.shell_angular[sk], ll = batch.shell_angular[sl];
  if (li > 2U || lj > 2U || lk > 2U || ll > 2U ||
      li + lj + lk + ll < 6U || li + lj + lk + ll > 7U) return false;
  const std::size_t dimension = batch.direct_nbf, physical = std::size_t(system) * dimension * dimension;
  const std::size_t spin = 2U * physical, ao_begin = std::size_t(system) * dimension;
  const auto first_count = shell_ao_pair_count(batch, first_pair);
  const auto second_count = shell_ao_pair_count(batch, second_pair);
  const auto count = first_pair == second_pair ? first_count * (first_count + 1U) / 2U
                                              : first_count * second_count;
  auto* diagnostic = direct_force_task_work(batch, task);
  auto* census = diagnostic ? &diagnostic->density : nullptr;
  MaterializedDirectPairWork* work = census ? &census->recurrence : nullptr;
  if (census && threadIdx.x == 0U) atomicAdd(&census->attempted_shells, 1ULL);
  bool full = true;
  unsigned components = 0;
  for (std::size_t ordinal = threadIdx.x; ordinal < count; ordinal += blockDim.x) {
    std::size_t first{}, second{}, third{}, fourth{};
    const bool admitted = decode_direct_tile_ao_ordinal(batch, task, ordinal,
        first_count, second_count, ao_begin, dimension, first, second, third, fourth) &&
        direct_ao_quartet_survives_schwarz(schwarz_bounds, physical, dimension,
            first, second, third, fourth, screening_tolerance);
    full &= admitted;
    if (admitted) components += direct_force_density_coefficient_scaled<Unrestricted>(
        dimension, physical, spin, density, first, second, third, fourth,
        coulomb_coefficient, 0.0) != 0.0;
  }
  if (!__syncthreads_and(full)) {
    if (census && threadIdx.x == 0U) atomicAdd(&census->mask_rejections, 1ULL);
    return false;
  }
  if (!__syncthreads_or(components != 0U)) return true;
  bool finite = prepare_direct_density_force_inputs<Unrestricted>(batch, first_pair,
      system, density, coulomb_coefficient * (first_pair == second_pair ? 0.5 : 1.0),
      density_shared.pairs[0]);
  finite &= prepare_direct_density_force_inputs<Unrestricted>(batch, second_pair,
      system, density, 1.0, density_shared.pairs[1]);
  if (!__syncthreads_and(finite)) {
    if (census && threadIdx.x == 0U) atomicAdd(&census->numerical_rejections, 1ULL);
    return false;
  }
  double gradients[3][Sources][3]{};
@TRAVERSAL@
  double total[3]{};
  finite = true;
  for (unsigned center = 0; center + 1U < centers; ++center)
    for (unsigned axis = 0; axis < 3U; ++axis) {
      const double reduced = reduce_cooperative_direct_force(
          gradients[center][0][axis], shared.warp_sums);
      if (threadIdx.x == 0U) {
        total[axis] += reduced;
        finite &= isfinite(reduced) && isfinite(total[axis]);
      }
    }
  if (!__syncthreads_and(finite)) {
    if (census && threadIdx.x == 0U) atomicAdd(&census->numerical_rejections, 1ULL);
    return false;
  }
  if (diagnostic) {
    if (threadIdx.x == 0U) {
      atomicAdd(&diagnostic->executed_shells, 1ULL);
      atomicAdd(&census->accepted_shells, 1ULL);
      profile_direct_force_primitive_products(batch, diagnostic, first_pair, second_pair);
    }
    if (components) atomicAdd(&diagnostic->executed_components,
                              static_cast<unsigned long long>(components));
  }
@FORCE_SCATTER@
  return true;
}
"""
