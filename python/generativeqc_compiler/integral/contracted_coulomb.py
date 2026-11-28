"""Compiler-owned s/p Coulomb pair jets for a source-driven frozen sweep.

The specialized shape has exactly three primitives per Cartesian shell. Inputs
and outputs are immutable or disjoint windows supplied by the native owner;
these leaves neither select a device nor allocate. This is a finite candidate
schedule, not a public integral or runtime-JIT backend.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from math import comb

from .cuda import CudaEmitter
from .expr import Expr, Graph
from .factorized_second import _jet_function
from .shell_class import (
    _component_quantums,
    _pair_expansion,
    build_coulomb_derivative_algebra,
)
from .shell_spec import cartesian_components

PRELUDE = r"""
#include <cuda_runtime.h>
#include "scf/cuda/boys_table.cuh"

namespace generativeqc::dft::hessian_aot::contracted_detail {
constexpr unsigned left_radials = @BL@, right_radials = @BR@;
constexpr unsigned primitive_products = left_radials * right_radials;
constexpr unsigned active_coordinates = 9;
constexpr unsigned threads = 128;
constexpr double two_pi_five_halves = 0x1.17e50a9dc6553p+5;

struct RadialIndex {
  unsigned char value, first[3], second[9];
  signed char sign;
};

__device__ inline double value(const double* jet) { return jet[0]; }

__device__ inline double gradient(const double* jet, unsigned coordinate, bool first_pair) {
  const auto center = coordinate / 3;
  if (first_pair) return center >= 2 ? 0.0 : (center == 0 ? 1.0 : -1.0) * jet[1 + coordinate % 3];
  return center == 2 ? jet[1 + coordinate % 3] : 0.0;
}

__device__ inline double mixed(const double* jet, unsigned first, unsigned second, bool first_pair) {
  const auto left = first / 3, right = second / 3;
  if (first_pair) {
    if (left >= 2 || right >= 2) return 0.0;
    return (left == right ? 1.0 : -1.0) * jet[4 + 3 * (first % 3) + second % 3];
  }
  return left == 2 && right == 2 ? jet[4 + 3 * (first % 3) + second % 3] : 0.0;
}

__device__ inline double slot_scale(const double* left, const double* right, unsigned coordinate) {
  const auto center = coordinate / 3;
  return center < 2 ? left[4 + center] : -right[4];
}
"""


def _product_code() -> str:
    """Emit two consumers of the same coordinate-product AD."""
    graph = Graph()
    names = ("left", "right", "radial")
    variables = {
        f"{name}{suffix}": f"{name}{suffix}"
        for name in names
        for suffix in ("", "_first", "_second", "_mixed")
    }
    leaves = {key: graph.variable(key) for key in variables}
    value = leaves["left"] * leaves["right"] * leaves["radial"]
    first = graph.differentiate(
        value,
        graph.variable("first"),
        {name: leaves[f"{name}_first"] for name in names},
    )
    second = graph.differentiate(
        first,
        graph.variable("second"),
        {
            **{name: leaves[f"{name}_second"] for name in names},
            **{f"{name}_first": leaves[f"{name}_mixed"] for name in names},
        },
    )
    return "\n".join(
        f"""__device__ __forceinline__ double product_{name}({", ".join(f"double {key}" for key in keys)}) {{
  double result;
  {_jet_function(graph, (root,), variables, ("result",))}
  return result;
}}"""
        for name, root, keys in (
            ("first", first, (*names, *(f"{key}_first" for key in names))),
            ("mixed", second, tuple(variables)),
        )
    )


KERNEL = r"""
extern "C" __global__ void contract_@KEY@(
    const double* __restrict__ left_geometry, const double* __restrict__ right_geometry,
    const double* __restrict__ left_jet, const double* __restrict__ right_jet,
    const double* __restrict__ left_basis, const double* __restrict__ right_basis,
    const unsigned* __restrict__ group_pairs, double* __restrict__ output, unsigned group_count,
    int* status) {
  constexpr unsigned left_states = @LS@, right_states = @RS@;
  constexpr unsigned left_components = @LC@, right_components = @RC@;
  constexpr unsigned hessian_fields = active_coordinates * (active_coordinates + 1) / 2;
  constexpr unsigned outputs = hessian_fields + active_coordinates * (left_components + right_components);
  constexpr unsigned lane_count = 32;
  extern __shared__ double radial[];
  const auto group = blockIdx.x;
  if (group >= group_count) return;
  const auto left_index = group_pairs[2 * group], right_index = group_pairs[2 * group + 1];
  const auto* geom_left = left_geometry + left_index * left_radials * 6;
  const auto* geom_right = right_geometry + right_index * right_radials * 6;
  if (threadIdx.x < primitive_products) {
    const auto product_index = threadIdx.x;
    const auto* bra = geom_left + 6 * (product_index / right_radials);
    const auto* ket = geom_right + 6 * (product_index % right_radials);
    const double inverse = 1.0 / (bra[0] + ket[0]);
    const double rho = bra[0] * ket[0] * inverse;
    const double dx = bra[1] - ket[1], dy = bra[2] - ket[2], dz = bra[3] - ket[3];
    const double prefactor = two_pi_five_halves * sqrt(inverse) / (bra[0] * ket[0]);
    double boys[@L@+1];
    generativeqc::scf::cuda_execution::boys_values<@L@>(rho * (dx * dx + dy * dy + dz * dz), boys);
    for (double& moment : boys) moment *= prefactor;
    radial_@KEY@(rho, dx, dy, dz, boys, radial + product_index);
  }
  __syncthreads();
  const unsigned lane = threadIdx.x % lane_count;
  for (unsigned field = threadIdx.x / lane_count; field < outputs; field += threads / lane_count) {
    const bool hessian = field < hessian_fields;
    unsigned first = field % active_coordinates;
    unsigned second = 0;
    if (hessian) {
      first = 0;
      while ((first + 1) * (first + 2) / 2 <= field) ++first;
      second = field - first * (first + 1) / 2;
    }
    const auto component = hessian ? 0 : (field - hessian_fields) / active_coordinates;
    const bool left_incidence = !hessian && component < left_components;
    // The owning shell pair is uniform across the warp. Specialize outside
    // the state loops so the common AD product can drop its zero derivative
    // terms without any runtime tests or loads inside the contraction.
    const auto contract_field = [&]<bool Hessian, bool FirstLeft, bool SecondLeft>() {
    const auto u = first % 3, v = second % 3;
    const double sign_u = first < 3 ? 1 : -1, sign_v = second < 3 ? 1 : -1;
    double sum = 0;
    for (unsigned product_index = lane; product_index < primitive_products; product_index += lane_count) {
      const auto bra_radial = product_index / right_radials, ket_radial = product_index % right_radials;
      const auto bra_pair = left_index * left_radials + bra_radial;
      const auto ket_pair = right_index * right_radials + ket_radial;
      const auto* bra_geom = geom_left + 6 * bra_radial;
      const auto* ket_geom = geom_right + 6 * ket_radial;
      const auto scale_first = FirstLeft ? bra_geom[4 + first / 3] : -ket_geom[4];
      const auto scale_second = !Hessian ? 0.0 : SecondLeft ? bra_geom[4 + second / 3] : -ket_geom[4];
      const auto* bra = left_incidence ? left_basis + ((bra_pair * left_components + component) * left_states) * 4
                                      : left_jet + bra_pair * left_states * 13;
      const auto* ket = Hessian || left_incidence ? right_jet + ket_pair * right_states * 13
               : right_basis + ((ket_pair * right_components + component - left_components) * right_states) * 4;
      const auto bra_stride = left_incidence ? 4 : 13;
      const auto ket_stride = Hessian || left_incidence ? 13 : 4;
      const auto* source = radial + product_index;
      for (unsigned bra_state = 0; bra_state < left_states; ++bra_state) {
        const auto* bra_terms = bra + bra_state * bra_stride;
        const auto left_value = value(bra_terms);
        const auto left_first = FirstLeft ? sign_u * bra_terms[1 + u] : 0.0;
        const auto left_second = Hessian && SecondLeft ? sign_v * bra_terms[1 + v] : 0.0;
        const auto left_mixed = Hessian && FirstLeft && SecondLeft
                                    ? sign_u * sign_v * bra_terms[4 + 3 * u + v] : 0.0;
        for (unsigned ket_state = 0; ket_state < right_states; ++ket_state) {
          const auto indices = indices_@KEY@[bra_state * right_states + ket_state];
          const auto* ket_terms = ket + ket_state * ket_stride;
          const auto right_value = value(ket_terms);
          const auto right_first = !FirstLeft ? ket_terms[1 + u] : 0.0;
          const auto radial_value = source[indices.value * primitive_products];
          const auto radial_first = source[indices.first[u] * primitive_products] * scale_first;
          if constexpr (Hessian) {
            const auto right_second = !SecondLeft ? ket_terms[1 + v] : 0.0;
            const auto radial_second = source[indices.first[v] * primitive_products] * scale_second;
            const auto right_mixed = !FirstLeft && !SecondLeft ? ket_terms[4 + 3 * u + v] : 0.0;
            const auto radial_mixed = source[indices.second[3 * u + v] * primitive_products]
                                    * scale_first * scale_second;
            sum += indices.sign * product_mixed(left_value, left_first, left_second, left_mixed,
                right_value, right_first, right_second, right_mixed,
                radial_value, radial_first, radial_second, radial_mixed);
          } else {
            sum += indices.sign * product_first(left_value, right_value, radial_value,
                left_first, right_first, radial_first);
          }
        }
      }
    }
    return sum;
    };
    double total;
    if (!hessian)
      total = first < 6 ? contract_field.template operator()<false, true, false>()
                        : contract_field.template operator()<false, false, false>();
    else if (first < 6)
      total = contract_field.template operator()<true, true, true>();
    else if (second < 6)
      total = contract_field.template operator()<true, false, true>();
    else
      total = contract_field.template operator()<true, false, false>();
    for (unsigned offset = lane_count / 2; offset; offset /= 2)
      total += __shfl_down_sync(0xffffffff, total, offset);
    if (!lane) {
      if (!isfinite(total)) atomicExch(status, 1);
      output[group * outputs + field] = total;
    }
  }
}
"""


HALF_CONTRACTION = r"""
extern "C" __global__ void contract_@KEY@(
    const double* __restrict__ left_geometry, const double* __restrict__ right_geometry,
    const double* __restrict__ left_jet, const double* __restrict__ right_jet,
    const double* __restrict__ left_basis, const double* __restrict__ right_basis,
    const unsigned* __restrict__ group_pairs, double* __restrict__ output, unsigned group_count,
    int* status) {
  constexpr unsigned lstate = @LS@, rstate = @RS@, lc = @LC@, rc = @RC@;
  constexpr unsigned tile = @TILE@, fields = 45 + 9 * (lc + rc), perblock = @THREADS@ / tile;
  constexpr unsigned local_fields = (fields + perblock - 1) / perblock;
  constexpr unsigned moments = @NS@;
  extern __shared__ double storage[];
  auto* radial = storage;
  auto* right_half = radial + (moments * tile);
  auto* left_half = right_half + 34 * lstate * tile;
  const unsigned group = blockIdx.x;
  if (group >= group_count) return;
  const unsigned left_index = group_pairs[2 * group], right_index = group_pairs[2 * group + 1];
  const auto* geom_left = left_geometry + left_index * left_radials * 6;
  const auto* geom_right = right_geometry + right_index * right_radials * 6;
  const unsigned lane = threadIdx.x % tile;
  double total[local_fields]{};
  for (unsigned begin = 0; begin < primitive_products; begin += tile) {
    const auto product = begin + lane;
    const auto bra_radial = product / right_radials, ket_radial = product % right_radials;
    const auto* source = radial + lane;
    const auto bra_pair = left_index * left_radials + bra_radial;
    const auto ket_pair = right_index * right_radials + ket_radial;
    if (threadIdx.x < tile && product < primitive_products) {
      const auto* bra = geom_left + 6 * bra_radial;
      const auto* ket = geom_right + 6 * ket_radial;
      const double inverse = 1.0 / (bra[0] + ket[0]);
      const double rho = bra[0] * ket[0] * inverse;
      const double dx = bra[1] - ket[1], dy = bra[2] - ket[2], dz = bra[3] - ket[3];
      const double prefactor = two_pi_five_halves * sqrt(inverse) / (bra[0] * ket[0]);
      double boys[@L@+1];
      generativeqc::scf::cuda_execution::boys_values<@L@>(rho * (dx * dx + dy * dy + dz * dz), boys);
      for (double& moment : boys) moment *= prefactor;
      radial_@KEY@(rho, dx, dy, dz, boys, radial + lane);
    }
    __syncthreads();
    if (product < primitive_products) {
      // Right partial jet: value, R and d, RR, dR, dd. R is the
      // product-center separation and d is the right shell separation.
      for (unsigned packed = threadIdx.x / tile; packed < lstate * 34; packed += perblock) {
        const unsigned st = packed / 34, field = packed % 34;
        unsigned first_axis = (field - 1) % 3, second_axis = 0;
        if (field >= 7) {
          first_axis = ((field - 7) / 3) % 3;
          second_axis = (field - 7) % 3;
        }
        double sum = 0;
        for (unsigned other = 0; other < rstate; ++other) {
          const auto& indices = indices_@KEY@[st * rstate + other];
          const auto* ket = right_jet + (ket_pair * rstate + other) * 13;
          unsigned kfield = 0, rfield = indices.value;
          if (field >= 25) kfield = 4 + first_axis * 3 + second_axis;
          else if (field >= 16) {
            kfield = 1 + first_axis;
            rfield = indices.first[second_axis];
          } else if (field >= 7) rfield = indices.second[first_axis * 3 + second_axis];
          else if (field >= 4) kfield = 1 + first_axis;
          else if (field) rfield = indices.first[first_axis];
          sum = fma(indices.sign * ket[kfield], source[rfield * tile], sum);
        }
        right_half[packed * tile + lane] = sum;
      }
      // Left partial jet is read only by the right Fock incidence and
      // needs no second derivatives. The Hermite sign belongs to ket.
      for (unsigned packed = threadIdx.x / tile; packed < rstate * 7; packed += perblock) {
        const unsigned st = packed / 7, field = packed % 7;
        const auto axis = (field - 1) % 3;
        double sum = 0;
        for (unsigned other = 0; other < lstate; ++other) {
          const auto& indices = indices_@KEY@[other * rstate + st];
          const auto* bra = left_jet + (bra_pair * lstate + other) * 13;
          const auto kfield = field >= 4 ? 1 + axis : 0;
          const auto rfield = field && field < 4 ? indices.first[axis] : indices.value;
          sum = fma(indices.sign * bra[kfield], source[rfield * tile], sum);
        }
        left_half[packed * tile + lane] = sum;
      }
    }
    __syncthreads();
    if (product < primitive_products) {
      const auto* bra_geom = geom_left + 6 * bra_radial;
      const auto* ket_geom = geom_right + 6 * ket_radial;
      #pragma unroll
      for (unsigned index = 0; index < local_fields; ++index) {
        const auto field = threadIdx.x / tile + perblock * index;
        if (field >= fields) continue;
        const bool hessian = field < 45;
        unsigned first = (field - 45) % 9, second = 0;
        if (hessian) {
          first = 0;
          while ((first + 1) * (first + 2) / 2 <= field) ++first;
          second = field - first * (first + 1) / 2;
        }
        const auto component = (field - 45) / 9;
        const auto u = first % 3, v = second % 3;
        const bool first_left = first < 6, second_left = second < 6;
        const double first_sign = first < 3 ? 1.0 : -1.0;
        const double second_sign = second < 3 ? 1.0 : -1.0;
        const double alpha = first_left ? bra_geom[4 + first / 3] : -ket_geom[4];
        const double beta = second_left ? bra_geom[4 + second / 3] : -ket_geom[4];
        double sum = total[index];
        if (hessian || component < lc) {
          const auto stride = hessian ? 13 : 4;
          const auto* bra = hessian ? left_jet + bra_pair * lstate * 13
                                    : left_basis + (bra_pair * lc + component) * lstate * 4;
          for (unsigned st = 0; st < lstate; ++st) {
            const auto* basis = bra + st * stride;
            const auto* half = right_half + st * (34 * tile) + lane;
            const auto q0 = half[0];
            const auto qu = fma(alpha, half[(1 + u) * tile], first_left ? 0 : half[(4 + u) * tile]);
            const auto pu = first_left ? first_sign * basis[1 + u] : 0;
            if (!hessian) sum += product_first(basis[0], q0, 1, pu, qu, 0);
            else {
              const auto qv = fma(beta, half[(1 + v) * tile], second_left ? 0 : half[(4 + v) * tile]);
              const auto pv = second_left ? second_sign * basis[1 + v] : 0;
              const auto puv = first_left && second_left ? first_sign * second_sign * basis[4 + 3*u + v] : 0;
              double quv = alpha * beta * half[(7 + 3*u + v) * tile];
              if (!first_left) quv = fma(beta, half[(16 + 3*u + v) * tile], quv);
              if (!second_left) quv = fma(alpha, half[(16 + 3*v + u) * tile], quv);
              if (!first_left && !second_left) quv += half[(25 + 3*u + v) * tile];
              sum += product_mixed(basis[0], pu, pv, puv, q0, qu, qv, quv, 1, 0, 0, 0);
            }
          }
        } else {
          const auto* ket = right_basis + (ket_pair * rc + component - lc) * rstate * 4;
          for (unsigned st = 0; st < rstate; ++st) {
            const auto* basis = ket + st * 4;
            const auto* half = left_half + st * (7 * tile) + lane;
            const auto pu = fma(alpha, half[(1 + u) * tile],
                                first_left ? first_sign * half[(4 + u) * tile] : 0);
            sum += product_first(half[0], basis[0], 1, pu, first_left ? 0 : basis[1 + u], 0);
          }
        }
        total[index] = sum;
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (unsigned index = 0; index < local_fields; ++index) {
    const auto field = threadIdx.x / tile + perblock * index;
    double result = total[index];
    for (unsigned step = tile / 2; step; step /= 2)
      result += __shfl_down_sync(0xffffffff, result, step, tile);
    if (!lane && field < fields) {
      if (!isfinite(result)) atomicExch(status, 1);
      output[group * fields + field] = result;
    }
  }
}
"""


def _half_primitive_tile(left: int, right: int, auxiliary_legs: int) -> int:
    if auxiliary_legs or right == 0:
        return 0
    return 8 if left == right == 2 else 16


def quartet_launch_shape(left: int, right: int, auxiliary_legs: int) -> tuple[int, int]:
    """Return block width and exact dynamic bytes of the selected leaf.

    Each tiled primitive product holds one radial jet, 34 right-incidence
    fields per bra state, and seven left-incidence fields per ket state.
    Their combined allocation stays under the prior 81 × 84 FP64 limit.
    """
    radial = comb(left + right + 5, 3)
    tile = _half_primitive_tile(left, right, auxiliary_legs)
    if tile:
        return 256, 8 * tile * (
            radial + 34 * comb(left + 3, 3) + 7 * comb(right + 3, 3)
        )
    return 128, 8 * (81 // 3**auxiliary_legs) * radial


def _early_radial_stores(
    graph: Graph, roots: tuple, variables: dict, targets: tuple
) -> str:
    """Schedule each radial store once its DAG root is available, sharing CSE."""
    emitter = CudaEmitter(graph, variables)
    lines = []
    for root, target in zip(roots, targets, strict=True):
        emitter.emit((root,))
        lines.extend(emitter.lines)
        emitter.lines.clear()
        lines.append(f"{target}={emitter.reference(root)};")
    return "\n".join(lines)


def emit_quartet_jet(left: int, right: int, *, auxiliary_legs: int = 0) -> str:
    """Emit one pair-degree class with no primitive-result global buffer.

    Output stores the inclusive lower triangle of the 9×9 relative Hessian,
    followed by the nine first derivatives for each bra and ket pair component.
    Auxiliary legs occupy the ket first, then the bra, and use three radial
    slots rather than nine. The native fold owns the physical atom map,
    diagonal shell-pair weight, and status publication.
    """
    if auxiliary_legs not in (0, 1, 2):
        raise ValueError("expected zero, one or two auxiliary legs")
    if not (
        0 <= left <= (1 if auxiliary_legs == 2 else 2)
        and 0 <= right <= (1 if auxiliary_legs else left)
    ):
        raise ValueError("unsupported Cartesian s/p pair degrees")
    max_order = left + right + 2
    key = (
        f"{left}_{right}"
        if not auxiliary_legs
        else f"fitted{4 - auxiliary_legs}_{left}_{right}"
    )
    algebra = build_coulomb_derivative_algebra(max_order)
    state_ids = {state: index for index, (state, _) in enumerate(algebra.roots)}
    bras = [state for state in state_ids if sum(state) <= left]
    kets = [state for state in state_ids if sum(state) <= right]
    axes = tuple(tuple(int(coord == axis) for coord in range(3)) for axis in range(3))

    def lookup(*orders: tuple[int, ...]) -> int:
        x_order, y_order, z_order = map(sum, zip(*orders, strict=True))
        return state_ids[x_order, y_order, z_order]

    table = []
    for bra, ket in product(bras, kets):
        first = ",".join(str(lookup(bra, ket, axis)) for axis in axes)
        second = ",".join(
            str(lookup(bra, ket, axis, other))
            for axis, other in product(axes, repeat=2)
        )
        table.append(
            f"{{{lookup(bra, ket)}, {{{first}}}, {{{second}}}, {(-1) ** sum(ket)}}}"
        )
    tile = _half_primitive_tile(left, right, auxiliary_legs)
    threads, _ = quartet_launch_shape(left, right, auxiliary_legs)
    store_stride = str(tile) if tile else "primitive_products"
    lower_radial = _early_radial_stores if tile else _jet_function
    root_body = lower_radial(
        algebra.graph,
        tuple(root for _, root in algebra.roots),
        {
            "rho": "rho",
            "difference_x": "dx",
            "difference_y": "dy",
            "difference_z": "dz",
            **{f"boys_{order}": f"boys[{order}]" for order in range(max_order + 1)},
        },
        tuple(f"radial[{index} * {store_stride}]" for index in range(len(state_ids))),
    )
    parts = {
        "KEY": key,
        "LS": len(bras),
        "RS": len(kets),
        "LC": 3**left,
        "RC": 3**right,
        "NS": len(state_ids),
        "L": max_order,
        "THREADS": threads,
        "TILE": tile,
    }
    kernel = HALF_CONTRACTION if tile else KERNEL
    for token, part in parts.items():
        kernel = kernel.replace(f"@{token}@", str(part))
    return (
        PRELUDE.replace("@BL@", "3" if auxiliary_legs == 2 else "9").replace(
            "@BR@", "3" if auxiliary_legs else "9"
        )
        + _product_code()
        + f"""
__device__ __constant__ RadialIndex indices_{key}[] = {{{",".join(table)}}};
__device__ __forceinline__ void radial_{key}(double rho, double dx, double dy, double dz,
                         const double* boys, double* radial) {{
{root_body}
}}
"""
        + kernel
        + "\n}\n"
    )


def emit_pair_jet(
    left_angular: int, right_angular: int, *, auxiliary: bool = False
) -> str:
    """Emit Cartesian s/p jets for normalized, density-weighted pair records.

    Records contain two 3D centers, three exponents and coefficients per
    shell, then the row-major pair density (including off-diagonal shell
    multiplicity). Weights are held fixed under differentiation. An auxiliary
    slot uses the same record layout but replaces the right shell by a constant
    function of exponent zero and coefficient one, not a normalized Gaussian.
    """
    if (
        left_angular not in (0, 1)
        or right_angular not in (0, 1)
        or (auxiliary and right_angular)
    ):
        raise ValueError("expected Cartesian s or p shells")
    states = tuple(
        (xx, yy, degree - xx - yy)
        for degree in range(left_angular + right_angular + 1)
        for xx in range(degree, -1, -1)
        for yy in range(degree - xx, -1, -1)
    )
    graph = Graph()
    axes = "xyz"
    decay = graph.variable("decay")
    decay_slope = graph.variable("decay_slope")
    inverse = graph.variable("inverse")
    fraction_left = graph.variable("fraction_left")
    fraction_right = graph.variable("fraction_right")
    bindings = {
        name: name
        for name in (
            "decay",
            "decay_slope",
            "inverse",
            "fraction_left",
            "fraction_right",
        )
    }
    left_shifts = {axis: graph.variable(f"left_{axis}") for axis in axes}
    right_shifts = {axis: graph.variable(f"right_{axis}") for axis in axes}
    variables = tuple(graph.variable(f"axis_{axis}") for axis in axes)
    seeds = []
    for index, axis in enumerate(axes):
        slope = graph.variable(f"log_{axis}")
        seeds.append(
            {
                f"left_{axis}": -fraction_right,
                f"right_{axis}": fraction_left,
                "decay": decay * slope,
                f"log_{axis}": decay_slope,
            }
        )
        bindings.update(
            {
                f"left_{axis}": f"-fraction_right * displacement[{index}]",
                f"right_{axis}": f"fraction_left * displacement[{index}]",
                f"log_{axis}": f"decay_slope * displacement[{index}]",
            }
        )
    labels = tuple(
        product(cartesian_components(left_angular), cartesian_components(right_angular))
    )
    coefficients = []
    for index, label in enumerate(labels):
        terms = defaultdict(list)
        for state, coefficient in _pair_expansion(
            graph, _component_quantums(*label, left_shifts, right_shifts), inverse
        ):
            terms[state].append(coefficient)
        coefficients.append(
            {state: graph.sum(terms[state]) * decay for state in states}
        )
        bindings[f"weight_{index}"] = f"density[{index}]"

    def gradient(expression: Expr) -> tuple[Expr, ...]:
        return tuple(
            graph.differentiate(expression, variable, seed)
            for variable, seed in zip(variables, seeds, strict=True)
        )

    bodies = []
    for index, state in enumerate(states):
        components = tuple(row[state] for row in coefficients)
        effective = graph.sum(
            graph.variable(f"weight_{index}") * component
            for index, component in enumerate(components)
        )
        first = gradient(effective)
        second = tuple(child for component in first for child in gradient(component))
        basis_roots = tuple(
            node
            for component in components
            for node in (component, *gradient(component))
        )
        targets = tuple(f"weighted[{field}]" for field in range(13)) + tuple(
            f"unweighted[{field}+{component * len(states) * 4}]"
            for component in range(len(labels))
            for field in range(4)
        )
        body = _jet_function(
            graph, (effective, *first, *second, *basis_roots), bindings, targets
        )
        bodies.append(f"case {index}: {{\n{body}\nbreak;\n}}")
    return f"""
#include <cuda_runtime.h>

extern "C" __global__ void prepare_{"aux_" if auxiliary else ""}{left_angular}_{right_angular}(
    const double* __restrict__ records, unsigned pair_count,
    double* __restrict__ all_geometry, double* __restrict__ all_weighted,
    double* __restrict__ all_unweighted, int* status) {{
  constexpr unsigned state_count = {len(states)}, components = {len(labels)};
  const unsigned item = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned primitive = item / state_count, state = item % state_count;
  constexpr unsigned radials = {3 if auxiliary else 9};
  const unsigned pair = primitive / radials, radial = primitive % radials;
  if (pair >= pair_count) return;
  const auto* record = records + 27 * pair;
  const auto first = {"radial" if auxiliary else "radial / 3"};
  const auto exponent_left = record[6+first], exponent_right = {"0.0" if auxiliary else "record[9+radial % 3]"};
  const double exponent_sum = exponent_left + exponent_right;
  const double reciprocal = 1.0 / exponent_sum;
  const double fraction_left = exponent_left * reciprocal, fraction_right = exponent_right * reciprocal;
  const double inverse = 0.5 * reciprocal;
  const double displacement[] = {{record[0]-record[3],record[1]-record[4],record[2]-record[5]}};
  const double reduced = exponent_left * fraction_right;
  const double decay = record[12+first] * {"1.0" if auxiliary else "record[15+radial % 3]"} * exp(-reduced *
      (displacement[0]*displacement[0]+displacement[1]*displacement[1]+displacement[2]*displacement[2]));
  const auto decay_slope = -2 * reduced;
  const auto* density = record + 18;
  if (!state) {{
    auto* geometry = all_geometry + 6 * primitive;
    geometry[0] = exponent_sum;
    for (unsigned axis = 0; axis < 3; ++axis)
      geometry[1+axis] = fraction_left * record[axis] + fraction_right * record[3+axis];
    geometry[4] = fraction_left;
    geometry[5] = fraction_right;
  }}
  auto* weighted = all_weighted + 13 * item;
  auto* unweighted = all_unweighted + 4*(state + components*primitive*state_count);
  switch (state) {{
  {chr(10).join(bodies)}
  }}
  for (unsigned field = 0; field < 13; ++field)
    if (!isfinite(weighted[field])) atomicExch(status, 1);
  for (unsigned component = 0; component < components; ++component)
    for (unsigned field = 0; field < 4; ++field)
      if (!isfinite(unweighted[4 * state_count * component + field])) atomicExch(status, 1);
}}
"""
