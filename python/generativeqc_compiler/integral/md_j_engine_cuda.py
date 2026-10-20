# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compiler-owned compact MD radial lowering and reciprocal J contraction.

Adapted from GPU4PySCF ``gpu4pyscf/lib/gvhf-md/md_contract_j.cu`` at
9af748a71ca8f4881c604332c2f0cc1632e033ac (Apache-2.0).
Modifications: generate dependency-safe descending in-place scalar stages,
retain GenerativeQC's shared Boys implementation and folded-density convention,
and replace upstream tile/half-weight policy with caller-owned direction flags.

This is a storage/schedule lowering of the usual Cartesian Coulomb recurrence,
using the existing scalar Graph and C emitter. It owns no runtime allocation,
screening, density preparation, shell normalization, or launch policy.
"""

from __future__ import annotations

from functools import cache
from math import prod

from .expr import Graph, RematerializationPolicy
from .scalar_c import ScalarCEmitter
from .shell_spec import cartesian_components

UPSTREAM_COMMIT = "9af748a71ca8f4881c604332c2f0cc1632e033ac"
MAXIMUM_PAIR_ANGULAR = 4
MAXIMUM_RADIAL_ORDER = 2 * MAXIMUM_PAIR_ANGULAR


@cache
def _hermite_states(order: int) -> tuple[tuple[int, int, int], ...]:
    """Reorder the shared Cartesian inventory to resident ascending t/u/v."""
    return tuple(
        sorted(
            (label.count("x"), label.count("y"), label.count("z"))
            for degree in range(order + 1)
            for label in cartesian_components(degree)
        )
    )


def _emit_radial_stage(stage: int) -> list[str]:
    """Lower R(n+1) -> R(n) without retaining the fourth simplex dimension.

    For the ascending t/u/v layout each source has a strictly smaller numeric
    slot than its destination, including the compacted previous-stage layout.
    Descending stores therefore consume every old source before overwriting it.
    This replaces upstream's parallel temporary copy with scalar dependency
    ordering, not a second radial mathematical model.
    """
    previous = {state: slot for slot, state in enumerate(_hermite_states(stage - 1))}
    current = _hermite_states(stage)
    lines = [f"  if constexpr (Order >= {stage}) {{"]
    for destination in range(len(current) - 1, 0, -1):
        powers = current[destination]
        axis = next(axis for axis, power in enumerate(powers) if power)
        source = list(powers)
        source[axis] -= 1
        first = previous[(source[0], source[1], source[2])]
        if first >= destination:
            raise ValueError("MD radial stage would overwrite a live dependency")
        graph = Graph()
        variables = {
            "coordinate": "difference." + "xyz"[axis],
            "first": f"output[{first} * Stride]",
        }
        expression = graph.variable("coordinate") * graph.variable("first")
        if powers[axis] > 1:
            source[axis] -= 1
            second = previous[(source[0], source[1], source[2])]
            if second >= destination:
                raise ValueError("MD radial stage would overwrite a live dependency")
            variables["second"] = f"output[{second} * Stride]"
            expression = expression + (powers[axis] - 1) * graph.variable("second")
        plan = graph.materialization_plan(
            (expression,), RematerializationPolicy.inline_single_use_values()
        )
        emitter = ScalarCEmitter(graph, variables, plan)
        emitter.emit_assignment(expression, f"output[{destination} * Stride]")
        lines.extend("  " + line for line in emitter.lines)
    lines.extend((f"    output[0] = seeds[Order - {stage}];", "  }"))
    return lines


def _emit_contraction_tables() -> str:
    """Generate exact 8-bit summed-state maps for all ordered s/p/d pairs."""
    mappings = []
    phases = []
    for bra in range(MAXIMUM_PAIR_ANGULAR + 1):
        phases.extend(-1 if sum(state) & 1 else 1 for state in _hermite_states(bra))
        for ket in range(MAXIMUM_PAIR_ANGULAR + 1):
            radial = {
                state: slot for slot, state in enumerate(_hermite_states(bra + ket))
            }
            mappings.extend(
                radial[
                    (
                        first[0] + second[0],
                        first[1] + second[1],
                        first[2] + second[2],
                    )
                ]
                for first in _hermite_states(bra)
                for second in _hermite_states(ket)
            )
    if len(mappings) != 70 * 70 or max(mappings) >= 256:
        raise ValueError("MD contraction map exceeded its bounded 8-bit inventory")
    indices = ",\n".join(
        "    " + ", ".join(str(index) for index in mappings[start : start + 24])
        for start in range(0, len(mappings), 24)
    )
    signs = ", ".join(str(sign) for sign in phases)
    return f"""// Cooperative component lanes read different summed-state addresses; keep the
// index map read-only global rather than serializing those reads in constant memory.
// Internal linkage avoids exported CUDA data and cross-owner device linkage.
static __device__ const unsigned char kMdEngineRadialIndices[4900] = {{
{indices}
}};
static __device__ __constant__ const signed char kMdEngineKetPhases[70] = {{{signs}}};
"""


def emit_md_j_engine_header() -> str:
    """Emit deterministic compact FP64 CUDA without runtime imports or probes."""
    stages = "\n".join(
        line
        for stage in range(1, MAXIMUM_RADIAL_ORDER + 1)
        for line in _emit_radial_stage(stage)
    )
    return (
        _PREAMBLE
        + _RADIAL_PREFIX
        + stages
        + "\n}\n\n"
        + _emit_contraction_tables()
        + _emit_primitive_norms()
        + _CONTRACT
    )


def _emit_primitive_norms() -> str:
    r"""Emit Coulomb norms for the *represented* Hermite Gaussian functions.

    For Lambda_t = d_P^t exp(-p |r-P|^2), evaluating the existing Coulomb
    recurrence at P=Q gives

      (Lambda_t|Lambda_t) = 2*pi**(5/2)/(p*p*sqrt(2*p))
                          * p**|t| * prod((2*t_i-1)!!) / (2*|t|+1).

    Cauchy--Schwarz and the triangle inequality then bound an AO primitive
    pair by sum_t |E_t| ||Lambda_t||, even for signed/general contractions.
    Positive arithmetic rounds outward. Unrepresentable intermediates fail
    open; a zero coefficient is the only structural-zero shortcut.
    Runtime ownership and the cumulative omission allowance stay native.
    """
    states = [state for l in range(5) for state in _hermite_states(l)]
    degrees = ", ".join(str(sum(state)) for state in states)
    factors = ", ".join(
        str(prod(prod(range(1, 2 * n, 2)) for n in state)) for state in states
    )
    return r"""
#if defined(__CUDACC__)
static __device__ __constant__ unsigned char kMdPrimitiveDegrees[70] = {@DEGREES@};
static __device__ __constant__ unsigned short kMdPrimitiveFactors[70] = {@FACTORS@};

/** Upper bound on a Hermite Coulomb norm, not a contracted-shell estimate.
 * 35 exceeds 2*pi^(5/2). This deliberately loose exact constant avoids a
 * libm/transcendental rounding assumption in the screening certificate. */
__device__ inline double md_engine_hermite_norm(double p, unsigned angular, unsigned index) {
  const auto prefix = angular * (angular + 1) * (angular + 2) * (angular + 3) / 24;
  const auto degree = kMdPrimitiveDegrees[prefix + index];
  const auto factor = kMdPrimitiveFactors[prefix + index];
  const double denominator = __dmul_rd(__dmul_rd(p, p), __dsqrt_rd(__dmul_rd(2.0, p)));
  if (!(p > 0.0) || !isfinite(p) || !(denominator > 0.0) || !isfinite(denominator)) return INFINITY;
  double square = __ddiv_ru(35.0, denominator);
  for (unsigned i = 0; i < degree; ++i) square = __dmul_ru(square, p);
  square = __ddiv_ru(__dmul_ru(square, double(factor)), double(2 * degree + 1));
  return isfinite(square) ? __dsqrt_ru(square) : INFINITY;
}
#endif
""".replace("@DEGREES@", degrees).replace("@FACTORS@", factors)


_PREAMBLE = r"""/*
 * Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Generated by GenerativeQC's integral compiler. Adapted from GPU4PySCF
 * gpu4pyscf/lib/gvhf-md/md_contract_j.cu at
 * 9af748a71ca8f4881c604332c2f0cc1632e033ac.
 * Modifications: compiler-generated descending in-place radial stages, shared
 * GenerativeQC Boys seeds, resident Hermite indexing, overwrite contraction
 * outputs, and caller-owned directions without the upstream self-pair half.
 */
#pragma once

#include <cmath>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/direct_md_j.hpp"

namespace generativeqc::scf::cuda_execution {

"""

_RADIAL_PREFIX = r"""/** Produce choose(Order+3,3) strided R(0,t,u,v) values, t/u/v ascending.
 * At stage s the same compact buffer holds R(Order-s,t,u,v), |tuv|<=s.
 * Descending stores preserve every prior-stage dependency. The only additional
 * radial storage is Order+1 shared-owner Boys seeds; no four-dimensional
 * auxiliary, allocation, lane synchronization, or global scratch is needed. */
// Stride interleaves independent quartets across adjacent source lanes. The
// default preserves the compact scalar layout used by independent math probes.
template <unsigned Order, unsigned Stride = 1>
__device__ inline void md_engine_radial(double rho, Vec3<double> p, Vec3<double> q,
                                        double* output, double prefactor = 1.0) {
  static_assert(Order <= 8, "MD s/p/d radial order must not exceed eight");
  static_assert(Stride > 0);
  const Vec3<double> difference{p.x - q.x, p.y - q.y, p.z - q.z};
  double seeds[Order + 1];
  boys_values<Order>(rho * (difference.x * difference.x + difference.y * difference.y +
                            difference.z * difference.z), seeds);
  // Linearity permits one source prefactor in the seeds instead of one
  // multiply for every output dot. A unit prefactor retains the original
  // unscaled oracle/compatibility lowering and its rounding order.
  double factor = prefactor;
  for (unsigned n = 0; n <= Order; ++n) {
    seeds[n] *= factor;
    factor *= -2.0 * rho;
  }
  output[0] = seeds[Order];
"""

_CONTRACT = r"""
/** Count the preceding pair-angular Hermite segments (hockey-stick identity). */
__host__ __device__ constexpr unsigned md_engine_hermite_prefix(unsigned angular) {
  return angular * (angular + 1) * (angular + 2) * (angular + 3) / 24;
}

/** Exact summed-state lookup in the shared compact radial workspace. */
template <unsigned BraAngular, unsigned KetAngular>
__device__ inline unsigned md_engine_radial_address(unsigned bra_component,
                                                    unsigned ket_component) {
  static_assert(BraAngular <= 4 && KetAngular <= 4);
  constexpr unsigned bra_count = md_j_hermite_count(BraAngular);
  constexpr unsigned ket_count = md_j_hermite_count(KetAngular);
  constexpr unsigned offset = 70 * md_engine_hermite_prefix(BraAngular) +
                               bra_count * md_engine_hermite_prefix(KetAngular);
  return kMdEngineRadialIndices[offset + bra_component * ket_count + ket_component];
}

template <unsigned KetAngular>
__device__ inline double md_engine_ket_sign(unsigned ket_component) {
  static_assert(KetAngular <= 4);
  return kMdEngineKetPhases[md_engine_hermite_prefix(KetAngular) + ket_component];
}

/** Unscaled forward dot; the native schedule owns publication and reduction. */
template <unsigned BraAngular, unsigned KetAngular, unsigned Stride = 1>
__device__ inline double md_engine_forward_component(const double* radial,
                                                     const double* ket_density,
                                                     unsigned bra_component) {
  double value = 0.0;
  for (unsigned j = 0; j < md_j_hermite_count(KetAngular); ++j)
    value += md_engine_ket_sign<KetAngular>(j) * ket_density[j] *
             radial[Stride * md_engine_radial_address<BraAngular, KetAngular>(bra_component, j)];
  return value;
}

template <unsigned BraAngular, unsigned KetAngular, unsigned Stride = 1>
__device__ inline double md_engine_reverse_component(const double* radial,
                                                     const double* bra_density,
                                                     unsigned ket_component) {
  double value = 0.0;
  for (unsigned i = 0; i < md_j_hermite_count(BraAngular); ++i)
    value += md_engine_ket_sign<KetAngular>(ket_component) * bra_density[i] *
             radial[Stride * md_engine_radial_address<BraAngular, KetAngular>(i, ket_component)];
  return value;
}

/** Replace independent lane-strided outputs from one already prepared source.
 * The density already includes the resident unordered-AO-pair folding. The
 * forward density sign is (-1)^ket; for the reciprocal consumer reversing
 * R_(bra+ket) gives (-1)^(bra+ket), cancelling its (-1)^bra density phase.
 * Both directions therefore consume the same signed radial source. A self
 * source is represented by reverse=false, with no additional half factor.
 * Cooperative callers provide lane in [0,stride), stride>0; they own barriers
 * around shared radial and output storage. No output element is accumulated
 * from a prior call. Disabled directions overwrite their outputs with zero. */
template <unsigned BraAngular, unsigned KetAngular>
__device__ inline void md_engine_contract_radial(
    const double* radial, const double* bra_density, const double* ket_density,
    bool forward, bool reverse, double* bra_potential, double* ket_potential,
    unsigned lane = 0, unsigned stride = 1, double prefactor = 1.0) {
  static_assert(BraAngular <= 4 && KetAngular <= 4,
                "MD engine accepts s/p/d shell-pair angular orders only");
  constexpr unsigned bra_count = md_j_hermite_count(BraAngular);
  constexpr unsigned ket_count = md_j_hermite_count(KetAngular);
  for (unsigned i = lane; i < bra_count; i += stride)
    bra_potential[i] = forward ? prefactor * md_engine_forward_component<BraAngular, KetAngular>(
                                                radial, ket_density, i) : 0.0;
  for (unsigned j = lane; j < ket_count; j += stride)
    ket_potential[j] = reverse ? prefactor * md_engine_reverse_component<BraAngular, KetAngular>(
                                                radial, bra_density, j) : 0.0;
}

/** Primitive-local entry point sharing the same cooperative contraction math. */
template <unsigned BraAngular, unsigned KetAngular>
__device__ inline void md_engine_contract(const MdJPrimitive& bra, const MdJPrimitive& ket,
                                          MdJView md, bool forward, bool reverse,
                                          double* bra_potential, double* ket_potential) {
  double radial[md_j_hermite_count(BraAngular + KetAngular)];
  const double rho = bra.exponent * ket.exponent / (bra.exponent + ket.exponent);
  if (forward || reverse)
    md_engine_radial<BraAngular + KetAngular>(rho, bra.product, ket.product, radial);
  const double prefactor =
      2.0 * pow(kPi, 2.5) / (bra.exponent * ket.exponent * sqrt(bra.exponent + ket.exponent));
  md_engine_contract_radial<BraAngular, KetAngular>(
      radial, md.density + bra.hermite_offset, md.density + ket.hermite_offset,
      forward, reverse, bra_potential, ket_potential, 0, 1, prefactor);
}

}  // namespace generativeqc::scf::cuda_execution
"""
