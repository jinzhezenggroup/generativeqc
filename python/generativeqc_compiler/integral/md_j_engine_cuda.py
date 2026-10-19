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
        first = previous[tuple(source)]
        if first >= destination:
            raise ValueError("MD radial stage would overwrite a live dependency")
        graph = Graph()
        variables = {
            "coordinate": "difference." + "xyz"[axis],
            "first": f"output[{first}]",
        }
        expression = graph.variable("coordinate") * graph.variable("first")
        if powers[axis] > 1:
            source[axis] -= 1
            second = previous[tuple(source)]
            if second >= destination:
                raise ValueError("MD radial stage would overwrite a live dependency")
            variables["second"] = f"output[{second}]"
            expression = expression + (powers[axis] - 1) * graph.variable("second")
        plan = graph.materialization_plan(
            (expression,), RematerializationPolicy.inline_single_use_values()
        )
        emitter = ScalarCEmitter(graph, variables, plan)
        emitter.emit_assignment(expression, f"output[{destination}]")
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
                radial[tuple(a + b for a, b in zip(first, second, strict=True))]
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
        + _CONTRACT
    )


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

_RADIAL_PREFIX = r"""/** Produce exactly choose(Order+3,3) R(0,t,u,v) values, t/u/v ascending.
 * At stage s the same compact buffer holds R(Order-s,t,u,v), |tuv|<=s.
 * Descending stores preserve every prior-stage dependency. The only additional
 * radial storage is Order+1 shared-owner Boys seeds; no four-dimensional
 * auxiliary, allocation, lane synchronization, or global scratch is needed. */
template <unsigned Order>
__device__ inline void md_engine_radial(double rho, Vec3<double> p, Vec3<double> q,
                                        double* output) {
  static_assert(Order <= 8, "MD s/p/d radial order must not exceed eight");
  const Vec3<double> difference{p.x - q.x, p.y - q.y, p.z - q.z};
  double seeds[Order + 1];
  boys_values<Order>(rho * (difference.x * difference.x + difference.y * difference.y +
                            difference.z * difference.z), seeds);
  double factor = 1.0;
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
template <unsigned BraAngular, unsigned KetAngular>
__device__ inline double md_engine_forward_component(const double* radial,
                                                     const double* ket_density,
                                                     unsigned bra_component) {
  double value = 0.0;
  for (unsigned j = 0; j < md_j_hermite_count(KetAngular); ++j)
    value += md_engine_ket_sign<KetAngular>(j) * ket_density[j] *
             radial[md_engine_radial_address<BraAngular, KetAngular>(bra_component, j)];
  return value;
}

template <unsigned BraAngular, unsigned KetAngular>
__device__ inline double md_engine_reverse_component(const double* radial,
                                                     const double* bra_density,
                                                     unsigned ket_component) {
  double value = 0.0;
  for (unsigned i = 0; i < md_j_hermite_count(BraAngular); ++i)
    value += md_engine_ket_sign<KetAngular>(ket_component) * bra_density[i] *
             radial[md_engine_radial_address<BraAngular, KetAngular>(i, ket_component)];
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
