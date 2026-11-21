# Decision: Fit integral legs before forming CUDA density-fitting cotangents

Status: implemented
Date: 2026-11-21

## Problem

The opt-in M06-2X and MN15 stationary endpoints reached native SCF and
analytic forces for spherical HF/cc-pVDZ with def2-universal-JKFIT, but
their Hessian preparation returned a generic invalid-source/cutoff error.
The independent Libcint metric had a minimum relative eigenvalue of
`5.72e-8`, more than 570 times the `1e-10` cutoff. Slurm job `1734`
located the actual rejection at the three-center cotangent symmetry audit,
not at the metric domain audit. In contrast, H₂⁺ passed with the full H
JKFIT set and HF passed with Direct J/K on the same binary.

## Decision

Use the symmetry of the physical AO-pair and auxiliary source spaces in the
compiler derivative. With `B = A M⁻¹`, both equivalent exchange pullback
legs produce the same dressed `B`; apply the inverse before contracting
with the density and before taking the auxiliary outer product. On the
full-rank branch, use `B` on both legs of the metric cotangent. The
fixed-rank branch still produces the inverse cotangent and applies the
separated spectral response in its native owner. Differentiate the same
reassociated weights for the action. In full rank `B M = A` gives
`dB = (dA − B dM) M⁻¹`; applying the final inverse after forming the
residual avoids an intermediate dense `−M⁻¹ dM M⁻¹` and its amplified
rounding error in the weak metric eigenspaces. Density cotangents and their tangents
use the adjoint on the symmetric density space, averaging contributions
from the two AO orientations in the compiler graph. Project the accepted
AO-pair inputs and the full-rank metric direction onto their physical
symmetric subspaces before the fit, without bypassing their pre-fit audits.

Keep native input, finiteness, source symmetry, cutoff, and raw assembled
Hessian checks at their existing tolerances. Distinguish the device-side
rejections without downloading tensors, installing a host fallback, or
adding a synchronization between the checks. The generated work/arena
planner still bounds every selected map for both spin conventions.

For fixed-geometry induced Fock, retain the accepted SCF whitening applied
once to the raw AO pairs, not two raw panels contracted through a preformed
`WᵀW`. The whitened J/K Gram uses a factorized compiler schedule, including
all validation work. The immutable short/full- and long-range radial
providers have separate whitened panels within the existing device budget;
they share only one raw staging panel. The second-order weights still use
the raw metric branch and its actual nuclear differential, not a frozen fit.

## Rejected alternatives

- Lowering the metric cutoff or relaxing symmetry tolerances would not
  address the amplification from applying the inverse to an already rounded
  density contraction. Replacing the fit with exact J/K would change the
  stationary energy being differentiated.
- Symmetrizing a published molecular Hessian cannot repair an inconsistent
  frozen Fock derivative or a response computed from the wrong fit. Retain
  the raw molecular symmetry gate and compare to reconverged gradients.

## Evidence

- `tests/python/test_wb97mv_df_response_compiler.py` checks fitted energy,
  Fock, source derivatives, and their motion against independent NumPy
  expansions/solves. It also tests factorized work admission.
- `tests/python/test_native_wb97mv_df_response.py` includes a rotated
  full-rank metric with condition number `2e7`, signed RKS/UKS densities,
  independent linear solves and complex-step checks of moving weights, plus
  failure-injection checks at the metric and source gates.
- The angular replay in `build/hessian-gpu-sm120/r2scan-orbital-f/` uses
  independently reconverged PySCF/Libxc gradients with central differences
  at `8e-4` and `4e-4` Bohr and Richardson extrapolation. HF and H₂⁺ use
  spherical cc-pVDZ, with Direct and full def2-universal-JKFIT providers.
- RTX 5090 Slurm job `1739` passed all eight model/spin/provider molecular
  oracles and 46 fitted-component tests. The largest Hessian error was
  `1.15e-9 Ha/Bohr²`; the raw symmetry, translation and dense/HVP replay
  checks also passed with no numerical host fallback. HF/JKFIT full dense
  endpoint (including its reference preparation, excluding the separate SCF)
  took `24.29–24.32 s` with 82 induced Fock actions and six raw-DF value and
  contraction passes each. The single-action endpoint took `4.22–4.23 s`
  with 24 induced Fock actions and one pass of each kind. These are single-run
  measurements, not a DF speedup claim.
- RTX 5090 Slurm job `1740` replayed the same eight independent molecular
  references through the public Python and native CLI Hessian and HVP
  endpoints. Job `1741` passed eight checked-in public parity tests that
  use a synthetic spherical d orbital shell and, for DF, a g auxiliary shell.
- Jobs `1743` and `1745` located the water/cc-pVDZ JKFIT failure at the
  accepted-Fock reconstruction audit. In `1748` the whitened path passed
  both water/JKFIT independent molecular oracles and public Python/CLI
  parity. Their largest Hessian error was `1.99e-9 Ha/Bohr²`; both use nine
  raw-DF passes of each kind and 113 induced Fock actions for the dense result.
- HF/cc-pVTZ with the same JKFIT basis no longer hits the induced-Fock
  term bound, but its nuclear fitted-map plan requires about `206 million`
  action terms at `44 AO × 95 auxiliary`, above the explicit `100 million`
  cap. That rejection is not evidence of a numerical failure. It does not
  qualify orbital f for the public split-hybrid interfaces.
- Job `1749` used a separate, local-only test CLI with a `224 million` term
  cap and replayed both cc-pVTZ RKS models. The maximum Hessian error was
  `6.78e-10 Ha/Bohr²`, with raw symmetry, translation and HVP parity intact.
  Dense end-to-end executions took about `142 s` at 156 induced actions and
  six DF value/contraction passes each; the HVP took about `25 s`, 26 actions
  and one pass each. Production caps and public shell admission were unchanged.
- The final production-configured build passed 24 public frontend and
  cross-model regressions and 46 fitted-component regressions in job `1750`.
  The public set includes both split hybrids, fitted PBE, r²SCAN+rVV10, and
  WB97M-V full-rank/fixed-rank fitted LR with cutoff rejection.
