# Proposal: reduce admitted residual-J contraction before more candidate tuning

Status: proposed; diagnostic experiments completed, no new contraction promoted
Date: 2026-10-18
Measured source: `80f2d1830caa95f3bdccf5d0d4285ed7a6b7c4e7`

## Problem

The residual stage accounts for roughly 60% of prepared resident MD-J kernel
time. That includes candidate selection **and** contraction; a large rejected
candidate count does not identify which part consumes time. The bounded tile
promotion in PR #2600 changes scheduling, not this scientific representation.

## Evidence and revised diagnosis

The [source-matched measurement record](../../../benchmarks/results/md-j-residual-20261018/README.md)
retains the scope, receipts and qualifications. On n1/RTX 5090, a common accepted
96-atom/1088-AO r2SCAN-3c density gives prepared-J medians 4.199605 seconds legacy,
4.107982 seconds tiled default, and 0.712396 seconds GPU4PySCF 1.8.1. This is not
complete SCF or force timing, and the screening contracts are not identical.

In the uncounted Nsight Systems trace, default residual kernels sum to 2.450146
seconds, reciprocal uniform kernels 1.019684 seconds, and higher-order oriented
uniform kernels 0.640492 seconds. Density bounds, transform and projection total
less than one millisecond. The largest residual class `(2,1)` alone takes
0.680100 seconds; this is not a failure to launch the existing reciprocal path.

A separate unchanged-owner legacy control keeps all 526,842,030 candidate tests
but substitutes zero density or the physical density scaled by `1e-200`. Both
admit zero tasks and evaluate zero radial roots. Residual kernel sums fall from
2.535387 seconds for physical density to 0.009439/0.010073 seconds. The scaled
case retains nonzero density entries, avoiding reliance solely on an exact-zero
special case. This falsifies candidate **count alone** as the explanation of the
physical residual cost. Different branch outcomes, scheduling and memory reuse
mean these controls are not an additive time decomposition; do not call 99.6% an
exact measured contraction fraction.

## Mechanisms worth addressing

`contract_source_task` constructs contracted public-AO integral components,
applies the per-orientation AO Schwarz mask, and publishes density-weighted
atomic contributions. The higher-order source fills Coulomb intermediates in
one lane, then performs two E/R contractions with group synchronization. The
uniform native path already transforms density first, but the partially screened
residual does not share that representation.

GPU4PySCF transforms the decontracted density into Hermite form before MD-J,
caches density/geometry across many primitive-pair products, and contracts into
Hermite output before the final AO projection. Its angular-specific batching
also reuses a radial source for both density directions. These are substantive
work/representation differences, not simply a different fixed tile dimension.

Native instrumented work counts report 2,225,962,468 uniform radial evaluations
plus 242,399,046 residual primitive products. A source-replayed GPU4PySCF census
predicts 1,304,707,584 recurrence slots, including padding/symmetry-zero slots.
All 21 observed launch names, grids and block shapes agree with the replay.
This is **not** an instrumented GPU4PySCF work counter. Native higher-order
oriented work can evaluate both directions separately; angular mixes and
screening contracts differ. Do not divide aggregate times by these counts to
attribute an exact screening-versus-instruction speedup.

Native density rejection budgets `min(screening, 1e-12)` over the full contracted
shell-pair census and uses public-AO bounds/summed density envelopes. GPU4PySCF
uses decontracted pair bounds, absolute-maximum density and a per-interaction
log cutoff. Matching the numeric tolerance strings does not match those error
contracts. Tightening primitive-level bounds under the native aggregate budget
is preferable to importing a looser cutoff without new independent gates.

## Proposed direction and invariants

Prioritize compiler-owned density-contracted residual work and reusable masked
Hermite representations, starting with `(2,1)` and `(2,2)`. Preserve every
per-orientation AO mask, nonsymmetric/UKS density behavior, error allowance and
normal fallback. The partially screened mask is not generally separable: blindly
reusing the uniform density transform is not a safe implementation. Measure
actual component/atomic work before choosing an exact mask factorization.

Reduce duplicated higher-order radial sources and consider tighter conservative
primitive bounds as separate experiments. Do not replace them with more fixed
tile tuning, lower precision, or screening relaxation justified only by a single
accepted J matrix. Keep independent raw-J, complete energy/force and actual Fock
count gates; the default tile's one completed endpoint pair is slightly slower
in aggregate and two other first-cold arms remain unresolved timeouts.

Nsight Compute hardware counters are denied (`ERR_NVGPUCTRPERM`, Slurm 10111).
The timeline and source replay do not establish achieved FP64 utilization,
bandwidth, occupancy or the separate costs of recurrence, synchronization and
atomics. No driver permissions or Slurm device visibility were changed.

XC is a separate representation experiment: PR #2597 is still open/draft at
`659a6635814bef090885c529bc70ad8ee27df7f7` on this date. This work neither repeats
its occupied-frame optimization nor claims its cancellation issue is qualified.

## Subsequent implementation trials

The [bounded implementation/timing trials](../implemented/performance/2026-10-18-md-j-high-reciprocal.md)
reject register/loop expansion and task-local J publication reduction, while
retaining angular-three uniform reciprocal reuse as opt-in only. Neither
residual experiment implements masked Hermite density, so this proposal's main
representation question remains open. Complete endpoint work counts and gates,
not just the prepared-J improvement, still govern any default promotion.

Subsequently, the user explicitly authorized
[default angular-three reuse](../implemented/performance/2026-10-18-md-j-high-reciprocal-default.md)
based on the measured prepared-J and complete-endpoint gains, with an explicit
opt-out. This does not change the residual representation proposal or close its
separate qualification questions.
