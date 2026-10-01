# Decision: reuse compiler-owned Coulomb roots across CPU ERI components

Status: implemented
Date: 2026-10-01

## Problem

#1667 already prepares geometry and Boys values once per primitive shell quartet,
but Cartesian components still reconstruct common Coulomb derivative roots.
Baseline whole-endpoint sampling attributes approximately 8.5% of the water and
formaldehyde def2-SVP endpoints to component algebra. This is a bounded remaining
opportunity: the earlier interpreter and geometry-reuse gains cannot be claimed
again, and sampled attribution is not a latency comparison.

## Decision

Reuse the shared compiler mathematical owner, `_coulomb_derivative`, to
expose the existing roots and emit a second value-only CPU schedule. Prepare the
requested graded root prefix once for each primitive geometry, then consume it
across the canonical shell components. Keep the component-local generated
schedule and existing derivative/f+ fallbacks. Native code remains responsible
for traversal, contraction, normalization, scatter, and bounded scratch, without
new recurrence formulas or separate HF/DFT scientific implementations.

Mixed f+ systems retain component-at-a-time `primitive()` calls for their s/p/d
quartets. Eagerly preparing every requested graded root inside that compatibility
wrapper would repeat the full table for each component. Keeping the byte-identical
local factored DAG avoids this work amplification. This concrete compatibility
need is the reason to accept the generated-code growth of two schedules.

The caller-owned scratch has 165 doubles plus an order field, approximately
1,328 bytes on the measured ABI. It initializes only the requested degree prefix:
1, 4, 10, 20, 35, 56, 84, 120, or 165 values for maximum orders zero through eight.
There is no cross-geometry cache and no molecule-wide primitive-quartet storage.

## Rejected alternatives

- A handwritten native Coulomb recurrence would create a second scientific owner
  and an avoidable CPU/CUDA or method-specific fork
- Removing the component-local schedule in favor of eager full-prefix preparation
  inside the compatibility wrapper would amplify root work for mixed f+ systems
  that still request one s/p/d component at a time
- Treating a theoretical arithmetic reduction or sampled PC share as an endpoint
  speedup would hide preparation/storage overhead and the remaining SCF work
- Keeping the first pilot's cold-speedup claim would confound scientific reuse
  with unequal Python bytecode preparation
- Adding a molecule-specific selector after noisy H2 results would not establish
  a general compiler demand or scheduling rule

## Invariants

The shared shell-class value/derivative equations remain the mathematical owner.
Two generated schedules do not authorize two independently maintained formulas.
Axis remapping, center permutation, and odd parity under bra/ket exchange must
follow the generated symmetry maps. Tests protect every supported root prefix and
the full s/p/d component inventory; derivative/f+ consumers keep explicit fallback.

Scratch must be prepared for the exact current primitive geometry before every
shared consumer. Matching maximum order only checks an order/resource precondition;
it cannot detect stale geometry. A reused scratch object must be re-prepared even
when the next geometry requests the same order. Uninitialized suffix values must
never be consumed.

The endpoint protocol remains fresh-core-guess FP64 exact in-core RHF, with
unchanged BSE basis decimals, DIIS history eight, 1e-10 Ha energy and 1e-8 density
RMS gates. No density reuse, density fitting, relaxed convergence, timing-based
sample removal, or iteration-based numerical filtering is allowed.

## Evidence

The primary qualification uses a fresh baseline build of the exact #1667 merge
`f678b01` and frozen candidate `a7afa3864f12a747b1851a25f97db5500c58c36c` on the same
host. Twenty interleaved rounds, four cases, three engines, and four endpoint
calls per process give 960 endpoints. All converge, native iteration counts stay
identical, and maximum all-repeat errors are 8.53e-14 Ha against baseline and
4.55e-13 Ha against independent PySCF/libcint. The fixed frozen endpoint script and
basis payload are unchanged from the earlier campaign.

Formaldehyde/def2-SVP warm pooled medians are 298.606 → 286.679 ms (4.0% lower
elapsed time, 1.042× speedup). The separate median process-pair ratio is 0.95849
with 95% percentile-bootstrap interval [0.92444, 0.98672], resampling processes
rather than correlated warm calls. Cold and changed-geometry paired intervals
also favor the candidate. Water warm intervals include one. H2 warm pooled
medians are 1.389 → 1.461 ms; its interval is wide and establishes neither a gain
nor non-regression. Keep the narrow formaldehyde finding rather than claiming
general speedups, PySCF parity, force/DFT scaling, or CUDA qualification.

The nine-round 432-endpoint pilot remains retained, but its cold measurements
were confounded by 57 baseline versus 289 candidate source-tree bytecode files.
All selected source trees were normalized to 477 files before the primary run;
the normalized baseline uses its actual build/source repository. No scientific
source or native library changed. Retain this failure mode so future comparisons
control interpreter-cache state as well as native identity and thread settings.

Baseline/candidate native suites pass 57 tests each; the candidate passes 39
codegen tests, 186 compiler regressions (44 skips), nine public HF/PBE/PBE0
energy/gradient checks, and final pre-commit. Independent review verifies 313
unchanged component-local function bodies, 6,334 cached-root reference sites,
six axis maps, and 26 exact value/first-derivative DAG identities. The auxiliary
untimed HF/PBE/PBE0 campaign passes six cases, retaining two failed oracle pilots
and one final STO-3G PBE self-refinement from its own oracle density. No oracle
density enters the native runs or timed endpoints.

Source-derived unscreened root-demand counts fall from 2,344,466 to 319,254 for
water/def2-SVP and from 18,759,192 to 1,995,222 for formaldehyde/def2-SVP, without
changing primitive-quartet/component counts. These counts are not runtime FLOPs.
The retained static arithmetic audit is a hypothetical DAG design estimate,
not the exact emitted two-schedule operation count or a speedup predictor.

## Consequences

Preserving both generated schedules increases the ERI header from 2,745,678 to
4,842,970 bytes (+76.4%) and native `.text` from 18,843,047 to 19,317,369 bytes
(+474,322). Complete build observations are 491.1 → 518.7 seconds and maximum
child RSS 3,771,236 → 4,460,064 KiB; differing qualification concurrency prevents
a causal compile-time or peak-memory comparison. No isolated build regression
claim follows. Independent generated-file review leaves 107 artifacts exactly
unchanged and eight Libxc files different only in source-bound identity strings.

The larger generated inventory and explicit scratch-lifetime contract buy a small
measured endpoint improvement, not a new asymptotic algorithm. Baseline sampled
profiles still point to Jet storage/lifecycle and native SCF as substantial work;
unknown-caller samples must not be silently attributed to a preferred subsystem.

## Revisit when

- Matched larger-system or additional-host endpoints establish whether reuse
  generalizes beyond the primary formaldehyde case
- Compiler demand pruning or a general resource-based selector can reduce small
  shell overhead or duplicated generated code without molecule-specific tuning
- A controlled isolated build study quantifies the code-size/compile-memory cost
- Storage/scatter or SCF work becomes the next independently qualified slice
- New consumers need persistent scratch, derivatives, range separation, or CUDA;
  re-establish scientific identity/lifetime and oracle gates before broadening use

## References

- `benchmarks/results/cpu-eri-coulomb-20261001/README.md`: compact evidence,
  identities, all campaigns/failures, and portable reproduction commands
- `2026-10-01-cpu-eri-shell-geometry-reuse.md`: preceding geometry-reuse decision
- `docs/maintainer/performance_engineering.md`: endpoint, work, and oracle policy
- #762 / #926: compiler-first shared scientific ownership
