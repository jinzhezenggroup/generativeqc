# Proposal: share native XC source work across independent response directions

Status: proposed; private component prototype only
Date: 2026-11-14

## Problem

The current #2844–#2848 full physical matrix request still fails its unchanged
1800-second cap after 17 of 18 completed coordinates. Its completed journal
contains 258 Arnoldi actions taking 341.3453 seconds and 257 fresh exact-residual
replays taking 340.1598 seconds. The interrupted next replay is not completed
work. The previously isolated induced-Fock bottleneck is XC, not J/K. Keep each
exact residual replay; do not change its cadence or substitute a linear
combination of saved operator images for a fresh physical action.

The retained production source identity remains
`2bbcc04c3142115739d58db9389b8d4d4ae2881eedb206237ccb4f3b64e61fd9`, with core
SHA256 `0ab7928b126aa821568ecdb45ac9e6e6cf43278c3107331693017bd839ac8263`.
Nothing below implements a production batched solve or supersedes that request's
failure. The full fix, installed endpoints, mixed heavy elements/spins, raw
Hessians/modes and the representative native Ni saddle remain incomplete.

## Investigated alternatives

A complete native projected frozen-integral diagnostic observes all
130,597,029 matrix calls, split 58,747,446 / 61,158,798 / 10,690,785 by one,
two and three physical owners. Its explicit zero W makes it an integral
component only. It forwards every evaluation to the unchanged native core;
no simulated cache supplies scientific data.

At 131,072 direct-mapped entries, exact-record/exact-order geometry reuse is
92.6272%, and exact-T-bit/exact-order positive Boys reuse is 94.1943%.
Nevertheless, owner-weighted repeated-input samples put geometry construction
on an approximately 11.835-second scale, and the moment subset on a
7.099-second scale, versus 285.294 seconds of sampled native leaves. These
are warm sampled cost estimates, not rigorous bounds or endpoint timings.
Lookup, key comparison, copying, allocation and actual streaming costs remain
unpaid. High hit rates alone therefore do not justify a new production cache.
Do not repeat this experiment merely because its hit rate looks attractive.

The instrumented component finishes in 397.9273 seconds. Its time must not be
compared with earlier uninstrumented physical frozen preparation as a speedup:
it performs additional geometry/simulation work, uses zero W, and is a separate
request. Initial compile and two pre-computation authentication failures remain
retained. The library-hash check needed the canonical regular ELF path, not its
SONAME symlink. Saved-state acceptance comes from the authenticated physical
recovery and independent validation receipts, not the original incomplete
whole-Hessian checkpoint's `accepted` flag.

Skipping only a fixed primal-field contraction saves little: on the actual
87,458-point, 81-AO grid, paired field costs are approximately 0.608 seconds
for primal+tangent versus 0.567 seconds for tangent only. A new cross-call field
cache would additionally need a sound reader/input/floating-environment proof.
It is not implemented or justified by these component observations.

## Private scheduling experiment

Instead, four independent directions within one call share each ordered AO-pair
product. SIMD lanes in the field kernel still represent rho/x/y/z, never terms
of an AO reduction. A complete XC batch evaluates the shared fixed-D primal
alongside the four tangents, retains the existing lazy vacuum branch and linked
scalar XC/AD and coefficient-response programs, and shares AO products in the
pullback. Pullback SIMD lanes are independent directions; each result still
accumulates points and rho/x/y/z in incumbent order. No FMA, reassociation,
symmetry repair, density screen, new derivative algebra or cross-call primal
field cache is introduced.

The four diagnostic matrices are a previously authenticated native D', native
D, a finite signed nonsymmetric variant, and a finite linear combination.
Saved D' is explicitly a component input, not a newly solved nuclear response
or a substitute Krylov vector. The normal native core remains loaded, with a
separately compiled, uniquely named private class: this is not execution of a
new production core.

The actual-grid field-only experiment compares 12,593,952 fields bitwise after
each individual kernel, including both orders of alternating pairs. Four serial
tangent contractions take a median 2.26845 seconds versus 0.96756 seconds for
the shared-product version. This interior ratio is not projected onto an
endpoint. An initial harness checked outputs only after the second kernel in
an alternating pair; its original record and explicit negative coverage audit
remain retained, rather than being treated as a complete bitwise control.

Complete four-direction fixed-grid XC actions give:

| Component configuration | Native sequential median | Private batch median |
| --- | ---: | ---: |
| Streamed native AO reader | 5.65869 s | 2.94133 s |
| Original native AO-prefix policy and response cap | 4.93057 s | 2.86677 s |

The latter is an observed 41.8572% component reduction. Both experiments use
three alternating warm pairs in one process, not independent repeated endpoint
trials. Each checks all four complete AO matrices after both paths, totaling
157,464 bitwise comparisons per experiment. All complete outputs also match
between streamed and prefix readers. Timings include AO reading, field and
coefficient work, scalar-map consumption, allocation, packing, pullback and
publication; they exclude J/K, orbital transforms, Krylov synchronization,
nuclear sources, SCF and full Hessian assembly.

The matched-prefix experiment explicitly prices 546,120 additional numeric bytes
for the private in-call workspace. The original 37,632-point / 98,456,992-byte
AO prefix shrinks to 37,376 points / 97,787,296 bytes. Composed reservations are
133,877,352 bytes for the baseline and 133,753,776 for the candidate, both inside
the original 134,217,728-byte response ceiling. Crucially, these reservations
still include only the original single GMRES workspace: they do not qualify
memory for four simultaneous Krylov contexts. Report actual provider work as
well as four fully produced physical directions, not an invented fourfold
reduction in scientific outputs.

## Production direction, not an implemented decision

If further qualified, expose bounded same-frame multiple-direction execution in
the native XC/induced-Fock provider and drive it from independently progressing
Krylov states. This is batching independent right-hand sides, not changing their
mathematical problem to block GMRES, assuming common convergence, or discarding
fresh candidate-residual actions. Each RHS retains its own initial residual,
preconditioning, Arnoldi/reorthogonalization, Givens solve, convergence target,
restart, stagnation, breakdown and failure status. Preserve the existing
absolute 1e-12 solve and independent 2e-10 physical-response gates.

Do not duplicate the compiler-owned orbital-density/Fock projection equations
from `response::spin_orbital_problem` merely to expose a batch. Keep one
auditable numerical controller for single and batched execution. Avoid a hidden
thread rendezvous around the synchronous callback: existing borrowed buffers,
busy guards, budget and failure propagation do not authorize concurrent calls.

Required work before promotion or another full-matrix request:

- Implement generic bounded batch admission with truthful numeric and semantic
  work accounting, including every simultaneous Krylov context and staging
  panel inside the original aggregate and response caps.
- Retain the ordinary sequential schedule on optional capacity, allocation or
  backend-support misses; never turn scientific failure into fallback success.
- Cover variable batch sizes, tails, UKS cross-spin response and a portable
  non-AVX2 path. The private prototype is four-direction RKS on this AVX2 CPU.
- Exercise output bounds, aliasing, failed/incomplete partial-bank preparation,
  mutation/invalidation, reentry, cancellation, failed lanes and recovery, with
  independent scientific and sanitizer qualification of the real consumers.
- Qualify newly solved coupled responses, including each fresh true residual,
  and measure complete all-coordinate work/time before claiming endpoint gains.
- Only a materially changed native integrated consumer may justify a new full
  raw-H/modes request under the existing cap. Do not rerun the unchanged failed
  full request or splice its missing column from another calculation.

## Evidence and retention

Ignored local cohorts:

- `build/qualification/host-geometry-reuse-profile-20261114`
- `build/qualification/host-fixed-fields-cost-20261114`
- `build/qualification/host-batched-xc-candidate-20261114`, including
  `matched-prefix` and separately instrumented sanitizer work.
- `build/qualification/host-geometry-and-batched-xc-retention-20261114`

The cross-cohort audit authenticates executed source/input/core hashes and work
counts. Sanitizer observations are separate from ordinary timing; do not infer
failure/lifetime coverage from a successful valid-input component run. The
separate ASan/UBSan run passes all 157,464 complete-output bitwise controls in
182.0777 seconds under its finite 240-second diagnostic cap. The new private
class, batch kernels, AO-prefix reader and driver are instrumented; the unchanged
shared core/providers are not. This is valid-input bounds/undefined-behavior
evidence, not allocation-failure, aliasing, reentry, cancellation or UKS coverage.
The
retained archive is a diagnostic delta requiring the previously sealed physical
state/private-continuation/production evidence archives, not a new self-contained
release, installed endpoint or external backup.

Disk pressure required moving only the reconstructible in-tree build ELF to
tmpfs and replacing its build pathname with a symlink. The identical durable
qualified baseline remains untouched, with its verified recovery binding stored
in `host-geometry-reuse-profile-20261114/build-artifact-storage.json`. No
scientific evidence or compiler cache was cleared.

## References

Implementation continuation: the subsequent
[independent native response batch decision](../implemented/architecture/2026-11-14-independent-native-response-batches.md)
records actual shared numerical/controller/nuclear/column integration. It does
not turn this proposal's private measurements into production endpoint evidence
or supersede the failed whole-matrix request.

- [Current native AO-prefix policy](../implemented/performance/2026-11-14-bounded-native-response-ao-prefix.md).
- [Current production integral lowering](../implemented/performance/2026-11-14-live-bin-canonical-owner-matrix-lowering.md).
- `src/dft/host_xc_response.cpp`, `src/dft/host_gga_ordered.hpp` and
  `src/response/native_gmres.cpp`.
