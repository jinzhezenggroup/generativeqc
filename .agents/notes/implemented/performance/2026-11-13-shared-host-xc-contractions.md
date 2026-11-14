# Decision: share bounded host XC contractions before and after AD

Status: implemented
Date: 2026-11-13

## Problem

The native CPU frozen-integral projection made the retained ICH3I qualification
frame fit its unchanged one-billion primitive cap. The next complete endpoint
attempt correctly refused the independent XC map-work cap: its actual 81-AO,
87,458-point grid and 18 nuclear columns required 1,969,741,089,108 generated
map terms, above the unchanged `2^40` limit. This was not a census bug or a
reason to increase the cap. The scalar XC model and all moving-geometry sources
were already present; repeated direct AO bilinears and their differentiated
copies accounted for avoidable constant work.

## Decision

Extend the existing generic TensorIR contraction planner with the explicit
`require_degree_reduction=False` policy. Its default remains strict symbolic
degree reduction. Only the CPU build-time feature/curvature maps opt into the
new policy through `shared_contractions=True`. Planning acts on the canonical
feature primal before differentiation; exact CSE acts before AD and on selected
first, mixed, adjoint and relaxed-adjoint programs afterward.

The same existing planner chooses binary contraction trees, subject to at most
one AO axis and one batch/point axis in each new intermediate. It can therefore
reuse density-times-AO panels without introducing point-by-AO-pair storage.
The density and its response remain arbitrary signed matrices; no PSD
factorization, density clipping, occupancy change, or separate XC formula is
introduced. The mixed frozen map still excludes both density tangents; relaxed
maps retain the supplied right density response exactly once. Other consumers
keep their default direct programs and logical equation identities.

The new boolean policies require actual booleans. Cached feature owners use
typed cache keys: otherwise a prior `False` call can make an integer `0` cache
hit bypass validation. Tests exercise this warmed-cache case.

## Invariants

- Preserve normalized AOs, public matrix layouts, signed weights, total RKS and
  occupation-one UKS conventions, and physical/nonfinite input validation.
- Preserve the exact linked scalar XC ingredient-Hessian domain, including
  vacuum, empty-spin and minority-spin rejection; do not floor a rejected field
  to make a larger calculation pass.
- Continue actual-shape generated work queries and complete numeric workspace
  admission before execution. Rank limits do not bound all live storage or time.
- Preserve complete-request caps, raw Hessian symmetry gates and transactional
  publication. A map census is not molecular H/mode qualification.
- No production reference engine, finite difference, runtime compiler or public
  ECP saddle admission is added by this schedule change.

## Rejected alternatives

- Raising the work cap would merely allow the same redundant schedule and hide
  the next real scalability boundary.
- Exact CSE on the direct n-ary contractions alone still exceeds the retained
  complete-request cap. Binary splitting without sharing before/after AD is
  likewise insufficient.
- A handwritten AO contraction, XC Hessian, new derivative algebra or
  positive-density factorization duplicates existing compiler ownership and
  risks breaking signed response semantics.
- Enabling degree-neutral scheduling for every consumer would change reduction
  order and generated identities without scientific qualification for them.
- Treating smaller map work as an endpoint speedup omits scalar XC, AO/grid,
  integrals, CPKS, ECP and source assembly. Complete endpoint timing is separate.

## Evidence

The source-bound, ignored cohort is
`build/qualification/host-shared-xc-maps-20261113`. Production source identity:
`c174b666cd1b08a8aa109fac26920711114d7aeb0fb42e42c54154430c7a505e`.
Its sealed parent is `host-projected-curvature-20261113`, archive SHA256
`75d2f04c6a16135d5fc830ea5b68df9728da25cd35b58cf7749550223a0880e6`.
The parent remains immutable and documents the genuine earlier refusal.

- `test_tensor_complexity.py` and `test_host_grid_maps.py`: 35 passing tests.
  Four isolated compiled namespaces compare canonical/shared first/curvature
  registries, including signed cotangents, zero extents, scalar JVP stencils,
  bounded intermediate fallback and transactional failures. Numerical parity
  uses the unchanged `3e-12` gates. Actual-shape compiled work queries require
  shared work below the unchanged cap and direct work above it, not an exact
  pinned instruction count.
- `test_host_xc_nuclear.py`, `test_host_xc_curvature.py` and
  `test_host_xc_column.py`: 65 passing independent normalized-AO/Libxc component,
  raw symmetry/translation, supplied-density, domain and safety tests.
- Ten native/API/CLI test groups and 63 public CPU/source-manifest tests pass.
  Compiler/SCF/electronic ownership checks, Ruff and whitespace checks pass.
- An opt-in parent-archive/current-source audit confirms 33 default logical
  equations unchanged at three AO/point shapes. This is migration evidence,
  not a permanent required-CI historical source fingerprint.
- The actual native plan and an independent emitted-loop census agree on
  867,660,477,948 complete XC map terms at the unchanged physical grid, below
  the unchanged 1,099,511,627,776 cap. First/curvature numeric queries are
  17,603,856 and 30,557,832 bytes respectively, not aggregate endpoint RSS.
  The independent integral census still reports 882,874,977 complete first
  plus projected ERI primitives under the unchanged one-billion cap. None of
  these metadata audits executes an electronic or derivative evaluation.

These component gates do not certify the actual stationary ICH3I raw Hessian,
unscaled frequencies, mixed Ni/ECP mode controls, a practical 25--40-atom
endpoint or the requested Ni transition-state optimization. Keep those gates
and public ECP saddle guards separate.

## Complete endpoint observation

The changed full native ICH3I attempt passed the old XC admission point. It
completed its actual cold stationary energy/gradient/orbital checkpoint at
546.476 seconds, then reached the unchanged finite 1,800-second wall limit
without publishing a complete raw Hessian or modes. Saved-state independent
validation passes: energy error `3.411e-13` hartree, maximum gradient error
`3.929e-12` hartree/bohr, density error `1.156e-11`, and minimum occupied-root
overlap `0.9999999999999969`. No new reference SCF or displaced-state call runs.
This is not stationary Hessian qualification or an endpoint speedup claim.

The cohort is sealed with SHA256
`254e90d11b7df30f0c6b68d6df929ce76492d5890be1b9483421bf5ef7f1983f`.
Do not rerun its exclusive preparation/retention scripts or repeat this
unchanged expensive endpoint. Required final validation comprises 271 passing
tests/groups, without skips. Additional default-grid controls have nine passes
and two expected optional-CUDA-probe skips; no GPU runs.

Completed stationary two-electron gradient work is separately journal-owned:
255,078,161 primitive evaluations in 195,391 shell submissions. Hessian counters
publish completed API results, not unfinished preparation. Their zeros cannot
be read as zero Hessian work. The later preparation/column stages lack
phase-resolved progress in this attempt, so missing projection records cannot
identify the particular hotspot. CPU perf sampling was denied by the existing
host policy; no kernel or ptrace policy was changed.

A separate, test-only first-leaf investigation in
`build/qualification/host-first-leaf-hypothesis-20261113` compares two already
linked implementations on eight s--f Cartesian quartet classes. The factorized
first specialization is 3.15--14.17 times faster in these isolated leaf timings,
with maximum derivative difference `7.55e-15`. The older derivative entry
evaluates its recurrence separately for each of three independent centers;
the factorized specialization shares their primal algebra. This makes reuse
worth investigating, but is not an independent scientific oracle, actual
molecular schedule profile or complete-endpoint timing result. No production
first-source backend is changed by that experiment. Qualify any future switch
against normalized-AO/independent primal controls and actual complete endpoints.

A separately qualified follow-up reuses the existing factorized first leaf in
the native integral first source; see
`2026-11-13-factorized-host-first-sources.md` in this directory. It supersedes
this earlier XC-only change's unchanged-first-source boundary, not its retained
XC schedule, caps or scientific qualification requirements.

## Revisit when

Actual endpoint profiling identifies remaining repeated map work, or a new
consumer can independently qualify its changed reduction order and complete
resource schedule. Reprice actual runtime extents rather than extrapolating
the compile-time template's contraction-tree score.

## References

- Current contract: `docs/developer/dft_hessian.md`, native CPU moving-grid XC.
- Existing projection rationale:
  `2026-11-13-frozen-eri-polynomial-density.md` in this directory.
- Qualification scripts `audit-xc-plan.py` and `audit-default-identity.py` in the
  ignored cohort retain independent work and compatibility evidence.
