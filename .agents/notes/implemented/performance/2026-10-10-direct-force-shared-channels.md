# Decision: share shell derivatives across nuclear J/K weight channels

Status: implemented
Date: 2026-10-10

## Problem

The native shell force source adapter shared primitive geometry and radial
moments but invoked the weighted derivative recurrence independently for J and
K. Its source loop also decoded each canonical AO symmetry orbit twice. This
duplicated invariant integral work even when both outputs consumed the same
primitive shell.

## Decision

Keep the scientific weighted ERI DAG as the sole recurrence/derivative owner.
Because its fixed external cotangents enter linearly, differentiating nuclear
gradient roots with respect to each cotangent exposes the existing unweighted
component derivatives. Emit those roots using one persistent scalar CSE state
per bounded subset, consuming each component immediately across all active
channels. No large derivative tensor, second recurrence implementation or
method-specific density formula is added to integral lowering.

Generate independent density-channel weights from the scalar native weight
equation with changed bindings only. The canonical symmetry orbit is decoded
once; each source still adds J then K in the original permutation order. The
retained scalar weight source is byte-identical to the parent implementation.

The shared shell adapter covers its nine order-zero-through-three classes and
the five/three admitted s/p/d order-four/five classes. Larger classes retain
64-component additive partitions and share within each partition. Existing
uncovered/high-angular owners and admission selectors are not changed.

Combined, long-range-only and single-active-source requests keep the retained
weight-first recurrence. For two active channels, shared evaluation publishes
transactionally. If an unweighted active component or contraction is nonfinite,
replay the original weight-first evaluator for each active source. This bounded
fallback preserves its intermediate dynamic-range behavior; final native audits
still reject a nonfinite result. No threshold, approximation or screening is
introduced by fallback or exact-zero channel/component handling.

## Invariants

- Fixed cotangents carry normalization and source coefficients exactly once.
- Canonical shell slots, Cartesian component order and AO symmetry multiplicity
  remain owned by the existing native adapter and generated weight equation.
- J and K remain independent even when their sum cancels; inactive channels do
  not read their weights, and zero weights do not multiply nonfinite integrals.
- Primitive accumulation order and translation/physical-atom recovery are
  unchanged. Shared component contraction reassociation is explicit.
- Generated outputs require regeneration/recompilation. Public APIs, task ABI,
  persistent numeric capacities, host/device ownership and stream leases do not
  change. Register/local-memory use is not claimed unchanged.

## Rejected alternatives

Calling the old weighted helper twice shares only geometry, not the derivative
integrals. A second handwritten derivative recurrence would split scientific
ownership. Retaining a shell/molecular derivative tensor would increase numeric
storage and complicate leases. Combining J and K weights would erase separately
observable sources and introduce unsafe cross-source cancellation.

## Evidence

Worktree: `/data/jzzeng/qc-ccsdt-cold-master-20261008-4385f727`, parent
`0e3bdbc72721c6d6a4d2f56d4822ccbed7060096`. The dirty main checkout is untouched.

- Host native low-order census: 19,540,224 coordinate comparisons, all shell
  orientations, both spins, repeated atoms, masks, screening, zero sources and
  resident-bra lease/fallback cases. Geometry/recurrence calls fall from
  1,311,456 to 1,043,280; 268,176 calls consume both channels. Maximum absolute
  difference from retained scalar source workers is 4.441e-16. Combined and
  single-source controls retain bitwise equality. Shared-channel comparison uses
  an explicit 3e-12 absolute-plus-relative gate instead of the old bitwise
  scheduling control, because component summation is intentionally reassociated.
- Independent displaced-value Hermite recurrence: 15,840 low-order coordinates,
  all components and dense signed weights, coincident/noncoincident centers,
  exponent scales 0.4/1/3, and two five-point steps. Maximum error 3.074e-10.
- Independent order-four/five gates retain the existing 2e-8 derivative and
  2e-10 adapter gates. Maximum displaced-value errors are 3.356e-10 / 6.844e-10;
  maximum native adapter differences are 1.666e-16 / 1.388e-16. All components,
  including every 64-component partition boundary, are covered.
- Compiled three-channel/subset/zero/publication tests and independent density
  orbit checks pass. A rejected shared evaluator exercises weight-first fallback.
- Compiler ownership: 505 modules, zero dependency errors. Focused native
  scheduling/lifetime/identity/ownership suite: 588 passed. Final focused union:
  648 passed; the separately executed order-four/five gates bring the distinct
  focused total to 650. CUDA ownership inventory: 339 files, check passed.
- Actual generated CUDA roots for all 17 classes pass on the RTX 5090 through
  Slurm on n1 (`node1`, main partition, `gpu:5090:1`, three-minute time limit,
  job 6853). All 1,224 comparisons pass, maximum difference 5.552e-16; assigned
  visibility was preserved. NVCC V12.9.86 targets sm_120 with `--fmad=false`.
  Compute Sanitizer memcheck of the same probe, also through finite Slurm, reports
  zero errors. Local and remote generated/native/probe SHA256 identities match.
  The existing ccache 4.5.1 cache is reused, with before/after receipts retained.
- The comparison probe contains both retained and shared evaluators. Its ptxas
  output reaches 255 registers and a 16,112-byte stack frame with nonzero spills.
  These are not isolated production-kernel measurements, but they explicitly
  prevent claiming unchanged occupancy/local memory or a net speedup.

Raw logs and the reproducible CUDA probe are retained under
`.artifacts/shared-shell-channels/`. Real-device qualification is separate from
the host Hermite oracle gates. The initial kernel qualification did not measure
a complete endpoint; the work census alone is not a speedup measurement.

### Complete-endpoint follow-up

The PR qualification worktree is
`/data/jzzeng/qc-shared-shell-channels-pr-20261010`, frozen on master
`2daa0aceaf6d140389ac5a183ead8b0ac3d3cd4a`. Native scientific changes apply without
conflict; current public PBE0 force qualification also builds the matching
`generativeqc_stationary_pbe0_rks_spd_manifest` target. An older common auxiliary
artifact was rejected by its scientific contract, not force-loaded or repaired.

Final Slurm job 6943 measures complete public PBE0/def2-SVP spherical energy and
all analytic force coordinates at 24/96 AO on one RTX 5090 allocation. Both
Release sm_120 libraries have identical nonempty 22-entry shell inventories,
the same precision/scientific settings and verified ccache launchers. ABBA order
provides two fresh processes per side, each with cold preparation, fixed-density
warm, changed-geometry and moved-warm endpoints. The explicit Separate route
consumes both channels; Combined is the retained weight-first control.

| AO | Separate warm baseline/candidate, seconds | Separate moved-warm baseline/candidate, seconds |
| --: | :-- | :-- |
| 24 | 0.169086 / 0.195760 | 0.167352 / 0.194614 |
| 96 | 0.873220 / 0.917002 | 0.879459 / 0.922607 |

This is a negative endpoint result: warm regressions are 15.8%/5.0%, moved-warm
16.3%/4.9%. Cold differences and smaller Combined-control changes do not justify
a speedup claim. Every one of the 64 final endpoints passes unchanged independent
energy/force gates of 1e-8/1e-7, with maxima 3.127e-12/2.628e-11. Iterations,
Fock builds, warm-use and fallback flags match in every phase/repeat; maximum
candidate-to-control force difference is 2.800e-13. Native force primitive and
recurrence census remains unavailable, not inferred from AO capacity bounds.

The first complete campaign, job 6939, is also retained: Separate warm was
0.170438/0.196278 s at 24 AO and 0.872928/0.938012 s at 96 AO. The follow-up removes
redundant per-component finite tests, keeping the final transactional audit:
under strict FP64, a nonfinite active contribution cannot regain a finite sum.
Zero/inactive guards and bounded weight-first replay remain. Source-shape and
forced-nonfinite tests protect this change. Cross-allocation timing changes are
not controlled causal evidence for the optimization.

Linked production resource receipts expose the tradeoff hidden by evaluator
counts. For the RHF order-five FullSources kernel, per-thread stack rises from
24,696 to 41,144 bytes, despite unchanged 255-register/global-maximum reports.
That does not isolate the full regression's cause or certify unchanged
occupancy, spill traffic or allocator peak. Raw component differentiation can
also lose the retained weight-first DAG's early contraction advantage.

Final core/adjacent requalification passes 13/536 tests; compiler/SCF ownership
checks pass 509/224 modules with zero dependency errors. Fresh actual generated
CUDA roots for all 17 classes pass 1,224 comparisons in finite Slurm job 6944,
maximum error 5.552e-16; Compute Sanitizer reports zero errors. GPU visibility
is preserved throughout. Neither allocator peak nor comparable cache-miss
compilation cost is claimed measured. Four setup failures are retained as such;
no complete failed force endpoint is relabeled as passing.

The [reviewed endpoint bundle](../../../../benchmarks/results/shared-shell-channels-20261010/README.md)
retains both campaigns, reconstruction patches, independent references,
all-repeat numerical gates, complete solver histories and linked resource
inventories. The PR remains a draft, not a merge-ready performance promotion.

## Consequences

The retained sample bundle requires lossless storage compaction because the base
checkout has only 5,220 bytes left under its 64 MiB evidence budget. Two original
DF-value reports are deterministically compressed, with decoded bytes verified
against the frozen base and both stored/decoded identities pinned in their
`storage.json`. Historical observations and acceptance are not changed. The new
large sample member has an exact storage-review identity; no retention budget,
scientific gate, or policy exception is relaxed.

Repeated expensive integral work no longer scales with the two output channels
in the covered source adapter. The raw component lowering increases generated
source and can increase compilation cost or register pressure. Keep the
weight-first single-source schedule and bounded numerical fallback; evaluate
complete cold/warm/moved endpoints before making a performance claim.

## Revisit when

More channels or much larger angular classes become production consumers, or
complete endpoint profiling shows that source-driven reuse loses to early
weight contraction despite fewer recurrence traversals. The measured dual-source
regression now triggers that condition: investigate liveness-aware scalar
materialization or sharing only weight-independent Hermite/Coulomb frontiers
while retaining channel-local early contraction before performance promotion.
