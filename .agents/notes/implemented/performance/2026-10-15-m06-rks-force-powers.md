# Decision: qualify bounded M06-2X powers at the RKS force consumer

Status: implemented
Date: 2026-10-15
Campaign started: 2026-10-14

## Problem

Imported M06-2X component emission leaves bounded integer powers in its scalar
graph, unlike curated semilocal emission. The M06 header contains 317 small
integer `pow` calls. Independent polarized AD must precede any equal-spin
binding; differentiating an equal-spin energy would lose independent partials.

The functional matters: M06 and r2SCAN have larger tau-dependent, piecewise
point programs than the simpler GGAs. This is an implementation opportunity,
not a reason to change the functional. The r2SCAN-3c direct-force gap also has a
different large-basis, high-order integral contribution and is not repaired by
a point-only improvement.

## Decision

Reuse `Graph.lower_small_integer_powers` after AD, and opt in only in the FP64
M06-2X **unpolarized/RKS stationary force** consumer. Preserve SCF ordering,
UKS consumers, float scalar graphs, other split hybrids and r2SCAN defaults.
Strict FP64 force fallbacks use the qualified FP64 graph; no float operation
ordering is newly qualified. Mathematical expression identities stay unchanged;
generated source and artifact identities deliberately change.

No new AO compaction, matrix tile, dddd acquisition or order-seven page policy
is selected. PR #2486 already records those negative or mixed experiments.

This proposal does not depend on or re-credit PR #2464's equal-spin branch CSE.
The qualification control is master `8c770b34c`, including merged #2486, and the
candidate scientific code is `8021e2d92`. Earlier endpoints on the separate CSE
source `786c31a48` remain historical experiments, not master measurements.

## Repairing the unstable proposal

Global point lowering increased every DF96 frozen warm replay from three to
four Fock builds in the earlier experiment and regressed complete warm time.
Move the new arithmetic out of SCF rather than relax any convergence gate,
change the frozen-density protocol, or subtract an extra build from timing.

The earlier force-only direct96 moved call also increased from 12 to 14 builds.
Two complete same-image reruns of **each** arm return 12 moved builds and retain
all 48 calls with their original gates. The baseline image's first cold
Hartree component itself differs by one ulp between runs, before any force
consumer executes. This demonstrates existing numerical variability, but does
not by itself isolate every source of the earlier extra iterations. Preserve
that original regression; do not call it free work or erase it as noise.

## Independent acceptance and boundaries

- Compiled host force consumers pass independent Libxc Cartesian rho/gradient/
  tau comparisons and displaced-energy finite differences, with independent
  polarized partials retained before equal-spin binding.
- Real-GPU force scalar acceptance uses the exact generated production body,
  Libxc 7.0.0, and authenticated upstream Maple with 113-bit arithmetic at empty
  spin boundaries. Clean source `8021e2d92`, Slurm n2/3021, passes the original
  scientific gates. The GPU probe's header substitution is explicit validation
  tooling, never a production SCF consumer.
- A byte-preservation receipt verifies M06-2X and MN15 double/float SCF bodies
  against master. It is campaign evidence, not a permanent source-hash CI test.
- Existing stricter diffuse-tail stress failures remain in the earlier capsule:
  both M06 arms fail four common derivative entries at 65,536 points against the
  quad oracle. The established scientific gates are not loosened or replaced by
  those diagnostics; neither are the diagnostic failures deleted.
- All real-GPU work uses finite Slurm allocations on permitted nodes. All local
  n3 work is CPU-only. Compiler caches and assigned device visibility are kept.

## Endpoint qualification

Use matched FP64 def2-SVP RKS energy **and host-returned analytic forces**, the
moving grid and unchanged DF model where applicable, cold + five frozen warm +
moved + five moved-warm calls. Time preparation as part of cold completion.
Require the unchanged independent gates: 1e-8 Eh and 1e-7 Eh/Bohr. Retain every
phase, force vector, reference, work count, image/source identity and failure.

The planned master cases are DF48, DF96 and direct96, plus a second complete
matched direct96 population. Raw work is not normalized across different SCF
iteration counts. Unavailable integral execution counters stay unknown.

All 96 source-matched master calls pass both unchanged independent gates.
Maximum observed errors are 2.0464e-11 Eh and 1.5725e-10 Eh/Bohr. The complete
warm E+F reductions are 9.23% (DF48), 4.54% (DF96), and 4.36% (direct96); the
second direct96 population gives 4.30%. Every warm/moved-warm work count matches.
All first-population cold/moved work counts also match.

The same master baseline image takes 19/13 cold/moved builds in its second
population instead of 18/12 in the first, while both candidate populations take
18/12. This independently reproduces baseline SCF work variability. Retain
both populations unchanged and do not credit the force optimization with a
solver improvement: the large second-population cold/moved timing differences
include different SCF work. Stable fixed-work warm endpoint gains are the
default-promotion claim, not guarantees for arbitrary noisy cold solves.

The new evidence includes 96 master calls and 48 historical same-image control
calls; the earlier 108-call negative experiment is kept separately. Host
validation has 124 passes
and 43 explicit optional skips; final policy-only validation has 17 passes.
Configured pre-commit and compiler ownership checks pass. The master README is
not rewritten from mixed-source historical timing populations.

## Evidence storage boundary

The first PR CI retention check found the two complete compressed capsules
exceeded the unchanged 64-MiB aggregate checkout budget by 533059 bytes. Keep
their exact originals in already-pushed ancestor
`3ca3fa0038fbea5350a8bb799cca26d678522678`; do not rewrite/drop that ancestor,
increase budgets, publish Releases or migrate unrelated campaigns. The owned
`full-evidence.json` snapshot binds both original capsules and their guides and
manifests by revision, SHA256, byte lengths and totals. Existing
`tools/restore_retained_evidence.py` restores all six files offline, verifying
before publication and refusing overwrite. No new external host is introduced.

Current capsules are explicitly endpoint ledgers, not full-vector archives.
They preserve every timing sample, phase range, E/error/gate/convergence value,
available semantic work counter, source/model/library identity and diagnostic
failure. Full vectors/references, point arrays, SCF histories, component
profiling, logs/cache/build receipts and scripts are restored from the pinned
originals. `tools/compact_m06_force_evidence.py` deterministically regenerates
the ledgers from the verified snapshot without rounding or filtering; tests
protect negative rows, null counters, timing values, all 252 calls and identities.
The historical rejection is not reclassified by this storage-only projection.

## References

- Historical rejected proposals:
  `.agents/notes/rejected/2026-10-14-mgga-integer-power-defaults.md`.
- Historical raw evidence:
  `benchmarks/results/mgga-integer-powers-20261014/`.
- New independent source-matched evidence:
  `benchmarks/results/m06-rks-force-powers-20261014/`.
