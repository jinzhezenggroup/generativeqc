# Decision: borrow class-major dd x dp pages for cooperative order-seven forces

Status: implemented; composed explicit endpoint and independent scientific gates pass
Date: 2026-10-10

## Problem

The preceding three-way prototype preserved science but scanned the entire
shell-pair domain three times instead of twice. Independently checked intrusive
48/96 diagnostics show the producer cost cancelling cooperative recurrence
savings. A new formal population is unwarranted without changing that work.
See `../../rejected/2026-10-10-bounded-order-seven-cooperative-split.md` for retained V1 evidence.

## Decision

The existing max-L=2/materialized-cache/compatible-recurrence/256-lane guard
proves every order-seven quartet is dd x dp. Borrow the same Coulomb plan's
immutable class-major topology, claim one dd bra and at most 32 same-system dp
kets per page, then reuse the unchanged generated cooperative math. No new
resident allocation, quartet list, CPU topology readback or reference work is
introduced. Cursor reset and output storage still belong to the caller.

Class offsets give a ragged page domain across arbitrary batch sizes. Each
CTA's cursor claims increase monotonically, so its scalar prefix position
advances across systems once rather than rescanning all preceding systems for
every page. Empty classes consume zero claims. Address products and prefixes
are 64-bit even though immutable ordered pair indices remain 32-bit.

## Invariants and fallback

- Preserve physical max/min pair orientation after ordered-index decoding.
- Pass actual force-owner density bounds, not topology's Coulomb-owner bounds.
- Keep exact Force screening, generated masks/overflow, active masks, profiles,
  both spins, Combined/Separate coefficients and sticky launch/reset errors.
- Keep all 256 coefficient lanes. Halving the cooperative CTA width loses
  coefficients in the generated `slot * 256 + threadIdx.x` indexing.
- Without coherent topology, cooperation or the existing resource proof, keep
  the qualified original generic-plus-dddd route, not a third full-domain scan.
- SR/LR/RSH, Fock and the qualification-only angular schedule remain unchanged.
- No changed scientific thresholds, algebra, precision or production oracles.

## Evidence and remaining gates

The actual address decoder is compiled against an independent rectangular
inventory: empty neighbors, reordered physical IDs, 31/32/33 and 63/64/65 tails,
random mixed-system counts, interleaved monotonically increasing CTA claims,
over-32-bit page counts and the UINT32_MAX pair-index boundary. Actual dispatcher
and class launch wrapper exercise both spins/source layouts, topology present
and absent, all existing resource guards/shared-memory maxima, and failures at
each launch/peek/reset. The focused host set passes 56 cases.

Require an actual cached native build, executed/no-skip independent libcint
gates, exact per-class admission/resource equality and measured descriptor
reduction before clean complete endpoint timing. Preserve all samples and the
strict gain > max(2%, twice-summed-relative-MAD) gate at 48/96 warm/moved.
Numerical gates remain 1e-8 Eh / 1e-7 Eh/Bohr with physical residual <=1e-10,
one warm Fock build and no fallback. Master-only unrelated commits do not
authorize repeating this qualification; investigate only changed consumers.

## References

- `2026-10-10-ks-implicit-cooperative-pairs.md`
- `../../rejected/2026-10-10-bounded-order-seven-cooperative-split.md`
- `../../../../docs/developer/direct_pair_recurrence.md`

## Executed device evidence

Corrected Slurm 6912 completed the cached native build and six independent
libcint cases (19.79 seconds, no skips). The original build failed for a
missing helper include before any device tests; its logs are quarantined and
objects/caches reused. Final library SHA-256 is
`9596a199a826d42bd5358978f5908254589a6a4bec10711fb3a5be0350a39bac`.
The resumed build records 397 cache hits and eight misses, not an uncached
whole rebuild. Raw receipts are under `.artifacts/force7-class-stage/evidence/`
and remote `evidence/force7-class-v1/`.

Diagnostic 6914 completed fourteen calls per size. Independent reference,
physical residual, exact class census, semantic work/resource equality and
per-owner seed stability checks pass. Maximum errors are
6.3665e-12 Eh / 2.4711e-11 Eh/Bohr at 48 and
1.0459e-11 Eh / 3.7521e-11 Eh/Bohr at 96. Seeds still differ across owners;
do not claim a matched-byte single-owner comparison.

Actual class domains are 136 dd x 1024 dp (139,264 candidates; 4,352 pages)
and 528 dd x 4096 dp (2,162,688 candidates; 67,584 pages). Total logical
pages rise only 1,033,088 to 1,037,440 and 4,476,288 to 4,543,872, rather
than the rejected V1's 1.5x full-domain inflation. Intrusive two-electron
intervals baseline/candidate are 1682.199/1614.628 and 1853.518/1706.639 ms
at 48 warm/moved, and 4610.631/4557.222 and 4612.604/4519.514 ms at 96.
These are modest mechanistic savings, not formal complete endpoint gains.
The class CTA has 214 registers, 416 static shared bytes, 42,984 dynamic
shared bytes and 144 local bytes; retain all 256 coefficient lanes.

The frozen diagnostic remains #2166-v7 composition on `125a4e33f`; it is not
relabelled as whole-master evidence. Before the single clean timing population,
Slurm 6917 builds the actual `88cfa7017` tree and verifies byte-identical
generated force/contraction headers. The #2153 K/convergence integration
executes 14 energy-change passes; two snapshot cases initially fail for missing
packaged PBE0 artifacts, not numerical assertions. Slurm 6918 builds precisely
those artifacts and rechecks only the two failures: both pass. The retained
union is 16 executed focused integration passes, with the original failure
receipt preserved. Slurm 6919 memcheck/racecheck report zero errors/warnings
for the new class kernel; an argument-preserving observer proves three actual
class launches per tool and performs no GPU work itself.

## Complete endpoint acceptance

Slurm 6921 is one clean r5 ABBA population, 48/96 atoms, both unchanged and
moved geometry warm E+F. Both arms use library
`1e2059954b4b003e717a0e30c68c55891ab0d7b26f0b7dd543539801cb4cc7a5`.
Baseline selects shell Hcore/cooperative force off; candidate selects implicit
cooperative Hcore/cooperative force on. Force selection is captured at owner
preparation. Each owner's converged density/coordinate bytes remain frozen;
cross-owner seeds differ, so this is not an identical-byte same-owner claim.

| Atoms / phase | Baseline median s | Candidate median s | Gain | Noise floor |
| --- | ---: | ---: | ---: | ---: |
| 48 / warm | 5.582238134 | 5.333044238 | 4.46405% | 3.94885% |
| 48 / moved-warm | 5.524045762 | 5.321095161 | 3.67395% | 2% |
| 96 / warm | 17.062031366 | 16.045000125 | 5.96079% | 2% |
| 96 / moved-warm | 17.046683248 | 16.009522241 | 6.08424% | 2% |

Independent verification accepts all 46 complete calls per size: construction,
setup, priming and every sample. Maximum energy/force errors are
6.13909e-12 Eh / 2.46954e-11 Eh/Bohr at 48 and
1.04592e-11 Eh / 2.95027e-11 Eh/Bohr at 96. All warm calls have one Fock
build/iteration, physical residual <=1e-10 and no warm fallback. No intrusive
observer is loaded in clean samples. The old failed Hcore-only cohorts and
work-inflating force split remain rejected; gains belong to this composition.

Master #2167 and #2173 change post-HF DF admission/CCSD replay, not these
PBE0 owners or mathematical emitters. Both are integrated without duplicate
GPU campaigns. The review source may advance while measured source stays
explicitly pinned. Actual isolated compile cost and peak-memory budgets were
not measured, so this scoped endpoint acceptance is not a fabricated shared
formal-promotion envelope. Cooperative force remains opt-in, and the prepared
Hcore default is preserved unless its mapping knob is explicitly selected.

Compact raw records, references, source reconstruction and qualification
receipts live in
[`benchmarks/results/pbe0-implicit-class-force-20261010/`](../../../../benchmarks/results/pbe0-implicit-class-force-20261010/README.md).
Full native seed blobs and binaries remain ignored locally/on n1. No Release,
tag or external backup is created.
