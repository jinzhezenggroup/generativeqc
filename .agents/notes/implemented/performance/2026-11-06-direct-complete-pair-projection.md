# Decision: extend exact complete-pair projection to the remaining low classes

Status: implemented
Date: 2026-11-06

## Problem

After the compact cooperative MD schedule, the A100 water96 r2SCAN-3c
Direct force diagnostic still attributes 2.278 s inclusive to generated-low
classes. This instrumented number is not a clean incremental saving. The
arbitrary-weight order-four/five contraction partitions repeat scalar work
and have substantial live state.

## Decision

Qualify the existing compiler's exact complete ordered-pair contraction for
PPPP, DSPP, DSDS, DPPS, DPPP, DPDS and DDPS. This is an exact outer product of
external pair weights, not a density approximation. Reuse the existing
scientific graph, relative differentiation, strict FP64, complete Schwarz-mask
proof and native pure-J/no-seed admission. DDSS has a trivial ket projection
and is not worth an additional helper. Arbitrary weights, exchange, mixed
precision, incomplete masks and failed finite checks retain strict bounded
fallbacks. Same-pair triangles newly possible for PPPP/DSDS require explicit
independent force qualification, not only the existing mask-proof tests.

The qualified candidate is enabled by default. The prepared owner captures
`GENERATIVEQC_DIRECT_FORCE_PAIR_SEPARABLE`: unset/1 enables the preference;
0/empty/unknown disables it. Resource/restart identity includes this control.
The measured candidate used the former experimental selector; changing its
name and default does not re-identify the measured image. Do not dispatch by
GPU model or change physical AO spaces, grids, screening or convergence.

## Rejected alternatives

- Single-use/rematerialization: current emitted baseline has 57,339 operations
  and maximum modeled live state464; alternatives give57,339/734 and68,606/655.
- Cross-partition CSE increases modeled live state, e.g. DPPP400→1023,
  DPDS430→848 and DDPS464→872. No GPU build justified.
- Earlier reverse-AD, arbitrary-weight pair-first, convolution and primitive
  accumulation experiments are not retried. They do not establish a benefit
  over the current relative-forward consumer.
- Do not promote the earlier small DPPP-only experiment using old measurements.
  This candidate changes the covered work and needs new qualification.

## Evidence and gates

Current factored pressure-aware, gradient-only emitter model:

| Class | Arbitrary-weight ops/live | Exact-pair ops/live |
| --- | --- | --- |
| PPPP | 5545/251 | 2062/122 |
| DSPP | 3114/200 | 1958/100 |
| DSDS | 2340/193 | 1834/108 |
| DPPS | 3513/216 | 1898/132 |
| DPPP | 15796/400 | 4135/178 |
| DPDS | 12115/430 | 3974/201 |
| DDPS | 12127/464 | 4299/324 |

Total modeled operations54,550→20,160; maximum modeled live464→324. These
are not measured register counts or endpoint timings. Local reproducible
diagnostic receipts are under `.artifacts/direct-low-lifetime-20261106/`.

Required gates: independent Libcint complete derivatives and signed weights;
compiled actual native admission, symmetry, fallback and no-partial-publication
tests; actual-device positive admission; same settings/work-count complete
E+force comparisons on scheduled A100 and5090. Timing runs must not contain
traces or work censuses. Retain original source/ELF and baseline identities.

## Accepted device evidence and boundaries

The retained [offline replay](../../../../benchmarks/results/direct-complete-pair-20261106/README.md)
contains every cold plus three warm sample, immutable producers, semantic
work counts, observable gates and authenticated source/core receipts.
Water96 r2SCAN-3c uses 1088 spherical def2-mTZVPP AOs, exact Direct E+analytic
forces including native D4/gCP. Baseline is the already-qualified compact-MD
image from PR3014; the new candidate adds these seven pair projections.

| Device / actual allocation | Baseline warm median | Candidate warm median | Change |
| --- | --- | --- | --- |
| A100 / ECNU LSF225765 | 7.962698835879564 s | 6.952339671552181 s | -12.688652% |
| RTX5090 / Slurm16007 | 11.085675675421953 s | 10.544288743287325 s | -4.883662% |

Cold single observations are 18.35358353704214→17.39413031935692 s on A100
and 29.316666338592768→28.656750690191984 s on5090; neither establishes a
cold distribution. Cold10/10 and warm1/1 iterations/Focks, all recorded
settings, anchor/delta/closure work and admission semantics match exactly.
Sixteen native/native observable pairs per device pass unchanged gates;
worst force differences are 1.3461e-11 and 1.3237e-11 Eh/Bohr respectively.
The prior5090 GPU4PySCF population is reused for science only, not cross-device
timing. Do not add savings from different campaigns to claim closure of the
original1.576s gap, or infer wb97m-v hybrid gains from pure-J evidence.

Independent qualification: 585 host tests pass with164 physically invalid
signature combinations explicitly skipped; six actual5090 Libcint cases pass
(RKS/UKS, Cartesian/spherical, Separate/Combined/replay). Pure-J dimer work
asserts acceptance of exactly classes5,8,9,11,12,13,16; hybrids retain ordinary
cotangents. Memory checking covers all seven classes and reports zero errors.
The separate racecheck hits180s and exits124 with no diagnostic: **inconclusive**.
Auxiliary Slurm16008 is FAILED, not passed, because its runner requires both.
This extension changes thread-private expressions/pair arrays, not shared
storage, barriers or scatter synchronization. Promotion relies on scoped
memory, independent numerical and endpoint gates; it is not a blanket
sanitizer-green claim. Do not rerun until the timeout happens to disappear.

Measured base is7fc4ae8f6a6e4e31509f0fa7b3409d7daa72d83d, source
4c54ccfb3d38e5ec86b147a019d8ef99e87c8b8247ca5a42bf50a84027dc61ed,
ELF3142176ecd915a1fe5eaaced6aa5cb4b3da3af6a1ff7b71b4cf6a2255f199799.
Master08c9a23f subsequently changes a disjoint nonlocal Direct batching branch;
the semilocal cohort's first branch is unchanged. Integration and formal
selector promotion are audited/tested, not claimed as a new full GPU build.
Pre-link resource dumps are not final-linked stack/register evidence; static
modeled live state is not a hardware register count.

## References

Issue2976; PR3014; proposed `2026-11-04-dppp-separable-mask-candidate.md`;
implemented `2026-11-05-direct-cooperative-md-radial.md`.
