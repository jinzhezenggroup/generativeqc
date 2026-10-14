# Decision: retain integer-power lowering as an explicit compiler experiment

Status: rejected as a default; explicit opt-in compiler schedule retained
Date: 2026-10-14

Superseding bounded decision, 2026-10-15:
`.agents/notes/implemented/performance/2026-10-15-m06-rks-force-powers.md`.
The original global and force-only observations below remain historical rejects
on their original common source. Separately qualified master FP64/RKS force
admission does not erase these regressions or promote the global SCF schedule.

## Problem

The master README retains several FP64 M06-2X and r2SCAN endpoints slower than
GPU4PySCF. Its publication combines independently qualified source populations;
it is not a fresh measurement of one master build. Functional complexity matters,
but does not make those gaps unavoidable: imported split-hybrid components did
not reuse the existing bounded integer-power lowering used by curated semilocal
programs. M06-2X emitted 317 `pow` calls with integer exponents of magnitude at
most four. Its exchange/correlation AD graphs and tau contractions are also
larger than the GGA cases. Canonical r2SCAN-3c additionally has a larger basis
(1,088 rather than 768 AOs at 96 atoms); its direct integral derivatives dominate
the large remaining gap. An XC point improvement alone does not close that gap.

## Candidate and boundaries

Apply `Graph.lower_small_integer_powers` after independently differentiating the
polarized component energy, then emit both polarized and producer-proven
equal-spin functions from the rebuilt graph. This reuses the existing
exponentiation-by-squaring/reciprocal implementation rather than inventing a new
algebra. Independent polarized partials, Libxc admission/work floors, global FHC,
Stoll stabilization and lazy branch selection remain unchanged. Mathematical
program identity stays unchanged; generated source/build identities change.
The shared body emitter accepts `lower_integer_powers=True` only as an explicit
FP64 compiler experiment and defaults to the old operation order. Both default
SCF and stationary force consumers remain unchanged; no production path enables
the experiment automatically. Default M06-2X/MN15 point and geometry source is
byte-identical to unmodified master for both double and float. A request for
lowering with float, or a non-boolean policy, fails closed. No CPU/reference work
enters production and no SCF tolerance or frozen-seed policy changes.

This candidate does **not** include or take credit for the independently
qualified equal-spin branch CSE on `786c31a48`. Point diagnostics include arms
with and without that CSE. The native endpoint experiment uses that retained
qualified source as its common baseline and applies only the explicitly recorded
integer-lowering ablation on top, with the same Release/auto AOT configuration
and matrix tile 16. It is not
a source-matched remeasurement of master `810f081ef` or of every README row.

## Rejected global lowering

The first full candidate lowered integer powers in both SCF and forces. All 72
returned baseline/candidate calls pass the unchanged independent E/F gates,
but the complete endpoints show why point speed alone cannot choose a default:

| M06-2X endpoint | Baseline warm / moved-warm (s) | Global lowering warm / moved-warm (s) |
| --- | ---: | ---: |
| DF/48 | 7.124970 / 7.118959 | 6.514494 / 6.519952 |
| Direct/96 | 17.251976 / 17.272345 | 16.272472 / 16.271191 |
| DF/96 | 29.957500 / 29.948927 | **31.703341** / 27.516511 |

Every DF/96 frozen post-cold replay changes from three to four SCF iterations
and Fock builds, producing a **5.8% complete warm regression**. Its moved-warm
population remains three builds and improves 8.1%. Direct/96 cold also changes
from 18 to 19 builds, despite a small complete cold improvement. Do not filter
replays, reset to the other arm's density, normalize times by iterations, or
relax the energy/density/residual tolerances to manufacture a win. The global
candidate is rejected as a default; all phases and work remain in the evidence.

Force-only lowering was separately tested to keep the SCF point emission's
operation order unchanged. This is a consumer-specific emission schedule, not
new SCF/method policy in scalar algebra. It improves all frozen warm populations,
but its direct/96 moved endpoint regresses too, so it is not selected by default.
The unmerged CSE prerequisite itself also has a retained r2SCAN-3c direct moved
regression (14 to 23 builds); using it as the common M06 baseline does not
authorize universal CSE promotion or retry that negative case.

## Rejected force-only default

All 36 force-only returned E/F calls pass the unchanged independent gates. The
common baseline, global candidate and force-only candidate total 108 accepted
returned calls, with maximum error 2.0464e-11 Eh / 1.5727e-10 Eh/Bohr.

| Endpoint phase | Baseline (s) | Force-only (s) | Actual SCF iterations / builds |
| --- | ---: | ---: | --- |
| DF/48 cold | 37.869489 | 37.431879 | 18 / 18 in both |
| DF/48 warm | 7.124970 | 6.670259 | 1 / 1 in every replay, both arms |
| DF/48 moved | 28.664111 | 28.288082 | 13 / 13 in both |
| DF/48 moved-warm | 7.118959 | 6.669358 | 1 / 1 in every replay, both arms |
| DF/96 cold | 123.680956 | 122.614808 | 18 / 18 in both |
| DF/96 warm | 29.957500 | 29.000015 | 3 / 3 in every replay, both arms |
| DF/96 moved | 92.432596 | 91.500335 | 12 / 12 in both |
| DF/96 moved-warm | 29.948927 | 28.989035 | 3 / 3 in every replay, both arms |
| Direct/96 cold | 136.580794 | 136.208958 | 18 / 18 in both |
| Direct/96 warm | 17.251976 | 16.772929 | 1 / 1 in every replay, both arms |
| Direct/96 moved | 94.625997 | **106.983009** | **12 / 12 to 14 / 14** |
| Direct/96 moved-warm | 17.272345 | 16.787956 | 1 / 1 in every replay, both arms |

Warm time improves 6.4%, 3.2%, and 2.8% respectively, but direct/96 moved is
**13.1% slower** and executes two more SCF builds. All five warm samples, ranges
and the moved regression are retained; no sample is replaced or normalized.
Source-byte equality for the SCF point emitter is not evidence of deterministic
SCF work across these separately built full images and independent populations.
The cause of the moved work-count change is not independently isolated here.
In particular, do not assert that force arithmetic necessarily caused it or
hide it as noise without repeated same-source full cohorts. The available
AO/grid work matches the baseline in every other force-only phase. Inner J/K
execution counters are unavailable in these clean timing populations; null
counts are not zero or proof of identical integral enumeration.

The observed pattern justifies investigating complete-endpoint SCF work
stability and scalar batching without reordering point arithmetic. It does not
justify loosening scientific tolerances or promoting either measured candidate.

## Alternatives not repeated

PR #2486 (`ca84681be`) already retained negative or mixed endpoint results for
device AO compaction, dddd class-major acquisition, smaller order-seven pages,
and matrix tile changes. None are retried or enabled here. The evidence in that
PR still motivates cross-AO-component primitive/ladder reuse for order six and
dependency/barrier-liveness work for order seven, not more scheduling knobs.
No occupancy/stall-counter claim is made: NCU permissions were denied there.

## Point evidence

All real GPU work uses finite Slurm allocations and preserves assigned device
visibility. Local work on maintenance node n3 is CPU-only generation,
compilation or host-oracle validation. CUDA 12.9.1, `sm_120`, FP64 and
`--fmad=false` are used for point probes, with four forward/reverse ordering
rounds of 100 launches and 128 threads/block. The seeded inputs cover densities
from 1e-9 to 10 and physical equal-spin sigma/tau scaling.

- RTX 5090, node n5, Slurm 1565, 65,536 points: master-style M06 baseline
  2.427364 ms, integer lowering 1.649378 ms; post-CSE baseline 1.131505 ms,
  post-CSE integer lowering 0.696010 ms. The latter is about 38.5% less point
  kernel time, **not** 38.5% less complete E+F time.
- RTX PRO 6000, node n2, Slurm 3013: corresponding post-CSE medians
  1.012830 / 0.537151 ms. This is different hardware, not a README timing.
- M06 static point resources change from 255 registers with spills to 224
  without spills, or 192 to 190 registers without spills after CSE. Static
  resources do not measure occupancy or identify a hardware bottleneck.
- An **unpromoted** r2SCAN experiment extends bounded addition chains from four
  to twelve only at backend emission, retaining its mathematical identity.
  PRO 6000 post-CSE medians are 0.444952 / 0.375838 ms for 65,536 points
  (approximately 15.5% less point time). Both still use 255 registers and the
  candidate reports more spills. There is no complete r2SCAN endpoint evidence
  for this experiment and no default r2SCAN change.

## Numerical and operational negatives

An additional diagnostic uses rtol 2e-10 (energy atol 1e-12, partial atol 2e-10),
tighter than the pre-existing split-hybrid acceptance gates. Binary64 Libxc
fails against the independent 113-bit upstream Maple oracle in diffuse tails:
M06 has two failing entries at 8,192 points and six at 65,536; r2SCAN has 44 and
256. Do not loosen the diagnostic or discard those rows. With the same strict
gate against the quad oracle, both M06 arms have zero failures at 8,192 and
four common failing derivative entries at 65,536 (worst tolerance ratios
2.0455 / 2.0513). Both fail this extra stress gate. Fourteen candidate entries
also exceed an extra baseline-pair rtol 5e-12 diagnostic at 65,536. These are
retained, not represented as passes. Both r2SCAN arms pass the strict quad gate
at both sizes and the retained empty-spin/exchanged-spin fixture at rtol 5e-12.

The existing split-hybrid oracle gates are not changed: 50 polarized points,
finite differences, restricted embedding, and M06 empty-spin neighbors against
the independent 113-bit oracle pass for M06-2X and MN15 on host and GPU.
GPU report n5/1566 deliberately remains globally `passed: false` because its
copied, uncommitted source has no clean Git identity; individual scientific
method gates pass. It is not a clean-release qualification receipt. Memcheck
of both 50-point methods on n2 reports zero errors.

The first n5 native baseline attempt returned no E/F because n1-built native
libraries require glibc 2.38 while n5 provides 2.35. Retain that failed receipt.
Use the same authenticated loader/runtime bundle already retained by #2486 for
both endpoint arms; do not patch libc symbols or change production code.
Process startup remains outside the endpoint timer. A first sanitizer invocation
also failed because its executable transfer was refused by SSH host verification;
the subsequent locally relayed transfer and finite Slurm rerun succeed. Neither
failure is a successful timing or sanitizer pass.

## Endpoint acceptance

Use the ordinary README runner: matched moving grid, full def2-SVP spherical
basis, matched DF metric/auxiliary basis where applicable, FP64 tolerances,
independent retained references at both geometries, cold + five frozen-seed
warm + moved + five moved-warm calls. Retain every sample, actual SCF iterations
and Fock builds, AO/grid/force work, source/consumer/library hashes, and Slurm
receipt. Endpoint gates remain 1e-8 Eh and 1e-7 Eh/Bohr. No profile or oracle
is inserted into a timed native call. Larger benchmarks keep the existing
1,800-second whole-point limit.

The global and force-only candidates and full ranges are retained in the endpoint
ledger `benchmarks/results/mgga-integer-powers-20261014/summary.json.xz`.
The exact original capsule, including vectors and build receipts, remains in
Git ancestor `3ca3fa0038fbea5350a8bb799cca26d678522678`, bound by the superseding
campaign's `full-evidence.json` and restored offline with
`tools/restore_retained_evidence.py`. This storage-only projection does not
change the historical rejection or discard negative observations.
Baseline Slurm jobs are n5/1568, 1569 and 1570; global candidate n5/1571;
force-only candidate n5/1572. Native timings use the common compatible runtime
bundle, Release/auto AOT, matrix tile 16 and separate arm-local compilation caches.
Cold includes construction and preparation using `complete_seconds`, not just
`seconds`. All returned energy/force data, source/library/consumer hashes,
scientific inputs and reproduction scripts are embedded in the compressed data.
After the storage-only projection described above, this refers to the original
full capsule in pinned Git history, not to the compact current-tree ledger.
The final unit cohort passes 96 tests and skips 43 explicitly optional tests;
skips are not qualification. Compiler structure, Ruff and formatting checks pass.
Do not update the public README using point-kernel timings or the different
hardware results.

## Revisit when

- Repeated complete native cohorts explain and bound moved-geometry SCF work
  variation while establishing a reproducible whole-population gain.
- The separately gated r2SCAN extended-power experiment has complete E+F evidence.
- A precision-specific investigation resolves the stricter M06 cancellation
  stress failures for both baseline and candidate; this experiment does not.

## References

- `python/generativeqc_compiler/xc/split_hybrid_codegen.py`
- `python/generativeqc_compiler/integral/expr.py`: bounded integer-power lowering
- `tools/validate_split_hybrid_cuda.py`, `tools/split_hybrid_wide_oracle.py`
- `tools/generate_r2scan_tail_reference.py`
- `benchmarks/results/readme-dft-20261013/README.md`
- PR #2486 retained bounded scheduling experiments
- Local raw experiment: `.artifacts/mgga-integer-powers-20261014/`
