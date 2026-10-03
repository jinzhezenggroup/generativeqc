# Decision: widen large integral-force admission blocks

Status: implemented
Date: 2026-10-03

First-process timing caveat: see the later
[lazy-library-load correction](2026-10-03-xtb-cold-library-timing.md). The
historical xTBloom first-process timings here omit an untimed native/provider
load; warm/changed timings and scientific gates retain their scope.

## Problem and decision

After removing synthetic bootstrap work, the production 96-atom integral-force
preflight still spent about 812 microseconds scanning complete directed S/D/Q
adjoints with one 64-thread block. Its ten matrix-component scans are independent
across AO entries and do not perform floating-point reductions.

The compiler now emits a host launch selector from
`integral/gfn2_force_schedule.py`. A rounded-up mean of at most 4096 matrix
elements retains 64 threads; larger means use 256. Selection uses division and
remainder to avoid integer overflow and needs no device metadata download. One
block still owns each actual ragged system. Atom, shell, primitive and matrix
scans, seed initialization, shared state, error gates and publication are unchanged.

Only the admission launch width changes. Scientific shell-pair evaluation and
its fixed 64-lane reduction must not inherit this width. Generated scalar helpers
are byte-identical to the parent after excluding the new host selector and its
integer include. CPU generated mathematics and all precision/tolerance policies
are unchanged.

## Work invariants and rejected alternatives

Each system still performs one atom/shell/primitive metadata pass and scans ten
times its squared AO count for finite adjoints. Every gradient seed is read once.
The existing four launches (sequence capture, preflight, scientific contraction,
publication), scratch capacities and synchronization count remain unchanged.

Duplicating the complete preflight across blocks would multiply validation and
race on seed initialization. Separating metadata and matrix scans could distribute
large work across SMs, but adds a launch and a failure boundary. Widening the
existing block gives a measured endpoint gain without those costs. Small means
retain the existing width to avoid unnecessary launch-resource changes.

## Qualification

The emitted policy is compiled and tested at 4095/4096/4097 elements, ragged-mean
ceilings and INT64_MAX. The native harness includes the actual production owner
to exercise private admission/publication kernels without adding a test API.
It compares retained and selected widths on 63/64/65/257-AO cases and skewed
s/p/d metadata with 3/257/17 shells. Every path is tested through ordinary launch
and two CUDA Graph replays.

Faults include late overlap/dipole/quadrupole NaNs, overflowing outer offsets,
late invalid angular and primitive data, nonfinite coordinates and seeds, invalid
request masks, inactivity, failed SCC, a closed sequence and pre-existing system
errors. An independent host fixture predicts the exact diagnostics. The harness
snapshots seeded scratch, inserts a known downstream payload, and verifies that
healthy members publish while failed/gated members preserve original bytes.
It deliberately tests admission and publication; independent tblite endpoint
tests cover the unchanged scientific contraction.

The native harness passes on n2 RTX PRO 6000. Compute Sanitizer memcheck reports
zero errors and racecheck zero errors/warnings. Host codegen tests include the
existing independent Gaussian-quadrature S/D/Q and derivative oracle. Compilation
uses ccache; all real GPU execution uses finite Slurm allocations.
All 15 host policy/codegen/provenance tests pass (one opt-in skip exercised
separately); the final n2 build passes 43 public/native tests through the actual
Python harness entry point. The first remote entry-point attempt lacked the
copied SDQ generator script; that export failure is retained separately from the
successful run. Compiler structure, ownership and pre-commit checks pass.

Both n1 RTX 5090 and n2 RTX PRO 6000 pass 41 public tests and 110 complete
energy/force samples (seven fixtures and water8/32/64, one cold plus five repeated
and five changed calls each). FP64, fresh SCC, 300 K, Broyden 8/0.4, maximum 300
iterations and energy/charge tolerances 1e-10/1e-8 are unchanged. Every numerical
and SCC-count gate passes. Energy is identical to the parent; maximum force
change is 5.552e-17 Eh/bohr. Maximum xTBloom differences are 5.684e-14 Eh and
5.156e-15 Eh/bohr.

## Complete endpoint evidence

Warm energy/force medians, milliseconds:

| GPU / atoms | Parent | Candidate | xTBloom |
| --- | ---: | ---: | ---: |
| n1 / 96 | 126.583 | 125.837 | 131.827 |
| n2 / 96 | 123.983 | 123.381 | 129.627 |
| n1 / 192 | 314.875 | 312.507 | 321.739 |
| n2 / 192 | 307.113 | 304.510 | 314.715 |

All ten first-singlepoint, warm and changed medians beat xTBloom in both cohorts.
Small cases are essentially unchanged versus the parent, including minor timing
regressions. The n2 production-route trace retains three preflight calls across
three public calculations, with median 812.102 -> 207.521 microseconds. Some
device-launched SCC kernels are absent from the summary; this is not a complete
SCC profile.

Corrected constructor plus first-call timing separates previous-calculator
cleanup, per the [loader timing decision](2026-10-03-xtb-loader-discovery.md).
192-atom cold totals improve 365.046 -> 362.292 ms on n1 and 350.943 -> 348.674 ms
on n2; xTBloom measures 400.434 and 385.864 ms. However, n1 96-atom cold total
regresses 235.723 -> 239.188 ms in its single cold sample. Combined cold totals
win only 7/10 on n1 and 8/10 on n2. First-process H2 remains slower than xTBloom
(431.109 vs 379.797 ms on n1, 389.550 vs 300.316 ms on n2). This is not universal
startup superiority, nor a statistical confidence claim for tiny timing margins.

Ignored receipts:

- `.artifacts/{n1,n2}/integral-preflight-256/`: all measured outputs and reports.
- `.artifacts/preflight-qualification/`: native, compiler and sanitizer receipts.
- `.artifacts/n2/preflight-{before,after}_cuda_gpu_kern_sum.csv`: kernel work and
  timing evidence.
- `.artifacts/libgenerativeqc-preflight-256.so`, SHA-256
  `fc23ea615764d458bb3281ceed72749ac0466bd27e32b62bbd0fdbb7fc467145`.
- Final source-provenance metadata refresh, with unchanged CUDA objects,
  SHA-256 `2bb5afe520e8c079785e759f6a621a32b0f2fc987c16cd90c1f7a3c37b12a408`;
  separate final qualification is in `.artifacts/preflight-qualification/n2-final.log`.
- Parent loader library SHA-256
  `7e93f097958fe2d5115448c587164c7b4b001271895922d7643f66e30cf9bde1`.
- xTBloom revision `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`, library SHA-256
  `6be47a7183a21c10c8204b6fbeb27e8587bc45d1958e59f5c7d61356777db53f`.

## Revisit when

Revisit multi-block admission if further endpoint measurements justify its extra
launch and failure boundary. Preserve single-pass validation, disjoint seed
initialization, bounded ragged traversal and the unchanged scientific reduction.

## Historical first-process timing qualification

The later [PR #1742 comparator correction](https://github.com/jinzhezenggroup/generativeqc/pull/1742)
found that identity verification triggered xTBloom's lazy native/provider load
outside both timed phases. The first-process comparisons above omit that
reference loading cost. The corrected v3 comparator includes lazy loading and
keeps cleanup separate. Warm/changed timing and scientific gates retain their
original scope, and the historical rows above are unchanged. The corrected
endpoint begins after Python package import, not OS process launch; no universal
startup superiority is established.
