# Decision: retain native FP64 force ownership for budget Focks

Status: implemented
Date: 2026-10-16

## Problem

Enabling FP32 Focks did not imply using the same complete execution graph as
strict FP64. Prepared native stationary-force admission required strict SCF
controls even when the selected derivative body itself remained FP64. A
budget-converged density therefore lost its otherwise qualified native force
capability and entered Python compatibility orchestration. Reconstructing
derivative requests and checking generated-artifact provenance could cost more
than the small-system electronic solve.

The source-bound eight-family comparison on node2's RTX PRO 6000 explicitly
retains the negative results. With three-repeat medians, Fast/FP64 endpoint
seconds for 3 atoms include PBE 2.052/0.516, PBE0 1.853/0.552, B3LYP 1.891/0.597,
r2SCAN 2.490/1.081 and M06-2X 2.366/1.236. At 12 atoms those five pairs are
4.087/3.437, 3.259/1.856, 3.360/2.023, 5.275/4.814 and 4.397/3.369. These
measurements do not establish a universal Fast advantage. Their existing disk
artifacts were not equally cold, and no converged density was reused.

A separate diagnostic PBE/3 profile attributes about 2.962 seconds to the
compatibility force consumer, including request reconstruction and file hashing.
It also finds a separate J limitation: approximately 241 ms mixed versus 32 ms
strict GPU J work over thirteen Focks. Instrumented times diagnose components;
they are not clean endpoint comparisons.

## Decision

Separate the admission of a stationary FP64 force body from the admission of
mixed derivative arithmetic. A restricted, all-electron, exact Direct budget
Fock may reuse its existing prepared native FP64 consumer only if all existing
complete-model and shape qualifications pass. Check the actual Fock module's
CUDA capability and finite positive SCF controls bounded by `1e-6`, `1e-4`,
`1e-10` for energy, density and screening respectively. Do not add functional
name dispatch or expand the generated complete-model qualification inventory.

The FP64 force path also retains the qualified spatial schedule available to
strict force consumers. Strict-density mixed-gradient qualification remains
separate and checks the strict control ceilings before either execution or
capacity planning. A budget-density native force must not silently acquire
FP32 geometry arithmetic, weighted products or mixed-gradient resident capacity.

Range exchange, nonlocal correlation, density fitting, corrected graphs,
unsupported shapes and absent linked bindings retain their existing consumers.
Final-state token checks, physical-Fock weight preparation, capacity admission,
finite iteration limits and independent observable gates are unchanged.

## Evidence boundaries

The incumbent packet-v4 source identity is
`9e709e551c8261fad08c1c0011fbb3cec4e1ae259c9ca466c78c07d5a76b3b28`;
its core SHA-256 is
`b6a06237afe3f03b0eb416b0dc9f64c13bece29ad86297b4a2730610601975f7`.
The native-force-v5 source identity is
`0d351ff90f03b77b6f3cad07fef225e4a4ac13f72c32384172043ae10c0b2410`;
its core SHA-256 is
`caf921d07b7909f501e715419b3563a9a4660930d0be7260467c293d9cd6dde8`.
The separately generated force binaries and their authenticated manifests are
unchanged between these two core builds.

The initial scalar replay test incorrectly required every displaced proposal to
publish a qualified mixed endpoint. A paired diagnostic on the unchanged
packet-v4 incumbent and native-force-v5 displaces one H by `0.01 bohr` in the
12-atom PBE case. Both reject the candidate's final physical-state proof and
return `observed_met` after a recorded FP64 fallback and audit. The old
compatibility consumer reports its detailed final-state commutator/density/etc.
refusal; native preparation returns numerical failure. Neither publishes the
rejected force. Retain the initial failed test receipt and both complete paired
diagnostics; do not count this strict recovery as successful FP32 execution.

The regression now separately verifies native ownership for scalar fresh-hcore
calls, process-artifact replay and moved geometries, and for genuinely retained
prepared warm/moved batches. Successful mixed returns require their actual
work; moved numerical failure requires explicit bounded strict recovery with
unknown failed-attempt work, not fabricated zero work. Every returned force
still passes the same independent moving-grid oracle. No production numerical
admission was relaxed to make the test pass.

Raw measurements, freezes, build logs, diagnostic profiles and failed attempts
remain in `../evidence-fast-goal-20261016/`. Node2 measurements cannot be divided
by node4 RTX 5090 measurements to manufacture a before/after speedup. The
source-matched packet-v4 96-atom result is historical evidence for that build,
not a timing measurement of native-force-v5.

The final same-card node4 comparison measures PBE/PBE0/r2SCAN at 3 and 12 atoms,
with first calls and three repeat pairs per point. Each snapshot alternates
FP64/Fast order in the middle repeat; the old and new snapshots are separate
sequential jobs, not interleaved or randomized arms. Native force
ownership reduces previous Fast endpoint medians by 67.5/73.1/55.6% at 3 atoms
and 37.3/45.6/29.6% at 12 atoms. Both arms retain the same observed SCF work
populations. These are complete endpoint gains, not FP32-kernel throughput.

The final eight-family matrix retains all 128 independently checked returned
calls and genuine mixed work in every Fast sample. PBE/r2SCAN at 3 atoms remain
slower than optimized FP64; at 12 atoms their FP64/Fast ratios are 1.272/1.232.
PBE0 is near parity, not a large demonstrated acceleration. Complete per-family
results and scope are in the [tracked compact report](../../../../benchmarks/results/shared-fast-20261016/README.md).

The final v5 source also completes the 96-atom/768-AO nonlocal endpoint under
Slurm 1647 on node5, in 810.377 seconds with 17 mixed and four strict Focks.
Its authenticated independent E/max-force/RMS-force errors are
`6.68e-10 / 3.86e-6 / 1.17e-6`. This is a final-build numerical/large-size
qualification pilot on a different physical GPU/driver, not a paired timing
ratio against the node4 packet-v4 observations.

## Remaining limitation

The shared mixed J route still uses canonical whole-shell recurrence rather
than strict J's density-contracted MD schedule. Opening a generated FP32
density-product contract is not an admissible shortcut: this Fast contract
retains FP64 density products and global accumulation. A future MD optimization
must preserve those invariants, actual-work evidence and independent matrix and
complete endpoint gates. Native force ownership alone does not promise that
every small-system Fast endpoint beats optimized FP64.
