# Decision: guarded complete-force response defaults

Status: implemented for the complete native DF molecular-force owner
Date: 2026-10-12

## Problem

The existing gap-free triples scalar fusion has positive scoped endpoint evidence,
and serial Lambda scratch sharing saves real device memory without a measurable
speedup. Keeping both behind explicit selectors prevents the ordinary complete
force endpoint from benefiting. Earlier observations used a different frozen
source and cannot qualify the combined current-source default by addition.

## Default policy

On master `2a6de26eea1f83e9fd1e8df01c189fea44058b6b`, with the retained-shell
derivative dependency from PR #2348, request scalar fusion and serial scratch
sharing in the complete native DF force owner. Keep lower-level triples and
standalone Lambda defaults unchanged. Explicit false selectors remain matched
oracles; benchmark `auto` follows the complete-force defaults.

Fusion preserves gap output demand, page and panel residency, strict FP64
equations, finite checks and the original reduction order. A tight complete
budget refuses optional V seeds before reducing residency. Free-VRAM preflight
also refuses optional storage. A typed preparation allocation failure destroys
its partial CUDA owner before exactly one unfused retry with the original
page/panel plan. Driver, numerical and submission failures are not OOM fallbacks.

Scratch shares only one owner's sequential producers and borrowed-output
consumers. Persistent inputs, cuts, seeds, publications and the core frontier
remain separate. Existing matrix/core/audit refusal and allocation retries retain
their bounded order. This policy does not authorize cross-stream reuse.

## Qualification

Host planning covers the existing optional-owner masks, budget boundaries,
rollback steps and alignment padding. The executable default/parser contract
covers both omitted controls and explicit opt-outs. Real-device qualification
must cover independent small complete forces, all water coordinates, signed
Lambda actions, scalar/matrix and conventional/DF paths, tail batches, mixed
precision composition, exact budget refusal and post-preflight allocation
refusal. Memcheck must include borrowed-output reuse and the partial-owner retry.

Performance qualification uses ten fresh complete ethane230 E+F processes on
one Slurm-owned RTX 5090, with fixed CCSD/Lambda Q32, FP64 and core reuse. Both
sides retain the single shell derivative traversal; only fusion and scratch
selectors differ. Independent energy and retained complete-force checks run
after each endpoint. Large independent finite differences cover only the retained
two components at two steps, not all large-force components. Record endpoint
and phase times, work counts, actual selected paths and sampled VRAM peaks;
do not call modeled materialization bytes measured hardware traffic.

## Next bounded Lambda experiment

The existing native arena analysis identifies a pair of redundant unit copies
in the matrix auxiliary graph. Both are only `o*v` singles slices, not large
Q-major doubles tensors. Their producer/final slot equality survives both arena
policies. An emitter candidate compiles, but the source change is reverted:
removing these small copies does not address the roughly 1.86 TB of reported
logical packing outputs. No device speedup is claimed for this unmeasured
candidate, and it is not producer-consumer fusion.

The next independent experiment compares requested Q48 with Q32 while holding
the new fusion/scratch defaults, core reuse and FP64 fixed. It must report the
actually admitted batch, complete endpoint time, GEMMs, batch count and peak
VRAM. Neither a larger request nor a successful fit proves an endpoint win.

## Source-matched observations

The compiled library is
`9bdb525460e365593be6a0d6869005f2fc735a79aebbc25f55b2fe1c78dd9a88`.
After reverting the small-copy emission experiment and rebuilding, the library
is byte-identical to the qualified and measured default binary.

Slurm job 1492 on n5 runs ten fresh complete endpoints, five per side, in
counterbalanced paired order. Both sides use one shell derivative traversal;
only fusion and serial scratch requests differ. Mean complete E+F time is
227.335725 -> 219.841134 seconds (3.2967% less time). Triples time is
39.347941 -> 31.819904 seconds; scalar kernels are 1320 -> 165. Lambda time
79.215928 -> 79.221516 seconds has no identifiable improvement. Sampled peak
VRAM is 26544 -> 25720 MiB (824 MiB saved), and owned Lambda device storage
falls by 860495616 bytes. CCSD 19 iterations, Lambda 21 iterations/22 actions,
416 reported Q batches and 14307 Lambda GEMMs remain unchanged.

Slurm job 7826 on n1 runs the qualification suite: 779 pass, no skips, including
host planning cases and real-device scientific tests. All water nuclear
coordinates have independent finite-difference checks. Memcheck passes 17
reuse/refusal cases with zero errors and zero leaked bytes. Early qualification
attempts exposed validation-interposer problems, not accepted scientific
evidence; the final interposer binds the vendor CUDA DSO explicitly because
ctypes loads the production library in local scope.

Slurm job 7829 separately compares Q32/Q48 on a different physical RTX 5090.
Four fresh ABBA endpoints give mean complete E+F 220.128467 -> 217.917966
seconds (1.0042% less time), and Lambda 79.083091 -> 76.944110 seconds
(2.7047% less time). Actual batches are 32/48, reported batch count 416/286,
and Lambda GEMMs 14307/10577. Both matrix audits/replays and core reuse remain
admitted. Sampled peak VRAM rises 25689 -> 29595 MiB, an additional 3906 MiB.
This two-sample-per-side, memory-expensive observation does not promote Q48;
Q32 remains the default. Do not pool timings across the two GPUs or add their
percentage reductions.

Every retained endpoint passes the original independent energy, complete
retained-force, response residual, stationarity and scoped large-FD gates.
Maximum retained-force error is below 8.164e-10. The large FD evidence still
covers only two components at two steps. The unpaired job-1490 endpoint is
retained separately: its cohort stopped on an incorrect validation assumption
that scratch-avoided bytes must be zero when serial reuse is off. Existing
matrix/audit sharing already reports a positive value; effective serial savings
must be demonstrated by paired capacities and sampled peaks instead.

No native DIIS/secant preconditioner, general Q-batch producer-consumer fusion,
f-shell component tiling or multi-output parameter VJP is delivered here.

## Evidence storage

The full compact publication is retained in the PR's Git history and bound by
`benchmarks/results/df-response-defaults-20261012/snapshot.json`. After the
retained-shell dependency merged, current master had only 3060 bytes available
under the unchanged aggregate evidence cap. The current tree therefore retains
only the new bundle's verified recovery manifest and concise summary. Existing
historical evidence is untouched; the shared offline recovery tool restores
and checks every archived file without requiring a Release or external archive.
Frozen measurements retain their original source identity after rebasing; they
are not relabeled as timings of the newer integrated master.

## References

- `docs/maintainer/performance_engineering.md`
- `tests/python/test_df_lambda_serial_scratch.py`
- `tests/python/test_df_occupied_triples_fock.py`
- `tests/python/test_df_complete_force.py`
- `benchmarks/results/df-response-defaults-20261012/README.md`
- Ignored source-matched execution receipts: `.artifacts/cc-response-defaults-20261012/`
