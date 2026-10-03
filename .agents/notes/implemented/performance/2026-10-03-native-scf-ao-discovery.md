# Decision: explicit native SCF sampled-AO discovery

Status: implemented, experimental; integration qualified, endpoint timing pending
Date: 2026-10-03

## Problem

The resident force caller can consume geometry-bound maps, but the native SCF
owner still supplies every AO to density and potential assembly on every tile.
The explicit compact-panel API alone does not change a complete SCF endpoint.

## Decision

An unused physical FP64 `CudaXcPlan` may explicitly discover sampled-jet maps
using its own immutable device basis/grid, the compiler's ordinary AO collocator,
and the same all-jet flag reducer used by resident discovery. Discovery performs
no density contractions or CPU AO work. It borrows the charged dense AO/work
panels and transfers only flags/errors; sorted host maps are copied once into
an appended, caller-owned device arena. A plan cannot change maps after any
physical or replay body has started, including unpublished replay work.

The SCF experiment is opt-in with `GENERATIVEQC_CUDA_KS_ACTIVE_AO=1`, admitted
only for device-fused FP64 WB97M-V, with an explicit sampled-jet cutoff of
1e-16. Unset or 0 retains dense SCF. Invalid switch values are rejected.
This is an experimental qualification control, not a promoted product default.
The native owner is rebuilt on changed basis/geometry/grid; density updates
reuse its immutable map. The force caller has a separate order-2 map and must
still pass complete energy/force gates when composed with this order-1 SCF map.

## Resources and fallback

Admission reserves one full global-AO map per tile, plus host indices, offsets,
and one tile's flags. The host setup peak is bounded by 64 MiB and conservatively
retained in the enclosing owner's numeric-capacity report. This fixed experiment
cap is not admission against the remaining public host budget. The current
Python KS resource planner rejects WB97M-V before preparation, so constrained
public host-budget support remains a promotion prerequisite. Mean selected AO
counts never reduce admission. Device allocations use the ordinary resource
allocator and any active device ledger.
Mandatory KS, VV10 and eigensolver allocations precede optional maps. A typed
device/budget OOM for the enlarged XC arena retries the original dense arena;
host registry OOM and unrelated errors propagate. Fixed host-cap misses select
the dense route without discovery. No external library or reference engine enters
production preparation.

The optional additive C diagnostic reports the current result's actual XC
submission count and its owner's immutable preparation work. Preparation time
is lifetime setup time, not a cost to add again on every warm sample. Counters
for point/AO and point/AO-square work describe one complete XC traversal.

## Qualification

Native tests derive independent expected masks from CPU AO values, then use
full-global-AO bilinears to validate selected rho/gradient, XC energy/electrons,
and the complete potential. Cartesian and spherical f cases span >32 AOs and
129/257-point tiles, with a small cutoff, partial maps, and all-empty maps.
Resource tests cover exact bounds, host/device misses, arena canaries and
post-evaluation refusal. The integration composition at `c06859af9` passed the
native discovery target, memcheck and synccheck (zero reported errors), and the
WB97M-V native target in n1 Slurm job 5571. That executable qualification does
not establish this standalone branch's complete SCF/force caller or a speedup.
Complete endpoint qualification remains pending.

The complete integration composition `0132d7584` subsequently passed seven
independent WB97M-V energy/force, changed-geometry and stale-snapshot cases with
both native SCF and force maps enabled, and the same seven with force-cache
allowance zero (n1 RTX 5090, finite Slurm 5575; 182.27/182.45 s, no skips).
Each group reported 66 successful native calls and 467 actual XC submissions;
every successful native call selected its geometry-owned SCF map. The zero
allowance group asserts no force discovery and dense force contraction work.
Native discovery, memcheck, synccheck and the native WB97M target also passed.
Host resource/compiler checks with the completed library passed 180 cases
(seven device cases skipped in that separate host-only command).

The loaded library SHA-256 is
`b82e468613c5c90e076aa10082b2389c8dda74335f625a7b77641c1c085ef3f9`;
all 1351 manifest inputs independently reproduce source identity
`de8a1684f0afbb4ec8545022c46d24c1e6c85c3c7851df237af436897bf1d862`.
The actual master `d442177a6` integration changes no build-manifest inputs from
the built source `c06859af9`. This is composition evidence, not an exact-head
standalone library qualification. Complete matched 24/48/96 timings are pending;
the numerical tests do not establish a speedup or public host-budget support.

SCF discovery runs in prepared-owner construction. A complete cold comparison
must include synchronized preparation plus the first energy/force execution
for both engines. The historical comparator's execute-only `native_cold` field
alone excludes this preparation. Preserve that field and report an explicit
complete-cold total; lifetime discovery time must not be added again to warms.

## Revisit when

Promote only after independent full cold/warm/moved-geometry energy/force gates,
constrained-resource fallback checks, actual work reporting, and matched large
endpoint timing establish a useful workload domain. The preliminary force-only
24-atom experiment did not establish an advantage over GPU4PySCF.
