# Decision: reuse immutable canonical values for repeated exact RHF actions

Status: implemented
Date: 2026-10-05

## Problem

Issue #1972 targets minute-scale reductions in the exact RHF response consumer
of complete correlation-only DF-CCSD(T) forces. Delivered #1901 checkpoint
GMRES already reduces large-system J/K actions from 28 to 17. Repeating that
controller work or relaxing screening is not the next useful optimization.
The remaining actions repeatedly evaluate ERIs of an unchanged geometry.

## Decision

The shared Direct provider can retain an immutable, geometry-owned source of
symmetry-unique full-range Cartesian/public-AO ERIs. Its existing canonical
recurrence dispatcher evaluates the source once. A recurrence-free replay
kernel shares value loads and traversal for J and K and calls the existing
compiler-owned distinct-orbit FP64 scatter. No second CC-specific J/K equation
or new integral recurrence is introduced. Restricted and unrestricted replay
share this source and retain their original spin normalization.

The lease follows the plan's immutable canonical angular-bucket and sorted-pair
row order. Geometry, basis, device and stream belong to that plan, not to a
dimension-only cache key. Plans do not share or persist these values across
changed geometries. Preparation fences and finite-audits the entire source
before publication; failure releases unpublished storage. Idempotent admission
still validates the owning device. Optional allocation OOM is cleared before
exact fallback; unrelated CUDA and nonfinite-source failures propagate.

For Cartesian source dimension `N`, storage is `8 * M * (M + 1) / 2` bytes,
where `M = N * (N + 1) / 2`. It remains quartic storage, admitted all-or-nothing
under both the requested ceiling and the remaining complete-endpoint budget.
Unavailable canonical storage, short capacity or allocation refusal retains
ordinary exact recomputation. Neither screening nor precision changes.

The initially opt-in route is promoted to conservative automatic selection for
relaxed, unscreened frames with at least 64 public AOs, capped at 8 GiB.
Explicit zero disables reuse; a positive ceiling requests it independently of
that crossover. The 64-AO gate deliberately excludes the small endpoints and
is not a claim of a universally optimal crossover. Domain and resource refusal
continue to select the original exact provider.

Final scalar-CUDA physical residuals explicitly recompute unscreened ERIs,
without trusting the lease. Positive fixed-mask actions also retain ordinary
recomputation. The published residual and stationarity gates are unchanged.

## Rejected alternatives

- Harder density-dependent screening changes the fixed signed-density operator;
  the retained geometry-only screening experiment also removed negligible work.
- Another CC-owned recurrence/scatter would duplicate scientific ownership.
- Unconditional quartic storage, partial publication, dimension-only reuse and
  ignoring CUDA faults would violate resource or physical-source contracts.
- Further strong-DF-preconditioner/strict-identity-recycling tuning repeats the
  negative large-system evidence accompanying #1901 without a new algorithm.

## Evidence

All real-device execution uses finite Slurm allocations on n2, partition
`main`, `gpu:pro6000:1`, preserving assigned device visibility. Other machine
names are optional resources, not a requirement to use every node.
Base: `ab403d8be2e54c4c8059e272973d8b21af1dd9b4`.

Complete 230-AO ethane / 488-auxiliary endpoint, seconds:

| Phase | Unmodified base, job 2385 | Matched recomputation, job 2411 | Automatic default, job 2428 |
|---|---:|---:|---:|
| RHF | 143.147 | 147.470 | 157.902 |
| DF source | 3.595 | 3.632 | 3.653 |
| CCSD | 145.721 | 149.143 | 147.464 |
| (T) | 109.951 | 112.160 | 110.913 |
| Lambda | 272.614 | 277.530 | 274.904 |
| factor/source response | 3.761 | 3.854 | 3.799 |
| orbital/nuclear response | 393.993 | 397.637 | 121.668 |
| complete endpoint | 1072.782 | 1091.425 | 820.304 |

Default saves 252.478 s (23.53%) versus unmodified base and 271.121 s (24.84%)
versus matched recomputation. These are individual complete samples, not a
statistical confidence interval. Parent timings expose clock/load variation;
do not attribute all movement in unchanged parents to this optimization.
Explicit 8-GiB admission, job 2412, independently returned in 793.182 s.

Default charges 17.693 s of source preparation to its 17.886 s response setup.
The J/K subset falls from 313.111 s wall / 313.117 s device to 20.228 s wall /
20.227 s device. Sixteen replay actions total 1.890 s wall / 1.889 s device;
one action still evaluates the independent final source. J/K actions remain
17, Z iterations 12 and solver operator actions 13. These nested quantities
must not be added to parent endpoint times.

The source retains 575,639,415 values / 4,605,115,320 bytes (about 4.29 GiB).
Replay reads 9,210,230,640 scalar values. Action recurrence evaluations fall
from 9,785,870,055 to 575,639,415; preparation separately evaluates another
575,639,415 values. Reads, inventory, radial evaluations and hardware traffic
are different quantities. Reported complete numeric capacity remains
7,107,898,249 bytes, but this is not evidence of unchanged actual peak VRAM.

Final independently recomputed Z residual is `1.35449e-13`, stationarity
`1.09824e-11`. Relative to matched recomputation, energy difference is zero
and maximum force difference is `1.86826e-9` Hartree/Bohr. Independent libcint
signed-density J/K, linearity and self-adjointness cover Cartesian/spherical,
restricted/unrestricted, through-f sources and capacity/allocation refusal.
Complete RHF/Pulay and nonzero-Z gates and independent complete CC energy/force
finite-difference gates explicitly require source reuse when requested.

Qualification: 49 CUDA response tests, 12 cached RHF/Pulay tests and 12 cached
complete-force tests pass. Memcheck passes 22 source/replay/refusal tests with
zero errors. Shared response and diagnostic input tests pass 44 CPU cases;
the native resource-policy test passes without CUDA initialization.
The 317-Cartesian-AO inventory exceeds 8 GiB and refuses without allocating a
value lease or executing response actions.

Automatic small/medium endpoint samples retain zero resident bytes:
water7 1.434 s versus 1.463 s control; water58 24.180 s versus 24.503 s control.
Explicit water58 reuse returns in 20.867 s but is deliberately not a default
crossover claim. Three successive signed-density actions reuse each admitted
source; a 0.05-bohr geometry change reconstructs its own source and agrees with
uncached evaluation. A one-byte ceiling retains the ordinary action timing.

An independent native action receipt measures about 0.119 s cached versus
18.359 s recomputed per 230-AO action, plus 17.677 s preparation. Nsight records
28 replay kernel launches: the gain is recurrence removal, not lower launch
count. Static sm_120 replay resources are 60 restricted / 48 unrestricted
registers and zero local, stack and shared bytes. Runtime API/copy/event and
owner transfer/synchronization receipts are retained. Pageable download API
time can include preceding kernel waits; it is not removable PCIe transfer
time. Native diagnostic wall times include test transfers; its device interval
excludes those transfers and is not a complete-force endpoint.

Profiling now observes ordinary production dispatch. Complete canonical census
coverage is explicitly unavailable for uncounted generated/generic channels;
a zero count is not evidence of zero work.

## Consequences and revisit conditions

Retain the explicit zero-capacity route and all bounded resource fallbacks.
Requalify automatic selection when device families, class-specific generated
consumers, basis families, warm-start behavior or available memory materially
change the crossover. No speed claim is made for an untested GPU. Avoid claims
of asymptotically bounded storage or bitwise reproducible atomic reductions.
Further work should address remaining measured parent costs, not repeat the
already-delivered checkpoint action-count reduction.

## Reproduction

Ignored evidence is under `.artifacts/issue1972/`; n2 retains jobs and inputs
under `/data/jzzeng/issue1972-20261005/`. It includes input/native binary hashes,
source patches, traces, phase/counter JSON, JUnit, memcheck, Nsight and ccache
before/after receipts. Configure explicit CXX/CUDA ccache launchers with
checkout-root `CCACHE_BASEDIR`, then rebuild both native probes and
`benchmarks/df_ccsdt_force_endpoint.cpp` against the same library.

Use endpoint selectors `1 1 1 1 8 8 8 0 1 2 1 30 0 0 1`; omit the final cache
ceiling for automatic selection, append `0` for matched recomputation, or
append `8589934592` for explicit reuse. Run through finite Slurm GPU steps.
`benchmarks/rhf_resident_jk.py` provides basis-only paired action receipts;
`--maximum-bytes 1` and `--displacement 0.05` qualify refusal and geometry
reconstruction without an orbital oracle.
