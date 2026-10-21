# Decision: bound streamed Coulomb work and restore CUDA force capacity

Status: implemented (remaining large-endpoint bottlenecks documented)
Date: 2026-10-21

## Scope and invariants

This follows the [endpoint repair](2026-10-21-scf-endpoint-redundant-work.md).
Keep reference defaults, native XC/VV10 grids and moving-grid response, metric
cutoffs, final-state gates and complete energy-plus-force timing unchanged.
The runs below are diagnostic populations, not new README measurements.

## Streamed value symmetry

The source-backed J build now has an explicit raw lower-pair adapter. Fold D
as D_ii on the diagonal and D_ij+D_ji elsewhere; this also supports signed,
nonsymmetric diagnostic densities. Reuse the existing dead K transpose scratch
for folded D and then packed J after charge contraction. Projection leases are
revoked before this reuse, all work remains ordered on the plan stream, and
neither K output nor live auxiliary charge aliases the scratch.

The full-rank factor-ordered metric solve and rank-truncated root path are
unchanged. `GENERATIVEQC_DF_STREAMED_COULOMB_PACKED=0` retains the dense source
fallback. Host-staged compatibility plans retain their original execution.

Twelve source/J tests pass against independent libcint and NumPy, including
Cartesian/spherical batches with different shell layouts, auxiliary g, rank
truncation, ragged panels and three source mappings. Twelve existing streamed
K tests pass. Twenty-four complete HF cold/warm/changed-geometry cases and two
DF-KS energy replay cases pass after restoring their intended streamed storage
domain. Dense value storage is explicit in those tests; HF uses 16 MiB for
energy and 20 MiB for forces, while the larger DFT fixture uses 20 MiB. An
inadequate budget or a now-resident plan is not a numerical failure or a pass
of streamed-route coverage. Earlier failed fixture attempts are retained.

For r2SCAN-3c96 (1088 orbital, 6816 auxiliary), each J build produces
64,606,519,296 bytes over 592,416 lower pairs, versus 129,094,385,664 ordered
bytes. Instrumented J time is about 67.4 s versus about 136 s. Slurm 11807
finishes SCF in 22 cycles but times out at its 45-minute limit during forces.
It supplies no complete endpoint timing or energy/force qualification.

## Pure Coulomb force work

The timeout identifies a separate amplification: the full-rank streamed force
reader forms A M^-1 AO panels, regenerating the entire raw tensor per output
auxiliary panel. A single panel can take about 44 s before its derivative
consumer. Pure J needs only q=M^-1(A:D), then bar_A=c D q and
bar_M=-c q q^T/2. Exchange has additional factor-consistency requirements and
must not use this reassociation.

The candidate reads each raw auxiliary slice once, contracts density, applies
Q^T, divides by retained eigenvalues, and rotates with Q. It forms no explicit
inverse or raw Gram. Existing charged workspace supplies the slice, vectors
and derivative panel. Admission requires a source-only full-rank J owner,
BLAS arithmetic, no tensor borrow and no single fitted tensor. Retained
forward factors keep their existing route; truncated metrics keep the spectral
Frechet response. `GENERATIVEQC_DF_COULOMB_RESPONSE=panels` retains the fitted
comparison path. This is not yet a claim of a completed large-case repair.

Twenty-one tests pass: eighteen extended-precision adjoint comparisons (positive
and signed coefficients, condition numbers 1e2/1e6/1e10, ragged and full panels)
and two independent PBE/r2SCAN-3c tetramer cold/warm E/F gates plus a displaced
PBE tetramer owner-rebuild gate, with default
auxiliary-g bases, grids and full moving-grid response. Unsupported exchange
and rank-truncation requests are rejected by the specialized interface.
Synthetic signed metric terms can cancel large outer products. Their forward
error is bounded against the sum of absolute contributing terms, with the
same 3e-9 relative factor and 2e-6 absolute floor. An initial cancelled-result
relative check failed at condition 1e10 (maximum relative error about 1.1e-7);
the component-scaled check and additional positive-coefficient cases are
explicit, not a changed molecular force tolerance. Molecular absolute force
gates remain 2e-6 maximum / 7e-7 RMS in these small endpoint tests.

Slurm 11823 completes the cold E/F endpoint in 2114.053 s (22 SCF cycles).
Its common dense-reference errors pass: energy 2.0400e-6 Eh/atom, force maximum
5.6255e-5 and RMS 2.4767e-5 Eh/Bohr. The warm call exceeds the total 45-minute
allocation. The receipt remains `running` as written before Slurm termination;
its separately retained outcome is a timeout, and the overall population is
unverified. Do not invent a warm value or report the earlier timeout as a
successful control. Raw charge preparation takes about 359.5 s; generic
three-center derivative panels cost about 20 s each. The source-only schedule
still excludes automatic shell derivative promotion; that remains a separate
qualification problem.

## WB97M-V DF96 driver memory

An independent CUDA allocation/launch/limit interposer finds about 14.35 GB of
explicit live arrays. The original public Schwarz kernel has 27,480 bytes of
local storage per thread, consuming about 6.9 GB of driver backing. The later
64 KiB CUDA stack request consumes about another 9.9 GB. Prepared free memory
is only about 1.6 GB on the 33.67 GB device. A response allocation requests
4,301,873,664 bytes with only about 862 MB free, despite its declared 4.315 GB
allowance. The missing capacity is primarily driver storage, not a hidden
multi-gigabyte XC array. Explicit DF allocation diagnostics now distinguish
declared-budget rejection and physical allocation failure.

An experimental Schwarz split launches only the even orders reachable from
the proven maximum shell angular momentum; unknown bounds retain all orders.
Its through-d public local frame peaks around 16,840 bytes. Twenty-four Direct
HF/streaming/high-angular cases pass independent energies/forces and geometry
replay. But the subsequent 64 KiB stack request still leaves about 1.56 GB free;
Schwarz splitting alone does not repair the endpoint.

A diagnostic interposer reducing the requested stack to 32 KiB leaves about
10.24 GB free and permits separate K/J responses in about 3.27/1.60 s. The subsequent long wait was initially attributed to LR force, but that
attribution was incorrect. Entry/return instrumentation shows LR returns in
seconds. Native stack sampling and owner metrics place the delay in subsequent
XC/VV10 grid/geometry force response. These probes are diagnostic populations. Twenty-four Direct tests and two explicit
CUDA mixed-DF force tests pass under Compute Sanitizer memcheck with this
interposer (zero reported errors). The original mixed-DF tests using `auto`
fail earlier on auxiliary-role admission; the explicit-CUDA diagnostic copies
are identified separately and do not claim to repair that unrelated issue.

The next candidate requests 32 KiB only with proven through-d metadata and
retains the original ceiling for f/unknown domains. Both HF and prepared KS
consumers use one serialized runtime read/raise transaction. A dedicated mutex
prevents a smaller request racing with and overwriting a larger one; it is
separate from the allocation-ledger mutex because driver calls can trigger
resource callbacks. No ordinal/context-limit cache survives a context change.
Host tests exercise competing 32/64 KiB requests, pre-existing larger limits,
new contexts and query/set failures on both NVIDIA and CuMetal test doubles. Historical d/f recurrence failures with a 1 KiB stack remain
relevant; this does not justify a universal small-stack policy. Forcing dense exchange did not remove the
packed owner's reserved projection and still failed its response allocation.

## Evidence and next gates

Local ignored `.artifacts/{packed-v1,raw-coulomb-v1,schwarz-v1,df-memory-v1,df-memory-v2}`
and matching `n1:/data/jzzeng/scf-performance-20261021/` directories retain
scripts, source receipts, actual library hashes, traces and failed attempts.
All GPU execution uses finite Slurm allocations and compiler builds use verified
sccache 0.16.0. Immutable libraries are separate from the mutable build tree.

## Composite grid capacity diagnosis

A complete WB97M-V DF96 native-checkpoint E/F call finishes in 650.497 s with
32 KiB diagnostic stack (Slurm 11824). Independent dense-oracle errors pass:
energy 3.0088e-6 Eh/atom, force maximum 1.9949e-5 and RMS 9.3917e-6 Eh/Bohr.
This is a checkpoint diagnostic, not a cold/warm benchmark. Direct96 cold/warm
calls at 32 and 64 KiB both pass; their timings are 283.611/31.273 and
281.024/44.049 s, respectively, with 28/1 SCF cycles. Single instrumented pairs
do not establish a performance ranking.

The DF composite planner selects eight point lanes, no retained center geometry
and no phased Becke storage for both stationary source owners. Each source
allocates 9,604,216 bytes. More than 117,000 eight-point batches take about
487 s. Its 1 GiB device cap includes a lazy LR derivative owner estimated with
`cartesian_nao = 2*nao`, although actual WB dimensions are 768 public and 800
Cartesian. This conservative shape-only envelope starves grid concurrency.
The capacity candidate accepts the exact per-item Cartesian count from known
basis metadata, preserving the old shape-only envelope when that metadata is
absent. KS applies it to first-order capacity queries; value-only allowances
are unchanged. Both cold and warm queries use the same bound. Optional-owner
allocation and bounded generic fallbacks remain constrained by that allowance.

A diagnostic disabling all MD J (Slurm 11830) times out before force owner
construction. It changes SCF execution too, so it cannot isolate the unused
MD reserve or support promoting that environment override.

The next immutable library is `resources-v1`, SHA-256
`a639a6014e2cfa75afd2ec8a01768592abb114e926015599dea5f1b9148629b7`.
Its source manifest verifies 5912 local production/test/generated inputs;
pre-existing historical benchmark evidence is outside the compiled-input set.
Host allocation/bridge/composite checks pass 1184 cases. Thirty-three current
streamed raw/J/adjoint/complete molecular tests pass on that exact library.
An explicit shell-schedule diagnostic adds three passing complete/rebuild
cases, without promoting source-only shell scheduling by default.

The exact-Cartesian native admission matrix passes all 32 s/p/d/f,
Cartesian/spherical, zero/first-order, shape-only/metadata-aware combinations
under Compute Sanitizer (zero errors). An initial test inferred J capability
from the composite schedule label; a tighter value-only allowance selected
canonical K alongside generated J. The test now queries actual J capability
and requires the retained first-order shell lease, instead of interpreting a
label as evidence of a missing J provider.

WB97M-V DF96 on the production candidate, with no stack interposer, completes
the identical native-checkpoint E/F call in 165.881 s instead of 650.497 s.
Both owners now select 256 lanes, 13,257,400 bytes and retained center geometry;
phased storage still does not fit. Independent errors are unchanged to
roundoff and pass the same gates. The fresh cold/warm pair (Slurm 11839) also completes and passes, at
721.468/113.579 s with 28/1 SCF cycles. It is still much slower than the desired
comparison, especially the roughly 17 s SCF iterations. This evidence resolves
the allocation failure, not the remaining full-endpoint performance problem.
The split real-device sanitizer groups pass 6 high-angular, 8 RHF and 8 UHF
cases with zero errors, preserving independent values/forces and geometry replay. The first combined 10-minute
sanitizer allocation times out after 19 cases without a final summary; it is
not counted as a completed test population.

## Final source integration

PR #2657 was merged upstream as `af6a2ecf7a89148fbea5b40734c19bbcadf74dfa`.
This follow-up is based on master `08dd9bd94b4cf6ca30ba9cc32b677482394c020c`.
Its intervening changes concern finite Array API reductions; the native
scientific source matches the qualified candidate. The updated master runtime
also passes all 33 streamed tests. Earlier receipt identities remain unchanged.

The final serialized-stack library is `resources-v2`, SHA-256
`abf855a9e595806bc5190d0aa9714dc17428e517c51e3d844b9c788bb0359d12`.
Local device-fact/concurrency and allocation checks pass 1158 cases, with the
real-GPU-only device-facts test skipped locally. The final library passes the 32-case native resource matrix under Compute
Sanitizer, three independent streamed complete/rebuild force cases and six
public semilocal force cases. Its WB97M-V96 native-checkpoint call completes
in 152.723 s (two SCF cycles) and passes the unchanged common gates. The
4944-file source snapshot matches this worktree; the later hybrid test-only
adjustment is retained separately with its execution log.

Six public semilocal RKS/UKS force tests pass. A supplementary four-hybrid run
initially has three missing-package failures (small s/p stationary artifacts
were absent from the diagnostic directory) and one pass. These are retained
as packaging failures. Matching generic PBE0/B3LYP/M06-2X artifacts are built
through verified sccache and packaged separately for the final recheck; existing
immutable diagnostic libraries/manifests are not overwritten. After packaging,
PBE0/B3LYP pass independent forces and replay but hit an obsolete assertion
requiring a Python stationary owner after default routing selects the native
prepared force. The generated-owner lifetime/work test now explicitly selects
its intended consumer; its fresh displaced call still uses default routing.
All four FP64 RKS hybrid tests then pass independent analytic forces, replay,
changed geometry, finite differences and source accounting on the final library.

Next: remove remaining large-source response/regeneration and long-range SCF
work, retain bounded fallbacks and then repeat every README method under the
final protocol. README/CBS and their timing figures are unchanged here.
