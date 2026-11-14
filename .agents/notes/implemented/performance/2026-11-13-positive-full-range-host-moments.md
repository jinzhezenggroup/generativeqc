# Decision: reuse the positive moment evaluator for explicit CPU Full consumers

Status: implemented; component-qualified, complete endpoint still times out
Date: 2026-11-13

## Problem and measured scope

The early physical-owner matrix source still exhausted its unchanged complete
1,800-second limit. Its new scopes locate the unfinished frozen subphase to ERI
enumeration/contraction, not S/T/V. See
`2026-11-13-physical-owner-host-matrix-sources.md`.

A separate test-only profiler interposes the actual frozen core's value/matrix
leaves and interrupts after a finite number of primitive calls. It reconstructs
RKS D from the saved native orbitals without SCF; W is explicitly zero for this
component experiment, not a claim to reproduce the stationary S/T/V operator.
All failures leave the public matrix unpublished. A four-million-call prefix
contains 2,831,716 scalar-only and 1,168,284 one-nonorigin-owner calls, with no
three-/four-physical-owner calls. It is not a representative full endpoint.

In that prefix, one-in-16 sampling estimates 6.293 s in leaves out of 6.986 s
in ERI enumeration/contraction. Full sampling gives 6.329 s in leaves out of
7.222 s; disabled sampling gives 6.991 s for the ERI region. This identifies a
leaf-cost hypothesis for the measured prefix, not every molecular quartet.
The profiler does not require denied perf permissions or security changes.

## Decision

The explicit CPU factorized value/first/mixed/matrix consumers previously obtained
Full moments through the generic bounded 64-node positive quadrature. Reuse the
existing `range_detail::long_moments` evaluator at boundary one for these Full
consumers: this is the same full Boys integral, not long-range exchange or a
change to the physical operator. Retain the established bounded quadrature as
an explicit failure fallback, and preserve all domain/order/storage checks.

Use an explicit CPU-only geometry/moment schedule. Do not silently change the
generic default or any CUDA/range-separated consumer lacking device evidence.
Do not copy a recurrence, introduce a second scientific algebra, screen tails
differently, change FP64, or alter derivative identities/normalization. Graph/AD
continues to differentiate moments through the next-order moment rather than
through a numerical evaluation branch. All original resource and scientific
gates remain unchanged; both public ECP saddle guards stay closed.

## Independent evidence

`build/qualification/host-positive-full-moments-hypothesis-20261113` retains a
test-only header/leaf prototype built from the sealed frozen generated sources.
Its 3,930 moment samples cover orders 0--14, T=0, a logarithmic 1e-300--1e308
domain, dense moderate arguments and both sides of every series/recurrence
transition. A 100-decimal-digit mpmath incomplete-gamma oracle gives maximum
relative error 2.1941625858586e-15 above a 1e-300 denominator floor, under the
predeclared 2e-14 relative gate. Underflow, finite values and exact write bounds
are checked, with invalid-domain/no-write controls.

All 182 independent s--f matrix samples and 14 nontrivial owner partitions pass,
including distinct completely coincident owners. Maximum independent matrix
error is 2.1625097545996397e-11; incumbent parity is a separate control, not the
oracle. Forty-four admitted first/mixed samples cover four-center Full/LR/SR
and full-range three-/two-center families through g auxiliary angular momentum;
independent value/gradient/fourth-order-gradient-HVP error is at most
3.4314676100599684e-11. Four unsupported range-DF refusals remain closed.
Geometry validation covers 16 invalid bindings and 60 valid cases; 15 owner-map
domain/write-bound controls pass. No reference/native SCF runs occur.

Seven interleaved isolated medians across 182 primitive records show median
baseline/candidate ratios 3.09 for value, 2.44 for first, and 1.91/1.41/1.24 for
one/two/three non-origin matrix owners. These are isolated primitives, not
molecular preparation or endpoint speedups. The separate interleaved component
prefix comparison must be retained with its complete apply and ERI times.
The seven interleaved four-million-call prefixes give median ERI times
7.013237755 s (incumbent) and 3.932085500 s (candidate). Complete interrupted
component apply medians are 18.980112906 s and 15.907653176 s, including the
unchanged S/T/V sweep. Primitive/map/incidence work is identical, all outputs
remain unpublished, and no SCF is restarted. This remains a bounded component
experiment with explicit zero W, not complete molecular timing or raw-H proof.

## Diagnostics and promotion boundary

The initial test-only fast geometry condition lacked the incumbent's full
argument/omega admission checks; that unqualified prototype is retained, not
promoted. The corrected version explicitly preserves finite T/rho and Full
omega-zero checks before the positive path, with the existing bounded routine
as fallback. A validator initially requested unsupported range-separated DF
operations; it was corrected to test the actual admitted domain and explicit
refusals, without expanding production capability or loosening numerical gates.
Initial exclusive-output and prerequisite-ordering diagnostics are also retained;
final receipts use new names rather than overwriting the original evidence.

Production now opts the CPU factorized second and physical-owner matrix emitters
into `make_eri_geometry`'s explicit positive-Full policy. The geometry default,
CUDA emission, prepared-geometry ownership and LR/SR paths remain unchanged.
The Full helper validates T/order/output before entering the existing recurrence;
the binder additionally retains the finite-positive-rho and zero-Full-omega
checks. The generic quadrature and established order-13 API are untouched.
Permanent CPU tests cover independent 100-digit moments and geometry
operator/domain/storage contracts; current-state documentation describes the
opt-in and fallback without treating the prototype as endpoint qualification.

The sealed profile archive has SHA256
`04c0a7f1218edeaa7f0adead63c3171295667a4940a1af922d05a332cc38df97`;
the sealed hypothesis archive has SHA256
`201bf3883a1b3325ef234e1cc44f1d750819ebf55240ed3bed7d0902bae0d6c0`.
Their production-unchanged labels are historical and must not be rewritten.

Complete source/core-bound component qualification and a materially changed
endpoint attempt remain required. No complete raw Hessian, response, unscaled modes, mixed Ni/I/Cs,
practical 25--40-atom workload or requested Ni transition state is qualified by
this experiment. Do not relabel a precursor or rerun the sealed old endpoint.

## Production component qualification

The new `host-positive-full-moments-source-20261113` cohort qualifies production
identity `aefd4636e91330f2de8eac1cbd756c0551dcb46917901bd2d4142314a96f9f67`
and core SHA256
`662fd3dccf3006083b7d98860ee3d94688ed380138b26c36b2977772b2e06608`.
All 506 tests/groups pass without skips: 84 positive-moment/domain, 109
factor/physical-owner, 231 native integral, 66 public/source/CMake, ten native
response/API/CLI and six documentation contracts. Sphinx warnings-as-errors,
formatting, Ruff and compiler/SCF/native ownership checks pass. The CPU-only
build uses verified sccache 0.16.0 and retains before/after cache statistics;
cache use must not be described as cache hits when the changed headers miss.

A one-off sealed-precursor comparison confirms all four CUDA emissions are
unchanged and the five CPU emissions differ only in explicit geometry policy.
The initial audit's wrong emitter import is retained separately as a validator
diagnostic; the corrected audit passes without changing production code.
Metadata-only probes retain the identical physical basis/grid and complete
plans: ERI 882874977 below 1e9, XC 867660477948 below 2^40, 81 public AOs,
six atoms, 87458 points, tile 256 and 18 columns. No electronic or derivative
evaluation occurs in those probes. A new source/core-frozen full native attempt
under the unchanged 1800-second cap remains separate from these component gates.

## Terminal changed complete-endpoint attempt

The new frozen native attempt exits 124 at complete wall 1800.03 seconds and
peak RSS 441608 KiB. The stationary energy/gradient checkpoint returns at
545.799988449 seconds. First-source preparation completes in 315.834755903
seconds; frozen-integral preparation now completes in 599.204973361 seconds,
including S/T/V 12.410291744 seconds and ERI 586.794065520 seconds. These are
source-bound subphase observations, not a completed-endpoint speedup claim.
The previous first preparation was 561.996026237 seconds, but its frozen ERI
phase never returned within that endpoint's remaining cap, so do not fabricate
a complete matrix/end-to-end ratio from that censored predecessor.

The completed frozen receipt records 130597029 ERI matrix calls, 639412191
requested coordinates and 3596397327 ordered second partials; the S/T/V sweep
records 9548100 primitive calls. These are execution counts, distinct from the
unchanged conservative admission plans. Nuclear columns zero and one complete
in 167.629852575 and 152.978219656 seconds. Column two remains incomplete at
timeout. Across completed GMRES action scopes (including that unfinished
column), 26 Arnoldi actions and 26 exact-residual replays consume 94.530563994
and 94.601172862 seconds inclusively. Those timers do not isolate XC from JK,
and an unfinished column is not zero work or a completed response receipt.
The complete raw Hessian, final response receipt and unscaled modes are absent.

Independent saved-state/gradient validation passes, without reference SCF or
displaced-state calls: energy error 3.410605131648481e-13, gradient error
3.928496281535229e-12, native gradient maximum 1.1870708816852198e-08,
density error 1.1554396328605776e-11 and occupied-root overlap
0.9999999999999969. Overall validator exit 1 correctly records the absent raw
Hessian. Both public ECP saddle guards remain closed and the full goal remains
incomplete. Do not rerun this unchanged expensive endpoint or extend its cap.

The next investigation must distinguish actual response-operator and nuclear
column work on this producer, rather than attributing all remaining time to
the now-completed matrix preparation. Existing point-local response fields
retain quadratic AO work; any BLAS/batched alternative needs explicit bounded
panels, nonsymmetric-density semantics, independent numerical/derivative gates
and complete-consumer evidence. No response solver tolerance, exact residual
replay, raw symmetry or final acceptance criterion is relaxed by this decision.
