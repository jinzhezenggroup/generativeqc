# Decision: qualify typed weighted four-center derivatives separately from SCF

Status: implemented as a private qualification candidate, not public Fast admission
Date: 2026-10-13

## Problem

The XC-only Fast prototype leaves all gradients in FP64. The upstream study in
[the gradient-first proposal](../../proposed/2026-10-13-gradient-first-budget-fast.md)
suggests narrowing derivative recurrences and local reductions while retaining
strict global accumulation. Our exact-PBE0 profile contains substantial native
weighted four-center derivative work; copying upstream's DF algorithm would
change the scientific model rather than just arithmetic precision.

## Decision

Emit a float sibling of the existing external-weight derivative DAG. The
ordinary shell queues, screening, density weights, Boys policy, component
partitions and translational scatter remain their existing owners. No Rys or
Hermite recurrence mathematics is copied into a second implementation.

- Geometry preparation, density/normalization weights and Boys moments: FP64.
- DAG leaves: explicitly converted before arithmetic, not after double products.
- Weighted derivative DAG, component partition sums and primitive/source sums:
  FP32, without fast-math, TF32, clipping or screening changes.
- Independent-center results widen once before FP64 translation recovery and
  FP64 atom accumulation/storage.
- The candidate covers existing weighted total-angular-order 0–5 consumers;
  higher orders, SCF, XC, one-electron/Pulay and response remain strict.

`DirectForcePrecisionRequest` travels through private prepared-owner ports and
is applied to a launch-local batch view. It does not mutate the retained
geometry/SCF identity or use a global environment switch. Public callers keep
the null request and existing strict gradients. A float return buffer is not
necessary for float arithmetic; conversely the unchanged double API says
nothing about arithmetic precision.

A nonfinite task returns before any atom scatter and is retried once through
the strict implementation. The dispatcher calls the two implementations as
siblings: nesting a strict recurrence inside the mixed frame would increase
worst-case CUDA stack liveness. All task sources retry together; no partial
J/K force can be published before the check. This detects nonfinite failure,
**not finite rounding error or underflow**. It is not a per-call error bound.

Optional device counters report mixed task attempts, primitive/source
evaluations and strict task retries. Their three words are charged alongside
the existing force cursor. Instrumented copies/fences are explicit and excluded
from clean timings. Higher-order strict work is not relabeled as mixed.

## Qualification and promotion boundary

`benchmarks/pbe0_mixed_gradient.py` uses fresh FP64-SCF owners and the complete
native stationary gradient for both comparisons. Its test-only bridge also
runs strict/mixed/strict gradients on the **same** final-state token. This
private endpoint is distinct from the previous public Python-force endpoint;
do not combine their times into a speedup. No DF substitution is permitted.

Numerical gates are `1e-7 Eh` energy, `1e-5 Eh/bohr` maximum force component and
`3e-6 Eh/bohr` RMS force component against independent moving-grid PySCF.
Those gates qualify only measured cases, not arbitrary user error budgets.
Public Fast is deliberately unchanged pending endpoint evidence, broader
cancellation/diffuse coverage and a method-owned admission/reporting policy.
NVE drift, Hessians, transition states and IRC are not qualified here.

## Evidence

- Host compiled typed-arithmetic, overflow and private-owner forwarding checks;
  the existing weighted-derivative suite also retains its finite-difference and
  component checks. No source-hash or implementation-spelling tests are added.
- Scheduled endpoint/oracle and retry tests live in
  `tests/python/test_mixed_force_endpoint.py`; execution evidence is recorded
  separately with source/library identity, first-owner work and timing scope.

## Revisit when

Promote only when complete same-route endpoints show a useful gain inside an
explicit application-budget domain. Extend the common generated algebra to
remaining derivative/XC hotspots rather than claiming this is an all-FP32
gradient. Reject an always-on audit that repeats every complete SCF/gradient
merely to make an approximate candidate look safe or fast.

## Measured continuation: do not promote this candidate

The complete numerical and timing evidence is retained in
[`benchmarks/results/pbe0-mixed-gradient-20261013`](../../../../benchmarks/results/pbe0-mixed-gradient-20261013/README.md).
On RTX 5090, exact-PBE0/def2-SVP at 192/384 AOs gives only 1.0127x/1.0085x
construction-through-gradient endpoint ratios, and 1.0241x/1.0130x gradient
ratios. All measured mixed errors pass `1e-5 Eh/bohr`, but the larger case's
`1.17e-6` maximum force error would fail a `1e-6` budget. Neither larger size
nor a dtype change alone makes this a useful Fast policy.

The native endpoint initially admits only s/p publicly. An explicit private
precision request now admits its existing d-shell providers for qualification;
the strict comparator sends the same private request with mixed arithmetic
disabled. Null/public requests retain the original admission. This must not
be confused with promoting d-shell native forces or enabling mixed forces in
the public Python endpoint. The latter has a distinct complete consumer and
packaged `pbe0_rks_spd` artifact contract; combining its historical times with
these measurements would be misleading.

Nsight's strict 192-AO capture attributes 79.9% of kernel-sum time to the still
FP64 native AO/XC/moving-grid geometry kernel. The affected integral kernel
improves by about 1.31x, from 0.5154 to 0.3930 s. The next precision experiment
must address the measured complete consumer's remaining AO/XC work, not promise
upstream DF-r2SCAN speedups for exact PBE0. Schedule improvements apply to the
strict comparator too. Public Fast remains unchanged.

Qualification checks now include same-state strict/mixed replay, actual mixed
work, unchanged FP64 SCF work, a tighter strict comparator gate and independent
oracle gates before publishing timing ratios. Optional counters add an explicit
fence and stay outside clean timing. Their storage must be charged in both the
generated owner's allocation inventory and the outer shape-only capacity
authority; changing only the local allocator omits 24 bytes from the envelope.
The CUDA test shim likewise needs four cursor words, not one. The host lifetime
tests execute the real transports and cover this receipt/fence and copy-failure
recovery rather than requiring one lexical `cudaMemcpyAsync` occurrence.

Water/tetramer GPU oracle tests and the overflow retry pass; water/retry memcheck
reports zero errors. An earlier sanitizer failure at `cudaDeviceSetLimit`
reported OOM and is retained separately: no invalid access was demonstrated,
and its root cause is not established by the capacity-accounting correction.
