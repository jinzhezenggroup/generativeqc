# Decision: expose paired physical inputs at the shared RKS batch boundary

Status: implemented; structural discovery remains opt-in
Date: 2026-10-28

## Problem

The [structural discovery campaign](../architecture/2026-10-28-xc-structural-parameter-discovery.md)
confirmed four small RKS batched point regressions: TH_FCO/TH_FCFO at approximately
3%, and BKL1/BKL2 at approximately 2%, for 16,384 points and 127-point tiles.
The bad restricted-batched kernels had the same register counts as literal
specializations. Aggregate near-parity did not qualify those individual cases.

The mathematical equal-spin Graph already shares work, but the common physical
traversal still accepts a runtime spin count. Its two transport channels reach
the shared scalar adapter as independent arrays. Parameter lifting also changes
native constant folding and instruction scheduling. Equal mathematical roots do
not by themselves communicate identical physical inputs to the CUDA compiler.

## Decision

Add a restricted batched adapter that explicitly reconstructs both physical
channels from channel zero after the host selector proves RKS. It calls the same
restricted mathematical body. The existing four point kernels per group remain
four: the new adapter replaces the restricted batched instantiation. Ordinary
and general adapters, parameter storage, 128-thread scheduling, runtime output
addressing, and the compact final-tile layout retain their existing contracts.

This is an execution-ABI rule for every selected sharing group, independent of
functional names, coefficient count, or source entry. It does not change group
discovery, mathematical derivatives, work floors, precision, or public admission.
In particular, pairing physical gradients does not replace the independent
polarized sigma-ab derivative. LDA continues to use its actual U1 RKS body;
disabling equal-spin optimization still selects the general GGA/MGGA worker.

## Evidence and rejected alternatives

The source-matched receipt and executable reducer live in
[`xc-shared-point-abi-20261028`](../../../../benchmarks/results/xc-shared-point-abi-20261028/README.md).
They retain all ablations, the rejected broad candidate, compiler/cache metadata,
instruction/resource reports, paired point populations, and complete endpoint
work ledgers. Baseline binaries and original measurements remain immutable.
The two campaigns share a compact receipt because the evidence corpus was at
its 80 MiB limit. All measurement populations, failures and dirty producer bytes
are preserved. Generated source bodies and raw disassembly stay in ignored run
storage; their hashes, recipes and operation/resource summaries remain tracked.
An opt-in source audit reconstructs CPU/CUDA products from a published ancestor
and exact overlays, independently of the current producer or unpublished PR head.
After the force/AOT PR advanced the target branch, its additional evidence made
the integration tree exceed the same budget. The existing runtime-cache campaign's
whole source/environment file censuses now use verified published Git recovery;
all its numerical inputs and measurements remain offline, and both replays yield
identical numerical/timing/work outputs. This follows the existing retention
policy rather than changing its budget or discarding XC qualification samples.

- Restricting registers or changing occupancy was not supported by the equal
  register counts. Changing to 64-thread blocks did not remove the regressions.
- Marking every parameter row `__grid_constant__` helped BKL but left TH slow.
  Combining it with paired inputs introduced approximately 3% in an APBE UKS
  batch. Resources already reported LOCAL:0 and STACK:96; there is no evidence
  of an eliminated thread-local parameter copy. The attribute changed generated
  instructions/scheduling. It is unnecessary for
  the accepted paired-batch fix and is not adopted.
- Fixing the entire runtime spin count removed TH's regression but did not
  consistently qualify BKL. The accepted change exposes physical input identity
  while preserving runtime tile/output addressing.
- Deduplicating opposite-sign coefficient columns did not fix TH/BKL.
- Applying input pairing to ordinary launches introduced a roughly 1% B86
  ordinary-case regression. Pair only the already distinct batched entry.
- A separate grid-constant RKS kernel entry also worked in the targeted pilot,
  but added an unnecessary storage distinction. Pairing alone sufficed.

Nsight Compute could not access device performance counters. Static disassembly
and paired CUDA-event measurements are the diagnostic evidence; do not describe
them as measured dynamic instruction counts or a profiler-proven stall cause.

The final three-arm campaign covers 120 cases and 1,080 process measurements
(three rounds, 50 event samples after 10 warm launches). The four original
failures have final/literal ratios 0.993252–1.000784. No final/before case exceeds
1.01. An unchanged ordinary UKS `GGA_X_APBE` control remains approximately 1.54%
above literal execution (before: 1.57%); this is not catalog-wide qualification.
The selected package retains 48 point kernels versus 292 literal kernels and
uses 7,972,672 object bytes versus 41,532,592. These are selected XC object sizes,
not complete library sizes. Default discovery promotion remains unjustified.

Host tests execute the generated launchers, scalar adapters, and grid-stride
traversal against independent Libxc, covering both selectors, U1/P2 layouts,
multiple coefficient rows, invalid inputs, and partial 16/127-point tiles.
Real-device qualification additionally exercises point/Fock/replay behavior and
public KS endpoints: 274 host checks, 657 GPU point/Fock checks, 23 independent
public KS checks, and 16 targeted device tail checks pass. Across 550 complete
endpoint calls, iteration/Fock counts and precision/work ledgers agree; maximum
paired energy error is 1.137e-13 Eh. Pooled per-cohort cold/warm/moved ratios span
0.985944–1.014979. The initial TH_FCFO RKS cold ratio 1.020619 remains visible:
five predeclared additional paired rounds plus a TH_FCO control independently
give phase ratios 0.988805–1.004964, with all original samples retained.

The earlier MVSB UKS nonconvergence on both literal and
shared paths remains a separately documented baseline limitation; this change
does not alter SCF policy or claim to fix it.

## Invariants and revisit conditions

Only a producer proving identical physical spin channels may use this adapter.
Arbitrary polarized arrays must never enter it. Public host validation and the
equal-spin selector supply that proof; matching numerical values at a sampled
point is insufficient. Input validation remains effective because the actual
RKS producer duplicates the same values into both channels.

Future compiler/toolchain changes should recheck per-case timings and complete
endpoint work, including ordinary and UKS controls. Do not widen the default
discovery policy, insert functional-name exceptions, or add redundant point
kernel variants merely because these local regressions have been repaired.
