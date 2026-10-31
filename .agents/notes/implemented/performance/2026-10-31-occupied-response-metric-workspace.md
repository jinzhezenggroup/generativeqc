# Decision: size occupied DF metric scratch by its live representation

Status: implemented
Date: 2026-10-31

## Problem

The stationary grid batching investigation found a warm-replay memory cliff:
WB97M-V DF96 needed 740,556,800 bytes of occupied response factors alongside
437,346,816 bytes of fixed response/density storage. A 2048-point experimental
force grid left 954,531,840 bytes free. The occupied route refused admission,
and its bounded fallback performed 4,489 fitted-panel GEMMs. The four-matrix
fixed layout reserved inverse and Frechet temporaries even when the full-rank
fitted consumer never touched them.

## Decision

A bridge-owned, full-rank fitted occupied response uses one metric matrix for
its adjoint. A spectral forward source without an explicit response root keeps
a second matrix to construct that root. Raw, borrowed, truncated and general
panel response retain their original four-matrix layout. The shared host layout
owns offsets for the adjoint, three AO temporaries, charges and potentials, and
rejects element-count overflow. Both admission and actual allocation use it.

The dispatch contract explicitly rejects compact storage for incompatible
consumers. Fitted occupied allocation refusal cannot fall through to a general
consumer with the smaller allocation. Existing bounded fallback, panel shrinking,
physical-memory replanning and authenticated projection recycling remain intact.
The earlier method-wide response reserve remains conservative: metric rank and
forward representation are not known when that capacity promise is prepared,
and its raw/truncated fallbacks still need the general layout.

No equation, numerical threshold, precision, BLAS ordering, quadrature, GPU-name
selection or public memory cap changes. A different admitted panel width may
change reduction grouping, so independent adjoint and molecular tests remain
required rather than assuming byte-for-byte results.

At 3,616 auxiliary functions the explicit-root layout saves 313,810,944 bytes;
the implicit spectral-root layout saves 209,207,296 bytes. These are resource
savings, not inferred endpoint speedups.

## Evidence

The rotated-metric probe allocates exactly the selected layout and checks both
physical adjoints against independent long-double equations for symmetric,
spectral and rotated explicit roots, signed terms, ragged panels, retained final
projection and condition numbers through 1e10. The molecular test uses independent
PySCF forces at original and displaced coordinates, including an allowance below
the old fixed-layout minimum. A spectral source must still budget its live root;
the first test allowance omitted that matrix's metadata headroom and correctly
failed allocation. That initial failure is retained separately.

All 24 ordinary ABBA populations (PBE0/WB97M-V DF96 on A100, RTX 5090 and
RTX PRO 6000) pass unchanged independent README gates. Warm medians differ by
less than 0.2%; no endpoint speedup is claimed for an already admitted response.
Each device passes 199 adjoint/budget tests and 34 existing response tests. Both
Blackwell devices additionally pass 81 compact-adjoint and one tight-budget
memcheck tests with zero errors and no API-error suppression.

The same-binary A100 64-MiB diagnostic now takes 23.625708 seconds warm at
2048 force points and 25.948267 at 256. Both one-cold/one-warm populations pass
the unchanged independent numerical gates. At 2048 points, free headroom remains
954,531,840 bytes, the response uses one metric matrix and omits 313,810,944
bytes, and the occupied response takes 0.920685 seconds with zero fitted-panel
GEMMs. The previous diagnostic took 75.572 seconds total and 52.846 seconds in
that response with 4,489 fitted-panel GEMMs. Its separate long-range response
remains present. These component observations do not promote larger default
batches or substitute for a repeated cross-device timing campaign.

Full build, test, sanitizer, endpoint, source and scheduler receipts are retained
in ignored `.artifacts/df-metric-workspace-20261031/`. The previous grid diagnostic
and its unpromoted settings remain in the linked investigation below.

## Revisit when

Joint force-grid/response planning may recover additional concurrency. Do not
raise grid limits globally solely because one A100 headroom case now fits.
Admission must account for live replay storage and retain complete endpoint
qualification on other devices, including the explicit tile contract.

## References

- [Stationary grid investigation](../../proposed/2026-10-31-stationary-grid-batching.md)
- [Compact evidence bundle](../../../../benchmarks/results/occupied-df-metric-workspace-20261031/README.md)
- PR #2888
