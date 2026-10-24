# Bind the force request to actual qualification work

Status: implemented, host validated; device qualification remains open
Date: 2026-10-16

## Problem

The point-family producer declared a fixed 256-point force policy while only
setting the independent SCF/XC tile option. The production Python force entry
resolved its workload and execution with `tile_points=None`, so actual receipts
could report `budget-auto` and a different geometry lane capacity. Equality of
such work between arms did not satisfy the original absolute-work contract.

Prepared native force capability can also select a path that never invokes the
Python force observer. A retained dictionary entry from the previous execution
must not become that next execution's apparent work receipt.

## Decision

The existing private stationary force composition entry accepts an optional
positive integer tile request. It binds the same request to active-AO workload
selection and ordinary/composite execution. Its default remains unspecified and
budget-selected; there is no new public method admission or environment control.

The producer supplies 256 to this entry, clears the capture before every E+F
execution and consumes each observed receipt once. It checks the actual typed
request and schedule. Missing selected-route receipts or ignored requests retain
an explicit failure stage; rejected raw work is preserved when available.

## Invariants and rejected alternatives

- SCF/XC tiling and force tiling are distinct controls.
- A prepared native route is not overridden to force a compatibility path. A
  bypass without the requested receipt fails closed.
- Declaration-only changes, accepting null/512 and reusing a prior receipt do
  not qualify a fixed request.
- The original independent consumer, numerical tolerances, absolute phased
  work counts, architecture guards and scientific algebra remain unchanged.
- A valid fixed request alone does not establish phased support on a device.
- Existing numerical feasibility and negative route evidence retain their
  original source/library identities. Changing the Python native input requires
  a new source-matched SDK; the previous installed library is not relabelled.

## Evidence

Eleven new host regressions fail before the repair: malformed explicit values,
unhonored or differently typed requests, an automatic schedule and a native
route bypass. After repair, the producer, packaged-AOT force and composite-route
host modules pass 127 tests with no skips. Packaged-AOT cases also exercise the
normal unspecified request and the explicit request through actual workload
resolution and execution handoff, without probing an NVCC compiler.

These are host orchestration and routing checks, not NVIDIA execution, complete
cold wallclock, dynamic local traffic or endpoint profitability evidence.

## References

- Refs #2072 and PR #2250.
- `benchmarks/pbe0_xc_tile_pairs.py`.
- `tests/python/test_pbe0_xc_tile_pairs.py`.
- `tests/python/test_stationary_aot_no_compiler.py`.
