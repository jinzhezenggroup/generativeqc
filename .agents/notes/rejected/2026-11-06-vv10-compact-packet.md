# Rejected: reduce the VV10 packet frame from four rows to two

Status: rejected
Date: 2026-11-06

## Problem and candidate

After rejecting explicit FMA spelling, the actual VV10 packet's136--176-register
frames suggested that reducing rows per lane might improve occupancy. A narrow
prototype changes four rows/128-point tiles to two rows/64-point tiles, retaining
the same canonical scalar roots, precision, symmetric bilateral accumulation,
positive-zero handling and O(N) output storage. Diagonal ordered redundancy is
smaller, not an intentional approximation; the pair count is
`(N*N + sum(tile_length*tile_length))/2 + positive_zero_rows*N`.

## Evidence and rejection

Only one bounded320000-point component comparison on actual RTX5090 was run;
no parameter sweep and no second full A100/5090 endpoint campaign:

- Feature four -> two rows:1.953467285 ->1.951296265s.
- Geometry four -> two rows:2.614718018 ->2.611764648s.
- Full-array max differences1.438849e-13 and1.172396e-13; same input and
  native pair roots. This is a scheduling parity diagnostic, not independent
  molecular/scientific acceptance or a new performance ranking.

The first scheduled launch failed while CuPy loaded NVRTC because the wrapper
omitted the CUDA library path; no timing samples completed. The corrected finite
Slurm16022 launch preserves scheduler visibility. The actual result, generated
header/wrapper and reproducible input seed/hash are retained in
`benchmarks/results/vv10-packet-fp64-fma-20261106/compact-probe.json.xz`.

Roughly0.11% component differences do not support a useful endpoint gain.
Revert the production prototype and its selector/test additions. Do not ship
a permanent opt-in or schedule broad molecular reruns to chase timing noise.

## Revisit when

A different algorithm actually reduces FP64 operations or repeated density-
current pair work, rather than changing packet width alone. Preserve exact
functional, grid/domain, force response, precision and convergence contracts;
qualify future changes independently and on complete actual A100/5090 endpoints.

## References

Issue #2976; [FMA rejection](2026-11-06-vv10-packet-fp64-fma.md).
Ignored producers/results: `.artifacts/direct-followup-20261106/compact-probe/`.
