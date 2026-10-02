# Proposal: stage compacted VV10 partners for ordered row consumers

Status: experimental; complete endpoint performance comparison pending
Date: 2026-10-03

Each ordered row reloads the same compacted partner's position, omega, kappa and
weighted density. Stage those six FP64 fields cooperatively in 128-partner
shared-memory tiles. The partner array is 6 KiB per CUDA block, with no new
resident/host arena or source lifetime. The row continues to consume ascending
compacted indices and perform every addition in the original order. Pair scalar
mathematics remain compiler-owned, including the bounded one-denominator closure
and its original ordered exceptional-domain fallback.

Only VV10 feature consumers with at least 128 grid points select this schedule.
Short grids, rVV10 and energy-only consumers retain direct partner loads. There
is no optional allocation or identity prerequisite. CUDA block size remains 128.
Masked or padded lanes must participate in cooperative loads and both barriers;
they perform no pair evaluation. An all-inactive block exits only after a
collective vote, so no live row is stranded. Negative-zero screened rows keep
zero outputs; positive-zero and negative quadrature-weight semantics are intact.

The source schedule shares partner load instructions across row lanes, but cache
broadcasts already affect the old route. Do not interpret the source load ratio
as a measured DRAM-traffic reduction or infer endpoint speedup from it. Actual
molecular active-pair counts remain unexported; dense grid squared is capacity.

## Qualification

33 host-executed signed-weight/row-mask tests pass for the direct fallback.
Slurm node1 job 5435 passes 50 real-device tests comparing every staged row output
exactly against the direct schedule: 127/128/129/257/383/513 points, both output
demands, signed weights, sparse/empty partners, an entirely screened block,
partial partner tiles, padded output canaries and exceptional failure publication.
All 50 tests also pass under both CUDA memcheck and synccheck with zero errors
in a separate finite Slurm GPU allocation.
The same job passes six independent complete RKS/UKS/displaced-energy WB97M-V
tests, and runs a sequential same-GPU allocation comparison against the
one-denominator baseline. Complete comparison results are still pending.

Measured candidate library is
`fa92488867552099dedb67bf453a05d21cf15ab0908473b3198b2e6b63be49c8`, with parent
`e2b19170b` plus the archived one-denominator and staging patches. It includes
#1716 and predates #1732. The current review branch is restacked onto master
`74c89369c`; keep native provenance distinct. A separate latest-master combined
experiment adds #1736's full/LR source decomposition; its six independent complete
tests pass in jobs 5439 and 5440, with 48/96-atom complete endpoints pending.
That combined run cannot isolate staging's speedup.

Build/ccache receipts, source archives, patches, real-device tests, sanitizer
receipts and comparison results live under ignored
`.artifacts/wb97m-vv10-staged/` in the composed worktree. Upstream GPU4PySCF's
Apache-2.0 source was inspected for its use of partner staging; no upstream
kernel code is copied. This implementation extends the existing traversal,
including GenerativeQC's stable active compaction and signed-zero row contract.

## Promotion and revisit conditions

Require complete endpoint improvement with every sample passing 1e-8 Eh and
1e-7 Eh/Bohr gates, a larger case, changed-geometry correctness and CUDA memory /
barrier validation. Reject or narrow selection if cache behavior or collective
loads make this slower on an admitted domain. Preserve a bounded direct-load
schedule and the ordered scalar/reduction semantics when changing tile shape.

Compiled sm_120 resources for masked feature/geometry staging are 72/98 registers,
7168 shared bytes and zero local/stack spill bytes. The partner fields themselves
occupy 6144 bytes; compiler-reported shared use, including overhead, is 7 KiB.
The corresponding direct-load kernels use 64/88 registers and no shared storage.
These are resource receipts, not endpoint performance evidence.
