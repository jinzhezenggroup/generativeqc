# Decision: compact the factorized MP2 derivative first stage

Status: implemented, bounded CPU qualification
Date: 2026-10-11

## Problem and scope

Conventional factorized MP2 derivative weights populate only structured regions
of the legacy `di*N^3` first shell buffer. The next transform traversed every
`q,r,s` slot, including known zeros. This is one #1626 slice; dense RCCSD(T),
second/third/local staging, dense orbital RHS and full UMP2 forces remain outside
its scope. The lexical inventory still reports six materializations. Five site
identities changed with the internal function signature/index spelling; this
is not evidence that five materializations were retired.

## Decision

For `O` occupied and `V=N-O` virtual orbitals, automatically choose a checked
`di*O*(2*N+V^2)` first allocation only when smaller than the admitted legacy
allocation. Keep an explicit internal staged layout for bounded comparison and
retain the conservative production resource plan.

The Coulomb region is `(iu*N+q)*O+i`; the exchange region starts at `di*N*O`
and uses `(iu*O+i)*N+q`. The shared `q==i` exchange contribution accumulates
into the Coulomb diagonal, preserving the original accumulation history. The
correlation region starts at `2*di*N*O` and uses `((iu*V+a)*O+j)*V+b`.

The next transform visits occupied `r` only. Its diagonal Coulomb reduction
retains full q order; its off-diagonal occupied exchange term precedes virtual
correlation terms, as in the legacy q reduction. First producer arithmetic and
zero gates, later transforms, callback domains and atom scatter remain intact.
The correlation region is still materialized and reused across sj. This is
compaction and removal of structurally zero consumer work, not complete fusion.

## Rejected alternatives

Preprojecting Fock with a matrix multiplication would regroup arithmetic and
needs separate acceptance evidence. Fusing correlation into each sj consumer
may remove storage but repeat producer work; no measured profitability supports
that choice here. Do not hide surviving staging by helper/index spelling or
replace measured resource telemetry with the admitted allocation plan.

## Invariants and evidence

The updated host harness binds the entire common-source SHA-256. Unmodified and
instrumented builds compare every local weight, gradient, callback/element count
and allocation lifetime. All compact/staged local weights and gradients must
match exactly; the independent long-double direct tensor oracle and expanded
dense gradient comparison retain the existing 3e-11 acceptance.

The 32 synthetic tests and 16 independent PySCF 2.14.0 ERI center-derivative
tests passed against the final formatted common source
`7cf56330b2c57adc4b4ab93ec992207da42c2befd119a19765333d11eeae1657`.
Real ERI fixtures include every center slot, Cartesian d, spherical d/f,
reordered shells, repeated atoms and spectator atoms. PySCF supplies only test
oracle data, never a production dependency.

Executed first-stage update counts are unchanged. Across all ordered shell
pairs, compact second accumulation counts are
`N^2*O*(N + N-1 + V^2)`, replacing `N^5`. Third and local remain `N^5` each.
The probe also measures scoped C++ allocation payload/lifetimes. Those counts
exclude zero-gate checks, rank-two BLAS, callback arithmetic and bookkeeping;
allocation payload excludes allocator overhead, borrowed inputs and oracle
storage. Neither substitutes for complete endpoint/device measurement.

Paired source-matched H2 public E+F builds passed all numerical/physical gates
and produced identical cold/warm/displaced results. Ordinary C++ payload peak
remained 18911 bytes, with 48 fewer total requested bytes per call. No complete
endpoint peak or speedup benefit is claimed. Source-matched LiH values also match
exactly, with all independent scientific gates passing; its ordinary C++ peak
remains 289735 bytes, with 7680 fewer total requested bytes per call. A fresh
GCC 11.4 HPC build passed 34 baseline and 50 candidate tests, including all
molecular cases, and repeated H2 scientific qualification. Later-collected
cache counters do not recover the earlier build server's hit/miss counts;
verbose launcher commands establish use, not a claimed hit rate. Raw records,
completed-pair timing/coverage, protocol and scoped work/allocation receipts are retained in
`benchmarks/results/mp2-factorized-first-20261011/`. Earlier attempted
candidate directories contained baseline bytes because git apply skipped paths
under a parent repository. Those results are invalid as candidate evidence.
Final snapshots use isolated Git roots, exact LF archives and per-file SHA
assertions. The original public qualification resource gate remains failed
when production reports its explicit unmeasured sentinel; an external payload
tracker/RSS experiment must be reported separately without relabeling that gate.

## Consequences and revisit conditions

Both CPU and CUDA derivative consumers receive the same host callback contract.
This host evidence makes no CUDA-device or full endpoint speedup claim. Keep
#1626 open for remaining producer/consumer fusion, staging and dense-RHS work.
Revisit the compact layout when measured reuse/fusion work, complete endpoint
cost and independent scientific gates justify replacing its retained regions.

## References

- https://github.com/jinzhezenggroup/generativeqc/issues/1626
- benchmarks/results/mp2-factorized-first-20261011/README.md
- tests/python/test_mp2_derivative_pullback_probe.py
- tests/native/mp2_derivative_pullback_probe.cpp
- docs/developer/mp2.md
- docs/maintainer/performance_engineering.md
- ../numerics/2026-10-09-common-derivative-pullback-validation.md
