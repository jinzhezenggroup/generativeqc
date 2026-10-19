# Rejected: reuse the uniform Hermite density inside residual tasks

Status: rejected
Date: 2026-10-19

## Problem and experiment

GPU4PySCF transforms density before its long-lived primitive tiles. The native
masked residual consumer still forms task-dependent density tails. A small
experiment attempted to reuse the already allocated uniform Hermite-density
buffer when a directed suffix starts at zero and its AO ranges are disjoint.
Partial suffixes and diagonal/orbit-sensitive cases retained explicit tails.

The density producer ran before residual contraction on each replay. Readiness
was carried only by a replay-local device view, not a persistent owner flag.
There was no new resident allocation, weaker screening, or reference fallback.
The source-matched experiment is frozen separately from the retained compiler.

## Decision

Remove this shortcut rather than layering further runtime branches onto it.
The shared-coarse/scalar-projection predecessor, not this experiment, is the
starting point for subsequent scheduling work. The failure does not reject
density-first MD-J: the shortcut retained task-local masks, tail construction,
short potential lifetimes and AO projection, unlike upstream's dataflow.

## Evidence

Both runs used node1 through Slurm, the same RTX 5090 assigned as visibility 4,
96 atoms, 1088 spherical AOs, the canonical explicit r2SCAN-3c basis and frozen
density SHA256
`a0b22059e12adcd39771e5dcfaac0951afb298d8252ef6674889ad0349ed6090`.
Each prepared-owner host-density-to-host-J benchmark used four interleaved
samples per method and a fresh independent CPU libcint J outside timing.

| Source-matched run | Native incumbent (s) | Candidate (s) | GPU4PySCF default (s) |
| --- | ---: | ---: | ---: |
| Scalar projections, Slurm 10341 | 4.023389 | 3.745616 | 0.690596 |
| Added full-density reuse, Slurm 10345 | 4.018491 | 3.828258 | 0.689302 |

The additional shortcut made candidate J about 2.2% slower than its predecessor;
the matched incumbent and upstream timings remained close. All methods passed
their independent J gate (`atol=3e-9`, `rtol=3e-11`). Native maximum absolute
error was approximately `2.456e-11`. Upstream and native screening semantics
are not identical.

Trace jobs 10343/10347 measured candidate residual `(2,1)` at
0.358864/0.348151 s and `(2,2)` at 0.404276/0.447359 s. A counted replay proved
that full-density reuse actually executed: 192513920 `(2,1)` and 144187380
`(2,2)` channel hits, alongside 414958144 and 460311500 explicit tail scalars.
Registers increased from 126/168 to 128/177 for these classes. These observations
do not isolate a unique hardware bottleneck.

The shortcut passed 76 real-CUDA oracle cases, including the native density
producer, nonsymmetric UKS inputs and a subsequent zero-density replay. Both
production modes passed all 114 normal molecular tests (Slurm 10346), and 88
host tests passed. Its sanitizer run, Slurm 10344, did **not** qualify: the
instrumenter could not attach within its default launch timeout while the
fixture compiled. Only that task's allocation was cancelled after checking the
live job. Do not report that attempt as sanitizer success. Future sanitizer
wrappers use a finite 300-second attach timeout and a shared pytest temporary
directory, while retaining a finite Slurm allocation.

## Source and receipt boundaries

The native experiments adapt compact-E PR #2600 commit
`aacbccc275274cb70e36b76a646750fe667acf3e`, not this checkout's older dense-E
native HEAD. The predecessor library SHA256 is
`ff6eb8f04c409e9af91e2550c580bf47fe30c0d56a14c6bc94baea379620b4b4`;
the rejected library SHA256 is
`8ef95507701ed910861b1e0e246bff0935809351f742b903c12270c885540ac9`.

Remote immutable roots on node1:

- `/home/jzzeng/generativeqc-qualification/scalar-hermite-20261019-49fc`
- `/home/jzzeng/generativeqc-qualification/density-first-hermite-20261019-49fc`

They retain source, compile commands/cache statistics, library identity, J
arrays/results, trace and work-count receipts. Local ignored counterparts are
`build/scalar-hermite/` and `build/density-first-hermite/`; rejected compiler and
test code is under the latter's `frozen/` subtree. Stationary AOT artifacts built
for the rejected compiler closure must not be reused as final-source evidence.

## Revisit when

A compiler schedule can keep Hermite potentials and density channels alive
across many partners with bounded exact-mask channels, rather than opportunistic
full-density reads inside the existing task-local consumer. Require fresh
source-matched endpoint evidence, not a lower count of explicit density sums.

## References

- [Upstream comparison](../implemented/performance/2026-10-19-gpu4pyscf-residual-j-comparison.md)
- [Compacted-source predecessor](../implemented/performance/2026-10-19-compacted-masked-hermite-sources.md)
