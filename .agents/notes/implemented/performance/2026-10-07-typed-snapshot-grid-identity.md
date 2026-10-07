# Decision: bounded typed identity for exact native snapshot grids

Status: implemented
Date: 2026-10-07 UTC (2026-10-08 Asia/Shanghai)

## Problem

Issue #2074 identified process-cold host amplification in exact-grid identity:
constructing a canonical JSON record expands millions of FP64 quadrature values
into Python lists, floats and text even when no interchange file is requested.
Retaining the resulting grid helps warm executions but not a fresh owner.

## Decision

Keep `ExplicitGrid` identity version 1 as the default for ordinary callers and
legacy imports. Add an explicit keyword-only version 2, selected by native
snapshot binding, which hashes immutable numerical buffers in bounded chunks.
Both versions have explicit, hash-verified interchange records; reading a v1
record does not migrate it, and relabelling a record's version fails verification.

Version 2 hashes the ASCII SHA256 of canonical metadata followed by C-order
little-endian FP64 points, FP64 weights and checked i32 owners. Metadata includes
schema/version, shapes, dtype, ordering, Bohr units and frozen provenance.
Signed zeros remain distinct. Endianness and layout of caller buffers do not
affect the normalized identity. JSON/list materialization occurs only in
explicit `record()`/`write()` calls. The fixed small-grid golden digest is
`4833bdb5b125e11031658df322ae8aebfc03d0f3bb60906126459cd741d07b48`.

## Invariants and rejected alternatives

- Retained hits compare every current source point, weight and owner. No
  pointer-only witness, stale mutable-buffer digest, skipped validation or new
  native generation/lifetime assumption is introduced.
- The existing 128 MiB retention budget and zero-budget uncached fallback remain.
  The fallback also selects typed identity, rather than reintroducing JSON cost.
- The grid, FP64 precision, density, convergence, screening, derivative ownership
  and analytic grid response are unchanged. No preparation moves outside timers.
- Removing the grid content hash entirely was rejected: exact source identity is
  a scientific contract, not optional diagnostic metadata.
- Making v2 the universal default or silently migrating old records was rejected:
  legacy canonical identities must remain stable.
- In the second population, rebuild native and packaged stationary artifacts
  separately for both variants. The sealed AOT contract includes `dft/grid.py`;
  replaying a modified Python compiler against a stale manifest is not a valid
  optimization qualification. Never override source hashes to manufacture a hit.

## Frozen qualification

PBE0 RKS, full spherical def2-SVP, FP64, RTX 5090/SM120 on n1 through finite Slurm
allocations. Unpruned `(48,16,32)` moving grids, no density fitting, full-density
reference Fock rebuilds, energy/density tolerances `1e-12`/`1e-10`, max100.
Native/reference screening is `1e-12`/`1e-14`. Timers include preparation and the
first synchronized complete host-return energy plus analytic force execution;
imports, CUDA context initialization and Calculator construction are excluded.
Every sample uses a fresh process/owner/density with persistent compilation
caches reused. Do not confuse this process-cold contract with an empty compiler
cache or an import-to-result timer.

The initial population uses `d84e4006c611a0fdfbf6e12cb9c1ff891e715ba6`.
Three interleaved samples per variant/size retain all completed samples and first
source/cache misses. Complete medians in seconds:

| Atoms | GPU4PySCF | Baseline | Typed candidate |
| ---: | ---: | ---: | ---: |
| 12 | 18.397 | 35.249 | 34.534 |
| 24 | 33.836 | 38.740 | 37.581 |
| 48 | 52.370 | 95.812 | 90.290 |
| 96 | 77.515 | 196.035 | 209.851 |

The first 96-atom complete median **regresses**. Baseline Fock builds are
`[26,25,29]`, candidate `[29,30,27]`; no iteration normalization or same-quartet
claim is justified. Its force-phase median nevertheless falls 28.965 to 24.278 s.
All 24 native cold and 16 candidate cold/warm/moved/moved-warm gates pass.

The second population freezes master
`fbc6b389fb28bc5e93bf8580f56f1f91e5e2fbc2`, including exact-hybrid stationary AOT
selection (#2082). Three interleaved fresh-process samples at 96 atoms/768 AOs:

| Variant | Complete raw seconds | Median | Physical Fock builds |
| --- | --- | ---: | --- |
| Baseline | 189.599, 221.473, 192.025 | 192.025 | 25,30,25 |
| Typed | 178.576, 184.890, 186.804 | 184.890 | 24,25,25 |
| GPU4PySCF | 87.658, 94.808, 78.819 | 87.658 | 43,47,38 |

The observed complete median reduction is **7.134 s / 3.72%**. Force-phase
medians fall **28.731 to 22.977 s**. These are small-population observations, not
statistical significance; differing SCF trajectories prevent attributing the
entire complete-endpoint change solely to hashing. This does not qualify J/K
improvements or GPU4PySCF parity.

Native library SHA256s for this matched-build population:

- Baseline: `78ac5365152b6d6d4e8b5669e90b47668c4ee66e02c5d9ec5cab8d0454d9b1fc`.
- Candidate: `d3e677edf5608fec9eef195a1e39dbba69c7fde43a5103cda09ccb8222fa67ab`.

All six second-population native cold gates pass, maximum energy/force errors
`7.9581e-11 Eh` / `2.5069e-11 Eh/bohr`. All 16 candidate replay gates at
12/24/48/96 atoms pass. Both variants pass packaged PBE0-RKS AOT qualification
for STO-3G and def2-SVP: cold/reuse/displaced endpoints forbid compiler discovery,
IR/AD and source generation. Stationary weight-program identities are identical
across all six native clean samples.

Pure-Python focused identity/response/resource/source/AOT-contract and production
boundary tests: 188 pass. Compiler structure: 493 modules, zero dependency errors.
Changed Python files pass configured Ruff lint/format checks. Native grid fixture,
response, identity and cache tests also pass (98 tests) in the matched candidate Slurm allocation;
the local worktree has no native library, so its corresponding library-dependent
tests are not evidence of native correctness.

## Diagnostic evidence, not additive clean spans

The first population's independent 2,359,296-point CPU construction benchmark
has medians **4.843 to 0.497 s**; separately instrumented tracemalloc peaks fall
**934,009,312 to 95,946,614 bytes**. Retained bytes remain **94,379,329** under
the unchanged budget. Separate cProfile construction/resolution observations
are **5.354 to 0.690 s** / **5.799 to 1.153 s**, with typed hashing 0.088 s.
Those are nested instrumented observations, not sums of clean execution time.
Diagnostic process peak RSS falls **3,277,196 to 2,622,044 KiB**.

The separately retained second-population diagnostics confirm construction
**5.224 to 0.683 s**, snapshot resolution **5.669 to 1.139 s**, and typed hashing
**0.088 s**. Diagnostic process peak RSS is **3,274,724 to 2,835,572 KiB**;
these whole-process peaks include SCF/native activity, not just grid allocation.
Clean stationary-owner preparation is **1.053 to 1.058 s** and its internal
stationary endpoint median is **19.614 to 19.578 s**. The primary demonstrated
saving is the outer grid/snapshot binding, not an accelerated derivative kernel
or a new reduction in source compilation.

The implementation files are byte-identical across both populations. Evidence
remains separated by base; do not pool the populations or relabel old results
after a master update. Later master changes to default GPU MINAO (#2086) and
build parallelism (#2088) are outside the frozen second-population timing claim.

## Retention and known limitations

Local receipts, full-precision samples, source/compiler-cache/native hashes,
replay matrices, profiles, process-memory receipts and audited summaries:

- `/data/jzzeng/qc-pbe0-96-opt-20261007-d84e4006c/.artifacts/pbe0-96-opt/`.
- `/data/jzzeng/qc-pbe0-96-opt-20261008-fbc6b389f/.artifacts/pbe0-96-opt/`.

The first copied harness initially retained an incorrect hard-coded source
label. Its exploratory profiles/cancelled partial population remain under
`results/initial-harness-label-error/`, not rewritten or pooled. The corrected
complete population was restarted for provenance, not to exclude slow samples.
See that population's `provenance-repair.md` and input checksums.

The first broader PBE0-only GPU build has 98 passes and five failures: four
missing non-PBE0 profiles and one small-system profile-selection assertion also
reproduced on unmodified baseline. They are not represented as passing tests or
fixed by this patch. The second population uses dedicated exact-hybrid AOT guards
instead of claiming these legacy selection assertions qualify artifact replay.

## Consequences and revisit criteria

Version 2 is a new content-identity contract; serialized records remain portable
but v1/v2 identifiers intentionally differ. IO still pays JSON expansion cost
when requested. The Python owner tuple and full source validation remain real
host work; a future immutable typed-owner representation requires its own
compatibility, budget and native-lifetime proof. Revisit if interchange evolves,
hash throughput becomes material, or a native exact-snapshot witness can prove
the full identity without weakening validation.

References: #2074 (host identity), #1895/#1965 (complete-endpoint accounting),
#1892 (independent J/K plans, not solved here), `docs/developer/dft_grid.md`.
