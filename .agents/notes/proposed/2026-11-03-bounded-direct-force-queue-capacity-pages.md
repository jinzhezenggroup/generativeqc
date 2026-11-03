# Decision: fill the bounded Direct force queue before draining

Status: proposed; frozen-image qualification and current-PR integration in progress
Date: 2026-11-03

## Problem

The portable class grouping in PR #2952 improves complete endpoints, but its
roughly 3–4% primary A100 improvement is not an explanation or solution for the
remaining Direct gap. Its 128-thread scalar force CTA usually receives only a
64-candidate indexed page; shell screening reduces useful participation again.
Sorting those small packets improves divergence without fixing their admission
and drain utilization.

The existing shared queue already holds 256 descriptors. Filling that capacity
requires neither another global quartet inventory nor device-model dispatch.

## Decision

Parameterize the private indexed pager width, preserving its 64-candidate default
for existing consumers. Only the spatial low-order force owner requests its
existing queue capacity. Each of its 128 lanes retains two admitted descriptors
and their classes, accumulating one 21-class histogram before a grouped drain.
Each candidate is visited and screened once. The shared queue remains bounded
at 256; no new retained device storage or timing-based selector is introduced.

The changed low-order launcher already requires a non-null indexed block domain
and the qualified s/p/d owner. Unindexed and unsupported owners retain their
existing fallback; they are not silently truncated to the wider packet size.
No GPU name, architecture number or device-generation threshold selects this
algorithm. The SM120 kernel genuinely changes and needs new qualification.

Recurrences, FP64 precision, source coefficients, generated-class ownership,
screening predicates, AO/primitive work, grids and SCF settings are unchanged.
Floating-point accumulation order can change, so observable gates still apply.

## Evidence and acceptance contract

- Baseline revision: `4a8794b57b889f7e8dc58d033d98500cdd1dfd50`.
- Baseline native source identity:
  `2c766f01f2b8a2d737628b5944c823f9b48e7f3e4505184924145f75ebccc127`.
- Two-packet candidate native source identity:
  `4f83d0317983fc5a257b3e60bab90ae04add02b7e7527ed99db5aa5b31272101`.
- Both normal production builds are Release, fast compile OFF, tests OFF and
  stationary AOT `auto`, with verified compiler caches retained. Every native
  benchmark owner checks the frozen source, package hashes and loaded ELF's
  source identity, and blocks reference-oracle imports during production work.
- Full qualification preserves the preceding six-method protocol: three fresh
  cold owners and nine frozen-own-cold-density warm samples per image/method,
  alternating arm order between rounds. These are complete energy/analytic-force
  endpoints including host export, not isolated kernel timings.
- The existing numerical gates remain energy/atom `1e-10`, maximum force `2e-9`
  and force RMS `1e-9` for every matched-phase pair. Independent default-grid
  gates remain `1e-5`, `1e-4` and `3e-5`, respectively. Existing independent
  GPU4PySCF receipts retain their original identities and are not new engine
  performance measurements.
- The full cold/warm median-ratio cap is 1.05 on both GPUs; both primary A100
  warm ratios must be at most 0.98. This gate does not establish that the larger
  engine gap is solved. Size/resource/moved-geometry and fresh CUDA safety
  controls remain separate requirements before promotion.

The initial three-method screens each contain one cold plus three warm calls
per image/method. They pass all matched-phase and independent observable gates,
equal SCF semantic work and complete per-class force-work comparisons:

| Method | A100 warm baseline → candidate (s) | RTX 5090 warm baseline → candidate (s) |
| --- | ---: | ---: |
| PBE0 | 5.124803 → 4.227095 | 6.873185 → 6.284061 |
| r2SCAN-3c | 9.991481 → 8.421843 | 13.568381 → 12.492154 |
| PBE | 2.955876 → 2.539775 | 5.038417 → 4.763908 |

These are screening results, not the full six-method qualification. Candidate
Libcint Cartesian/spherical RKS/UKS source-channel checks pass six cases on each
GPU. Host tests compile the production pager over widths 32/64/128/256 and CTA
widths 128/256 against an independent multisystem pair-inventory oracle, with
empty/repeated/sparse rows and diagonal/tail pages. The class-permutation test
now covers every occupancy through the actual 256-descriptor capacity.

While this frozen experiment runs, the PR branch advances to `9cd4adc09`,
including the resident restricted J/K coalescence change from #2900. That changes
the actual native total-gradient consumer. The `2c766...` / `4f83...` receipts
remain measurements of their original images; they must not be relabelled as
qualification of the newly integrated PR source. Promotion requires a fresh
source-matched baseline/candidate comparison after integration.

### Page counts are not scientific work counts

For the 96-atom PBE0 screen, traversal route-zero pages change from 2,756,832
to 1,078,224 while its 159,560,480 candidate visits remain equal. Other routes
and every per-class admission, screening and primitive count are equal.
Route zero includes 518,688 unchanged relative order-six pages; only its 139,884
indexed block products change from sixteen 64-candidate pages to four
256-candidate pages. Do not require a fourfold reduction of the mixed aggregate
or treat fewer pages as fewer primitive products. The local auditor verifies
the page decomposition and unchanged non-page traversal counters separately.

### Causal profiling limits

Slurm 15760 profiles the authenticated baseline's first warm native endpoint
through `PreparedBatch.execute` and the CUDA profiler API, with both application
diagnostics unset. On RTX 5090, low-order force kernel time is 1.5055 s of the
6.8156 s PBE0 kernel sum and 2.9219 s of the 13.4546 s r2SCAN-3c sum. Order-six
force contributes another 0.3636 s and 2.5448 s, respectively. Grid partition
work is also substantial. These profiler sums include instrumentation overhead;
they do not replace clean endpoint timings or establish the A100 breakdown.

`GENERATIVEQC_DF_TRACE` on non-captured full-range hybrid value calls requests a
work census and thereby disables the ordinary MD-Coulomb path. Earlier traced J/K totals cannot be used
as the clean endpoint's causal profile. Likewise, Python snapshot hooks do not
necessarily intercept the current C++ stationary-controller force owner.

### Restore a genuinely immutable A100 baseline

The first A100 screens correctly refuse the old baseline because its main ELF
no longer matches its recorded checksum. Do not replace the checksum to make a
guard pass. A fresh cached rebuild restores exactly the originally recorded
`abcc4ab3bb2bae6f555bff8a51836df8670d16d7bd042952624cc2b56a3de56d` image
under a new task-owned package, which is then made read-only.

CPU LSF 225155 reproduces the cause on a disposable copy: the earlier control's
GNU objcopy 2.27 `--dump-section` invocation without an output ELF rewrites the
input file. Its first dump changes the copy to exactly the old package's
`d593274724a0f29fb73a89f12f9d06efab59f5b354cd9cb56b440fe5982f908b` bytes.
Its fatbin and host `.text` hashes match the preceding recorded comparison, and
the fresh production baseline remains untouched. Use a disposable copy or an
explicit output ELF for such inspections; never assume a section dump is a
read-only operation. New controls omit that historical comparison entirely.

## Rejected alternatives and retained limits

- A GPU-name/SM80 whitelist hides the scheduling problem and is not retained.
- Repeating angular-specific admission would change candidate work and is not
  needed to fill the bounded queue.
- A one-packet 128-candidate intermediate has native source identity
  `b407d9364dd7d5926810cb3dadef60465edbe5374bcea0f889c1ac61c8846db6`.
  Its RTX 5090 clean screens improve warm endpoints by about 5–8%, but its
  diagnostics stop on an invalid zero-repeat argument. Its A100 baseline guard
  fails before timing. Those partial receipts are not relabelled as two-packet
  qualification.
- The old 192-atom baseline nonconvergence is not evidence of a passing larger
  endpoint. No SCF/accuracy gate or scientific setting is loosened here.
- A 14–18% A100 screening gain is meaningful but is not a claim of closing the
  entire Direct engine gap. Remaining targets require the actual clean native
  endpoint's phase/kernel evidence, not another unsupported tuning assumption.

## Reproduction and raw receipts

Ignored raw inputs, frozen packages, commands, source-specific receipts and
auditors are under `.artifacts/direct-bottleneck-20261103/` in the existing
`/data/jzzeng/direct-autotune-20261102` worktree. A100 GPU execution uses finite
LSF allocations on gpu22; RTX 5090 execution uses finite Slurm `srun` on node1,
preserving scheduler-assigned visibility. No GPU work runs on maintenance node3.

- A100 candidate build 225128: archive SHA256
  `73e60cbd075498c19e19e2e3817e6de0ebc088a4958e3ed85215dc9f57372ef1`.
- Rebuilt baseline 225148: `320fe949eb6f83ea6c6071f776fedc472b464a9735e6a2980e2a4e18d2cc5fa9`.
- A100 screen 225149: `f93921cc472d86fa64cb88ab58e628b79b52116e2099be1492ce64c9fbf7157e`.
- RTX 5090 screen 15767: `4dbccd7029918edfef687568dc0a5c3d10af8301bf1dcb59c12a9fe762973a91`.
- Disposable objcopy reproduction 225155:
  `e5206a75790f4ea7639e886d748168b5f658426c6688fe474b6173f8e067b28c`.

Historical context: [portable class grouping](../implemented/performance/2026-11-03-portable-direct-force-class-grouping.md).
