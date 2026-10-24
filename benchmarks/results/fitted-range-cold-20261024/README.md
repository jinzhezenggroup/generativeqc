# Bounded fitted-primary range cold endpoints

The qualified water96 WB97M-V DF candidate is source
`7dcfdc50eb2c0fcb47635e2094c9a00918f44e54`, core SHA-256
`a7fe54048d25483a9fc26094a11355546f99960577a0abf6a410e3ecacf965d2`.
PR code revision `045e42536` has identical production/build roots; the retained
bridge names those roots. Archive changes are not relabeled as measured source.

## Complete endpoints and scope

Five alternating fresh-process native/GPU4PySCF pairs run on the same
Slurm-owned RTX 5090. Each includes setup and the first energy plus analytic
force call, followed by one warm call using that owner's converged state.
Both engines retain the declared stock grids and independent solver policies.
Native explicitly enables `GENERATIVEQC_SCF_VALIDATION_GEMM=1` and
`GENERATIVEQC_KS_RANGE_RESIDENT_VALUES=1`. These results do not establish the
all-method default Direct/DF cold target.

| Phase | Native median / MAD (s) | GPU4 median / MAD (s) | Reduction |
| --- | ---: | ---: | ---: |
| Cold | 330.262210 / 0.061367 | 393.184624 / 0.084509 | 16.0033% |
| Warm | 45.183987 / 0.014207 | 135.656251 / 0.020362 | 66.6923% |

Both exceed `max(0.02, 2*(native_MAD/native_median +
GPU4_MAD/GPU4_median))`, which is 2% for this population. All twenty calls
pass the existing independent energy and full-force gates. No profiled,
moved, pilot or failed observation enters these paired statistics.

The historical population contains 17 records / 36 qualified calls: Stage26/27/28
pilots (10), Stage29 pairs (20), and Stage29 original/displaced compatibility
(6). The displacement is atom 1 z +0.001 Bohr; moved and moved-warm native
calls use their separately retained independent reference. Original raw
records retain full forces, settings, environment, iteration/Fock work and
source/binary identities. The Stage26/27 fixture-only corrections have their
own manifests; they do not rewrite the original built source.

## Mechanism, work and bounded fallbacks

Fitted-primary KS may retain exact long-range values under its explicit cache
control. A proved final occupied factor remains available to energy-only and
force consumers. Host shell metadata selects the existing through-d LR force
call graph without unreachable orders nine through twelve. Automatic live
budget admission checks the actual planner's complete occupied rank, instead
of mistaking resident packed B with rank zero for complete U. Explicit caps,
failed capacity probes, unproved factors, high angular momentum and unsupported
operators retain their bounded fallback paths. No precision, basis, grid,
screening, recurrence or convergence changes are made.

The separately instrumented Stage29 trace covers cold/warm/moved/moved-warm.
Each first force-response scope has occupied rank 160, one reused final fitted
projection, one projection BLAS call and 3,616 products (142,186,905,600 FLOPs).
There are zero fitted-panel-call events across all eight independent root
force-response trees. The reproducer reconstructs parent/child relationships;
it never adds child timings to root/endpoint time.

Final-source real-device qualification retains 24 Direct/high-angular tests,
36 range tests, six fitted-WB Fast tests, one force-rebuild failure-injection
test and native independent composition/warm/cold-factor checks. Sanitizer
receipts retain their actual selected cases and error counts. Host PR checks
passed 819 tests with one skip. GPU work used finite Slurm allocations with
device visibility preserved; builds used verified sccache 0.16.0. Oracle work
never supplies a timed production density.

## Retained larger failure and current capacity fix

The original water108 candidate fails at force admission with `stationary CUDA
byte budget is not representable`; it has zero complete samples. Its raw record
has no `native_build` field because the driver publishes that field only after
successful execution. The retained launch recipe selects the immutable Stage29
package, but is not a completed endpoint binary attestation. This failed record
is preserved separately and excluded from qualification/timing populations.

A fresh two-grid water108 independent oracle passes its stability gates. A dry
capacity query identifies an unused 128 MiB MD-J reserve in the derivative-only
long-range lease. Stage33 removes that reserve while keeping the original
1 GiB cap and SCF allowance. Source
`ebe88719525b70aaf00b9c62267e165778c0adeb`, core SHA-256
`5f31db0ea769f548accc73d860f9a457cdf7459339b76c2c92e45b7c1ce94a11`,
passes 14 new independent calls: water108 native/GPU4 cold/warm, water96
Standard/Fast/GPU4 cold/warm, and native original/warm/moved/moved-warm.
PR code `ba3b52ab7` has identical production/build roots to this measured source.
The archive now totals **23 records / 50 qualified calls**, plus the original
zero-sample failure. The new single observations remain separate from the
historical five-pair statistics above.

| Capacity-fix bridge | Native cold / warm (s) | GPU4 cold / warm (s) |
| --- | ---: | ---: |
| water96 Standard | 312.862 / 38.328 | 379.866 / 130.868 |
| water96 Fast | 312.893 / 38.377 | same allocation/reference arm |
| water108 Standard | 685.370 / 55.518 | 501.614 / 173.594 |

Water108 now completes and passes the independent full-force gates, but its
cold endpoint remains slower than GPU4. This is a capacity/correctness repair,
not a larger-size speedup claim. Water96 displaced native calls take 265.231 /
39.317 s and pass their independent moved reference; they are not timing pairs.
The Stage33 immutable build also repeats the 24 Direct/high-angular, 36 range,
six fitted-WB Fast, one failure-injection and native composition/factor checks.
All pass. Host capacity-policy checks pass 38 tests. Benchmark and accuracy
drivers are byte-identical to Stage29; the explicit larger-oracle reuse contract
and original oracle are retained and rechecked offline.

## Current-parent integration boundary

The PR subsequently integrates parent `9363e9f93`, including upstream staged
ordered DIIS, affine coefficient validation, occupied-XC defaults and automatic
fitted-primary range admission. Their original implementations and qualification
remain authoritative. The remaining production delta is the lazy derivative
capacity repair and the fitted high-angular cache fallback. Historical source
identities and all 50 calls above remain unchanged; they are not reassigned to
this newer combined tree. A fresh integration bridge is required before claiming
its complete endpoint timing or parity.

## Offline reproduction and retention review

```bash
PYTHONPATH=python:. python benchmarks/results/fitted-range-cold-20261024/reproduce.py
```

The stdlib/NumPy reader authenticates each raw receipt, manifest delta, source
patch and support member; re-runs all independent observable gates; checks pair
identity/ordering; recomputes median/MAD and the 2% gate; and verifies the moved
work trees and the explicit failed record. It reuses the retained
`readme-final-20261022` numerical oracles and `xc-bounded-cross-20261024` base
source manifest. No GPU, external archive or native library is required.

`endpoints.json.xz` stores full original endpoints and the moved reference.
`provenance.json.xz` stores lossless source-manifest deltas, exact source patches,
build/cache/allocation/test/sanitizer receipts, runners, the complete work trace
and the separate larger oracle. These are the minimum retained ingredients for
independent reconstruction of this staged claim. Large raw build directories,
binaries and transient artifacts remain ignored. The large provenance member
is hash/size registered in `benchmarks/legacy-evidence-review.json`; no storage
or change-review limit is raised and no release or external backup is created.

Fourteen historical RTX 5090 JSON records are stored as deterministic gzip,
recovering 617,245 bytes. `history-storage.json` pins each
original Git blob and both byte representations; decompression is exact.
Historical observations and their acceptance decisions remain unchanged.
