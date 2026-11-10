# Proposal: qualify exact-response defaults in the current VV10 domain

Status: proposed; compatibility selectors remain default-off, no native-default promotion
Date: 2026-11-10

## Problem

PRs #2494 and #2495 retain exact density-keyed integral and geometry responses.
Their October 14 complete endpoint evidence is valid for its recorded source,
but predates the [VV10 density-domain change](../implemented/numerics/2026-10-21-vv10-default-density-domain.md).
The historical protocols use `vv10-molecular-rho-ge-1e-8-v1`; master
`d35ca77a4c7142f8d6d3f0cbc1e8e96718d9af30` uses
`vv10-molecular-rho-ge-1e-10-v2`. Old energy/force vectors are therefore not
independent acceptance oracles for a current-domain promotion.

## Decision

Keep both compatibility defaults off while requalifying. Preserve the historical
archives byte-for-byte and require exact scientific-protocol agreement for any
reused reference. Do not lower the native density domain, edit old protocol
labels, weaken force gates, or use native off/on agreement as an independent
oracle. Fresh independent calculations may supply accuracy-only receipts;
their diagnostic durations are not reference benchmark populations.

Exercise unset, explicit off and explicit on for both RKS and UKS, geometry
changes, explicit disable, bounded failure recovery and actual authenticated
exact-density hits. Isolate integral reuse from geometry reuse, then exercise
the composition. A warm SCF density that changes by one ULP must miss even when
its forces remain numerically close.

## Evidence

Both retained readers pass: each checks 72 complete E/F calls, six original
populations and qualification outcomes. Authenticated extraction of the 48-atom
and partial 96-atom independent receipts reproduces their stored SHA256 values.
The current H/O def2-SVP basis file is byte-identical to their input, with
SHA256 `329c34dff451fcbc2f770e99b050b07f9e418049f260ebf9501ec21f7e79a4bc`.
Comparing the complete 48/96-atom protocols changes exactly
`nonlocal_density_policy` and `nonlocal_density_threshold`. The benchmark
correctly rejects this mismatch, including when missing references are allowed.

### Execution ownership must precede timing

The first scheduled controls (Slurm **17110**) failed all six tests before
qualification: both integral fixtures observed no `ks_integral_response` events,
and all four geometry fixtures found no Python composite execution owner.
Successful E/F calls had used native prepared-force capability instead. These
failures are retained in `control-receipt/`, not overwritten or relabeled as
numerical failures or passes.

`PreparedBatch.execute` queries native force capability per item and invokes
the Python compatibility consumer only when that capability is absent. Native
stationary CUDA defaults to the force-AOT build option, itself ON; the CLI
preset explicitly enables it even with Python-consumer AOT disabled. The native
controller consumes its resident final D/W directly, not the snapshot response
cache or Python composite geometry cache. Explicit snapshot API calls can still
exercise integral retention, but that does not make the selector effective for
the normal native E/F endpoint.

Native VV10 force-seed reuse is already a guarded default under
`GENERATIVEQC_VV10_COMPONENT_REUSE`; shared semilocal/nonlocal AO panels are also
already default under `GENERATIVEQC_STATIONARY_SHARED_GRID`. These are not exact
equivalents of either historical response cache, and must not be promoted twice.
See the [native reuse decision](../implemented/performance/2026-11-01-stationary-vv10-force-reuse.md)
and [shared-grid decision](../implemented/performance/2026-11-06-shared-stationary-force-grid.md).

The isolated controls live in `tests/python/test_integral_response_reuse_cuda.py`
and `tests/python/test_geometry_response_reuse_cuda.py`. The latter covers geometry
reuse with integral reuse explicitly off and on. They now explicitly model a
library without native force capability, preserving real hit, lifetime and
invalidation coverage without misrepresenting fallback timing as production
timing. `tests/python/test_response_reuse_routing_cuda.py` separately exercises
native capability and compatibility, with fresh independent current-domain RKS
and UKS energies/forces, both geometries, cold/frozen warm and reconverged
energy differences. Its five selector arms include unset, off, each selector
alone and their composition. Native ownership is required rather than silently
redirected to compatibility; actual native pair/grid work is retained.
The scientific-domain boundary
is executable in `tests/python/test_readme_dft_benchmark.py`.

The source-matched Release build is staged separately on n1 under
`/data/jzzeng/ks-response-reuse-20261110-d35ca77a4`. Its frozen source archive
SHA256 is `6def5b62702113c10dc84b283c0c32b900fcc16271de2d021ffbd78f11574471`.
Compilation uses verified sccache 0.16.0 without clearing the cache or stopping
the shared server. Build logs, commands and before/after cache statistics are
retained with the ignored task artifacts. Build job **17102** passed. Compiled
source identity is
`4c94e7388a633cab91f157605d30593930091b05b7f807540fad25a937b608e7`
(2,127 authenticated inputs); library SHA256 is
`c724f98e181ac9b4f1f990ae6e3a75189de5965100069de67b30f21dba7a015c`.
Test-only edits do not change those native identity inputs. Build success is
not execution or independent scientific qualification.

The first route-aware retry (**17111**, `route-controls/`) passes all 30 native
RKS water E/F calls and its reconverged energy-difference gate. Compatibility
tests cannot yet run because building only the core target did not link the
Python stationary plug-ins. Six existing native VV10 controls pass, including
minimum capacity, changed schedule and displaced geometry; its injected failure
case cannot execute because this Release build disabled native test hooks. A
separate qualification build links the four WB97M-V plug-ins and enables the
existing test hooks; its distinct binary identity must accompany later results.

The OH/def2-SVP UKS independent diagnostic also fails before changing either
selector: native energy `-75.65846698056328` versus independent
`-75.65873091631605` Eh, a `2.6393575e-4` Eh discrepancy. No cause or fix is
established here. Preserve that negative receipt and do not count same-source
UKS cache parity as its independent qualification. The subsequent molecular
matrix uses the existing physical NH2 doublet geometry as an additional UKS
fixture, not a resolution of the OH discrepancy.

Fresh independent 48-atom original/displaced references complete in jobs
**17108/17109** (45/47 iterations) and pass current-domain protocol, convergence,
shape and provenance validation. Their diagnostic durations are accuracy-only.
The subsequent 96-atom calculations were deliberately cancelled, not timed out
or accepted, once native ownership invalidated the proposed default-promotion
matrix. Only these two task-owned jobs were cancelled; partial records and the
cancellation rationale remain under `reference-cancellation/`. No current-source
48/96-atom native performance populations are claimed.

### Completed current-source molecular controls

Qualification build **17112** passes with the same authenticated source identity
and test-hook-enabled library SHA256
`f39de825d7913f6517bcd4d36ed40a7a97596e455fd56442ec4a0f3d29b77011`.
The second retry (**17113**) passes both native molecular fixtures and all seven
native nonlocal controls. Its compatibility tests still fail because linking
plug-ins alone does not produce their required identity manifests. The final
run builds the four existing `generativeqc_stationary_wb97mv_*_manifest` targets;
do not bypass the loader's manifest, contract or binary identity checks.

Final scheduled test commands in **17114** pass **4 independent-route tests,
6 isolated compatibility controls and 7 native VV10 controls**, all on n1's
5090, with finite Slurm allocation and unchanged device visibility. The four
independent populations each contain 30 complete E/F calls: five selector arms,
original/displaced geometries and three cold/frozen-warm executions. Each route
also reconverges the six displaced energies for its directional derivative.

| Fixture / owner | Maximum energy error (Eh) | Maximum force error (Eh/bohr) | Geometry-cache hits |
| --- | ---: | ---: | ---: |
| Water / native | 1.346e-11 | 3.721e-10 | not this owner |
| Water / compatibility | 1.343e-11 | 3.721e-10 | 8 |
| NH2 / native | 1.603e-11 | 5.172e-9 | not this owner |
| NH2 / compatibility | 1.603e-11 | 5.172e-9 | 0 |

All 120 calls satisfy the unchanged `1e-8` energy and `1e-7` force gates against
fresh independent references. Directional derivative errors are at most
`4.044e-8` Eh/bohr at step `1e-3` and `3.298e-9` at the smaller steps. On these
small grids, both native populations execute 30 pair sweeps regardless of the
legacy selector arms; native capability is required and Python compatibility
dispatch is forbidden in those cases. NH2 warm densities need not match bitwise,
so the absence of its geometry hits is not relabeled as a cache success. The
isolated compatibility tests separately require authenticated frozen-density hits
for both spins, explicit disable, moved geometry and bounded failure recovery.
The seven native tests independently cover their distinct force-seed policy,
including actual failure injection and recovery with the enabled test hooks.

The final outer shell driver exits 1 after the passing tests because its last
aggregation mistakenly reads the first failed run's hardcoded receipt directory.
Preserve this original exit code and `used-driver.sh`; it is not a successful
whole-job exit. Fixing that path to consume the selected output directory passes
a separate post-run aggregation check. An independent reader verifies all three
actual pytest exit codes, XML results, test-source hashes, binary/source identity
and all 120 raw reference comparisons. It writes `scientific-audit.json`; the
wrapper error is not hidden by rewriting a receipt or weakening any scientific
gate. All raw outputs and readers are retained under the ignored local
`.artifacts/ks-response-reuse-20261110/` directory, with matching remote receipts.

Focused host validation passes **275 tests** with 10 GPU-only skips. Ruff,
strict Sphinx, whitespace checking and the 111-entry / 177-control promotion
inventory pass. These are accuracy, cache-lifetime and ownership results, not
clean endpoint timings, resolution of the OH baseline discrepancy, A100
qualification or permission to promote either compatibility selector.

## Revisit when

### PR integration applicability (2026-11-11)

PR preparation advances the integration base from measured commit
`d35ca77a4c7142f8d6d3f0cbc1e8e96718d9af30` to master
`3c2c6a376`. The intervening #3060 changes force stationarity and pure-UKS
final-state proofs. The GPU receipts, binary identities and OH discrepancy above
remain observations of the original measured source, not qualification or a
diagnosis of this later integration base. Host checks can be repeated after
integration, but independent GPU and complete-endpoint acceptance must not be
transferred across these production changes. This PR changes documentation,
inventory and tests only; it neither promotes a default nor claims that #3060
resolves the recorded OH case.

First establish which public owners still consume the compatibility selectors.
Current-domain independent molecular and displaced-energy checks, actual hits,
invalidation/failure/resource behavior and complete 48/96-atom endpoints pass
with source/binary identities. Keep original cold, five fixed-density warm,
moved and five moved-warm work/timing populations separate. Explicitly state
any missing GPU coverage and protect cold/first-warm costs; historical warm
improvements are neither additive nor a claim about a new binary.

## References

- [Integral reuse proposal](2026-10-14-exact-two-electron-response-reuse.md).
- [Geometry reuse proposal](2026-10-14-exact-geometry-response-reuse.md).
- `benchmarks/results/exact-integral-response-20261014/`.
- `benchmarks/results/exact-geometry-response-20261014/`.
