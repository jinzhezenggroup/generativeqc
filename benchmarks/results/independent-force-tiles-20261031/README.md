# Independent CUDA stationary-force tiles and fitted AO maps

`KsOptions.force_tile_points` independently bounds stationary CUDA force tiles.
SCF tiles remain 256 in the public endpoint campaign; the production 16-MiB
geometry scratch cap, grids, screening and numerical gates are unchanged. The
option is explicit. Neither a larger automatic tile nor an all-README performance
win is promoted by this campaign.

The fitted Python force consumer previously disabled its exact-jet AO map for
explicit tiles other than 256, even when budget-auto selected the same tile.
The fix gives both requests the same fitted producer admission. It preserves
the cutoff, resource/work/occupancy guards and dense fallbacks. Native Standard
force controllers have separate map admission and did not have this defect.

## Measurements

Each population contains one complete cold constructor/preparation/SCF/force/export
call and three warm calls with a frozen engine-local cold density. Public PBE,
PBE0 and WB97M-V/DF96 use default/1024/1024/default populations; other methods
and 12-atom controls use sequential default/1024 observations. Fast and Standard
remain separate. No sample is removed because it is slow.

Same-binary Fast r2SCAN/DF96 ABBA experiments isolate the fitted map predicate at
fixed 1024 points. All twelve populations independently qualify. Medians pool
all six warm calls per arm:

| Device | Dense control (s) | Exact-jet map (s) | Warm time reduction |
| --- | ---: | ---: | ---: |
| A100 | 6.75148 | 5.86981 | 13.1% |
| RTX 5090 | 8.61721 | 5.19095 | 39.8% |
| RTX PRO 6000 | 8.01960 | 4.59324 | 42.7% |

Actual selected point×AO² work is 67,377,718,336 versus 636,028,452,864 dense work
on each device. SCF/Fock counts remain identical. The largest paired force
change is below 6.5e-9 Eh/bohr. Cold samples, variability and every observed map
selection remain in the receipts. Final production-source controls are retained
separately under `map-final/`; the ABBA timings are explicitly labelled diagnostic
policy ablations, not timings relabelled as that later build.

The coherent public API Standard/DF96 force1024 populations compare with fresh
same-device GPU4PySCF 1.9.0 controls as follows. Ratios are native/reference warm
medians; values above one are slower:

| Functional | A100 | RTX 5090 | RTX PRO 6000 |
| --- | ---: | ---: | ---: |
| PBE | 1.178 | 0.916 | 0.958 |
| PBE0 | 1.039 | 0.570 | 0.584 |
| B3LYP | 1.036 | 0.576 | 0.591 |
| r2SCAN | 0.981 | 0.716 | 0.743 |
| M06-2X | 0.910 | 0.513 | 0.527 |
| r2SCAN-3c | 1.267 | 0.737 | 0.761 |
| WB97M-V | 0.816 | 0.458 | 0.369 |

These sequential static-geometry observations do not establish every README
size/mode or performance on unmeasured GPUs. The unchanged independent energy
gate rejects all three fresh HF/DF96 controls (about 2.10e-5 Eh/atom). Their
complete timings remain visible without a qualified HF ranking. Earlier full
matrices and failed accuracy gates remain in
[`ecnu-portable-df-20261031`](../ecnu-portable-df-20261031/README.md).

## Negative experiments and limits

PRO 6000's first final-build timings coincide with a heavy host CPU workload;
its first final-build ABBA also overlaps other GPU test work. Those measurements
remain visible and numerically qualified, but do not enter the headline table.
The separate `fixed-map-serial/` ABBA waits for oracle/sanitizer work to finish.
All four populations qualify; under the remaining host load, its six-sample warm
medians are 9.01809 seconds dense and 5.57137 seconds mapped (38.2% lower).
This same-build result supports the repair without pooling timings across loads.
Slurm on that node warns that CPU binding is unsupported; a requested binding
must not be mistaken for enforced resource isolation.

A100 SCF-tile sweeps at 256/512/1024/2048, holding force1024 fixed, give PBE warm
medians of 2.66801/2.60367/2.58422/2.58843 seconds and r2SCAN-3c medians of
5.10951/5.11352/5.16367/5.31911 seconds. All eight populations qualify; increasing
SCF batching does not close the remaining gaps.

The separate public force2048/4096 sweep retains the production geometry cap.
All eight populations qualify, but warm PBE takes 4.687/4.728 seconds, PBE0
5.273/5.330, B3LYP 5.204/5.248 and r2SCAN-3c 7.175/7.271. These larger explicit
bounds encounter the geometry schedule's bounded fallback. This is negative
evidence for blindly promoting larger tiles, including an A100-wide fixed choice.

Instrumented A100 PBE/r2SCAN-3c force1024 calls retain native DF traces separately
from timed three-repeat populations. Their warm force-response times are about
0.618/1.406 seconds, including 0.369/0.823 seconds in three-center derivatives
and 0.156/0.420 seconds in metric derivatives. The trace calls are one-cold/one-warm
diagnostics and do not enter the ranking table.

Fast DF can still use the generated force owner where Standard uses the native
FP64 controller. Sharing the authenticated DF-XC budget contract with native
FP64 force admission remains separate work. This fix does not widen mixed-gradient
qualification or claim Fast universally beats Standard.

## Reproduction and provenance

`observations.json.xz` retains all cold/warm samples, variability, executed
settings, SCF/Fock counts, original receipt identities and independently computed
observable errors/gates. `provenance.json.xz` retains runners, source/build
receipts, validation outcomes, diagnostic trace extracts and setup failures.
The unchanged repository size budget requires compact summaries: full original
JSON receipts and force vectors remain in the shared workspace's ignored
`.artifacts/force-batch-planning-20261031/`, with no external artifact host.
They are not embedded in the Git summary, and summary-only replay is not an
independent numerical requalification.

The default report checks recorded sample statistics and ratios. To authenticate
every original receipt and recompute all full-vector oracle gates locally:

```bash
PYTHONPATH=python:. python benchmarks/results/independent-force-tiles-20261031/report.py \
  --verify --raw-root .artifacts/force-batch-planning-20261031
```

The retained set has 188 complete or failed populations: 183 independently
qualified, three failed HF reference gates, and two pre-alignment AOT failures.
It contains every population from coherent public API campaigns,
integrated controls, fixed-map ABBA, final production map controls and SCF/force
tile sweeps, including failures. Earlier instrumented pilots and successful
pre-alignment A100 trials remain in ignored local artifacts; their receipt
identities are listed as superseded. They supply no published ranking here.
This selection is by source/campaign, never by measured speed.

The independent oracles remain the original
[`readme-refresh-20261026`](../readme-refresh-20261026/README.md) records.
Receipt-bound reuse manifests audit the unchanged method, geometry, auxiliary
basis and reference equations without rewriting producer identities or relaxing
observable gates. Reference packages are GPU4PySCF 1.9.0, PySCF 2.14.0/Libxc 7.0.0
and CuPy 13.6.0.

Source boundaries are explicit:

- The aligned API matrix uses native source identity `433f5ac300ea1f9e68079c388dd062429e7d1938a16ff2bff972e22afb02429e`, reconstructed from `87f2f3d67` and the retained production patch.
- D4-integrated controls use commit `1885f4970e6ee332dfee30ebf572f14d7a77ca9e`, source `9357eb6603444fc76d7c0dc792908b11c316d80c73e8692782b3cd8dd40f5737`.
- The map fix uses source `9fbd98e19bbbb5733c40b47bee84bbc5e75f59e4f60e09525f5b09061e602189`.
- Fixed-map ablations use source433 on Blackwell and source9357 on A100; each pair has an identical native binary and retained diagnostic driver.

Initial A100 Fast r2SCAN failed AOT contract authentication because only the core
library had been rebuilt. Both failures remain retained. Coherent builds rebuild
all stationary companions and manifests through CMake; no contract/hash check
is bypassed. ECNU uses the environment's glibc sysroot loader. Setup failures,
including unavailable oracle packages in the native-only test environment, are
retained as failures/skips rather than claimed as GPU passes.

A100 runs in finite LSF `gpu_chem` allocations; Blackwell runs through finite
Slurm `main` jobs on n5 (5090) and n2 (PRO 6000), preserving assigned device
visibility. Each device passes all 16 final-source mapped-force oracle/rebinding
tests; the two Blackwell devices additionally pass one unfiltered memcheck case
with zero errors each. A100 has no new sanitizer claim for this map repair.
Moved-force tests cover RKS/UKS, derivative orders one/two, independent
oracles, repeated/moved calls, and cache/occupancy fallbacks. Separate public
force-budget tests exercise 48/128 atoms, force4096, planned/prepared capacity
parity, minimum-budget refusal and inherited bounds. Per-device logs state the
actual executed coverage; skipped tests are not GPU qualification.

Rationale is in the [force-tile decision](../../../.agents/notes/implemented/performance/2026-10-31-independent-force-tiles.md)
and [fixed-map repair](../../../.agents/notes/implemented/performance/2026-10-31-fixed-force-tile-ao-maps.md).
