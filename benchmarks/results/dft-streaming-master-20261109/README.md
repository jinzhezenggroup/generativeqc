# New-master native DFT streaming qualification

This cohort qualifies the independent DFT optimization in PR #3051 against
master `2bf3fd98da9c6cfba46ca83345e63e1d934e4780`, including its new derivative/AO
owners. It does not relabel the older image in
`../dft-shared-rys-streaming-20261108/`. The source-matched normal Release
images, build/cache/restore receipts and exact complete-ELF transport are retained.

## Complete endpoints

Each Direct96 method/card uses four fresh processes in control/candidate/
candidate/control order. Each process measures constructor/preparation, SCF,
analytic forces and host export, followed by its own three frozen-density warms.
No CPU/reference engine enters the native production process. Defaults, grids,
precision, screening and convergence are unchanged.

| Device | Method | Control cold mean / s | Candidate cold mean / s | Change |
| --- | --- | ---: | ---: | ---: |
| A100 | PBE0 Direct96 | 12.662143 | 11.734812 | -7.32% |
| A100 | B3LYP Direct96 | 12.779418 | 11.867471 | -7.14% |
| RTX 5090 | PBE0 Direct96 | 16.677469 | 16.828197 | +0.90% |
| RTX 5090 | B3LYP Direct96 | 17.062185 | 17.063496 | +0.008% |

Both A100 improvements exceed the predeclared 2%/noise gate. All 24 individually
paired warm comparisons pass the 3% nonregression gate; A100 ratios are
0.943085–0.962295, and the largest 5090 ratio is 1.005151. Positive 5090 cold
changes remain visible; this is bounded nonregression, not a zero-overhead claim.

Every complete native/reference/profile population passes the existing independent
dense-grid, grid-stability and CPU/GPU spot gates. Direct48 PBE0 checks also pass.
Maximum cross-arm native energy/force differences are 1.82e-12 / 5.25e-13.
Native cold target work stays 9 iterations/Focks with the same additional 7/7
preliminary initial-guess work; every own-density warm has one target Fock. Full
settings, component histories and incremental-build diagnostics remain retained.
Missing quartet/recurrence evaluation counters are unknown, not measured zeros.

## Remaining gaps and limited DF diagnostics

The same-allocation B3LYP Direct96 GPU4PySCF cold is **11.699543 s**: the new
native mean still trails it by **1.44%**, about 0.168 s. Its warm median also
remains below native. This cohort is not an all-README leadership claim.

GPU4PySCF PBE0 Direct96 cold is **165.394061 s**, while its warms are about
3.17 s. The PBE0 DF96 reference cold is **30.152892 s**. Both anomalous cold
observations are retained and are **not used to declare those gaps solved**.
Their cause was not measured; JIT/cache/compute explanations remain hypotheses.

One additional native/reference DF96 pair per functional measures all four
complete calls and passes paired plus independent E/F gates:

| Method | Native cold / s | Reference cold / s | Interpretation |
| --- | ---: | ---: | --- |
| PBE0 DF96 | 8.459426 | 30.152892 | Reference cold anomaly; no victory claim |
| B3LYP DF96 | 8.537541 | 8.701143 | Single paired point, not ABBA qualification |

The streaming selector does not change DF production policy.

## Actual cold owners

Separate PBE0/B3LYP CUPTI collections stop before the first warm. All loss/error
counters are zero. A100 executes each of the six streaming classes once per
actual cold Fock, while 5090 executes none. The compiled profile name is not a
GPU-product profitability rule: admission uses actual scalar FP32:FP64 throughput,
declared compiled capability and actual kernel resource/occupancy checks.

PBE0/B3LYP cold kernel sums are 9.917269 / 10.046366 s. Among the remaining
owners, DPPP streaming is 0.976224 / 0.981677 s, PPPP 0.709649 / 0.739630 s,
spatial-SPD low forces 0.692894 / 0.694513 s, and AO radial evaluation
0.477824 / 0.478102 s across 10540 calls. Becke response also contributes several
substantial owners. These sums are diagnostics, not seconds that may simply be
subtracted from an endpoint or proof of a future optimization.

## Provenance and replay

- A100: finite LSF230450 on gpu22, completed worker and remote exit 0.
- 5090: finite Slurm16863 on node1 through `srun`, exit 0; visibility preserved.
- Normal CPU builds16861/16862 use verified sccache0.16.0. The earlier16857
  internal timeout124 and exact borrowed-runtime restoration are retained.
- Source and full-ELF SHA-256 pairs are in `cohort.json`; the candidate source
  remains identical through metadata-only catalog reconciliation.
- An optional late `bmod` transport has no confirmed remote return. It is not
  evidence that the original finite run limit changed or that a second job ran.
- 183 original text members are byte-exactly authenticated. Full AOT stdout stays
  in the original ignored build archive, indexed by size/checksum; no endpoint,
  independent array, collector row or failed scientific sample is dropped.
- Prior executed raw-K/sanitizer evidence remains historical. The regenerated
  fixture executable body is identical, but no fresh GPU fixture run is claimed.

Hosted CPU fallback probes needed the two new captured throughput-fact fields in
their fake CUDA/property owners. Updating these doubles restores all 1541 existing
allocation/fallback tests without changing production. The sm120 AOT gate measured
102 kernels with exactly unchanged resources, toolchain, flags, headers and shard/
preprocessed sources; only the full manifest digest changed. Its authenticated
CI artifact, original failure report, previous ledger and official comparison
preflight are retained. Deliberately rebinding that one digest does not weaken
the resource gate or claim another GPU run.

```bash
PYTHONPATH=python:. python benchmarks/results/dft-streaming-master-20261109/replay.py
```

Replay needs no GPU, native library or private cluster paths. It authenticates
both cohorts separately, recomputes all new independent/per-warm/cold-noise gates,
checks normal build identities and restoration, and preserves the limited scope.
The existing 85 MiB aggregate evidence limit is unchanged; no Release assets or
external archive is required.
