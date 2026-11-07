# Large restricted GGA indexed matrix qualification

Measured2026-11-08, Asia/Shanghai. This is complete **cold** endpoint work,
not a kernel-only or warm-only claim. The frozen64-pair A100 audit in
`../readme-a100-20261108/` remains56/64 cold and59/64 warm point wins.
This followup does not relabel that baseline or claim universal superiority.

## Default behavior

Native restricted PBE-family/PBE0 and B3LYP one-jet panels can request the
existing indexed strict-FP64 density and potential GEMM providers. Admission
still requires≥512 AOs,≥2^19 grid points, compute major≥8 and a positive
single/double throughput ratio≤2. There is no GPU product-name branch.
The original resource, precision, lifetime and bounded generated fallbacks
remain; no new UKS, mixed-precision, public-budget or other GGA domain is
qualified. `GENERATIVEQC_CUDA_XC_GGA_GEMM=0` rolls back both GGA providers;
unset/1 cannot force admission. No grid, convergence or accuracy gate changes.

## Complete endpoint results

A population is one fresh process: constructor+prepare+SCF+analytic forces+
host export, followed by three own-density frozen-cold warm calls. All timings
include complete endpoint work; GPU4PySCF never enters the native process.

Final GGA-only default image, A100 LSF228380, seconds:

|Method/mode|Rollback cold|Default cold|GPU4PySCF cold|Cold reduction|Default/reference warm medians|
|---|---:|---:|---:|---:|---:|
|PBE Direct|7.325541|6.537924|8.691721|10.75%|2.308549/2.568002|
|PBE0 Direct|13.323914|12.596131|11.259502|5.46%|3.786889/3.322012|
|B3LYP Direct|13.464607|12.871374|11.961932|4.41%|3.887638/3.324542|
|PBE DF|6.631756|5.854885|7.678227|11.72%|2.030353/2.213122|
|PBE0 DF|9.134051|8.499987|8.485574|6.94%|2.884039/3.114260|
|B3LYP DF|9.225342|8.486766|8.865632|8.01%|2.906697/3.121033|

**B3LYP DF96 becomes a timing point win** in every measured screen.
PBE0 DF96 remains a near-tie: the first screen is8.390786/8.347527s
(native0.52% slower), both later cohort samples win narrowly, and the final
default is0.17% slower. Preserve all of those observations; these samples
do not establish a statistical ranking. Hybrid Direct and HF still need work.

Before default promotion, LSF228356 measures all six cases in both orders,
off→on and on→off, plus a fresh same-allocation reference. All12 cold pairs
improve4.86–11.57%. The final default independently confirms each case on its
own actual image; it is not the old opt-in executable with a new label.
Every cold pair clears2%; every individual warm ratio stays≤1.03.

RTX5090 final-image Slurm16753 covers the same six cases, cold+three warm
off/default. The largest cold fluctuation is+0.55%, the largest positive warm
fluctuation+0.27%; every ratio is below1.03. This is a capability-negative
device and retains its original arithmetic schedule. The first5090 screen
16747 is also retained, not replaced by this confirmation.

## Scientific qualification and work

The five GGA cohorts retain **70 populations/280 complete endpoints**:
A100 first screen6,5090 first screen4, A100 two-order cohort30, A100 final18,
5090 final12. Every GGA population passes the unchanged independent dense/CPU
accuracy contract. All16 cold/warm cross-comparisons of each arm pair pass.
All energies/Cartesian force vectors and every warm timing are retained.

Independent observations are reused explicitly from
`../readme-refresh-20261026/`, not recalculated inside a timed endpoint:
exact original producer/driver identity, geometry, orbital/auxiliary basis,
representation, charge/spin and recorded package environments are checked.
Replay validates original tighter-oracle controls, quarter-gate two-grid
stability and CPU/GPU small-system spots before applying each complete
endpoint's common E/F gates. It does not use historical GPU timing or density.
This qualifies the measured static geometries, not displaced trajectories or
every molecule eligible for the heuristic.

Semantic work is unchanged in paired populations: PBE10 target iterations/
Fock/J applications; PBE0/B3LYP9 target iterations/Fock/J/K applications;
every warm1. Preliminary Hückel work, grid1,078,336 points, exact-exchange,
precision operator inventory and physical publication work are compared too.
Native defaults stay energy1e-9, density1e-5, screening1e-12, FP64, with
moving-grid force response. The speedup is matrix execution, not less SCF,
grid or force work. Shared binding/repreparation/cleanup/publication/indexed
host coverage has934 passes/1 skip after restoring the required Python path;
the initial collection error is retained as an environment failure.

## Provenance and retention

- Experimental image: source`73bf0922e8a334da061cb1525b444dc74efe16d4d92fdf33209ba96dda80084e`,
  ELF`3bc4644e120ec35ac39a2ec8ee56784b8fc6e991c95c9645d0f5f48a8181ad85`.
  First A100228334, two-order228356 and5090 Slurm16747 use this identity.
- Final GGA-only image: source`843a2dae51b7a8539e8732b30e627f8950e8ebb6a8c6849db4cf6a73557aac63`,
  ELF`e25c91dfbcb19f69ef4e9fe5c85c0636f5c8761f62a4f797cb393bfcac303e3f`.
  Normal SM80-real+SM120-real Release CPU16752, LSF setup228379/A100228380,
 5090 Slurm16753. Native delta is only`src/dft/cuda_ks.cpp`; HF prototypes
  in the earlier experimental image are excluded. Explicit uncommitted-patch
  revision suffixes remain unchanged in raw receipts, not fabricated commits.
- Both normal builds verify private foreground sccache0.16 and restore the
  borrowed runtime/source. Actual allocations preserve scheduler visibility;
  no GPU runs on login nodes or outside the appropriate LSF/Slurm scheduler.
- Cohort raw archive SHA256
  `3f6be27fbe9c2597dfe7e573e3b1b18767e389aa4f62a080546f78d3a01d9d2d`;
  final A100 raw archive
  `9f386c3beb7c7c0447abc0784653da2200289aff335aa0a822e3fd1e2d6f5235`.
  Both downloads, final payload and negative-HF download retain numeric
  SCP exit0 receipts. Owned transfer listeners were verified and retired;
  the first cleanup query's expired LSF history failure is also preserved.
- `fp64.xz` plus `receipts.json.xz` losslessly reconstruct333 original texts
  and147,848 FP64 words. They also retain the rejected HF48/96 prepared/delta
  experiment's eight populations/32 endpoints separately; those are **not**
  counted as GGA qualification or accepted optimization. No raw failure or
  early loss is dropped, no evidence budget is raised, no Release is created.

The inherited checkout initially exceeded its85MiB cap by306,538 bytes after
adding this capsule. The historical H100 fixed30 packet is therefore encoded
with the existing nested exact-text/FP64 format, saving460,848 bytes while
reconstructing its original44,542,496-byte container and every member exactly.
No other owner's evidence is retired, no reader acquires implicit Git/network
recovery, and no numerical or acceptance data changes. Its former compressed
identities remain in its cohort manifest; the historical replay still checks
every original blob and the unchanged scientific report.

Replay and executable corruption tests:

```bash
python benchmarks/results/gga-indexed-gemm-20261108/replay.py
python -O benchmarks/results/gga-indexed-gemm-20261108/replay.py
PYTHONPATH=python:. pytest -q tests/python/test_gga_indexed_gemm_evidence.py
```

See the [default decision](../../../.agents/notes/implemented/performance/2026-11-08-gga-indexed-matrix-default.md),
the [rejected HF experiment](../../../.agents/notes/rejected/2026-11-08-hf-fixed-coarse-delta.md),
issue2976 and PR3041. The HF fixed coarse delta does not ship as a new opt-in.
