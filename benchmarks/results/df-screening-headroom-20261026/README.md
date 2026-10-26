# DF PBE0-96 shell-screening capacity cliff

The [README refresh](../readme-refresh-20261026/README.md) exposed an affected
RTX 5090 whose native DF PBE0-96 endpoint was about 2.4 seconds slower than
earlier qualification. Reserving optional norm storage during response planning
removes that difference. Original README samples and figures remain unchanged.

## Integration scope

The measurements below belong to the frozen candidate at
`46b916ac6df0082d56fbc48639bc1261aecc7d04`, built on `c10120db` plus the retained
source patch. The later dependency integration brings in the HF explicit-opt-in
containment and restored incumbent defaults from #2800. It does not change these
receipts, source pins or the original README figures. In particular, the retained
HF96 timing/parity comparison describes that historical candidate, not a new
qualification of the integrated default. The DF headroom planner source is
unchanged by this integration; no new GPU measurements are claimed here.

## Complete endpoints

Three fresh processes per arm/card, one cold and three frozen-density warm
energy-plus-force calls per process, using the **#2786 default-grid framework**.
Cold includes constructor, preparation, SCF, forces and host export. All repeats
are retained, including the 28.1189-second GPU4PySCF observation. Warm statistics
pool nine calls. These are descriptive medians/ranges, not confidence intervals.

| Card / arm | Cold median [min, max], s | Warm median [min, max], s |
| --- | ---: | ---: |
| A baseline | 26.7521 [26.7244, 26.8313] | 10.5745 [10.5411, 10.6242] |
| A candidate | 24.3199 [24.2453, 24.3578] | 8.1910 [8.1570, 8.2294] |
| A GPU4PySCF | 26.3067 [26.2791, 26.3851] | 13.4166 [13.3884, 13.4904] |
| B baseline | 24.3483 [24.3235, 24.3715] | 8.1736 [8.1594, 8.1909] |
| B candidate | 24.3884 [24.3301, 24.4959] | 8.1822 [8.1382, 8.1902] |
| B GPU4PySCF | 26.3928 [26.3260, 28.1189] | 13.4345 [13.4024, 13.7890] |

Card A is PCI `0000:21:00.0`, UUID
`GPU-d0b9e1b7-2083-6198-da50-c3e55693691c`; B is PCI `0000:41:00.0`, UUID
`GPU-57db1e6a-5ff9-570d-67b0-6965fb3a77e6`. Slurm job 14482 used node1/main,
two RTX 5090 GPUs, two tasks, `--gpus-per-task=5090:1 --gpu-bind=single:1`,
eight CPUs per task, 64 GB and a 25-minute limit. Each task ran its three arms
sequentially on one assigned GPU, reversing order in the middle round. Device
visibility remained Slurm-owned. CUDA 12.9.1, one-thread BLAS/OpenMP,
`GENERATIVEQC_CUDA_XC_SPATIAL_ORDER=2` and
`GENERATIVEQC_SCF_VALIDATION_GEMM=1` match the previous campaign.

## Cause and work

Lightweight timers isolated native DF integral forces: 6.4950 seconds on the
affected card versus 4.1130 on a fast card. The final traced Coulomb response
took 3.7368 seconds with `screening_shell_capacity_fallback=1`. The available
4,325,728,307 bytes admitted a maximal 412-column panel but could not also hold
1,189,120 norm bytes and the existing 4,718,592-byte borrowed-density reserve.
Reducing the panel to 411 leaves room for screening, still with nine panels.

The candidate final response takes 1.3617 seconds in a separate diagnostic.
It executes 110,089,450 primitive products and skips 281,968,918; it skips
57,959,926 of 87,964,800 shell tasks. Scratch stays within the existing ceiling.
These intrusive work counters and timers are **not clean endpoint timings**.
Different panel boundaries can change auxiliary-shell visits. Norms remain
per-geometry; minimum-capacity and allocation-failure strict fallbacks remain.
See the [decision](../../../.agents/notes/implemented/performance/2026-10-26-df-screening-headroom.md).

Both native versions perform 9 target cold Focks plus 7 preliminary guess Focks,
then 1 target Fock per warm call. GPU4PySCF reports 9 cold iterations / 10
effective-potential builds and 1 / 2 warm. Equal iteration counts alone are not
equivalent work. All native settings and work match between baseline/candidate.

## Numerical and regression gates

All **18 populations / 72 calls** independently qualify against the retained
dense-grid and CPU/GPU references, with unchanged gates: energy 1e-5 Eh/atom,
maximum force 1e-4 and RMS force 3e-5 Eh/Bohr. All-repeat baseline/candidate
energy differences are zero; maximum force difference is 7.59e-13 Eh/Bohr.
The 96-atom moved-geometry diagnostic agrees within 3.36e-13 Eh/Bohr. HF96
parity passes and cold remains 18.968 → 18.973 seconds.

Slurm regression: **167 passed, 1 deselected**, plus the native occupied-response
executable. Coverage includes independent PySCF forces, finite differences,
changed geometry, constrained budgets, minimum-capacity recovery and fallback,
host planner boundary tests, and benchmark metadata. The deselected benchmark
test expects Git metadata absent from the archived checkout. A separate
120-atom auxiliary attempt was stopped after its baseline produced no endpoint;
it provides no candidate performance or correctness claim.

## Provenance and replay

Baseline core SHA-256 begins `9ad12bbe8e66`; candidate begins `1dc2803d680c`.
Full hashes, commands and retained object identities are in `build.json.xz`.
The candidate rebuilds the changed CUDA translation unit using verified sccache
0.16.0 and independently relinks 478 verified parent objects/archives. Replaying
the original bridge object reproduces the frozen parent library byte for byte.
The attempted full build was cancelled; this is incremental-build qualification.
Embedded source identity inherits the parent metadata object. The measured
candidate is base `c10120db0b00b49368bbe474c3630e2d49f8db9a` plus the retained
patch, not an unmodified base. The initial source archive predates final
formatting/tests; final compiled bridge/header hashes match `manifest.json.xz`.

```bash
PYTHONPATH=python:. python benchmarks/results/df-screening-headroom-20261026/recheck.py \
  --output .artifacts/df-screening-headroom-recheck.json --verify
```

`endpoints.json.xz` retains byte-recoverable primary and auxiliary receipts;
`diagnostics.json.xz` retains exact final warm force-response lines, lightweight
timers, device identities, exploratory timings and clock histograms. Initial
exploratory raw outputs and full traces remain ignored with retained hashes.
Nsight changed the affected-card symptom; its measurements are diagnostic only.
The initial profiler import failure, failed/aborted attempts and later passing
gates remain identified in the bundles. `scripts.json.xz` preserves original
commands/scripts, including those failed attempts, rather than silently fixing
their history. No release asset is needed for replay.
