# CBS Fast: GMRES capacity and Lambda tiling

This campaign corrects an excessive host GMRES workspace reservation. It changes
neither equations nor convergence/precision policy. At restart 30, the leading
capacity term falls from 171 to 72 dimension-sized vectors when diagnostics are
disabled. This releases budget for existing DF Lambda tiles and optional caches.
The Python facade also queries the same diagnostic payloads that it executes.
See the [decision note](../../../.agents/notes/implemented/performance/2026-10-22-cbs-lambda-workspace.md).

## Complete endpoint observations

The baseline is master `976f756b391bc65edb79f76df1a8c3fe807518a7`, including the
previous Fast Lambda implementation. Each comparison below uses the same GPU,
scientific inputs and 32 GiB numeric budget for prior and new Fast.

| Cohort | Prior Fast (s) | New Fast (s) | Time reduction |
| --- | ---: | ---: | ---: |
| C7, n4, baseline first | 1225.765 | 1093.350 | 10.80% |
| C7, n1, candidate first | 1224.092 | 1094.678 | 10.57% |
| C6, n5, one pair | 500.025 | 488.979 | 2.21% |
| C4, n5, medians of three interleaved pairs | 68.067 | 68.054 | effectively unchanged |

The C7 pairs use different GPUs from one another; they are not pooled into a
cross-device median or statistical speedup claim. C6 has only one untraced pair.
Clocks are not locked, and other scheduled workloads can share host resources.
The gain depends on resource admission; no universal speedup is claimed.

The [homepage table](timings.md) contains fresh C1–C7 Standard/Fast measurements
from the candidate binary, with the same GPU within each pair. Standard C7 takes
1447.813 s and Fast 1093.350 s (1.324×). Those curves contain one complete call
per policy and molecule. The first C7 candidate call is also the first ablation
candidate: 14 curve receipts plus 12 ablation receipts represent **25 unique
calls**, not 26. Two additional trace-enabled C6 diagnostic calls are excluded
from timing cohorts.

## Causal work and capacity

In the first C7 pair, Lambda takes 645.109 → 513.805 s. Actual Q increases from
4 to 16 and FP32 GEMMs fall from 69,660 to 17,496. Both calls perform 12 FP32 and
8 FP64 shared transpose actions, with 11/6 mixed/strict iterations and
937,959,470 cast elements. Independent audits remain separate from these action
counts. Composite energies are identical; maximum force difference is
1.676e-13 Eh/Bohr.

The released host reservation is approximately 6.9 GiB for C7. The planner uses
it for larger GPU tiles: Lambda owned device capacity rises from 7,870,153,736
to 15,714,564,872 bytes, while its conservative numeric capacity rises from
33,014,414,891 to 33,459,900,883 bytes. These are capacities, **not measured peak
VRAM**. This is a reduction in unused reservation, not actual device memory.

C6 uses Q32 in both variants. Separate complete diagnostic calls show that the
existing invariant-core cache is admitted only with the corrected reservation.
The retained closed Lambda trace subtree has no overlapping parent GMRES totals:

| C6 diagnostic work | Prior Fast | New Fast |
| --- | ---: | ---: |
| Core reuse preparations | 0 | 1 |
| FP64 GEMMs | 10,177 | 9,550 |
| Generated kernels | 102,651 | 100,694 |
| Q batches | 1,270 | 1,270 |
| Logical packing bytes | 5,328,330,790,456 | 5,286,279,907,160 |
| Semantic summands | 124,145,401,772,058 | 121,332,688,165,650 |
| Host-to-device bytes | 2,218,056,928 | 2,218,056,928 |
| Device-to-host bytes | 2,301,262,048 | 2,301,262,052 |
| Synchronizations | 34 | 35 |

Semantic summands and logical packing bytes are not hardware FLOPs or measured
DRAM traffic. Complete E+F and nested phase timers must not be added together.

## Scientific qualification

The alkane protocol is all-electron cc-pVDZ/cc-pVTZ MP2 extrapolation plus the
DZ DF-CCSD(T)−DF-MP2 correction, HF exponent 1.63 and MP2 exponent 3. Reference,
response and gradients consistently use RI-JK/JKFIT. QM7 indices are
0, 1, 8, 48, 183, 293 and 5604; the dataset SHA-256 is
`fd6c26c1086887310518e6a6da778253bc5d3ceb8166007a679a740fff9ee6ea`.
Exact geometries, basis identities and settings remain in every receipt.

Fast retains composite allowances of 1e-5 Eh energy, 1e-4 Eh/Bohr maximum force
and 3e-5 Eh/Bohr RMS force. Constituent allowances divide by the combined
coefficient L1 norm. Every curve point and ablation passes all constituent and
composite controls against candidate Standard. Standard is not an independent
oracle. C2 also passes the preceding campaign's reconverged PySCF energy and
two-step directional-force oracle after exact scientific-protocol equality.
Independent full-vector alkane coverage is not claimed for the other sizes.

Twenty new complete water endpoints are checked against freshly reconverged
PySCF full-vector central differences at 3e-5 Bohr. The STO-3G/def2-SVP pair is
an algebraic test, not a physical CBS basis pair. Both exact-HF/correlation-only
DF and full RI-JK models are checked at original and displaced geometries,
including cold construction, process-warm replay and 256 MiB constrained calls.
All constituents and composite values pass. Maximum composite errors are
6.293e-9 Eh energy, 3.886e-8 Eh/Bohr maximum force and 2.422e-8 Eh/Bohr RMS.
The complete oracle driver and all displaced energies are embedded in the
water receipt so finite differences can be reconstructed independently.

Native tracked-ownership, ASan/UBSan and production-DSO GMRES tests pass,
including restarts, optional diagnostics, exhaustion, nonfinite returns and
pre-allocation refusal. The expanded Python response/CBS/README suite passes
345 tests. Nine independent full-DF GPU endpoint tests and 754 Lambda tests
pass; the latter run explicitly enables all 56 optional GPU cases, with no
skips. An earlier 698-pass/56-skip run remains recorded. Compute Sanitizer runs
both complete Fast endpoint tests and reports `ERROR SUMMARY: 0 errors`.

Initial candidate screens were cancelled when exact-budget CPU tests exposed
the Python diagnostic-flag mismatch. Their source patch, completed/partial
receipts, eight CPU failures and cancelled Slurm IDs remain separately under
`qualification.interrupted_screen`. Every final candidate endpoint was rerun
after the facade fix; screen timings do not enter the curves or ablations.

## Provenance and reproduction

`summary.json.xz` retains raw receipts, exact source/library/driver identities,
independent checks, nonoverlapping diagnostic trace subtrees, compiler/cache
observations and validation logs. `qualified-source.patch.xz` reconstructs the
measured production changes from the pinned master above. Builds use GCC
11.4.0, CUDA 12.9.1, Release/sm120 and verified ccache 4.5.1. Shared cache
statistics are observations, not isolated campaign hit counts. Python/preloaded
DSO startup, basis/dataset snapshots, serialization and oracle work are outside
the timer; calculator construction, both native basis groups and returned host
forces are inside it. Profiling is off for timing cohorts, with one CPU thread
per numeric pool.

All real GPU work uses finite Slurm allocations on n1, n4 or n5 and preserves
assigned device visibility. No GPU work runs on maintenance node n3. Final
curve C1–C6 pairs run on n1 and C7 on n4; C4/C6 ablations run on n5 and the
reverse C7 pair on n1. Device UUIDs, drivers, power limits and job IDs are in
the receipts. Each native benchmark process has a 3600-second limit.

Use separate baseline/candidate source trees and runtime libraries. Apply the
decompressed source patch only to the candidate. Build both with a verified
compiler cache, retaining cache statistics before/after. The measured key CMake
options are Release, `GENERATIVEQC_ENABLE_CUDA=ON`,
`GENERATIVEQC_CUDA_ARCHITECTURES=120`, `GENERATIVEQC_CUDA_FAST_COMPILE=OFF`,
`GENERATIVEQC_BUILD_TESTS=OFF`, `GENERATIVEQC_BUILD_CLI=OFF`, and explicit CXX/CUDA
compiler launchers pointing to ccache. Adapt the preserved absolute paths in
the environment/preload/job scripts to the local deployment. Install each
`*-environment.sh` as its variant's `environment.sh`, place `python-preload.sh`
in both variant roots, and place the job scripts alongside those roots.
Extract the exact water driver from
`qualification.water.driver_source` as `water-qualification.py`, and the
variant manifests from `qualification.source` as `source-manifest.json`.
The independent oracle additionally requires PySCF and the frozen basis inputs.

For example, run the three C7 endpoints on an allowed node:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=01:10:00 bash c7-job.sh
```

Replay all retained scientific gates and regenerate the figure without CUDA:

```bash
PYTHONPATH=.:python python benchmarks/results/cbs-fast-memory-20261021/verify.py
PYTHONPATH=.:python python -m tools.render_readme_cbs \
  --summary benchmarks/results/cbs-fast-memory-20261021/summary.json.xz \
  --output .artifacts/cbs-fast-memory-replay --readme-only
```

The publication manifest binds all retained files and accepts the scoped
numerical qualification. It does not certify a universal generated-kernel
performance result. Large binaries and working logs stay ignored locally;
no Release, tag or external archive was created. The
[previous Fast campaign](../cbs-fast-lambda-20261021/README.md) remains intact.
