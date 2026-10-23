# Fast proposal-source floor qualification

This measures the incremental change from `1e-8` to `1e-7` on
`46f0501a92c289675647d594fec14d04ca365639`, PR #2712's integration of the public Schwarz and DF raw-J
work. Both policies are rebuilt from that same source; the earlier pilots use
`cc9b0c041` and are retained separately. This is stock-grid solver qualification
on sampled def2-SVP spherical water geometries, not a universal method or
quadrature-convergence ranking.

The [decision note](../../../.agents/notes/implemented/performance/2026-10-24-fast-proposal-source-work.md)
records the rejected stronger cutoffs and unchanged admission/fallback rules.
Only the temporary Fast full-range hybrid integral source uses the new floor.
The primary provider, fresh FP64 closure and analytic forces retain their
requested screening. DF and the automatic size/model domain are unchanged.

## Complete endpoints (initial cohort)

Five fresh alternating baseline/candidate pairs at each size include constructor,
preparation, cold energy/analytic forces and two frozen warm calls. Warm timings
first take the median within each process, then the median across processes.
All samples are retained. The floor for gain/nonregression is max(2%, twice the
sum of relative median absolute deviations).

| Atoms | Phase | 1e-8 median (s) | 1e-7 median (s) | Time reduction |
| ---: | --- | ---: | ---: | ---: |
| 12 | cold | 2.095577 | 2.087023 | 0.41% |
| 12 | warm | 0.373354 | 0.374638 | -0.34% |
| 48 | cold | 9.028520 | 8.685979 | 3.79% |
| 48 | warm | 2.306222 | 2.311384 | -0.22% |
| 96 | cold | 23.520148 | 22.281159 | 5.27% |
| 96 | warm | 6.530802 | 6.539532 | -0.13% |

Only gains exceeding that floor establish a performance improvement. Full
process samples, MADs and gate outcomes are recomputed by the replay below.
The 96-atom cold mixed/strict Fock counts remain 7/2. The separate source-work
trace records 1,035,084,412 versus 813,497,268 FP32 values,
a 21.41% reduction and
886,348,576 fewer stored bytes. Instrumented
traces do not enter the clean timing population.

The 51 processes contain **183 complete E/F endpoints**: 90 clean, 60 lifecycle,
18 excluded-owner, six tiny-capacity, three explicit rollback and six diagnostic
calls. Lifecycle cases include PBE0/B3LYP/M06-2X at 12 atoms and PBE0 at 48/96,
with original, frozen, moved, frozen-moved and reset states. All calls pass
independent full-density GPU4PySCF references; no reference density enters native
production. Maximum errors are `1.84173e-10 Eh/atom`,
`1.13915e-05 Eh/Bohr` maximum force and
`3.38419e-06 Eh/Bohr` RMS force, below unchanged gates
`2.5e-6`, `2.5e-5`, and `7.5e-6`. Native incremental, DF cold-factor and
Fast anchor suites plus eight public GPU cases pass, including unavailable and
forced residency, two-owner batches, resource ledgers and invalid selectors.

## Source and execution provenance

Baseline core SHA256: `f4efb7a34800f15675c8302d77d67f4da62d4a527840593b614486c86ef1b99e`.
Candidate core SHA256: `f5c2266aff69abb524038cca97fffca108183dc047e5bef35becf0a467e8dc79`.
The final 2,797-input source map has baseline digest
`8ac733907d7d7760278069d9754b1a6d022d8a5c8c3a40d4623eca2fe5ade348` and candidate digest
`43e2f7a44c12e54d901cabafd4b0c32fc162b448021647aaa5a5ba8f5e3e94dd`; its sole production
change is `src/dft/application_precision.hpp`. The build uses GNU 13.3,
CUDA 12.9.86, Release `120-real`, PCH off, and verified sccache 0.16.0 CXX/CUDA
launchers. Before/after cache statistics are shared-server observations, not
per-build hit attribution. Three matching PBE0/B3LYP/M06-2X force AOT manifests
and actual binary hashes are retained.

Allocations (all finite `srun`, one numerical host thread):

```text
final-controls: job=13654 visible=1 host=node1
final-tests: job=13655 visible=5 host=node1
final-timing: job=1297 visible=0 host=node4
```

n4 reuses the n1 binaries with a private GLIBC loader and a coherent CUDA 12.9
userspace preload; its driver and Slurm visibility are preserved. The initial
n4 pilot failed 15 processes before calculation because an older Python
cuSOLVER wheel lacked a required symbol. Their logs are retained as setup
failures. The successful pilots contain 69 E/F calls, including the rejected
`1e-6`/`1e-5` proposals; they are separate from the final timing population.

## Replay

```bash
python benchmarks/results/fast-source-floor-20261024/reproduce.py
python -O benchmarks/results/fast-source-floor-20261024/reproduce.py
```

Both commands independently recompute all observable errors from the retained
energy/force arrays, verify exact populations and binary identities, and enforce
the timing gates. Optimized Python cannot disable these checks. The archive
contains full raw endpoint records, independent references, provenance, failed
setup logs and finite-Slurm runners. Large binaries, full compile logs and raw
progress streams remain in ignored `.artifacts/` and
`n1,n4:/data/jzzeng/fast-source-next-20261024/`.

## Storage prerequisite

The full receipt initially exceeded the unchanged 64 MiB aggregate evidence
budget by 296,064 bytes. This branch reuses the exact storage-only patch from
[PR #2711](https://github.com/jinzhezenggroup/generativeqc/pull/2711), commit
`26c8522cb8cd552243266418ee11316a0e7b229f`, saving 1,152,291 bytes in the
historical XC #161 family. Its 85 restored files match this branch's parent
`46f0501a92c289675647d594fec14d04ca365639` byte for byte. All 2,797 qualified
production/build/driver inputs remain unchanged. This reuses the existing
compaction work; it adds no new scientific or performance claim for XC #161.

## Integrated master requalification

After #2712 merged and #2700 added DF charge partitioning, rebuild both policies
at master `937102f71218a7f50e19fd8bb4b2697d0df3dd08`. The only baseline production-input changes from the first
cohort are `src/scf/cuda/df_md_j.cuh` and `src/scf/cuda/df_md_owner.cu`; the three
force AOT binaries remain byte-identical. Repeat the entire protocol on the
integrated source: five alternating pairs per size, all lifecycle/exclusion/
capacity/rollback/diagnostic controls, three native suites and eight public GPU
tests. This adds a separate 51-process / 183-endpoint cohort; the two cohorts
are never pooled for timing. All 366 qualified endpoints pass their independent
observable gates, in addition to the 69 separate pilot endpoints.

| Atoms | Phase | 1e-8 median (s) | 1e-7 median (s) | Time reduction |
| ---: | --- | ---: | ---: | ---: |
| 12 | cold | 2.232567 | 2.223318 | 0.41% |
| 12 | warm | 0.377756 | 0.378566 | -0.21% |
| 48 | cold | 9.128436 | 8.764538 | 3.99% |
| 48 | warm | 2.315570 | 2.311254 | 0.19% |
| 96 | cold | 23.599389 | 22.378540 | 5.17% |
| 96 | warm | 6.525197 | 6.526264 | -0.02% |

The integrated cohort passes the same gain/nonregression floors. Its maximum
errors are `1.84177e-10 Eh/atom`,
`1.13915e-05 Eh/Bohr` maximum force and
`3.38419e-06 Eh/Bohr` RMS force. The 96-atom diagnostic still
records 1,035,084,412 versus 813,497,268 provisional values and seven mixed/two
strict Focks. Both builds' compiler-cache snapshots and source verification
receipts are retained.

Baseline core: `f6554472ba59f464f5a2f3192df28fb19f1a0f9ce0947cfe0dd34129072ea02d`.
Candidate core: `6bdbeb44ad347ea7126c6576bd39321a718118dc1eda4915849eca46ae60da19`.
Baseline source digest: `4482ccbc1c243a16920b79018a587be13f613c5f1c60109d445fba5c06e77014`.
Candidate source digest: `3028c113b89f191c929792ba5653434278e0e0dadb5f100cbd0a00a62d38e257`.

```text
controls: job=13657 visible=1 host=node1
tests: job=13658 visible=2 host=node1
timing: job=1298 visible=0 host=node4
```

`integration.json.xz` stores every integrated endpoint and its provenance.
It shares the original receipt's authenticated references and unchanged source
map entries, retaining the exact source delta. The replay verifies the shared
receipt digest, reconstructs and checks both source identities, and independently
replays both populations under ordinary or optimized Python. The original
receipt and original timing table above remain unchanged.
