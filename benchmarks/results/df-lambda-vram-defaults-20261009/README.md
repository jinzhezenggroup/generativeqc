# Current-master VRAM-guarded Lambda defaults

## Scope and provenance

This is a separate source/binary/GPU epoch from the
[historical opt-in 2x2 ablation](../df-lambda-batch-replay-20261009/README.md).
Do not pool their timing samples. The new policy is an explicitly user-authorized
conditional batch-thirty-two/original-equation matrix-replay request. Both
publications remain numerical-only: two pairs do not satisfy the shared five-pair
performance-promotion contract, and no measured peak-memory, complete compilation
cost or full solver-history evidence is claimed.

Pinned master: `cdb2131a47aaeb003bbc85fc76cf172848b8044b`, which already includes
merged #2156. Apply only this directory's `measured-source.patch.gz`; do not
reapply the old dependency. The patch reconstructs production/compiler source,
endpoint drivers and validation fixtures. The measured archive tree/checksum
and the post-compilation host-test-only diagnostic-fixture update are explicit
in `summary.json` source provenance. All production/compiler/native-GPU/driver
bytes match the frozen compiled archive. Documentation, notes and historical
evidence are excluded from the source patch.

The Release build uses CUDA 12.9.1, sm120, `portable_cuda` AOT, fast compile off,
and verified ccache 4.5.1 with explicit CXX/CUDA launchers and checkout-root
`CCACHE_BASEDIR`. Cache receipts and full build logs stay ignored. The frozen
library SHA256 is
`b35fde13be7beaa6bba28b3d119a0835d4967cc501e10d1ce948c5c0b463ad97`;
the endpoint is
`0e12fa5ce7ee4c2578b764b5f319b0d3bd9d55dc050e97b681059d82befb296d`.
`summary.json` binds the exact GPU UUID/model/driver/power limit, input, both
drivers, library, executable and independent reference identities.

## Complete endpoint and gates

The cold spherical ethane energy-plus-analytic-force workload uses aug-cc-pVTZ /
aug-cc-pVTZ-RI, 230 AO, o=9, v=221, q=488, unscreened conventional RHF,
correlation-only DF, symmetric metric inverse square root with relative cutoff
`1e-10`, and a 64 GiB complete numerical budget. Production consumes molecular
input only; pinned PySCF 2.14.0 energy and the original two physical directional
finite differences are read after native execution.

Slurm job 2780 on n2/node2 owns one PRO6000 GPU with unchanged assigned
visibility. It runs four fresh clean processes in order `b8`, `b32_p1`,
`b32_p1`, `b8`, then two separately instrumented fresh processes. OMP, OpenBLAS
and MKL each use two threads; Slurm reserves eight CPUs. The old baseline is
explicit batch eight plus scalar fresh replay, not omitted flags. The candidate
explicitly requests the same batch thirty-two/matrix-replay schedule as the
defaults. CCSD batch eight, strict FP64 W, cadence thirty, core retention and
independent matrix audit remain fixed.

All six complete endpoints retain 21 Lambda iterations/22 actions and pass the
unchanged energy (`3e-9`), two directional-force (`3e-7`), component (`3e-7`,
plus existing `3e-9` sanity), translation (`3e-8`), Lambda/Z/stationarity (`1e-9`)
gates. `validation.json` retains quantitative errors, original tolerances and
all six independently evaluated final residuals. Clean endpoint/phase times and
semantic work counts are in `summary.json`; instrumented samples never add to
the clean timing population.

The two-pair clean timing medians are descriptive, with all observed ranges
retained rather than dropping the slower reverse-order baseline:

| Requested schedule | E+F median (range), seconds | Lambda median (range), seconds |
| --- | ---: | ---: |
| b8, scalar replay | 531.898 (518.303–545.493) | 122.673 (120.889–124.457) |
| b32, matrix replay | 473.173 (472.739–473.607) | 74.592 (74.573–74.611) |

The observed median differences are 11.04% E+F and 39.19% Lambda, not a promised
or formally promoted speedup. The first forward pair saves about 8.79% E+F;
the reverse baseline is slower in non-Lambda phases too (reference 128.815 to
136.012 s and orbital response 122.029 to 134.701 s). Other Slurm jobs overlap
on n2, although this job retains its assigned GPU; a contention explanation is
plausible but not established. Preserve that ordering drift and the distinction
between the observed median and the schedule's causal contribution. All four
clean samples retain 19 RHF and 20 CCSD iterations, besides unchanged Lambda
iterations/actions. Formal performance remains not-run.

## Device admission and qualification

Defaults do not require every optimization to fit. Owner admission takes the
smaller of free device memory and the optional independent device ceiling,
alongside the existing complete host/device numerical budget. Optional cuts,
matrix batches, shared replay/audit storage, core retention and the provider
allowance are charged. Q-clamped batches halve until they fit. Refused enlarged
replay storage retries audit-only storage; scalar replay/audit remain explicit.
Allocation-time shortage retains bounded fallbacks even after a successful
snapshot. Non-resource CUDA and numerical errors still fail without publication.

GPU job 2779 passes 102 allocation-owned tests, including native omitted
defaults, full/partial Q batches, source/no-source, device-only batch shrinkage
with bit-exact old-schedule parity, and mandatory-storage refusal without
publication. Compute Sanitizer memcheck passes 17 selected cases with zero
errors. The separate host/default/ablation/lifetime/evidence regressions pass
210 tests. Compiler dependency, CUDA ownership and provider-boundary checks pass;
the advisory structured-materialization scan adds no findings.

An earlier job 2778 exposes missing forwarding of the two new admission
diagnostics. The forwarding and the conventional host mock are fixed; original
numerical assertions and tolerances are not weakened. Failed-attempt logs remain
ignored, and no failed/incomplete timing observation enters this publication.

`lambda_available_device_bytes` and `lambda_device_limit_bytes` are admission
snapshots/limits. `lambda_device_bytes`, `lambda_capacity` and arena bytes are
conservative capacity reservations, not measured live/peak VRAM or hardware
traffic. Capacity evidence does not establish performance on every GPU/shape.

## Storage and reproduction

`samples.json.gz` losslessly shares equal endpoint fields once; after gzip
decompression reconstruct each full endpoint with
`{**shared_endpoint_fields, **observation["endpoint"]}`. Variable forces, gates,
work counts, admission snapshots, process times and order remain intact.
`profile-summary.json.gz` retains parent-consistent phase traces and nested
action work. Full logs, binaries and source archives remain ignored local
artifacts. No release, tag or external archive is published. The 64 MiB aggregate
and hard 1 MiB/file evidence caps are unchanged.

Start a detached worktree at the pinned master and apply the source patch. Build
the CUDA library with the normal prerequisites, verified cache launcher and
explicit Release/sm120/AOT settings above; then compile
`benchmarks/df_ccsdt_force_endpoint.cpp` as shown in the historical bundle's
build recipe. Run with the appropriate CUDA/dependency/library search paths and
`PYTHONPATH="$PWD/python:$PWD"`. The retained publication command is:

```bash
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --time=01:10:00 \
  python benchmarks/df_lambda_batch_ablation.py \
  --endpoint build/endpoint --library build-cuda/libgenerativeqc.so \
  --input benchmarks/results/rccsd-diis-ring-1900/ethane230.input \
  --oracle benchmarks/results/df-lambda-cost-2136-20261009/oracle-energy.json \
  --finite-differences benchmarks/results/df-lambda-gemm-20261004/oracle-energy-fd.json \
  --output .artifacts/lambda-vram-defaults/reproduction-clean \
  --batch-limits 8 32 --default-candidate --repetitions 2
```

For separate explanatory traces, use a new output directory with
`--repetitions 1 --screen --instrumented`. To preserve the exact original
allocation cohort, run both commands sequentially inside one finite `srun` job,
as the qualification does. Never override Slurm device visibility or schedule
on node3. Different binary/GPU/input identities are a new cohort, not additional
matched repetitions. See the
[default decision note](../../../.agents/notes/implemented/performance/2026-10-09-df-lambda-vram-defaults.md)
for durable rationale and revisiting conditions.
