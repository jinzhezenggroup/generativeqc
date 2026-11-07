# r2SCAN-3c complete cold: qualified indexed FP64 density products

The scoped automatic default saves about **2.9 s / 16.7%** on the recorded A100
96-atom r2SCAN-3c/Direct endpoint. RTX5090 retains its generated density route and
passes complete cold/warm non-regression. This is a capability/work/resource gate,
not a product-name rule or a new opt-in. The rollback is
`GENERATIVEQC_CUDA_XC_DENSITY_GEMM=0`; candidate processes genuinely leave it unset.

| Device / actual allocation | Order | Rollback cold / s | Default cold / s | Rollback warm median / s | Default warm median / s |
| --- | --- | ---: | ---: | ---: | ---: |
| A100 / LSF228050 | forward | 17.412127 | 14.509533 | 6.982215 | 6.561433 |
| A100 / LSF228050 | reverse | 17.421643 | 14.499977 | 6.967975 | 6.558612 |
| RTX5090 / Slurm16704 | forward | 28.500869 | 28.426446 | 10.517211 | 10.544306 |
| RTX5090 / Slurm16704 | reverse | 28.454144 | 28.434399 | 10.501042 | 10.510058 |

Each arm is a fresh process: first constructor+prepare+SCF+analytic-force+host-export
cold, then three same-owner calls using only that engine's frozen cold density.
All ten cold Focks, J/K applications, grid/tile work, occupied witness, strict
handoff and original convergence gates remain unchanged. Grids are stock;
moving-grid force response is explicitly enabled and disclosed in the retained
protocol. The 5090 warm maximum ratio is1.004643; its sub-percent cold differences
are noise-level protection evidence, not a claimed 5090 optimization.

## Main owner, not a kernel-only claim

Separate intrusive A100 LSF228063 traces reduce cold matrix projection from
**3456.553 to610.226 ms**. The ten explicit matrix FMA counts and mapped AO work
are unchanged. Each Fock retains1054 tiles,300677440 active point-AO visits and
89166908992 point-AO-square work; strict submissions retain356667635968 logical
FMA pairs and the seven proposals retain203855036416 explicit matrix-route pairs
each. Additional internal occupied-refusal work has no independent census and
must not be silently included in these explicit-matrix counts.

The candidate actually reserves9,469,952 matrix-cache bytes and100,663,296 provider
allowance; potential allowance is unchanged. The total optional allowance is
capped at128 MiB. Public resource ledgers, explicit reference budgets, unsupported
hardware/work, mixed schedules and provider refusal keep the bounded incumbent.
The provider uses typed FP64 Dgemm; no TF32/FP32 input conversion, CPU production
oracle or full AO/jet cache is introduced. Direct force remains about3653 ms,
with nested order-six about1786 ms. Event regions include stream gaps, are not
Nsight kernel-busy metrics, and nested force times are not additive.

## Numerical gates and failures

The independent full-AO CPU bilinear fixture exercises24 plans /48 density
evaluations on each device: PBE/r2SCAN, RKS/UKS, full/noncontiguous/empty maps,
changed densities and a one-byte fallback. It checks original E/V/feature gates
without test-only provider offers. RTX5090 memcheck has zero errors; racecheck has
zero hazards/errors/warnings. No oracle imports enter native production endpoints.

All paired exports pass unchanged energy1e-8, force max1e-7 and RMS3e-8 gates;
A100 maximum paired force difference is about1.03e-11. All independent exports
pass the original static common gates: energy/atom1e-5, force max1e-4 and RMS3e-5.
Observed independent errors are at most3.91e-12 energy/atom,1.35e-6 max force and
4.50e-7 RMS force. These are not a grid-convergence or displaced-geometry accuracy
certificate, and the common reference gates are not the tighter paired gates.

**Preserved failure:** LSF228050 completes all four native populations but its
fresh GPU4PySCF reference fails before any physical sample because NVRTC cannot
find `cuda_fp16.h`; its real scheduler exit is1, not0. Environment-only repair
LSF228063 sets the existing CUDA header root, completes the fresh reference and
separate intrusive profiles, and exits0. It does not repeat clean native calls.
The repaired reference cold is14.595587 s on a separate allocation. Native14.50 s
is close to it; do not claim a same-job race, statistical superiority, or a
universal production ranking. The 5090 reference completes in the original job.

Four earlier Nsight deployment failures reach no production endpoint. Initial
missing-native and retained-fixture host failures are not passes; the final100 CPU
XC tests and20 host owner tests pass, plus40 offline corruption/replay cases.
The original trace's SCP byte audit is retained. A legacy non-job-specific helper
overwrote that historical numeric client receipt; it is explicitly unavailable,
not reconstructed as success. Final transfers have job-specific receipts and
independently matching remote archive checksums. The owned HTTP listener is closed.

## Source, replay and limits

Measured master base: `6572be1904d370bd156a754a149acce982ab0ced`.
Executed source: `7fa6df22bf7bbf8912375aeab23b1e2e1149ea711373d336e833d1ca2dbf4931`.
Executed ELF: `3f902abe8e5082c8644e1e42d43a5ce540e3610875d77aa58d2ef89269c08847`.

`receipts.json.xz` and `fp64.xz` use the existing exact-FP64 codec, preserving
original endpoint record bytes, complete vectors, real scheduler/source/ELF,
build/cache/restoration and cleanup receipts, failures and reconstructible source
patch. `sha256.txt` authenticates both streams. Named-owner attribution summaries
retain every submission's complete work counters and per-region sums. Full per-tile
CUDA-event streams and verbose build output remain in ignored artifacts, with
their original sizes/checksums retained; summaries are explicitly derived and
cannot recover individual per-tile timings. No force or energy value is rounded.
The source patch was actually reconstructed outside any Git worktree and reproduces
the executed source identity; this one-off integrity receipt is not a permanent
test against today's production function spelling. The earlier e8 pre-master build
is compile-only, not relabelled as the final measurement. No runtime or licensed
profiler binaries, credentials, or transient namespace identifiers are published.

Run `python benchmarks/results/r2scan-density-gemm-20261107/replay.py` to recompute
all gates and every paired latency. Its checks remain active under Python `-O`.
Builds use normal SM80-real+SM120-real Release and private foreground
sccache0.16.0;11 cache misses, existing cache retained. Borrowed source/runtime is
restored. Two single-cold order checks are not confidence intervals; other
FP64-matrix devices and eligible workloads need additional performance evidence.

The checkout was within about34 KiB of its existing84-MiB evidence cap. Four old
recovery inventories therefore use the already supported authenticated compressed
manifest envelope. Every decoded original byte, Git reference, record and recovery
path is unchanged and verified. No numerical observation is deleted, external
archive published, limit raised, or scientific gate weakened. The retained
`campaign/retention-compaction.json` receipt names these metadata-only changes.

After measurement, PR #3041 also integrates master
`4c2334c143d98a594d7a89c26618e257729bb107` (#3038). Its production changes
are restricted to the separate HF owner, not the RKS density path qualified
here. The inherited evidence cap is now85 MiB; the84-MiB limit above describes
the measured campaign. No GPU samples were rerun or relabelled after integration;
the source/ELF identities and reconstruction base above remain historical facts.

See [the durable decision](../../../.agents/notes/implemented/performance/2026-11-07-r2scan-indexed-density-gemm.md),
[current contract](../../../docs/developer/xc_native_cuda.md) and tracking issue#2976.
