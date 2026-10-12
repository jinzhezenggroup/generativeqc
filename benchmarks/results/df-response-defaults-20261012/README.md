# Guarded complete-force defaults and a separate Lambda batch experiment

Frozen master `2a6de26eea1f83e9fd1e8df01c189fea44058b6b`, with the retained-shell
derivative dependency from `016475b85c7b5c4fea978fcdfd5e7ad8c0cd85ed` and the
guarded response-default patch. Both cohorts use exactly the same compiled
library and endpoint, strict FP64, ethane230 (230 AO, o=9, v=221, Q=488), CCSD
Q32, one shell derivative pass, Lambda core reuse and original-equation matrix
audits. Independent references are checked after every native endpoint.

## Combined default qualification

Slurm job 1492, n5, one assigned RTX 5090, ten fresh complete E+F processes in
order B,A,A,B,B,A,A,B,B,A. B explicitly disables triples scalar fusion and serial
Lambda scratch; A selects the complete-force defaults with `auto`. Five paired
observations establish this scoped result, not general hardware/workload coverage.

| Mean or sampled metric | Explicit off | Default on |
| --- | ---: | ---: |
| Complete E+F | 227.335725 s | 219.841134 s |
| Triples phase | 39.347941 s | 31.819904 s |
| Lambda phase | 79.215928 s | 79.221516 s |
| Whole-endpoint sampled peak VRAM | 26544 MiB | 25720 MiB |
| Owned Lambda device storage | 10282441664 bytes | 9421946048 bytes |
| Triples scalar kernels | 1320 | 165 |

Complete time falls 3.2967%; sampled peak VRAM falls 824 MiB. Scratch is a
memory optimization, not an observed Lambda speedup. CCSD 19 iterations,
Lambda 21 iterations/22 actions, 416 reported auxiliary batches and 14307
Lambda GEMMs remain unchanged. Scalar fusion retains its larger stage workspace;
complete budgets, free VRAM and a typed preparation-allocation retry preserve the
unfused page/panel plan. Gap-output consumers retain the original schedule.
Standalone Lambda and lower-level triples defaults are unchanged.

Slurm job 7826 on n1 runs 779 passing tests, no skips. This suite includes host
planning cases as well as real-device Lambda, triples and complete-force tests;
779 is not a claim of 779 distinct GPU numerical cases. All water coordinates
have independent FD acceptance. Seventeen memcheck reuse/allocation-refusal
cases pass, with zero errors and zero leaked bytes.

## Lambda Q48, not promoted

Slurm job 7829, n1, a different assigned RTX 5090, four fresh ABBA endpoints.
The new fusion/scratch defaults remain on; only requested Lambda batch changes.
Actual Q48 is admitted, including core reuse and both matrix audits/replays.

| Mean or sampled metric | Q32 | Q48 |
| --- | ---: | ---: |
| Complete E+F | 220.128467 s | 217.917966 s |
| Lambda phase | 79.083091 s | 76.944110 s |
| Reported auxiliary batches | 416 | 286 |
| Lambda GEMMs | 14307 | 10577 |
| Whole-endpoint sampled peak VRAM | 25689 MiB | 29595 MiB |

This saves 1.0042% complete time and 2.7047% Lambda time, but costs 3906 MiB
additional sampled peak VRAM. Two samples per side do not qualify a general
default; Q32 remains the default. Iterations/actions stay unchanged. These are
separate GPU cohorts: do not pool times or add the reductions.

## Evidence boundaries and reproduction

All retained endpoints pass independent energy, retained complete-force,
Lambda/Z residual, stationarity, translation and large-FD gates. Maximum force
error is below 8.164e-10. Large independent FD covers two components at two
steps, not every component. Raw endpoint outputs, controls, process times,
sampled VRAM traces, gate results, allocation/GPU identity, and final qualification
receipts are compressed in `raw-receipts.json.xz`. `summary.json.gz` retains all
sample times, counts and GPU identities. `validation.json.gz` records the
quantitative numerical gates; `publication.json` authenticates the selected
files and source reconstruction. The raw bundle also retains the initial
unpaired job-1490 endpoint and its incorrect scratch-metric gate; it is excluded
from the registered paired cohort, not discarded for a slower timing.

GPU trace projections retain every timestamp and memory-use sample; unused
utilization/power/clock columns remain in ignored original telemetry. The bundled
`trace-storage.json` binds original and projected identities and row counts.

`source.patch.xz` reconstructs every modified executed production/compiler,
benchmark and GPU-test source on the pinned master. Rebuild Release/sm120 with
CUDA 12.9.1 and verified ccache, disable automatic XC and stationary CPU/CUDA
force AOT, and use checkout-root `CCACHE_BASEDIR`. The raw bundle retains the
original finite Slurm scripts, build/cache completion receipts and inputs; preserve
Slurm-assigned device visibility. Production never reads the reference inputs.
The retained `run-benchmark.sh` reproduces the default cohort after rebuilding
and restoring the original input/root layout from the raw bundle; its historical
absolute paths name that layout, not an installed production dependency.
The library SHA256 is
`9bdb525460e365593be6a0d6869005f2fc735a79aebbc25f55b2fe1c78dd9a88`.
The restored final build is byte-identical to the measured/qualified binary.

The small-copy Lambda emitter experiment is reverted: its two copies are only
`o*v` singles slices, not the large Q-major materializations. No GPU performance
claim is made for it. No native DIIS-history preconditioner or new general Lambda
producer-consumer fusion is implemented by these runs.
