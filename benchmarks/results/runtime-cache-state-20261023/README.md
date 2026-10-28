# Fresh-process runtime cache-state E+F

All **30 formal processes** pass the independent numerical and cache-state
checks. Five full priming endpoints and a separate three-process WB97M-V pilot
also pass. This is the five-case timing slice of #2680; the earlier module
inventory is separate. Qualification covers these static geometries and SDK.

## Observations

Standard defaults, one Slurm RTX 5090, three fresh interpreters per case/state.
Water-12/48 are **atom counts**. Full ranges, separate process-wall times and
per-process driver spans/counts are in [timings.md](timings.md).

| Case | Empty E+F median, s | Primed E+F median, s | Observed driver calls/process, empty / primed |
| --- | ---: | ---: | --- |
| water-12 Direct PBE0 | 3.080376 | 3.055188 | 0 / 0 |
| water-12 Direct WB97M-V | 50.039351 | 12.283036 | 3 / 0 |
| water-48 Direct PBE0 | 11.324743 | 11.423876 | 0 / 0 |
| water-48 Direct WB97M-V | 125.983378 | 87.977974 | 3 / 0 |
| QM7 C2 full-DF CBS | 10.591641 | 10.633802 | 0 / 0 |

Counters cover original Python compiler/cache adapters. Actual GCC/NVCC CPU
probes establish compile/link, artifact replay and private scoped misses/hits;
formal traces retain original commands and results. **Native NVRTC, embedded
AOT dispatch and internal NVCC passes remain UNAVAILABLE.** Zeros are scoped to
observed adapters. Historical artifact `compile_seconds` is separate.

WB97M-V has about 25 seconds of observed driver wall per empty process. Its
E+F difference includes other first-use work; nested spans are never added to
an endpoint or called compiler CPU time. PBE0/CBS show no general benefit from
priming in this cohort. No default promotion or universal speedup is inferred.

Work is identical across empty/primed repeats. PBE0/12 reports 10 iterations/10
Fock builds; PBE0/48 reports 9/9. WB97M-V/12 reports 30 total iterations/31
Fock builds, including 25 provisional and 5 target nonlocal Focks; WB97M-V/48
reports 28/29, including 23 provisional and 5 target Focks. Both retain four
pair-cache probes, one hit and four sweeps. These scopes are not added together.
Original history/operator scopes and response/nonlocal diagnostics are retained.
CBS retains two basis groups, five unique constituents and complete convergence,
capacity, correlation and Lambda/operator work. Standard is not an oracle.

## Independent accuracy

Water uses spherical def2-SVP, zero charge/spin and unchanged public default
XC/VV10 grids, convergence and complete moving-grid response. Every endpoint
passes original E <=1e-5 Eh/atom, Fmax <=1e-4 and Frms <=3e-5 Eh/Bohr gates,
plus independent grid-stability and CPU/GPU spot quarter gates. Maxima across
pilot/priming/formal water calls are 4.856205e-7 Eh/atom, 2.248031e-5 and
1.085801e-5 Eh/Bohr. Oracle producer
`02337e0cf3d94915a352d5056f6d09d27fe353f2` and audited exact-input reuse
contracts are preserved; every new sample is independently compared.

CBS uses QM7 C2H6 index 1, all-electron cc-pVDZ/cc-pVTZ MP2 extrapolation plus
the DZ DF-CCSD(T)-DF-MP2 correction, RI-JK/shared JKFIT, HF exponent 1.63,
MP2 exponent 3 and a **32 GiB numerical capacity**, not measured VRAM. The
complete scientific protocol exactly matches the reconverged PySCF reference.
Constituent/composite E errors are <=3.964829e-12 Eh against 5e-8, two-step
directional errors <=5.454e-9 against 5e-6 Eh/Bohr, with translation <=1e-6.
**Directional checks are not full-vector or CBS-limit qualification.** Actual
projections use the unchanged original seed-709 direction and 1e-4/3e-5 Bohr
steps. Regeneration only checks identity with component-relative norm rounding
bound `gamma_n+4u`, `gamma_n=n*u/(1-n*u)`, `n=24`, `u=eps/2`; source, geometry,
protocol, receipt hashes and observable thresholds retain exact checks.

## Frozen software and execution

Git source `ececb48cd56a00a56fbb6223dc53bc61dfe6e11d`: all 12,353 committed
LF files, Git blobs and executable modes are verified before/after build.
Native source identity:
`58eb35df7cd4cc871485b06b44a78096930b7ea811ceaee45acb557d28d975c8`.
Core SHA-256:
`688b0e152215809bf493d6388e08da52cbeebc22f1db34c938ca98df8c47e5db`.
Layout: frozen source-tree Python plus CMake-installed native SDK. Release,
sm_120/120-real, FastCompile OFF, default native/ordinary/SPD stationary AOT;
all 24 AOT DSO/manifest pairs verified. CLI/tests are outside this build.
All 597 Ninja translation-unit commands use verified sccache 0.16.0; original
commands and private before/after stats remain recoverable.

Python 3.12.3, NumPy 2.4.6, PySCF 2.14.0/Libxc 7.0.0, CuPy 13.6.0,
GPU4PySCF 1.8.1; public CUDA 12.9.86/GCC 13.3. The 13,924-file non-bytecode
environment is unchanged before SDK freeze and after measurement, manifest SHA
`6da3a11b6ae0ace154745fd546895d13a8e7768f8a9f1193628d1d4d537680cd`.

Each empty process gets a new no-clobber artifact/semantic, compiler/JIT-slot,
profile/tensor, CuPy and CUDA-driver root. Primed processes reuse only their
case's completed byte/mode-verified root. The explicit local compiler config
and environment whitelist exclude credentials, remote caches and experimental
flags. No shared cache is cleared/stopped. AOT is identical; OS caches are not
flushed; no converged density survives across interpreters.

Process wall runs immediately before `Popen` through exit/wait; constructor E+F
ends after host force return/synchronization. Verification, observation, import,
cleanup/export remain in process wall. Nested spans are not added/subtracted.
GPU clocks are unlocked, before/after samples retained. Fixed balanced order
uses physical UUID `GPU-57db1e6a-5ff9-570d-67b0-6965fb3a77e6` throughout.

Owned root `/data/srxu/projects/gqc-runtime-cache-state-20261023`, USTC node1.
Build 13562 and SDK freeze 13569 completed 0:0. GPU **13581** completed 0:0
at **2026-10-23 22:23:08 Asia/Shanghai**, after 24m31s, 4 CPU/64 GiB,
`gres/gpu:5090:1`, actual visibility 2, driver 580.95.05, 32,607 MiB.
Slurm released it naturally; no GPU lease remains.

Initial bootstrap 13561 was cancelled for an observed public-PyPI bottleneck;
Slurm `CANCELLED/0:15` and inconsistent EXIT-trap `0` are preserved. One bounded
continuation used the [documented USTC mirror](https://mirrors.ustc.edu.cn/help/pypi.html)
with the same source/environment/cache and no global configuration change.
The unused 4a archive remains immutable; no old sm90 binary acceptance is reused.

## Raw recovery

The current self-contained capsule restores **339 files / 32,411,772 bytes**:
all original numerical vectors, requests, exits, logs/failures, compiler/link
output, scoped stats and consumed source/reference/mode identities. Every
retained raw file is byte-identical to the original capsule. Full source and
environment file censuses, including two identical environment rechecks, use
the verified [published snapshot](source-snapshot.manifest.json); they are
historical provenance rather than inputs to numerical replay. Their counts,
original byte lengths and hashes remain in the compact manifest and above.
Configuration source-hash references are expanded so replay needs no history.
Original ordering, floats, signed zeros, UTF-8, LF and CRLF are byte-checked.
No arbitrary byte chunks or rounded vectors are used.
Installed/opaque cache binary bodies remain immutable in the owned root with
full identities/census; complete consumed geometry and basis records are here.

Windows verifies all bytes and recomputes every original numerical/cache gate.
CPU Slurm **13596** restores/checks original POSIX modes and repeats both
replays, without executing GPU endpoints. Source/host preparation, sampled
scientific acceptance and protected delivery remain separate scopes.

```bash
PYTHONPATH=python:. python benchmarks/results/runtime-cache-state-20261023/reproduce.py \
  --output .artifacts/runtime-cache-state-replay
```

This refuses existing output, restores raw files and emits fresh quantitative
gates, timing/work tables and the validation envelope. Before/after storage
replays give byte-identical qualification, timing, work and validation outputs.
No GPU/network or historical Git objects are needed for this numerical replay.

The original full capsule (**343 files / 45,634,766 raw bytes**), including the
Windows inventory's original CRLF, is recoverable from published ancestor
`cbeca43a0d29503dca4fd727b5d49275a4c42639`. Every one of its 249 stored members
was restored and SHA-256/length checked before retiring the census parts:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/runtime-cache-state-20261023/source-snapshot.manifest.json \
  --output .artifacts/runtime-cache-state-history
```

The helper never fetches implicitly. Run `reproduce.py --bundle` against the
restored `benchmarks/results/runtime-cache-state-20261023` directory when auditing
the full historical capsule. The retained Linux/Windows recovery receipts and
the original raw-file index describe that full historical population.
