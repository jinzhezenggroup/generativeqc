# Runtime cache-state E+F experiment

Status: **INCOMPLETE; no GPU measurement or new scientific acceptance yet.**
This family implements the remaining timing slice of issue #2680. The earlier
runtime inventory delivery is separate from these measurements.

## Frozen production and preparation

The production source is Git revision
`ececb48cd56a00a56fbb6223dc53bc61dfe6e11d`, exported with committed LF bytes.
The 12,353-file inventory retains Git modes, blob identities and SHA-256 values.
The source archive is 61,394,408 bytes, SHA-256
`c6d6f674dd1955286cfc1f79d667062a9a308ddecd87f0b73e0eca5a417cb606`;
the source inventory SHA-256 is
`f2dd734bda6daa6eef4a517f6dfc27a8aa0183e3c12dc8eec4590515351cf769`.
The driver has its own recorded identity and does not alter production source.

The owned preparation root is
`/data/srxu/projects/gqc-runtime-cache-state-20261023` on USTC node1.
CPU-only Slurm job **13561** began at **2026-10-23 20:14:11 Asia/Shanghai**,
with 8 CPU, 64 GiB and a 4h30 deadline of **2026-10-24 00:44:11**.
That first attempt was cancelled at **20:35:39** after a measured PyPI download
bottleneck; Slurm records `CANCELLED / 0:15`, while its original EXIT trap wrote
`0`. Both are retained; the trap value is not a completed bootstrap.
The public USTC mirror's bounded 1 MiB control returned HTTP 206 at 13.5 MB/s.
One CPU-only continuation, **13562**, began at **20:36:22**, with the same
8 CPU/64 GiB and a four-hour limit ending **2026-10-24 00:36:22**. No requeue
or extension beyond the original deadline is enabled. It reuses the verified
source and owned environment/cache, without clearing or re-extracting them.
The environment has finished installation; native SDK compilation is running.
This is not a completed SDK, GPU allocation or scientific result. The original batch
script SHA-256 is
`ebeb7be01139b2f80422dc1a20cee956008172fd26bb2c137d610fd6a2ee589c`.
The continuation script SHA-256 is
`6ae11a60d0cff3203b7845b96fe2f3200cc60ecf21ea5cb5158e8fb5cb3e8c30`.
The build uses the public CUDA 12.9.86 toolkit and GCC 13.3, with verified
sccache 0.16.0, private configuration/cache/socket, Release/sm120,
FastCompile OFF, native stationary force execution and the complete default
ordinary/SPD stationary AOT inventory. Build concurrency is bounded to
one AOT compile, two native CUDA compiles and four total tasks.
The native CLI and native tests are outside this Python E+F SDK build.
Scientific Python was frozen separately within allocation 13562: Python 3.12.3,
NumPy 2.4.6, PySCF 2.14.0/Libxc 7.0.0, CuPy 13.6.0 and GPU4PySCF 1.8.1.
The 13,924-file environment manifest is 3,332,002 bytes, SHA-256
`6da3a11b6ae0ace154745fd546895d13a8e7768f8a9f1193628d1d4d537680cd`.
Its read-only handoff permits another owner to use a separate CPU Slurm
allocation and output root; it conveys no native SDK or GPU qualification.

The older 4a783a27 source archive is retained unchanged in this root as an
unused handoff artifact. No sm90 binary or old source acceptance is assigned to
the new sm120 build. Other owners' environments, builds and compiler caches
are not used. The pinned public QM7 dataset was copied read-only into the owned
root and independently rehashed before use.

## Measurement contract

Only Standard is measured. The five cases are Direct PBE0 and WB97M-V on
water-12 and water-48, where 12/48 are **atom counts**, plus QM7 C2, zero-based
index 1. CBS uses all-electron cc-pVDZ/cc-pVTZ MP2 extrapolation, the DZ
DF-CCSD(T)-DF-MP2 correction, RI-JK/shared JKFIT auxiliaries, HF exponent 1.63,
MP2 exponent 3 and the original 32 GiB numerical memory budget.

Each case has a full E+F priming process. The formal cohort then runs three
fresh processes for each empty/primed state: **30 measured endpoints**, plus
five priming endpoints. A separate water-12 WB97M-V pilot includes its own
priming, empty and primed processes and must pass independent accuracy and
cache checks before the formal cohort. Pilot results are not formal samples.
The balanced empty/primed order is fixed by the driver. One physical GPU UUID
and one complete frozen SDK must cover every formal process.

Every empty replicate creates a unique task-owned root for artifact/semantic,
tensor/profile, compiler/JIT-slot, CuPy and CUDA-driver caches. A primed process
reuses only its case's private root, with exact cache-content verification
against the previous completed endpoint. No shared cache is cleared, disabled
or stopped. Explicit empty compiler configuration and an environment whitelist
exclude inherited remote storage, credentials and experimental flags.
Installed AOT assets remain prebuilt and identical in both states; operating
system file caches are not flushed.

Two nested timers remain separate: process launch through exit, and calculator
construction through returned host energy/forces. Process wall includes
imports, SDK identity checks, observation, cleanup and receipt export; the
provenance-verification span is separately recorded. Compiler driver and
subprocess spans overlap with these endpoint timers and are not added to them.
Scoped compiler-cache stats use the actual private live server and are retained
before/after each compile/link call. Observation overhead stays in measured
wall time. Driver invocations are not internal NVCC pass counts or CPU time.
Unobserved native NVRTC compilation remains `UNAVAILABLE`. Historical artifact
metadata `compile_seconds` is not included in this run's compiler wall.

Full SCF histories, Fock counts, response diagnostics, VV10 work, returned
forces, source/binary identities, commands and unsuccessful process logs are
retained. Unavailable counters are not replaced with zero. A zero count applies
only to the explicitly observed Python compiler-driver functions.

## Independent numerical gates

The retained README oracle producer and raw reuse contracts are preserved.
Each new water endpoint must pass energy error <=1e-5 Eh/atom, maximum force
error <=1e-4 Eh/Bohr, RMS force error <=3e-5 Eh/Bohr, plus original reference-grid
stability and CPU/GPU spot quarter gates. Scientific inputs, exact geometry and
reference package versions must match before reuse.

CBS reuses the original independently reconverged PySCF C2 receipt only after
exact scientific-protocol and receipt identity checks. Every constituent and
composite retains energy <=5e-8 Eh, translation <=1e-6 Eh/Bohr, and normalized
seed-709 directional checks at 1e-4 and 3e-5 Bohr with error <=5e-6 Eh/Bohr.
These are directional force checks, not independent full-vector qualification.
The actual projection direction is always the unchanged original receipt.
Regeneration is only an identity check with the floating-point norm-reduction
bound `gamma_n + 4u`, `gamma_n = n*u/(1-n*u)`, `u = eps/2`, `n = 24`.
This bound never applies to source hashes, geometry, protocol or observables.

## Preparation and replay tools

- `build-sdk.sh` performs the finite CPU Slurm bootstrap/build, preserving
  source verification, compiler commands, cache statistics and exit status.
- `resume-sdk.sh` retains the measured network repair using the
  [documented public USTC PyPI mirror](https://mirrors.ustc.edu.cn/help/pypi.html).
- `freeze_environment.py` inventories an already completed scientific Python
  environment for a separate read-only handoff. It does not qualify native CUDA.
- `freeze_sdk.py` requires a completed build, validates the full ordinary/SPD
  AOT catalog, checks core source identity, freezes basis metadata and binds
  independent reference receipts.
- `python -m benchmarks.runtime_cache_state run --config CONFIG --output RAW
  --pilot` runs the separate pilot within a real Slurm GPU allocation.
- The formal command omits `--pilot` and requires `--pilot-receipt PILOT_RAW`.
- `python -m benchmarks.runtime_cache_state qualify --campaign RAW --output
  REPLAY.json` replays every numerical/cache gate without executing an endpoint.

Raw execution output belongs outside `benchmarks/results`, under the owned
experiment root or ignored `.artifacts`. Final publication remains dependent
on actual measurements, lossless recovery checks, evidence budgets, required
CI, exact-head formal approval and protected merge. None is claimed here.
