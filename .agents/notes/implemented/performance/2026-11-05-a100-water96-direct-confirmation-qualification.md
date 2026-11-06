# Qualification: repaired water96 full Direct on A100

Status: completed for the identified repair image; no default promotion
Date: 2026-11-05

## Scope and decision

The preceding [stationarity decision](../numerics/2026-11-05-vv10-confirmation-stationarity.md)
retains a complete native water96 endpoint on 5090, but its large independent
reference was capped. The user correctly requested actual A100 qualification,
not extrapolation from 5090 or substitution of the historical fitted-LR/DF case.

Reuse the exact repaired source and compiled SM80+SM120 image for one bounded
A100 population: independent small default/strict science, then full-Direct
water96 cold and own-density warm energy-plus-force endpoints and an independent
stock-default GPU4PySCF comparison. Keep the original native 540-second and
reference 360-second bounds. Do not change tolerances, device dispatch, force
tiles, production ownership or the Direct algorithm to obtain a result.

The measured source is the retained repair at `e58bc8b74`, not the subsequent
merged PR head `f49d98428` or later master. PR #2992 merged on November 5, 2026
at 10:26:27 Asia/Shanghai. Its later integrations are not implicitly qualified
by reusing this older image. This follow-up changes evidence only.

## Actual allocation and identity

- ECNU LSF job **225489**, queue `gpu_chem`, node `gpu22`, eight CPU slots,
  one exclusive-process A100-PCIE-40GB, finite 25-minute allocation.
- Actual `CUDA_VISIBLE_DEVICES=0` is preserved. Driver: `535.154.05`.
  LSF records terminal `DONE`; driver, native and reference exits are all zero.
  The driver completion marker is 13:17:16 Asia/Shanghai; LSF finalizes at
  13:22:12 after a surviving child exits. Retain that distinction rather than
  replacing complete endpoint timers with allocation elapsed time.
- Native input identity:
  `25555f38303c47ae970413639b16d208775763a22dd91c34f2b429379ae645e2`.
- Core ELF SHA256:
  `53a73ff59f0210c53d6f21afd641cb8c004c3994247e69f06474f478a076e9ee`.
  Core and both stationary companion libraries retain the exact 5090 bytes;
  there is no new native compile or identity-only relink to claim.
- Endpoint runner SHA256:
  `8231b80c16936d25e6598ebb1b409258ee45d155321708f3c92bad7fc91e0d2d`.
  Method, geometry, orbital basis, representation, charge and spin match the
  retained 5090 case. This is WB97M-V/def2-SVP, 96 atoms, 768 AOs, full Direct.
- Python 3.12.16, NumPy 2.5.4, CuPy 13.6.0, SciPy 1.18.1,
  PySCF 2.14.0, GPU4PySCF 1.9.0. Actual module paths and 155 reference-module
  byte identities are retained. Reference code is not patched for the trial.

## Complete endpoint result

Cold includes calculator construction, preparation, SCF, analytic forces and
host export. Both engines freeze only their own converged density for warm
execution. The native endpoint rejects PySCF/GPU4PySCF imports.

| Engine and phase | Complete seconds | Public SCF iterations | Additional work record |
| --- | ---: | ---: | --- |
| Native cold | 131.602106877 | 18 | 19 total Fock builds |
| GPU4PySCF cold | 88.213780578 | 10 | 11 effective-potential calls |
| Native warm | 21.680656832 | 1 | 1 total Fock build |
| GPU4PySCF warm | 28.299404789 | 1 | 2 effective-potential calls |

Native Fock builds and reference effective-potential calls are different work
counters, not interchangeable measures of J/K evaluation. Native cold work is
19 J/38 K, 13 provisional plus 5 target SCF Focks and 4 VV10 sweeps. Warm is
1 J/2 K, 1 target Fock and 0 VV10 sweeps. Actual default force tiles resolve to
256; no tile override is applied.

The actual large input records two blocked old-AO-hold/MO-candidate mismatches,
one valid density hold, six physical MO checks and two accepted canonical
proofs. There are no rejected proofs or unfinished native trace regions.
Thus the repaired escape is observed on A100, not inferred from a host model.

Both large reference endpoints finish within the original 360-second bound.
All four native-cold/warm versus reference-cold/warm numerical pairs pass the
existing common gates: energy error per atom at most `1e-5` Ha, maximum force
error at most `1e-4` Ha/Bohr and force RMS error at most `3e-5` Ha/Bohr.
Across those four pairs the observed maxima are:

- energy error per atom: `7.119638212316204e-12` Ha;
- maximum force error: `3.55212193881016e-7` Ha/Bohr;
- force RMS error: `1.5570658988521788e-7` Ha/Bohr.

The stock-default large reference is independent but is not an additional
tight/grid-extrapolated oracle. Reported grid counts are 1,078,336 native and
1,078,528 reference; retain these actual counts rather than claiming identical
reported arrays. Both use the declared level-3 rules, grid-response forces and
the aligned VV10 density threshold. The resolved physical MO target is
`sqrt(1e-9)`; public native density tolerance is not the MO norm. Native starts
from its Huckel specification; stock reference starts from MINAO. Different
iteration counts alone do not establish another stopping-threshold mismatch.

Independent tight same-grid small-system default/explicit-strict E/F, own-warm
and moved-geometry gates also pass: **2 tests in 28.80 seconds**, with actual
default MO admission and strict isolation. Exact-runtime selected host tests:
**114 pass in 7.14 seconds**. Cross-device native agreement is below
`1.9e-14` Ha/atom and `8.4e-13` Ha/Bohr maximum force error; this cross-device
check is not used as the independent oracle.

## Performance interpretation and next targets

This one progress-instrumented A100 pair has native cold **49.2% longer** than
reference, while native warm uses **23.4% less time** (reference/native 1.305).
These describe the retained sample, not a clean statistical speedup population,
a general GPU ranking, a current-master qualification or a 5090 regression gate.
The prior 5090 native diagnostic remains 214.366170/18.774825 seconds; in
particular A100 warm is not faster than that separate native sample.

The native A100 diagnostic identifies:

| Region | Cold seconds | Warm seconds |
| --- | ---: | ---: |
| Iteration J/K | 31.074503 | 1.881928 |
| Iteration XC | 35.112981 | 1.071905 |
| VV10 pair work, inside XC | 22.179505 | 0.000826 |
| Resident LR value preparation | 21.867797 | not repeated |
| Stationary forces | 27.494400 | 18.686835 |
| Direct RSH force, inside stationary forces | 7.358421 | 7.350472 |

These regions overlap/nest and must not be summed as independent endpoint
components. Investigate cold work count/initialization, LR preparation and
stationary-force operators next; warm forces occupy about 86.2% of native warm.
Do not pretend these measurements attribute the complete native/reference gap
without matching reference phase instrumentation, or remove required density
updates to suppress legitimate VV10 work.

## Environment recovery and rejected shortcuts

The old rootlesskit PID was gone, but the owned VPN container still survived.
Verify its owner, full container cgroup identity, user/network namespace IDs and
VPN route before entering that namespace. Preserve the existing pinned SSH host
key. A task-local helper uses the host filesystem, not the container's different
known-hosts path. Credentials are read only in memory; no password is placed in
commands, scripts, logs or GitHub. Host routing and unrelated VPN processes stay
untouched.

Home quota prevents reliable shared-home staging. Initial environment probes
225483/225484 retain exit 1 and quota diagnostics; 225484 only obtains allocated
GPU/environment metadata, not an electronic-structure result. Inspection 225485
retains its incomplete quota-affected output. Stage public task artifacts in
login02 `/tmp`, transfer to node-local `/tmp` over a finite internal-interface
service, and return immutable result archives. Do not delete other user data to
make room, or treat login02 `/tmp` as shared with gpu22.

Initial qualification 225487 exits 139 before host/science/endpoint tests. ABI
diagnosis 225488 proves that the retained Python executable's forced RPATH
selects private glibc 2.28 even underneath the newer loader. Merely supplying
`--library-path` therefore mixes incompatible loader/libc versions. A task-local
copy of Python receives the matched glibc-2.39 loader/RPATH, explicit Python home
and complete original OpenBLAS hashed dependencies. Native/companion scientific
ELF bytes and the original environment are not modified. The matched loader's
search path is scoped to Python, not globally exported over system tar/curl or
compiler subprocesses. This is runtime compatibility, not a CPU electronic-
structure fallback. The first failed launch is not counted as a benchmark
population or omitted from retained evidence.

Job 225491 is a read-only log checkpoint while 225489 runs, not another GPU test
or endpoint population. No cap is extended and no successful population is
repeated. No fake Slurm metadata, visibility override, GPU-name guard, production
CPU oracle, tolerance loosening, Direct-to-DF replacement, release/tag or asset
publication is used.

## Durable receipts and references

- Final immutable result archive SHA256:
  `4c84cf7136e55364a2e1a044947051b4d695a950a13df83b92c36c9e48da9a8f`.
- Final CPU-only audit SHA256:
  `4641d162dd694ea4ac1f3717954eddbe375e74e57a3ad623d47e4b02ec55bdb0`.
- Task-local ignored `.artifacts/issue2987/a100-results-225489/`,
  `a100-final-audit.json`, adapter/transport scripts and earlier failure receipts.
- Node-local `/tmp/generativeqc-ecnu-a100-tzhu-e-20261029/issue2987-hold-fix-20261105/`.
- login02 `/tmp/issue2987-a100-20261105/results-225489.tar.gz`.
- External engineering memory: issue #2976; repaired convergence PR #2992.
  Earlier 5090 large-reference caps remain incomplete, not retroactively timed
  or accepted by this independent A100 result.
