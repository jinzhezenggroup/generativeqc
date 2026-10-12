# Bounded CUDA DFT mixed warmup: retain FP64 default

This is **exploratory negative default-promotion evidence**, not an official
DFT-MP-v1 product/default PASS. The early-handoff implementation is optional;
the public precision default remains FP64. Positive early controls are admitted
only for restricted owners; UKS retains its incumbent unset/zero-control AUTO
schedule and explicitly rejects positive controls. No scientific kernel, functional
capability or physical precision ABI is broadened by this experiment.

## Source and boundaries

- Frozen master: `160bf3d2358847ec2e9ee101ba80e0283b117abc`.
- Candidate native edits: `native.patch`, SHA-256
  `a65ed744b68d1fd0bbc06d18e4771fd4a064998c4b97060187ae3045e2c7fef3`.
- Clean-master library SHA-256:
  `1b90a7be9a072504b528f354bc46ab9b4d5c0b10b74025c30a8182aab01468e2`.
- Candidate library SHA-256:
  `10f21528693ff7d405e3f798a2c351d67c3bb0ba9db8fc00acbe24e9341e3107`.
- Measured runner SHA-256:
  `24bb2d7e3053e0781f4fccd34f3b06a68bbcb9af7c2eddad9cd575ef386092bf`.
- Final RKS-only source: `qualified-source.patch`, SHA-256
  `f807b4d887cd4aaf875367a0910f31ecfe747a23d88ac12273a52cc4f23b6fe1`;
  rebuilt library SHA-256
  `9031beb46b1e82a2a26370d9feee351829f055d0f7a5f5c34c0a409682fd5049`.
  The original water timings bind the first exploratory native patch, not this
  later admission-guard revision. The larger-atom appendix binds the guarded
  source/binary. Neither dataset claims default-promotion acceptance.
- Slurm n1 RTX 5090, CUDA 12.9.1/SM120, portable CUDA; one GPU per allocation.
  GPU visibility remains scheduler-assigned. Driver 580.95.05. Host compiler
  GNU 11.4.0; verified ccache 4.5.1 for host builds and sccache 0.16.0 for
  runtime cache-miss preparation. Stationary force AOT binaries/manifests are
  built before timing and their actual hashes are recorded per sample.

The frozen DFT-MP-v1 inputs/common moving grid and direct spherical def2-SVP are
unchanged. Energy/density targets are `1e-10`/`1e-8`, screening `1e-12`, primary
budget 100. The approximate cap is 24, with thresholds
`1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001` on the pre-DIIS **maximum
absolute AO commutator entry**, not the diagnostic's maximum per-spin RMS.

Cold timing synchronizes and includes Calculator construction, preparation and
the public energy+forces execute. CUDA context initialization, Python imports,
hashing, offline scientific compilation and owner destruction are outside that
boundary for both routes. Warm timing includes same-owner execute; changed
timing includes its actual frozen changed-geometry rebuild and execute. Every
sample is a fresh process; same-owner calls are separate observations within it.
Independent oracle calculations never enter production or timed work.

## Results

The earliest tested threshold (1) has the lowest AUTO cold time in all completed
water screens. It still loses to FP64. These are medians of **five alternating
FP64/AUTO pairs per functional**, complete energy+force seconds:

| Functional | FP64 cold | AUTO cold | FP64 warm | AUTO warm | FP64 changed | AUTO changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PBE0 | 4.421898 | 5.209005 | 0.394462 | 1.623256 | 1.940221 | 2.390437 |
| PBE | 4.133195 | 4.922751 | 0.367383 | 1.395456 | 1.705299 | 2.138011 |
| r2SCAN | 5.678951 | 6.917469 | 0.570047 | 2.638212 | 2.956207 | 3.530724 |

PBE0 cold uses 3 mixed + 9 strict Fock builds rather than FP64's 10 strict.
PBE/r2SCAN cold use 3 mixed + 10 strict. All three AUTO warm calls use 2 mixed
and 6 strict, versus FP64's one strict build. Actual FP32 operator work in this
build is Coulomb J recurrence, not density contraction, exact exchange or XC
algebra; the complete raw precision/operator timelines are retained.

A separate **single** clean-master convergence-only AUTO PBE0 water E+force
sample takes 33.212925 seconds, with 100 mixed + 5 strict builds. This shows
why stopping the approximate tail matters, but is not an interleaved legacy
speedup qualification. Zero controls on the candidate preserve handoff criteria,
not the old cross-precision energy baseline.

PBE0 water8 FP64 E+force completes in 22.831387 seconds (prewarm) and
22.810829 seconds (screen); AUTO threshold 1 hits its 600-second subprocess
watchdog. A scheduled GDB attach is refused by ptrace policy, so this sample is
exploratory timeout evidence, not a clean unprofiled promotion trial or a phase
diagnosis. The following threshold 0.1 is administratively cancelled. The rest
of that larger sweep is explicitly **not run**, not silently passing.

## Numerical scope

All completed paired water samples pass strict residual RMS `1e-9`, energy
difference `1e-8` Eh and full-force maximum difference `1e-7` Eh/Bohr. An invalid
comparator, missing forces/lifetime boundary or failed repeat invalidates its
policy summary instead of being filtered from a favorable median.

Independent PySCF 2.14.0 uses the same offline basis and explicit moving grid,
full-density rebuilding and tighter SCF targets. Cold/warm native energies agree
within `1e-8` Eh. Independent central directional force differences at `1e-4`
and `5e-5` Bohr agree with native projections within `5.53e-9` Eh/Bohr across all
completed water screens and pairs; the two-step gaps are below `3e-9` Eh/Bohr.
Coarser reference differences are retained rather than hidden. This is not a
full independent Cartesian-force/finer-grid qualification. The larger CPU
oracle has no completed accepted result; cancellation/noncompletion is retained.

Additional OH open-shell PBE/PBE0/r2SCAN probes pass native convergence but their
independent PySCF reference fails its strict convergence target. A diagnostic
50-cycle Newton retry after 150 DIIS cycles, with unchanged tolerances/grid,
also fails. Both qualification failures and the attempted reference source are
retained. That retry is not shipped and the oracle gates are not relaxed; early
UKS handoff is withheld instead of treating native convergence as independent
proof or removing negative evidence.

The native policy CTest passes. Final adjacent Python tests report 50 passed,
28 GPU-dependent skips locally. The initial finite Slurm run reports 16 GPU
numerical tests passed. The final guarded source/binary reports **21 GPU tests
passed**: the same 16 numerical/control cases plus three explicit unqualified-UKS
refusals and two incumbent UKS preparation checks (FP64 control isolation and
zero-control AUTO). These are not 21 independent positive precision oracles.
Sphinx HTML documentation builds with warnings treated as errors.

## Retained records and reproduction

`storage.json` hashes each retained member. `captures.json.xz` is an exact-byte
capture mapping: each entry has original UTF-8 text, length and SHA-256. It keeps
raw scan/pair JSON (including all energies, forces, timing samples, identities
and work), native stdout/stderr, initial missing-AOT failures, administrative
outcomes, reference outputs, measured runners/jobs, compiler/cache records and
test receipts. `summary.json.xz` contains derived gates, all per-repeat semantic
work counts and the complete single legacy observation. `qualified-source.patch`
adds the final RKS-only admission boundary to the original native change.
Compression is lossless;
no unfavorable sample is removed to meet storage limits.

Decode/audit with the standard library:

```python
import hashlib, json, lzma
from pathlib import Path

root = Path("benchmarks/results/dft-mixed-warmup-20261012")
captures = json.loads(lzma.decompress((root / "captures.json.xz").read_bytes()))
for record in captures.values():
    raw = record["text"].encode()
    assert len(raw) == record["bytes"]
    assert hashlib.sha256(raw).hexdigest() == record["sha256"]
summary = json.loads(lzma.decompress((root / "summary.json.xz").read_bytes()))
```

To reconstruct the measured native owner, use the frozen base plus `native.patch`
and CUDA 12.9.1 Release/SM120 portable-CUDA build. Set tests OFF for that minimal
reconstruction patch (the full change additionally supplies its registered native
test). Build the native library and
`generativeqc_stationary_{pbe0,pbe,r2scan}_rks_spd_manifest` targets; install those
force binaries/manifests beside the selected library. The initial clean build
also pre-generates the declared weighted-ERI outputs before library compilation
because of the existing unrelated generated-target ordering issue. Reuse the
verified compiler cache. Exact measured scripts/argv and source-matched controls
are in the captures; an example paired trial inside a finite Slurm allocation is:

```bash
python benchmarks/dft_mixed_thresholds.py scan --method pbe0-rks --case water \
  --forces --lifecycle --thresholds 1 --warmup-max 24 --repeats 5 \
  --source-commit 160bf3d2358847ec2e9ee101ba80e0283b117abc \
  --source-patch benchmarks/results/dft-mixed-warmup-20261012/native.patch \
  --output /absolute/ignored/new-water-pairs.json
```

Use `qualified-source.patch` **instead of**, not on top of, `native.patch` to
reconstruct the final guarded owner. Keep its new binary/source identities for
new trials; the historical table is not silently relabeled as that revision.

Retain source/binary/runner identities for new trials rather than relabeling
this snapshot as a newer master. The frozen base has only 3,060 evidence-budget
bytes free; this lossless bundle's storage integration requires the already
merged retention repair `f2cf62d9b5bee139b3d8815ba78272f9770aa3c2` (or a
descendant), not a raised cap, exemption or deletion of unrelated evidence.
Default promotion still requires the unchanged
[DFT-MP-v1 gates](../../../docs/maintainer/dft_mp_v1_contract.md) across all
mandatory cases, five interleaved pairs, memory/lifecycle bounds and a reviewed
production-revision receipt.

## Larger-atom screening on the guarded owner

The requested larger systems use the final RKS-only library `9031beb4...`
and `qualified-source.patch` `f807b4d8...`, not the original water candidate.
The measured native runner is SHA-256
`2725be1392e26b6a0610c084b356615c1c6c5560eb5980dc6741b25e0bbcb5ec`.
It adds flushed preparation/execution phase records and atom/declared-AO counts.
The frozen geometry, spherical def2-SVP, common moving grid, stationary force
artifacts and physical stopping targets remain unchanged. Each row below is
**one fresh FP64/AUTO pair**, not a five-pair speed qualification. Complete cold
FP64 times retain Calculator construction, prepare and synchronized energy+force
execution; they do not include Python imports, hashing or destruction.

| Case | Atoms / spherical AOs | Method | FP64 SCF iterations / total Focks | FP64 complete E+force (s) | AUTO, requested cap 2 |
| --- | ---: | --- | ---: | ---: | --- |
| water16 | 48 / 384 | PBE0 | 16 / 17 | 55.154451 | 240 s process watchdog |
| water32 | 96 / 768 | PBE0 | unknown | incomplete: 240 s process watchdog | 240 s process watchdog |
| caffeine | 24 / 246 | PBE0 | 21 / 22 | 50.214392 | 240 s process watchdog |
| ace_glygly_nme | 26 / 247 | PBE0 | 21 / 22 | 43.501979 | 240 s process watchdog |
| caffeine | 24 / 246 | PBE | 25 / 26 | 50.706046 | 180 s process watchdog |
| caffeine | 24 / 246 | r2SCAN | 25 / 26 | 95.923291 | 180 s process watchdog |

The residual trigger is 1 for every AUTO row. A second caffeine/PBE0 pair tests
the minimum positive **cap 1**: FP64 completes in 51.263004 s with 21 SCF
iterations/22 total Focks, while AUTO reaches the 300 s process watchdog. Thus
the negative result does not depend on asking FP32 to converge. Requested caps
are not observed work counts: killed endpoints have **unknown actual mixed and
strict counts**, no returned force/energy audit and no qualifying timing. A
watchdog measures the whole subprocess, not a completed endpoint, so its limit
must not be reported as a speedup or substituted for an endpoint median.

The water32 **energy-only** follow-up also reaches the FP64 600 s subprocess
watchdog. Its remaining AUTO attempt is administratively cancelled after that
failed comparator, not recorded as another numeric/timeout result. No accepted
96-atom SCF iteration count or complete force result is claimed.

### Diagnosis and independent checks

Finite Slurm-launched GDB diagnostics separate preparation from execution for
water8: preparation completes in approximately 2 s. The ordinary asynchronous
snapshot lands in a density/XC launch and cannot identify preceding queued GPU
work. A separate diagnostic with `CUDA_LAUNCH_BLOCKING=1` stops after 70 s in
`launch_independent_jk_kernel`, called through
`enqueue_cuda_direct_jk_device_mixed_j`. Blocking/debugger timings are **not**
benchmark samples. This confirms a mixed-J Direct-J/K obstruction for that
diagnostic; it does not quantitatively apportion every larger timeout.

The frozen `src/scf/cuda/direct_jk.cpp` explains this legal fallback: DFT AUTO
admits FP32 ERI recurrence with FP64 density products and accumulation, whereas
the currently generated mixed-Fock capability also narrows the density product.
Mixed J therefore refuses that broader generated contract, resident strict
values and the strict MD-J path. Removing those guards merely to obtain a fast
time would change the qualified arithmetic. Qualify a recurrence-only generated
owner and its complete endpoint first; threshold tuning alone cannot fix the
cost of this fallback.

The separate PySCF 2.14.0 water16/PBE0 and caffeine/PBE0 oracles each reach their
600 s watchdog after entering SCF. Their targets remain `conv_tol=1e-13`,
`conv_tol_grad=1e-10`, maximum 150 cycles, full-density builds and the identical
offline basis/quadrature. Four CPU threads and 8192-point export tiles reduce
reference dispatch overhead without changing the model. Exact point/weight
parity against the original exporter is tested for both a small grid and the
full frozen water grid. No accepted independent larger-system oracle or full
Cartesian-force qualification is fabricated from these incomplete references.

The unchanged guarded native binary passes **24 GPU tests** in a separate finite
Slurm allocation, including three new one-observation handoff energy/oracle
checks. Final local adjacent checks report **54 passed, 33 GPU skips**; the native
policy CTest also passes. These receipts are correctness checks, not larger-system
speed acceptance. The current runner hash is
`903f2395f5c545fc4dec065a6765134dcd48d056a928cc1937d430d6b5ed85fe`;
historical native times retain the earlier measured runner hash above.

### Additional retained records

`large-captures.json.xz` is a separate exact-byte capture mapping with the same
text/bytes/SHA-256 schema as `captures.json.xz`. It retains all larger scan JSON,
native stdout/stderr, debugger traces, measured runner revisions/job scripts,
reference watchdogs, administrative cancellation and validation attempts.
`large-summary.json.xz` derives gates, complete endpoint times and semantic work
without filtering failed policies. The original water capture/summary archives
remain byte-identical. `storage.json` audits the additional members too.

Reproduce a larger screen inside a finite Slurm allocation with the guarded
library, the prebuilt force artifacts and the matching retained runner:

```bash
python benchmarks/dft_mixed_thresholds.py scan --case water16 --method pbe0-rks \
  --forces --thresholds 1 --warmup-max 2 --repeats 1 --sample-timeout 240 \
  --source-commit 160bf3d2358847ec2e9ee101ba80e0283b117abc \
  --source-patch benchmarks/results/dft-mixed-warmup-20261012/qualified-source.patch \
  --output /absolute/ignored/water16-new-screen.json
```

The current runner adds the bounded larger reference exporter; its changed hash
must be recorded for new samples rather than relabeling this historical screen.

## Provider bug repair qualification

The large-system stalls were provider bugs, not evidence that FP32 recurrence
must be slower. Mixed J now uses budgeted canonical sources and whole-shell
recurrence reuse, while products, projection and K remain FP64. All seven shell
prefix planes cover orders 0--12. FP64 refinement can again admit its optional
resident-value lease; mixed epochs cannot read it. Warm/seed/budget restrictions
and AUTO's host-controlled schedule remain intact.

The final complete E+force screens bind library
`46d599c1303d7d5b6c2a6f439a16193ad7dbe80fc2513bb48401ad120c067603`,
source snapshot
`c24aa94dcf161befaebc083dab77303fdeded2f657a66c46a9a340b8c63d776c`
and runner `903f2395f5c545fc4dec065a6765134dcd48d056a928cc1937d430d6b5ed85fe`.
The retained source snapshot is self-contained relative to the frozen master;
earlier provider patches overlay the guarded warmup snapshot. Prebuilt force
artifacts have their own identities in each record. These are single-pair
screens, not repeat-qualified speedup/default evidence:

| System / method | FP64 E+force (s) | AUTO E+force (s) | Mixed / strict Focks |
| --- | ---: | ---: | ---: |
| caffeine, 24 atoms / PBE0 | 50.456 | 79.022 | 3 / 29 |
| water16, 48 atoms / PBE0 | 54.487 | 73.336 | 2 / 17 |
| caffeine / PBE | 50.152 | 70.057 | 2 / 37 |
| caffeine / r2SCAN | 94.796 | 116.227 | 2 / 29 |

All rows converge and pass unchanged strict residual, energy and full-force
gates. Independent PySCF 2.14.0 caffeine references cover all three functionals.
The repair removes the ordered-work cliff and restores strict reuse, but these
screens still do not justify changing the public precision default. FP64 and
AUTO retain their existing distinct cold-guess policies; do not attribute the
iteration differences solely to arithmetic precision.

Slurm qualification passes native mixed source/orbit/CPU-oracle and strict K
gates, memcheck (zero errors), 24 warmup GPU tests and 8 phase-source GPU tests
(PBE0/B3LYP, resident/fallback, cold/warm/external seed, bounded capacity and
chunk requests). Host validation passes 128 focused tests, compiler inventory
(534 modules, zero errors), native warmup CTest and Sphinx warnings-as-errors.
The old generated-value capability assertions in the full canonical-values
fixture reproduce against the earlier canonical library; that suite is not
reported as passing.

`fix-captures.json.xz` publishes exact final raw endpoints, source snapshot,
independent references, sanitizer failure/pass and final validation logs within
the existing campaign budget. `fix-summary.json.xz` retains every intermediate
policy outcome and artifact identity, including the invalid prefix prototype
and force-artifact mismatch. It identifies a content-addressed ignored local
archive containing **all** intermediate raw captures. No attempt is discarded
or relabeled as valid; historical payload bytes are preserved unchanged.
Profiler times are diagnostics only, never complete endpoint benchmarks.

### Historical payload recovery

After integration with master `ef225164173b3b20a5f5666370ad81a314a707da`,
the repository-wide evidence total would exceed its unchanged 64-MiB cap.
Only this campaign's historical water raw archive is retired from checkout;
`history.json` identifies its exact existing commit, path, size and SHA-256.
The original summary, larger captures and all final repair raw records stay
checked out. No unrelated evidence or numerical/safety test is removed.
Recover the original bytes locally, without publishing an archive or Release:

```bash
git show b0cf631e9d72982314365703718a4c1f785d1997:benchmarks/results/dft-mixed-warmup-20261012/captures.json.xz > .cache/dft-mixed-water-captures.json.xz
sha256sum .cache/dft-mixed-water-captures.json.xz
```
