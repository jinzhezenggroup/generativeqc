# H100 paper environment qualification

Issue: #2726. Owner: environment Chat `01a16c8b-b852-7703-b597-d422a4b6b6d6`
(worktree `9dcc`). This family consumes the source and scientific contracts from
#2725. It does not own production/runtime changes or GPU scheduling.

Current status: **The public-project route returned all three source-bound H100
setup E+F samples. Original IEEE/index replay and the available independent
reference gates pass. C2's missing displaced-energy originals and independent
full-force oracle remain UNAVAILABLE, so complete scientific acceptance remains
INCOMPLETE. Publication/review/merge are verified separately from these data.**
Package installation, imports and builds do not establish
scientific acceptance. No performance result is reported here.

## Actual public H100 setup qualification

The sole public Job `h100-paper-smoke-public-20261024-9dcc-r1` used an actual
H100 80GB HBM3 / sm90 on `qb-prod-gpu075`, driver 595.58.03. All three fresh
processes returned rc0 without timeout, each with one complete finite E+F sample.
The original atomic-project r1 release timeout and cancelled queued r2 remain
negative/zero-call records; neither was rerun as an extra scientific sample.

| Setup arm | Independent checks | Remaining limits |
| --- | --- | --- |
| Native HF water-3 | Complete E/F and CPU/GPU quarter-spot gates pass | Main Fock count unavailable; detailed precision inventory incomplete |
| GPU4PySCF HF water-3 | Complete E/F and quarter-spot gates pass | Actual contraction engine is CuPy; historical engine remains unavailable |
| Full RI-JK CBS C2, 32 GiB | Six constituent/composite energy, retained-direction and translation checks pass | Original displaced energies and independent full-force vector unavailable |

These are setup qualifications, not the #2727 matrix or performance/default
promotion evidence. Raw precision completeness flags remain false where
reported. Invalid/opaque zero counters and timers are not interpreted as zero
work or absence of fallback; repeated component timings are not added.

`public-device-originals-v1.json.xz` preserves 27 exact output/control files,
including all original numeric bytes and indexes. Reconstruct and replay them
without a GPU, compiler or new reference calculation:

```console
python benchmarks/results/h100-paper-environment-20261024/restore_public_smoke.py --destination NEW_DIRECTORY --replay
```

The destination must not exist. The original pinned references are already
retained in `readme-repaired-20261021` and `cbs-fast-memory-20261021`.
`public-device-qualification.json` separates qualified, unavailable and
incomplete axes; `public-independent-replay-v1.json` retains the actual errors.
CPU preparation, transport failures and the exact public caller/plan bytes are
retained in the separate public capsules. Large environment-input inventories
remain at their explicitly hash-bound persistent task locations.

## Historical CPU preparation and admission

The checkout retains `hf-source-identity.json` and `cbs-source-identity.json`,
not the two regenerable full-repository file inventories. HF is the clean
`cbd4d807` Git tree; CBS is `976f756b` plus the existing qualified patch in
`cbs-fast-memory-20261021/qualified-source.patch.xz`. The four CBS changed-file
before/after identities, source/archive/patch hashes, pinned checks, original
inventory hashes and unavailable provenance are retained.

Each compact identity binds every source path, mode, byte count and SHA-256 by
one `source_tree_sha256`: SHA-256 of the UTF-8 JSON path-to-identity object with
sorted keys and compact separators. `qz_smoke.checked_source_tree` still reads
and hashes every actual source file before any device or endpoint call; missing,
extra, renamed, modified and mode-changed files are refused. It also retains
support for the original full inventories.

The frozen plans, transport capsules, publication failures and numeric originals
are unchanged. Their historical inventory hashes identify the original records,
not these compact replacements; do not silently rebind a frozen plan. This
storage change supplies no new build/device result or scientific acceptance.

The coordinator transferred the existing `i2679-cbs-energy-1023` CPU Notebook.
Fresh platform status and two container process/FD samples confirmed its
identity and allocation. These observations establish neither a future resource
reservation nor exclusive access to shared compiler caches.

The new task root is
`/inspire/ssd/project/chemicalreaction/czxs25220150/h100-paper-environment-20261024`.
Its private `env`, `source`, `build`, `output`, `assets`, `lease`, `scripts` and
`jit-slots` directories are separate from all historical campaigns. The root was
created with no-clobber semantics and contains the owning Chat identity.
`/opt/generativeqc-reuse` and the old #2681 root remain read-only inputs.

Python 3.11.16, CUDA 12.9.86 and sccache 0.16.0 were verified by executable
commands. The private environment uses the existing restore wheel directory and
its pinned requirements lock. Additional GPU4PySCF 1.8.1 / CuPy 13.6.0 wheels
are prepared on the CPU Notebook; the GPU workload must install nothing from
the network. Wheel names, sizes and hashes belong in the asset inventory.

Software versions differ from the historical measurements. The selected tiny
native HF, GPU4PySCF HF and CBS C2 archival receipts all report Python 3.13.9 and
NumPy 2.4.6, while the
qz base uses Python 3.11.16 and NumPy 2.2.6. Each engine's actual versions must
remain in its raw receipt; these are software/host confounders in any later
cross-system comparison.
The inspected old records do not identify the contraction backend; this remains
UNAVAILABLE. Current CPU import selected CuPy before failing for lack of a CUDA
driver. At that CPU-only stage the H100 choice was pending; the actual public
return subsequently recorded CuPy without an override or backend installation.

## Execution prerequisites

The coordinator accepted protocol commit
`df5c44a81c5d73210a8d2f7f09ba4930d7f67210` for CBS CPU reconstruction/build.
Its manifest has Git LF SHA-256
`dd96c17e702101d21352deb4f867ab0fbb7bce66700222d49fc67bbdef23185e`.
This prepared-contract acceptance does not authorize GPU execution.
HF/DFT uses measured `cbd4d8074de35105717c8aac7cd1f86babbe90ec`.
CBS uses base `976f756b391bc65edb79f76df1a8c3fe807518a7` plus the qualified
patch with decoded SHA-256
`5a1a974bab09e0fe31e6186e6ec2129abf26961113aafcd3d9c81616eb2d7cad`.
The archive commit, current upstream, old sm120 binary and unrelated H100 SDK
cannot replace these identities. Build targets must be sm90.

Both historical command-line drivers require Slurm metadata; the HF CLI also
rejects zero warm repeats. A qz smoke therefore requires a separately identified,
hash-bound transport adapter calling the original scientific functions. It must
record both original driver and adapter hashes, preserve actual qz allocation
evidence, and never invent `SLURM_JOB_ID`. The coordinator authorized preparing
this adapter; its concrete GPU launch still requires the coordinator's plan.

The recovered CBS source contains 12,145 files. Exactly four files changed by
the authenticated patch; all six archived pinned file hashes match. The derived
Git LF base tar has SHA-256
`1fdae9fb620e839b5bdd5f1cf06686d2c7bf7d8bcfbeb3742f559ea216d60e69`.
It is separately identified from the historical archive hash, not presented as
the original archive bytes.

The actual frozen-source host preflight rejects `sm_90 / auto`: no tuned or
compatible production profile exists. Its original explicit `portable_cuda`
branch resolves sm90 successfully, with `tuned=false` and no tuned selections.
The coordinator accepted this as a separate **same-scientific-source /
portable-schedule H100 cohort**, not an identical-schedule hardware-only replay.
The original failure and the initial probe's misspelled-API ImportError remain
in the evidence. Production code, model, precision policy and numeric budget
are unchanged. The first bounded CPU build targets the public `generativeqc`
DSO with Release, GCC 12.3, CUDA 12.9.86, sm90 and explicit portable profile.
Additional optional DFT artifacts do not inherit qualification from this target.

CBS completed its first actual build after the separately retained preconfigure
harness failure: configure/build returned 0, the owned processes exited, and
static inspection found sm90 images without sm120. All source files still match;
678 generated files, actual launcher/compiler commands, complete logs and cache
statistics are retained. The new 1,013,114,992-byte DSO has SHA-256
`fea001df0f1a7c8bd6cbf5188fa0decb818e349da7e64d38465adb0b5aad57b6`.
This is CPU artifact qualification, not device/scientific acceptance.

After that natural terminal and static validation, the independent HF source was
restored to `source/hf-dft-cbd4`. Its derived Git LF archive SHA-256 is
`21fe3643a42d982a4a18fa18e4319be7d44db45f5f8d50ab1199f8c9c0485026`.
The 12,130-file snapshot and six selected source hashes were verified. Actual
HF host preflight rejected auto/sm90 and resolved the original portable branch;
the coordinator authorized one sequential, bounded build in its own directory.
The historical 5,852-file package/archive remains UNAVAILABLE. The CBS DSO is
not used as the HF implementation.

The HF build subsequently completed with configure/build return codes 0 and all
owned processes gone. Its independent 1,013,114,968-byte DSO has SHA-256
`5645dacffe534cff08bd5cc042eefc9ab1f5335cced0f16220387eefd61a38df`.
Post-build source verification, sm90 image inspection, ELF/configuration checks,
678 generated-file identities and complete compiler/cache provenance passed the
CPU static checks. Its build-preflight canonical SHA-256 is
`839adc6f2fe7a55c0d2b947a21070f8aadc8672311130ab3c86d637a8967fa89`.
No new H100 call follows automatically from this outcome.

`prepared-smoke-plan.json` binds the two actual SDKs, source inventories,
environment, caller and stopping caps. Its Job name and allocation-proof fields
are null until the coordinator obtains a real allocation and creates a separate
final plan. The prepared snapshot is not an executable resource authorization.

## Actual r1 failure and revised proof contract

The real r1 preflight records one H100 80GB HBM3 / SM90, driver 595.58.03,
85019590656 runtime memory bytes, CPU cgroup limit 20 and host memory 200 GiB.
Its exact file SHA-256 is
`02429c0e41e6fdadf6d70f5f9dece3d1f611c0df96868532af1b61a7f82c98a2`.
No release marker was published. The gate exited after 301.740433 seconds with
`endpoints_started=0`, `attempts=[]`; platform status became FAILED. All originals,
complete platform logs and frozen r1 transport inputs remain retained. This is
actual allocation/preflight followed by control-gate failure, not E/F or science.

Supported `job instances` exposes only status/type/node/rank; it exposes neither
an opaque instance ID nor hostname. The v3 nonempty-ID predicate cannot honestly
accept it. V4 explicitly records platform Job/instance IDs as null/UNAVAILABLE,
and binds exact Job name/status, rank-0 locator, node and the unique hash-bound
runtime preflight hostname. Raw response envelopes are preserved. Workspace is
root's hash-bound CLI query scope, not an invented raw API field. The root-only
finalizer rejects failed envelopes, wrong resource/project/group/workspace,
nonunique/nonrunning worker, fabricated ID, changed preflight, expired/terminal
gate and old marker. It rechecks time immediately before no-clobber publication.

Frozen v3 reproduced the missing-ID RED without a native call; v4 accepted the
actual-shaped synthetic CPU fixture with explicitly unavailable IDs. V4 caller,
gate and finalizer passed 15, 6 and 5 host tests respectively on qz in normal and
optimized Python. These fixtures are not actual allocation/science evidence.
Sources, native runtime, DSOs, model, 32 GiB budget and three-call scope remain
unchanged. Root alone may approve and submit a fresh bounded r2 Job after
independently reviewing the frozen revision.

The intended setup qualification is exactly one tiny native HF E+F call, one tiny
GPU4PySCF HF E+F call and one full RI-JK CBS C2 call at the archived 32 GiB numeric
budget. It is not the core timing matrix. The coordinator alone submits the
finite GPU Job after reviewing scripts, source/environment hashes, offline
assets and per-process/job caps. Provider-admission failures remain raw negative
evidence and require the separate production owner; allowances are not raised.

## Evidence handling

CPU readiness and asset observations are retained as JSON in this family.
Connection stdout/stderr is captured before display, with transient URLs and
credential fields redacted before persistence. No `.env`, account credentials,
full environment dump, shared cache clearing or platform lifecycle action is
part of preparation. Missing telemetry remains unavailable, never zero.

GPU4PySCF CPU import returned `cudaErrorInsufficientDriver`; this is retained
as a CPU no-driver failure, not converted into a GPU import PASS. Other tested
CPU imports and the installed dependency check passed. The first binary-only
dependency download failed because geometric had no matching wheel; a pinned
pure-Python wheel was then built offline. That wheel build initially used pip's
default `/root/.cache/pip`; later package operations explicitly use task-owned
`PIP_CACHE_DIR` and `TMPDIR`. No global cache cleanup was performed.

The adapter stores each returned sample's original floating-point bytes and a
field/offset/dtype/shape index before JSON conversion or acceptance checks.
Python floats retain binary64 bytes; NumPy arrays/scalars retain their dtype and
C-order bytes. This preserves NaN payloads, infinities and signed zero even when
the display JSON uses explicit nonfinite tags. Hash-bound binary/index files
are required to claim byte preservation; tags alone are not original bytes.

The current prepared adapter v3 passed 13 CPU transport/integrity tests on the
qz CPU instance in both normal and optimized Python. It also reverified all
12,145 real source files against the authenticated inventory. Earlier prepared
snapshots remain immutable. These tests do not exercise a GPU endpoint.
The independent oracle reader passed four additional CPU tests in both modes,
using retained data and synthetic perturbations. Actual unmodified CLI checks
also verified the zero-warm rejection and both Slurm guards, with exit code 2,
no created measurement output and no scientific calls.
