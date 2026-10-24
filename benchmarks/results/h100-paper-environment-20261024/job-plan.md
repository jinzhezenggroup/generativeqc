# Prepared setup Job contract — not submitted

This is a setup plan for coordinator review, not execution authority. No GPU
resource has been created, started or stopped by the environment owner.

The coordinator owns resource/account/project/quota/image selection and a fresh
live availability/budget check. Use one actual allocated H100, highest authorized
priority, the verified reusable image, scheduler-assigned visibility and finite
Job lifetime. Never fabricate Slurm metadata or relax provider admission.

## Frozen inputs before submission

- Protocol lineage: prepared df5c44a81 / dd96c17e… plus the cdd488524 profile
  addendum / a86940ae…; retain both identities. The CBS execution cohort is
  `cbs-frozen-science-portable-cuda-sm90.v1`.
- CBS source: base 976f756b… plus decoded patch 5a1a974b…, six archived source
  hashes verified. Bind the full 12,145-file mode/hash inventory, actual build
  config and generated inventory, sm90 portable resolution, new DSO and cache
  provenance. The SDK build completed with CPU static checks; its DSO SHA-256 is
  `fea001df0f1a7c8bd6cbf5188fa0decb818e349da7e64d38465adb0b5aad57b6`.
- HF source: measured cbd4d807… and original driver 74f61cc0…. The coordinator
  accepted this independent source cohort for CPU reconstruction/preflight and
  exactly one bounded SDK configuration/build, sequentially after the current
  CBS attempt reaches its natural terminal and its artifact is validated.
  Actual HF reconstruction/profile/SDK static checks now passed; its independent
  DSO SHA-256 is
  `5645dacffe534cff08bd5cc042eefc9ab1f5335cced0f16220387eefd61a38df`.
  The historical
  5,852-file package/archive provenance remains UNAVAILABLE. The CBS DSO cannot
  substitute for this source.
- Private environment and offline packages: bind the installed lock and wheel
  inventory. Record Python 3.11.16 / NumPy 2.2.6 confounders, actual GPy 1.8.1 /
  CuPy 13.6.0 and actual contraction backend. No network package installation
  or new image publication is part of the GPU Job.
- QM7: task copy SHA-256
  `fd6c26c1086887310518e6a6da778253bc5d3ceb8166007a679a740fff9ee6ea`.
- Adapter/reader/test code: freeze final bytes after coordinator review. The
  current staged caller is `scripts/adapter-prepared-v3/qz_smoke.py`, SHA-256
  `a540bc5a6da30f3b605a9a56c91668ac8a896b44126972c733e111420f46400b`.
  Earlier prepared snapshots remain immutable. The caller verifies full source
  inventory and package drift, and observes the already loaded GPy contraction
  selector without setting a backend override.

## Actual allocation proof

The coordinator captures real Job status and instances after scheduling, with
Job identity, RUNNING lifecycle, instance identity, node, hostname, project,
workspace, actual GPU/CPU/memory allocation and observation time. Hash-bind the
sanitized raw platform records and the complete normalized proof. A manually
constructed provider/name/GPU-count triple is rejected by the adapter.

The normalized proof carries `provider`, `name`, `status`, `observed_utc`,
`gpu_count`, `platform_status`, `instances` and `resources`. Every `instances`
entry preserves actual `id`, `node` and `hostname` fields from the platform.
The adapter independently records runtime hostname, one visible H100/sm90,
allocated device UUID/PCI/memory/driver/power and cgroup CPU/memory limits. Runtime
hostname must identify exactly one recorded platform instance/node. Do not treat
historical allocation metadata or notebook existence as a current proof.

The final hash-bound plan supplies the actual proof path/hash, Job name, source
and environment bindings, protocol snapshot/hash, new libraries and their
hashes. Missing bindings block execution. A readiness wait, if required to obtain
the real allocation before finalizing the plan, must itself have a finite cap
and cannot initiate calculations before the coordinator supplies the final plan.

## Three fresh processes

After the coordinator accepts the concrete plan, run exactly these setup calls
in separate fresh processes, without a warm loop:

1. `qz_smoke.py --arm native-hf`: original frozen HF `run_native`, water-3,
   Direct, spherical def2-SVP, Standard, complete E+F and host export.
2. `qz_smoke.py --arm gpu4pyscf-hf`: original frozen `run_reference`, identical
   water-3/input, GPy defaults and complete E+F.
3. `qz_smoke.py --arm cbs-c2`: original frozen `geometry/setup/native_call`,
   QM7 index 1, full RI-JK, cc-pVDZ/cc-pVTZ, Standard and 32 GiB numeric budget.

Every invocation also requires `--plan <final.json> --plan-sha256 <actual SHA>`
and a unique `--output <new attempt directory>`. This setup adapter bypasses only
the historical CLI transport wrapper by calling its unchanged scientific
functions; the original Slurm guards and production admission checks remain.

Use the protocol's upper bounds: at most three processes, at most 3,600 seconds
per process, at most 14,400 seconds for the Job, and at most 30 seconds of owned
cleanup. These are stopping bounds, not predicted runtimes or spending approval.
Automatic retries are zero. The coordinator may tighten these bounds before
submission. Keep one math thread and task-owned JIT/cache lease slots outside
the shared compiler-cache tree. Keep the verified compiler launcher available
for any actual runtime cache miss.

## Frozen root release gate

The coordinator requested a device/host-only boot followed by a maximum
300-second wait. The separate `gate-wait-v1/gate_wait.py` launcher is staged with
SHA-256 `6fe47489440dbc3975032eb75d9825cd25a3b6c017f44b1433afceeb7405f0f5`;
its sibling cleanup helper is
`8df2f022d90ffe47b71264446842456d5b5abcbfcf9742b81cb235b313c51bad`.
The entry script has SHA-256
`dd9ced61e6eb8ec60cc6da39bf248b2d3c94041f519944180ba8a4883ae559fe`.
The exact finite command is retained in `job-command.txt`.

Before release, the gate writes only `preflight.json` containing device/host
observations and starts no endpoint subprocess. It claims a fresh directory,
rejects a pre-existing marker, and waits at most 300 seconds. Root atomically
publishes `release.json` with schema
`generativeqc.h100-environment-release.v1`, `coordinator_chat`,
`preflight_sha256` (actual preflight file bytes), `plan_path`, and `plan_sha256`.
The gate refuses changed frozen inputs, increased caps, a non-RUNNING proof or
the excluded #2677 node `qb-prod-gpu095`. Only then can the three fresh-process
arms run. No resource lifecycle action is performed by this launcher.

Six synthetic CPU tests passed on qz in normal and optimized Python, including
real POSIX process timeout/reaping, three distinct fixture processes after
release, no process before release, stale marker and no-clobber behavior,
excluded-node/input refusal, and credential/URL redaction. The fixture outputs
are not actual allocation or scientific evidence. The entry passed `bash -n`
and was not executed. A separate two-sample inode/mtime/hash and `sed` audit
confirmed stable files after root observed an unexplained read failure. Root
must independently re-read these frozen bytes before platform dry-run and its
single submission.

## Outcomes and replay

Each call persists returned E/F and numeric byte/index originals before
acceptance checks. Output directories never overwrite a prior attempt. Preserve
stdout/stderr, errors, timeout/OOM/signal/exit outcome and absent-result status;
capture/redact connection URLs and credentials before logging. If the original
scientific function raises before returning data, report E/F as UNAVAILABLE.

CPU `replay.py <attempt directory>` verifies hashes, restores returned floating
values from originals and recomputes finite complete-E/F outcomes. It explicitly
leaves scientific acceptance INCOMPLETE. Apply the source-matched independent
oracle/gate vector separately, including quarter/CPU spot checks where applicable,
and preserve the C2 displaced-originals limitation from #2705. Setup timings are
excluded from performance cohorts. A blocked preflight or finite return does not
close #2726 by itself.
