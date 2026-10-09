# Quartet-parallel Rys-K: complete PBE0 Cold qualification

Measured on 2026-10-08, frozen GenerativeQC base
`4385f72751b829883407c01106186917c344317b`, RTX 5090 on the approved n1 host
(Slurm node `node1`, partition `main`, `--gres=gpu:5090:1`). Every GPU operation
runs under finite `srun`, without overriding Slurm device visibility.

These receipts measure **explicit opt-in**, not default promotion. The original working
checkout has different VibeQC history and unrelated pending changes; the
implementation is in `/data/jzzeng/qc-rys-task-k-20261008`. At qualification,
no commit, PR, release or tag had been created. The paired controls use the same candidate-enabled binary,
with incumbent mathematics byte-identical to the frozen base.

Later actual-default qualification is separately labeled in the
[five-class frozen-base report](../rys-task-default-20261008/README.md) and the
[three-root current-base report](../rys-task-master-20261008/README.md).

## Complete endpoint results

All times include calculator/context construction, preparation, the first
strict SCF, analytic forces, host force return and owner teardown. Imports and
input decoding are excluded. Each sample has a fresh process/calculator/owner,
no supplied density and no CUDA priming. Persistent compiler/artifact caches
are reused equally; this is execution-Cold, not a cache-empty compilation test.

PBE0 RKS, spherical offline def2-SVP, 48 radial x 16 polar x 32 azimuth grid,
FP64, screening 1e-12, energy tolerance 1e-12, density tolerance 1e-10, max 100
iterations. The 96-atom water cluster has 768 AOs. J remains the default MD-J.

| Cohort | Fresh processes per mode | Incumbent median (s) | Rys-task median (s) | Less complete time | Slurm job |
| --- | ---: | ---: | ---: | ---: | --- |
| 96 atoms, holdout | 5 | 113.301060 | 109.245257 | 3.58% | 6644 |
| 96 atoms, independent confirmation | 3 | 113.489543 | 109.482571 | 3.53% | 6653 |
| 48 atoms, holdout | 3 | 51.198094 | 48.740725 | 4.80% | 6646 |

Means also improve: 116.668951 to 109.232215 s (96 holdout), 116.242208 to
109.369306 s (96 confirmation), and 51.196319 to 48.699103 s (48).

Actual Fock counts, without excluding slow trajectories:

- 96 holdout: incumbent `[17,17,19,19,17]`, candidate `[17,17,17,17,17]`.
- 96 confirmation: incumbent `[17,17,19]`, candidate `[17,17,17]`.
- 48: both modes `[19,19,19]`.

Both 96 medians correspond to 17-Fock samples; the 17-Fock control subset in
the first cohort also shows about 3.1% complete-time benefit. This subset is
diagnostic only: every 19-Fock sample remains in the aggregate. Do not equate
the kernel benefit to all of the mean benefit or pool different Slurm cohorts
through a summarizer that requires matched allocation identity.

Every complete sample passes the cached independent GPU4PySCF oracle. Across
all 22 samples, maximum errors are 9.09e-12 Ha energy and 3.79e-11 Ha/Bohr force,
below the 1e-8 / 1e-7 gates. Exact reference provenance, versions, protocol and
forces are retained in `reference-48.json` and `reference-96.json`.

## Separate fixed-density diagnostic

Six processes, three repeats each, sequence incumbent/old-Rys/new-Rys/new-Rys/
old-Rys/incumbent, Slurm job 6646, same library/device/input matrices. Wall time
includes upload, transform, K, projection, export and synchronization; owner
preparation is excluded. Event/census observations are outside clean timing.

| Case | Incumbent K (s) | Old component Rys K (s) | New selected Rys K (s) | Admitted quartets |
| --- | ---: | ---: | ---: | ---: |
| 48 atoms | 0.959678 | 1.609953 | 0.825779 | 23,480,495 |
| 96 atoms | 1.683338 | 4.369086 | 1.456879 | 81,907,624 |

The selected new K takes 13.45% less time at 96 atoms. Per-class admission
vectors match all three paths, including density scales 1, 1e-3, 1e-6 and a
zero-admission 1e-14 case; matrix oracle errors remain below 1e-8. Equal shell
admission does **not** prove equal primitive/root work, and the alternative
inventories differ. These diagnostic times are not added to Cold times.

## Implementation and gates

The selected classes are `psps,ppps,dsss,dpss,dsps`. A complete quartet belongs
to one lane of a 32-thread packed CTA, without primitive/root-loop barriers.
The existing pair cache, K screening and cross-chunk queue remain authoritative.
The shared Rys axis program supplies all recurrence algebra. Restricted raw K
contracts locally before atomic publication; other consumers/spins retain the
generic scatter. Unsupported/unselected classes retain the incumbent.

- 384 Libcint matrix cases pass: eight capability classes, three geometry/
  primitive variants, RHF/UHF, combined/J/raw-K/HF-K, and one/33 task tails.
- Memcheck/synccheck: zero errors. Racecheck: zero hazards/errors/warnings.
  Each tool covers 64 selected matrix cases under job 6646.
- Integrated public and prepared-SCF K: 48 inputs pass all/five/none class
  masks, both spins, asymmetric/symmetric/full/scaled/zero/restored densities,
  and environment mutation after preparation; maximum error 6.37e-12.
- CPU/compiler suite: 260 passed, 20 skipped; final focused checks: 24 passed.
  Compiler ownership: 497 modules, zero errors; shared SCF: 223, zero errors.
- All retained incumbent/old-Rys/old-K-block emitted source hashes match the
  frozen base. The intentional bundle/registry addition has reviewed golden
  hashes. Final Python formatting leaves generated CUDA/registry bytes unchanged.

Nsight Systems per-class diagnostic profiles reject `ssss` and `ppss` and show
only a minor `psss` benefit. Selecting all eight classes is not a winner: the
earlier full-inventory feasibility Cold regresses 113.55/17 Focks to
129.53/21 Focks. V1's approximately 1.2% median improvement also has a regressing
mean. These development results are not omitted from the rationale.

NCU counter collection fails with `ERR_NVGPUCTRPERM`. Source confirms removal
of inner-loop barriers, but active lanes, barrier-stall contribution and
writeback-cost fractions have **not** been quantified. No global driver or
counter permission change is attempted. An earlier selector-build failure used
a stale library; that invalid cohort is quarantined remotely under
`results/invalid-stale-selector/` and excluded for build provenance, not timing.

## Reproduce

Apply the qualified source capsule in `validation.tar.gz` to the frozen base;
later checkout schedules have separate qualification and do not reproduce this
exact binary. Build with ccache and the `sm_120` AOT profile, including
`generativeqc_stationary_pbe0_rks_manifest` and
`generativeqc_stationary_pbe0_rks_spd_manifest`. Keep both force companion
libraries/manifests beside `libgenerativeqc.so`; copying just the main library
does not reproduce a complete force endpoint. Do not clear compiler caches.

Set `PYTHONPATH` to `python:.`, `GENERATIVEQC_LIBRARY` to the build and native
provider-library paths appropriately. Inside one finite Slurm allocation:

```bash
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export GENERATIVEQC_AOT_RYS_TASK_FOCK_SHELL_CLASSES=psps,ppps,dsss,dpss,dsps
for item in incumbent0 rys-task0 rys-task1 incumbent1 incumbent2 rys-task2; do
  python -m benchmarks.rys_task_cold --mode "${item%[0-9]}" --atoms 96 \
    --reference benchmarks/results/rys-task-cold-20261008/reference-96.json \
    --output "$item.json"
done
python -m benchmarks.rys_task_cold --samples incumbent*.json rys-task*.json \
  --output summary.json
```

Run that shell using, for example, SSH n1 with
`srun --partition=main --nodelist=node1 --gres=gpu:5090:1 --nodes=1 --ntasks=1 --cpus-per-task=16 --time=00:20:00`.
Each worker explicitly selects its mode, so the control must not inherit a
candidate mode from an already-prepared owner. For ordinary candidate use set
both `GENERATIVEQC_DIRECT_K_FOCK_LOWERING=rys-task` and the five-class selector.
`incumbent` is the rollback. `rys` is the old, separate experiment.

## Retained evidence

- `summary.json`: complete gated cohorts, all timings/Fock counts, device/build
  identity, fixed-density summaries and artifact SHA256s.
- `observations.tar.gz`: all endpoint records/forces/diagnostics, all fixed-K
  records/counters, binary mask preflight, integrated gates, sanitizer results,
  independent references and Nsight/NCU logs.
- `validation.tar.gz`: CPU/structure/source-hash receipts, qualified changed
  compiler/native source capsule and exact remote driver/probe scripts.
- Remote ignored artifacts: `/data/jzzeng/qc-rys-task-k-20261008/results` on n1;
  fixed-density inputs remain under `source/.artifacts/inputs` there. NPZ SHA256:
  48 `b7c6c88b1b846876252a19c698dfbd78e9994adf473de2c6f20d7068abee398a`,
  96 `adab459881fe707c33fc05f53c6fd5ae8f064746abefe88869d48e31676caccb`.

Qualified library SHA256:
`7fe0cec3a72ceed6d35c65da69b909d30ca19a5e67dfc12e3cbb70b0189de030`.
The prepared preflight masks are five-class 1236, all-capability 1247 and none 0.
The final formatter touches only a Python constant's layout; the original
qualified compiler/native inputs are preserved in the source capsule.

See the [current contract](../../../docs/developer/direct_rys_tasks.md) and
[durable rationale](../../../.agents/notes/implemented/performance/2026-10-08-rys-task-k.md).

## Exact historical archive recovery

The observations and validation archives are retained byte-for-byte in Git at
`dee3d522b71df9fb55c91478df3c44b7af454488`. Their complete member names, sizes and hashes,
archive hashes and copyable recovery commands are recorded in
[`../unified-k-work-20261009/storage-recovery.json`](../unified-k-work-20261009/storage-recovery.json).
Recover those archives before following the historical extraction commands above.
This storage compaction changes no observation, gate or qualification claim.
