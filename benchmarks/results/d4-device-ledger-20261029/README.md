# D4 retained CUDA allocation authority

PR #2870 is an allocation/lifetime prerequisite for #2227 R2. Production source
`4fc8782cfa3b8ddbcaad8bbda4eed002e8aea5c8` is compared with master
`228a2a68c08d8b27ffec1798e9c3addfb200153f`. Every D4 buffer uses the shared
prepared allocator and matching release; partial construction remains owned
across host exceptions. This does not add global D4 ResourcePlan admission,
new scientific/force domains, or R3 work.

Release CPU/CUDA builds passed. CPU: 127 CTests and 18 focused Python checks.
Slurm 14847 completed on node1/main/RTX 5090 with exit 0 in 5m33s under a finite
one-hour allocation and unchanged assigned device visibility. Five native CUDA
programs cover independent D4 E/gradient/charge and finite-difference gates,
4,100-item ragged isolation and cooperative schedules. The production program
also exercises exact/below capacity, partial failures, host exception recovery,
cross-thread destruction and detached ledger retirement. Compute Sanitizer
reports zero errors and zero leaked bytes. All 21 selected GPU/Python checks pass.
Installed public-header CPU/CUDA SDK checks ran with no Python/compiler on PATH.

The matched complete E/E+F matrix covers PBE/PBE0 D4, RKS/UKS, scalar and ragged
fleet protocols, cold/warm/genuinely moved coordinates, construction and cleanup.
Eight alternating ABBA/BAAB groups retain all 1,792 calls. Maximum median time
ratio is 1.0042326378469892, below the unchanged per-case 1.05 limit. Work is
identical; max energy/force differences are 9.947598300641403e-14 Eh and
5.684341886080802e-14 Eh/bohr. Python correction-owner construction is poisoned.
These are matched-input non-regression results, not a new production ranking.

The complete untimed CUDA profile has identical 172,628 kernels, no graph calls,
5,648 explicit allocations, 119,201,475-byte explicit dynamic peak and zero final
live dynamic bytes. Static bytes and all copy counts/bytes are identical.
`profile-summary.json` states the exclusions and profiler database hashes.

`identities.json` binds binaries, sources, controls, scripts and local debug
receipts. `qualification/run-gpu.sh` preserves the exact environment/commands;
submit it through a finite `srun` after reconstructing its frozen installations.
`qualification/qualify.py --help` exposes the source/library arguments. It uses
(24,12,24) grids and def2-SVP spherical; independent correction references have
their own fixture inputs at the recorded production source. The CPU code path
was unchanged by this CUDA-only cutover; no CPU speedup claim is made.

Raw accepted samples remain in the existing ancestor identified by
`history.json`. Recover without network or remeasurement:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/d4-device-ledger-20261029/history.json \
  --output .artifacts/d4-ledger-restore
```

All original sample bytes, order and failed/successful local diagnostic populations
are preserved. `storage.json` records lossless compression of one historical
CPU-autotune JSON; its values and acceptance are unchanged. This keeps the 80 MiB
checkout budget without publishing a Release or moving evidence to another host.
