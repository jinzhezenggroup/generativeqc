# Native DFT shared-Rys XY streaming qualification

Run `PYTHONPATH=.:python python benchmarks/results/dft-shared-rys-streaming-20261108/replay.py`.
Replay uses complete retained FP64 energy/force/settings/work receipts and the
unchanged independent README dense-GPU, CPU/GPU spot and grid-stability oracles.
No GPU, reference execution, network or `.git` lookup is needed.

| Direct96 complete cold mean | Control | Candidate | Change |
|---|---:|---:|---:|
| A100 PBE0 | 12.686775 s | 11.760297 s | -7.30% |
| A100 B3LYP | 12.864098 s | 11.876885 s | -7.67% |
| 5090 PBE0 | 16.803305 s | 16.909077 s | +0.63% |
| 5090 B3LYP | 17.226281 s | 17.207576 s | -0.11% |

LSF229664 and Slurm16855 run fresh-process ABBA, two cold endpoints per arm
and three warms per process. Every paired warm meets the unchanged3% guard.
Target iterations/Focks remain `[9,1,1,1]`, initial guess separately reported.
All native parity and original independent accuracy gates pass; no looser grids,
convergence, precision, CPU production oracle or interposer is used. A cold-only
collector proves six shared-XY kernels each execute nine times on A100 and zero
times on5090. Quartet/recurrence totals were not acquired.

Source `7fbcd2d2e35fac47b64517f70bba10e6af2d65c3e74112801a6c159a5a7bc51c`,
normal Release ELF `9a37512dc5f96f8a5d0ca78cae46c33323cb8a693620ff3bc93958f33c82534f`,
build16854. Source receipts preserve the tested DFT-only overlay and control
identity rather than including unfinished HF adaptive/default work.

The independent positive raw-K fixture passes144 matrix predicates on each
card, with all5090 sanitizer gates passing. Reproduce source preparation with
`PYTHONPATH=.:python python -m tools.generativeqc_validation.shared_rys_xy_fixture --consumer raw-k --classes dppp pppp dpds ddps dpdp ddpp --output /tmp/raw-k-fixture`.
Compilation and actual GPU execution require separately scheduled allocations.
Original executed oracle arrays are retained. The formal source preparer emits
the same executable body; regeneration on the current BLAS changes independent
array last bits by at most1.39e-17, not an extra GPU execution claim.

## Boundaries

B3LYP still trails the same-allocation GPU4PySCF11.637084 s by about2.06%.
PBE0 reference cold142.657065 s is retained as an unexplained outlier; it is
not a basis for declaring a tenfold win. This is not all-README leadership or
new DF/HF/omega-B97M-V qualification. Weighted-Z admission is unchanged.

The checked-in capsule includes every final native/scientific observation,
independent raw-K array, sanitizer result and normal-build identity/configuration/
cache/restore receipt. Full compiler stdout remains checksum/size-indexed in the
original ignored build archive; the complete diagnostic capsule additionally
retains it and the earlier interposer/weighted-Z/attribution pilots and unmeasured
intermediate builds. None is deleted or relabelled as final qualification.
No Release, external asset or budget increase is used. Historical seed evidence
is losslessly repacked, with every original text byte authenticated, to make room.

Decisions and further limits are recorded in issue2976 comment6722254219 and
`.agents/notes/implemented/performance/2026-11-08-dft-streaming-shared-rys-xy.md`.
