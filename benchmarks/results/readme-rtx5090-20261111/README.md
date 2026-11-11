# RTX 5090 HF/DFT default refresh — 2026-11-11

Implementation `8c9acd542840f5c059ce866dc2e9f11c0aaf245d`, based on master
`d35ca77a4c7142f8d6d3f0cbc1e8e96718d9af30`: 8 methods × Direct/DF ×
3/12/48/96 atoms; 128 selected engine processes, 512 complete energy+force
endpoints. These single-process observations are not statistical rankings.

| Mode | Cold wins | Warm wins | Qualified cold wins | Qualified warm wins |
| --- | ---: | ---: | ---: | ---: |
| Direct | 32/32 | 32/32 | 31/31 | 31/31 |
| DF | 30/32 | 32/32 | 26/27 | 27/27 |

Direct HF48/96 use 10/11 cold physical Focks (strict: 14/15 in the prior
cohort). All cases remain plotted, including DF HF96 cold (37.030 vs 34.913 s)
and DF r2SCAN-3c96 cold (23.548 vs 23.229 s). Direct ωB97M-V96 warm is
**9.37×** faster. CBS is unchanged.

## Conditions and accuracy

Cold includes construction, preparation, SCF, analytic forces and host export.
Warm is the median of three complete calls from each engine's frozen cold
density; bars show min–max. Caches are retained, not emptied. Native processes
reject reference-engine imports. Finite Slurm jobs on node1/main pair engines
consecutively on the same assigned RTX 5090, balanced first-engine order 32/32,
math threads=1. Release, CUDA 12.9.1, SM120, fast compile off, device link on,
sccache 0.16.0. PySCF 2.14.0, GPU4PySCF 1.9.0, CuPy 13.6.0, NumPy 2.5.4.

GPU4PySCF settings are entirely unchanged. Unset native convergence selects
owner-bound physical MO-gradient convergence for standard-control all-electron
FP64 Direct CUDA RHF; tighter controls and unsupported default owners retain
strict behavior. `GENERATIVEQC_CUDA_RHF_CONVERGENCE=strict` restores AO-max
convergence. DF HF/KS policies, default XC/VV10 grids and full grid response
are unchanged. This is not a 1e-7 maximum-force guarantee.

Unchanged independent gates qualify **58/64 pairs** for both engines: energy
≤1e-5 Eh/atom, max force ≤1e-4 Eh/Bohr, RMS force ≤3e-5 Eh/Bohr. Failures:
DF HF at all sizes; Direct/DF M06-2X at 3 atoms. All 1024 cross-engine vector
comparisons pass but do not replace the independent gates. The
[historical oracles](../readme-refresh-20261026/README.md) are reused only after
exact scientific-input and explicit environment checks; original quarter-gate
stability and CPU/GPU checks remain. No old timing or weaker gate is used.

## Failed-pair retry and replay

The original GPU4PySCF DF ωB97M-V48 process hit `cudaErrorLaunchTimeout` on its
second warm call (exit -6). Exactly one fresh **paired** retry succeeds, with
the original engine order and unchanged inputs/settings/image. Both original
native four and reference two completed calls remain in
`superseded_populations`; no selection is based on timing. Total acquisition:
130 processes, one failed process, 518 completed calls; selected figures: 512.

`samples.json.xz` retains every unrounded timing/error/work sample and raw
receipt hash, plus allocations, software, build/cache and identity-recheck
receipts. Replay with Matplotlib 3.10.9 from the repository root:

```bash
PYTHONPATH=python:. python -m benchmarks.readme_defaults_summary \
  --campaign benchmarks/results/readme-rtx5090-20261111 \
  --output .artifacts/readme-rtx5090-replay
```

This replays scalar gates/plots, not fresh full-vector subtraction. Full
vectors, the independent audit, original failures and build logs remain in
`.artifacts/readme-rtx5090-default-20261110/`; the private acquisition root is
`n1:/data/jzzeng/readme-rtx5090-default-20261110-d35ca77a4`. The
[decision note](../../../.agents/notes/implemented/numerics/2026-11-10-default-hf-orbital-gradient.md)
records device/strict-override guards and the corrected outer-shell identity
check; expected source and measured ELF identities match.
