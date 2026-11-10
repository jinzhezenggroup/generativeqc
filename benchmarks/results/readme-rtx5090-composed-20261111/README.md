# RTX 5090: combined #3064/#3065 — 2026-11-11

Measured source `6eb5fceda4d7c0e40aea371929eef8c05db773a8` combines master
`3c2c6a376`, #3064 `ebe21c712` and #3065 `3782e5124`. Exact source/ELF,
software, allocation and build/cache identities are in `samples.json.xz`.

| Mode | Cold wins | Warm wins | Qualified cold | Qualified warm |
| --- | ---: | ---: | ---: | ---: |
| Direct | 29/32 | 32/32 | 28/31 | 31/31 |
| DF | 27/32 | 32/32 | 23/27 | 27/27 |

All 64 pairs / 128 processes / 512 complete E+force calls succeed; no retries
or old timings. Eight cold slow points remain plotted, all at 96 atoms.
Warm Direct ωB97M-V96 speedup is 9.38×. CBS is unchanged.

Four finite node1/main Slurm workers pair engines consecutively on each assigned
5090, first-engine order balanced 32/32, math threads=1. Cold includes setup,
SCF, forces and host export, not import/context startup or cache-miss compilation.
Warm is median of three frozen-own-cold-density calls; bars are min–max.
Caches are retained. Native imports reject reference engines. Both native
selectors are unset; GPU4PySCF SCF/convergence/grids remain unchanged, with
the original common full-grid-response force contract.

Both engines independently qualify 58/64 pairs: E/atom ≤1e-5 Eh, Fmax ≤1e-4
and Frms ≤3e-5 Eh/Bohr. Failures remain all DF HF sizes and Direct/DF M06-2X3.
All 1024 cross-engine vector comparisons pass, not a substitute for these gates.

Replay scalar gates/figures with Matplotlib 3.10.9:

```bash
PYTHONPATH=python:. python -m benchmarks.readme_defaults_summary \
  --campaign benchmarks/results/readme-rtx5090-composed-20261111 \
  --output .artifacts/readme-composed-replay
```

Full-vector audits/builds/failures remain in
`.artifacts/readme-rtx5090-composed-20261111/` and the matching
`n1:/data/jzzeng/readme-rtx5090-composed-20261111`.
The [interpretation note](../../../.agents/notes/implemented/performance/2026-11-11-composed-readme-5090.md)
explains extra cold work, historical storage-only compaction and the separate
UKS test-hook prerequisite failure. Compact replay is not fresh vector subtraction.
