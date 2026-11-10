# RTX 5090: combined #3064/#3065 and aligned RKS defaults — 2026-11-11

Measured source `886556ce5504dffa10679e624c963e795eb12e64` combines master
`3c2c6a376`, #3064 `ebe21c712`, #3065 `3782e5124` and default RKS orbital
stopping. Exact source/ELF, software, allocation and build/cache receipt
identities are in `samples.json.xz`.

| Mode | Cold wins | Warm wins | Qualified cold | Qualified warm |
| --- | ---: | ---: | ---: | ---: |
| Direct | 32/32 | 32/32 | 31/31 | 31/31 |
| DF | 31/32 | 32/32 | 27/27 | 27/27 |

All 64 pairs / 128 processes / 512 complete E+force calls succeed; no retries
or old timings. The only cold slow point remains plotted: DF HF96 takes
18.182 s native versus 17.742 s GPU4PySCF. Warm Direct ωB97M-V96 speedup is
9.35×. These are individual observations, not statistical rankings or a
universal speedup guarantee. CBS is unchanged.

Four finite node1/main Slurm workers pair engines consecutively on each assigned
5090, first-engine order balanced 32/32, math threads=1. Cold includes setup,
SCF, forces and host export, not import/context startup or cache-miss compilation.
Warm is median of three frozen-own-cold-density calls; bars are min–max.
Caches are retained. Native imports reject reference engines. Native RHF/KS
convergence and XC-order selectors are unset; GPU4PySCF SCF/convergence/grids remain unchanged, with
the original common full-grid-response force contract.

Qualified restricted CUDA RKS energy-only and force endpoints now share
`|delta E| < 1e-9` AND `||2 C_vir^T F[D] C_occ||_F < sqrt(1e-9)`, matching the
stock GPU4PySCF SCF candidate norm and threshold. Density change is diagnostic,
not an additional candidate stopping gate. The separate native canonical
publication proof remains mandatory. Explicit tighter E/D requests,
`GENERATIVEQC_CUDA_KS_CONVERGENCE=strict` and full-range physical-reference
exports retain strict admission; unsupported owners keep their existing gates.
The compact DF HF owner is unchanged. See
[application precision](../../../docs/user/application_precision.md) for the
complete policy. Default SCF thresholds do not promise `1e-7` force repeatability;
that stringent original-grid regression explicitly selects the historical
strict policy and retains its original numerical gates.

Both engines independently qualify 58/64 pairs: E/atom ≤1e-5 Eh, Fmax ≤1e-4
and Frms ≤3e-5 Eh/Bohr. Failures remain all DF HF sizes and Direct/DF M06-2X3.
All 1024 cross-engine vector comparisons pass, not a substitute for these gates.

Replay scalar gates/figures with Matplotlib 3.10.9:

```bash
PYTHONPATH=python:. python -m benchmarks.readme_defaults_summary \
  --campaign benchmarks/results/readme-rtx5090-composed-20261111 \
  --output .artifacts/readme-composed-replay
```

Full-vector audits and source-matched build/qualification receipts remain in
`.artifacts/readme-rtx5090-aligned-20261111/` and the matching
`n1:/data/jzzeng/readme-rtx5090-aligned-20261111`.
The preceding 56/64-cold cohort remains recoverable at commit
`28ac57e819405bd9bae06e8fae60ac25b98d14a1`; no historical timings supply this
new cohort. The [default-policy decision](../../../.agents/notes/implemented/numerics/2026-11-11-ks-default-orbital-stopping.md)
separates ordinary stopping from stringent force repeatability. Compact replay
recomputes recorded unrounded scalar errors; it is not fresh vector subtraction.
