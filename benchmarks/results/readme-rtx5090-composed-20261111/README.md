# RTX 5090: #3064/#3065, aligned HF/KS defaults, 2026-11-11

Source `5a1ae34addac124423a22490d1d7dda0f90fc0f4` combines master
`978efb34d`, #3064 `ebe21c712`, #3065 `bd92afd10`, RKS orbital stopping,
compact DF RHF alignment and the concurrent formatter fix. `samples.json.xz`
records source/ELF, software, allocation and build/cache identities.

| Mode | Cold wins | Warm wins | Qualified cold | Qualified warm |
| --- | ---: | ---: | ---: | ---: |
| Direct | 32/32 | 32/32 | 31/31 | 31/31 |
| DF | 32/32 | 32/32 | 27/27 | 27/27 |

All 64 pairs / 128 processes / 512 E+force calls succeed, with no retries or
old timings. Warm Direct ωB97M-V96: 9.37×. These are observations, not
statistical rankings or universal speedups. CBS is unchanged.

Four finite node1/main Slurm workers pair engines consecutively per 5090;
first-engine order is balanced 32/32, math threads=1. Cold includes setup,
SCF, forces and host export, excluding import/context startup and cache-miss
compilation. Warm is the median of three frozen-own-cold-density calls;
bars are min–max. Caches remain; native imports reject reference engines.
Native RHF/KS convergence and XC-order selectors are unset. GPU4PySCF
SCF/convergence/grids and the common full-grid-response force contract are unchanged.

Qualified restricted CUDA HF/KS energy and force endpoints share
`|delta E| < 1e-9` AND `||2 C_vir^T F[D] C_occ||_F < sqrt(1e-9)`, matching the
stock GPU4PySCF candidate gate; density change is diagnostic. Native canonical
publication proof, including DF RHF physical final-state checks, remains mandatory.
Explicit tighter E/D requests,
`GENERATIVEQC_CUDA_RHF_CONVERGENCE=strict` and
`GENERATIVEQC_CUDA_KS_CONVERGENCE=strict`, physical-reference exports,
no-DIIS and unsupported owners retain strict stopping. Compact DF RHF uses
the same candidate norm with unchanged AO DIIS history. See
[application precision](../../../docs/user/application_precision.md) for the
policy. Default SCF thresholds do not promise `1e-7` force repeatability;
the original-grid regression selects historical strict policy with unchanged gates.

Both engines independently qualify 58/64 pairs: E/atom ≤1e-5 Eh, Fmax ≤1e-4
and Frms ≤3e-5 Eh/Bohr. Failures remain all DF HF sizes and Direct/DF M06-2X3.
All 1024 cross-engine vector comparisons pass, not a substitute for these gates.

Replay scalar gates/figures with Matplotlib 3.10.9:

```bash
PYTHONPATH=python:. python -m benchmarks.readme_defaults_summary \
  --campaign benchmarks/results/readme-rtx5090-composed-20261111 \
  --output .artifacts/readme-composed-replay
```

Full-vector and source-matched build/qualification receipts remain in
`.artifacts/df-hf-aligned-final-20261111/` and the matching
`n1:/data/jzzeng/df-hf-aligned-final-20261111`.
Prior 63/64 and 56/64 cold cohorts:
`e25b70c24d8464c32444bec25bccd33d2f494795` and
`28ac57e819405bd9bae06e8fae60ac25b98d14a1`.
Initial `0b74eb57f` cohort: ignored
`.artifacts/df-hf-aligned-20261111/` and matching node1 storage. Formatting
changed native identity; every final-source point was reacquired.
Same-image strict/default DF HF ablation has equal iteration/Fock counts
at every size: matching thresholds proves neither equal trajectories nor
causal speedup. The
[DF HF decision](../../../.agents/notes/implemented/numerics/2026-11-11-df-hf-orbital-stopping.md)
and [KS decision](../../../.agents/notes/implemented/numerics/2026-11-11-ks-default-orbital-stopping.md)
separate ordinary stopping from stringent force repeatability. Replay uses
recorded unrounded errors, not fresh vector subtraction.
