# RTX 5090: #3064/#3065, aligned HF/KS defaults, 2026-11-11

Source `a50c80f627455a801e1e2196b26259e64e5e9de2`: #3064/#3065, aligned HF/KS
and public Standard defaults. `samples.json.xz` binds exact source/ELF,
software, allocation and cached-build identities.

| Mode | Cold wins | Warm wins | Qualified cold | Qualified warm |
| --- | ---: | ---: | ---: | ---: |
| Direct | 32/32 | 32/32 | 31/31 | 31/31 |
| DF | 31/32 | 32/32 | 27/27 | 27/27 |

All 64 pairs / 128 processes / 512 E+force calls succeed, with no retries or
old timings. Warm Direct ωB97M-V96: 9.39×. These are observations, not
statistical rankings or universal speedups. CBS is unchanged.
Cold DF HF96 remains slower: 17.860 s versus 17.328 s, with native target
iterations/Focks 11/12 versus reference iterations/effective potentials 10/11.

Four finite node1/main Slurm workers pair engines consecutively per 5090;
first-engine order is balanced 32/32, math threads=1. Cold includes setup,
SCF, forces and host export, excluding import/context startup and cache-miss
compilation. Warm is the median of three frozen-own-cold-density calls;
bars are min–max. Caches remain; native imports reject reference engines.
Profile, RHF/KS convergence and XC-order selectors are unset. GPU4PySCF
SCF/convergence/grids and the common full-grid-response force contract are unchanged.

The Standard arm omits `policy`, matching default `PrecisionCalculator` and
`PrecisionPolicy()`. No SCF, initial-guess, grid or tile controls are supplied.
Method/basis, spherical AOs and Direct/DF auxiliary inputs define the model;
these model choices must match when reproducing a plotted case.

Qualified restricted CUDA HF/KS energy and force endpoints share
`|delta E| < 1e-9` AND `||2 C_vir^T F[D] C_occ||_F < sqrt(1e-9)`, matching the
stock GPU4PySCF candidate gate; density change is diagnostic. Native canonical
publication proof, including DF RHF physical final-state checks, remains mandatory.
Strict/tighter E/D, physical-reference, no-DIIS and unsupported owners retain
their gates; DF RHF keeps AO DIIS history. See
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
`.artifacts/readme-public-default-20261111/` and the matching
`n1:/data/jzzeng/readme-public-default-20261111`.
The preceding 64/64 cold cohort is retained; every new-source point was reacquired.
The [default decision](../../../.agents/notes/implemented/compatibility/2026-11-11-public-default-standard.md)
retains older cohort identities, the Balanced mismatch and source validation.
Strict/default HF ablation has equal work counts: matching thresholds proves
neither equal trajectories nor causal speedup. Replay uses recorded scalar
errors, not fresh vector subtraction.
