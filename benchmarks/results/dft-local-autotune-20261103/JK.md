# Native full-range DFT J/K policy qualification

`jk-receipts.json.xz` retains every candidate's paired samples, independent
PySCF/libxc E/F oracle, native allocation ledger, observed work, fresh incumbent
holdout and actual accepted-recipe replay. Source commit: `9cc91893f`, stacked
on the XC work. Slurm job 1443 runs on n4 RTX 5090/SM120 with CUDA 12.9.86 and a
normal source-matched Release library, fast compilation disabled.

All three water/def2-svp spherical, Treutler-level-3 workloads select the native
policy `GENERATIVEQC_DISABLE_MD_J=1` with
`GENERATIVEQC_DIRECT_J_FOCK_LOWERING=rys`. Native capability/bucket fallbacks
remain enabled. Exchange schedule/lowering candidates are also timed, but
none supersedes this Coulomb-policy winner; this is not evidence of faster K.

| Workload | Fresh holdout baseline → candidate, ms | Speedup | Lower 95% |
| --- | --- | ---: | ---: |
| PBE-RKS, neutral singlet | 82.379 → 71.069 | 1.159 | 1.151 |
| PBE0-RKS, neutral singlet | 113.435 → 102.822 | 1.103 | 1.091 |
| PBE0-UKS, cation doublet | 218.904 → 196.401 | 1.115 | 1.112 |

Every row uses five alternating-order pairs of frozen-density warm complete
energy plus analytic-force replays. Preparation, cold convergence and warmup
are excluded. The selected native binary and bounded ResourceBudget XC policy
are fixed on both sides; this does not benchmark combining XC and J/K winners.
Every sample passes unchanged work/XC, independent numerical, strict schedule
parity, translation and complete allocation gates. Native accepted recipes
also replay successfully with source/device/library and raw-holdout checks.

The opt-in GPU suite independently qualifies every proposed native policy for
semilocal RKS, hybrid RKS and hybrid UKS (3 parametrized tests). The focused
CPU suites pass 106 tests with 9 optional GPU cases skipped locally. Full local
pre-commit ownership/contracts/retention checks pass.

To reproduce, extract each report's `workload` geometry (Bohr), charge/spin,
basis and batch. Under a finite Slurm allocation run the matching checkout's
`python -m generativeqc.tune_cli` with `--component jk`, `--grid-level 3`,
`--representation spherical`, `--repeats 5`, `--max-candidates 6` and an isolated
profile cache. Replay `qualified_recipe` with
`python -m generativeqc.execution_tune recipe.json replay.json` in a new finite
Slurm job before rebuilding its referenced library. No recipe changes global
Calculator policy or the active profile index. No speedup is generalized to
other molecules, basis sets, phases or GPUs.
